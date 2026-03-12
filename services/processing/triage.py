"""
Triage Service — Classifies documents by type, department, and quality.
Uses a 4-tier classification system:
  1. Folder path (95% confidence)
  2. Filename keywords (85-92%)
  3. Text content heuristics (75-88%)
  4. LLM-based classification for unknown documents (80-85%)
Auto-discovers new categories when content doesn't match known patterns.
"""
import re
from typing import Dict, Any, Tuple, List
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class TriageService:
    """
    Analyzes document text and structure to automatically classify document types,
    departments, and calculate quality metrics. Can create new categories dynamically.
    """

    # Known types that map to departments
    TYPE_DEPT_MAP = {
        "invoice": "finance",
        "receipt": "finance",
        "purchase_order": "operations",
        "shipping_order": "operations",
        "inventory_report": "operations",
        "contract": "legal",
        "agreement": "legal",
        "policy": "hr",
        "report": "general",
        "email": "general",
        "form": "general",
        "memo": "general",
        "presentation": "general",
        "spreadsheet": "finance",
        "manual": "operations",
        "specification": "operations",
        "proposal": "general",
        "letter": "general",
        "certificate": "hr",
        "resume": "hr",
        "unknown": "general",
    }

    KNOWN_DEPTS = {"finance", "legal", "hr", "operations", "general", "engineering", "marketing", "sales"}

    def __init__(self, quality_threshold: float = 0.60):
        self.quality_threshold = quality_threshold

    def _run_inference(self, text: str, file_name: str, file_path: str = "") -> Tuple[str, str, float]:
        """
        Core classification engine using folder path, filename, text heuristics,
        and LLM fallback for unknown documents.
        Returns: (document_type, department_category, triage_confidence)
        """
        text_lower = text.lower()[:3000]  # Cap for performance
        file_lower = file_name.lower()
        path_lower = file_path.lower()

        # ── Priority 1: Folder path (most reliable) ──────────────────
        folder_match = self._classify_by_folder(path_lower)
        if folder_match:
            return folder_match

        # ── Priority 2: Filename keywords ─────────────────────────────
        filename_match = self._classify_by_filename(file_lower, text_lower)
        if filename_match:
            return filename_match

        # ── Priority 3: Text content heuristics ──────────────────────
        content_match = self._classify_by_content(text_lower)
        if content_match:
            return content_match

        # ── Priority 4: LLM classification for unknown docs ──────────
        llm_match = self._classify_by_llm(text_lower, file_lower)
        if llm_match:
            return llm_match

        return "unknown", "general", 0.50

    def _classify_by_folder(self, path_lower: str):
        """Priority 1: Folder path detection."""
        folder_signals = {
            ("shipping",): ("shipping_order", "operations"),
            ("purchase",): ("purchase_order", "operations"),
            ("invoice", "invoices", "billing"): ("invoice", "finance"),
            ("inventory", "stock", "warehouse"): ("inventory_report", "operations"),
            ("contract", "contracts", "legal", "agreements"): ("contract", "legal"),
            ("hr", "human_resource", "personnel", "employee"): ("policy", "hr"),
            ("report", "reports", "analytics"): ("report", "general"),
            ("receipt", "receipts"): ("receipt", "finance"),
            ("proposal", "proposals"): ("proposal", "general"),
            ("manual", "manuals", "documentation"): ("manual", "operations"),
            ("marketing", "campaigns"): ("report", "marketing"),
            ("sales",): ("report", "sales"),
            ("engineering", "technical"): ("specification", "engineering"),
        }

        for keywords, (doc_type, dept) in folder_signals.items():
            for kw in keywords:
                if f"/{kw}" in path_lower or f"\\{kw}" in path_lower:
                    return doc_type, dept, 0.95

        return None

    def _classify_by_filename(self, file_lower: str, text_lower: str):
        """Priority 2: Filename keyword matching."""
        filename_rules = [
            (["invoice", "inv_", "receipt", "bill_"], "invoice", "finance", 0.92),
            (["purchase", "po_", "procurement"], "purchase_order", "operations", 0.90),
            (["shipping", "ship_", "delivery", "dispatch"], "shipping_order", "operations", 0.88),
            (["stock", "inventory", "warehouse"], "inventory_report", "operations", 0.85),
            (["contract", "agreement", "nda", "mou"], "contract", "legal", 0.88),
            (["policy", "guidelines", "handbook"], "policy", "hr", 0.85),
            (["report", "summary", "q1", "q2", "q3", "q4", "annual"], "report", "general", 0.80),
            (["resume", "cv_", "curriculum"], "resume", "hr", 0.88),
            (["proposal", "rfp", "quotation", "quote"], "proposal", "general", 0.85),
            (["memo", "memorandum", "notice"], "memo", "general", 0.82),
            (["manual", "guide", "instruction"], "manual", "operations", 0.82),
            (["spec", "specification", "requirement"], "specification", "engineering", 0.85),
            (["certificate", "cert_", "license"], "certificate", "hr", 0.85),
            (["presentation", "ppt", "slide"], "presentation", "general", 0.80),
        ]

        for keywords, doc_type, dept, conf in filename_rules:
            if any(kw in file_lower for kw in keywords):
                return doc_type, dept, conf

        return None

    def _classify_by_content(self, text_lower: str):
        """Priority 3: Text content regex patterns."""
        content_rules = [
            # Finance
            (r'\b(invoice\s*(number|no|#|date)|bill\s*to|amount\s*due|payment\s*terms|totalprice|balance\s*due|subtotal|tax\s*amount)\b',
             "invoice", "finance", 0.85),
            (r'\b(purchase\s*order|po\s*number|vendor\s*name|delivery\s*date|ordered\s*by)\b',
             "purchase_order", "operations", 0.82),
            (r'\b(shipping\s*(details|order|date)|ship\s*(name|to|from|date)|tracking\s*(number|id)|bill\s*of\s*lading|consignee)\b',
             "shipping_order", "operations", 0.82),
            (r'\b(stock\s*report|units\s*(sold|in\s*stock|ordered)|inventory\s*(level|count|report)|warehouse|reorder\s*point)\b',
             "inventory_report", "operations", 0.80),
            # Legal
            (r'\b(hereby|whereas|parties\s*agree|confidentiality|termination\s*clause|indemnif|governing\s*law|jurisdiction)\b',
             "contract", "legal", 0.82),
            # HR
            (r'\b(employee\s*handbook|code\s*of\s*conduct|benefits|leave\s*policy|compensation|performance\s*review)\b',
             "policy", "hr", 0.80),
            # Reports
            (r'\b(executive\s*summary|key\s*findings|analysis|conclusion|recommendations|quarterly\s*report|annual\s*report)\b',
             "report", "general", 0.75),
        ]

        for pattern, doc_type, dept, conf in content_rules:
            if re.search(pattern, text_lower):
                return doc_type, dept, conf

        return None

    def _classify_by_llm(self, text_lower: str, file_lower: str):
        """
        Priority 4: Use Groq LLM for documents that don't match any heuristic.
        Extracts category from content intelligently.
        """
        if not text_lower.strip() or not settings.GROQ_API_KEY:
            return None

        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)

            # Take first 1500 chars for classification
            sample = text_lower[:1500]
            known_types = ", ".join(sorted(self.TYPE_DEPT_MAP.keys()))
            known_depts = ", ".join(sorted(self.KNOWN_DEPTS))

            response = client.chat.completions.create(
                model=settings.GROQ_MODEL_NAME,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a document classifier. Given document text, respond with EXACTLY two words:\n"
                        "DOCUMENT_TYPE DEPARTMENT\n\n"
                        f"Known types: {known_types}\n"
                        f"Known departments: {known_depts}\n\n"
                        "If the document fits a known type, use it. "
                        "If it's a new type not listed, create a descriptive snake_case type name. "
                        "Respond with ONLY two words, nothing else."
                    ),
                }, {
                    "role": "user",
                    "content": f"Filename: {file_lower}\nContent:\n{sample}",
                }],
                temperature=0.0,
                max_tokens=20,
            )

            answer = response.choices[0].message.content.strip().lower()
            parts = answer.split()

            if len(parts) >= 2:
                doc_type = re.sub(r'[^a-z0-9_]', '', parts[0])
                dept = re.sub(r'[^a-z0-9_]', '', parts[1])

                # Validate department, fall back to general
                if dept not in self.KNOWN_DEPTS:
                    dept = "general"

                # Auto-register new type
                if doc_type and doc_type not in self.TYPE_DEPT_MAP:
                    self.TYPE_DEPT_MAP[doc_type] = dept
                    logger.info(f"Auto-discovered new document type: '{doc_type}' → dept '{dept}'")

                if doc_type:
                    return doc_type, dept, 0.80
            elif len(parts) == 1:
                doc_type = re.sub(r'[^a-z0-9_]', '', parts[0])
                dept = self.TYPE_DEPT_MAP.get(doc_type, "general")
                if doc_type:
                    return doc_type, dept, 0.75

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return None

    def _assess_quality(self, text: str, original_length: int) -> Tuple[float, bool, List[str]]:
        """
        Calculates a deterministic quality score based on text properties.
        Returns: (quality_score, is_degraded, detected_issues)
        """
        issues = []

        if original_length == 0 or not text.strip():
            return (0.0, True, ["empty", "unclassified"])

        score = 1.0

        if original_length < 50:
            score -= 0.3
            issues.append("low_text")

        alphanumeric_count = sum(c.isalnum() or c.isspace() for c in text)
        clean_ratio = alphanumeric_count / original_length if original_length > 0 else 0

        if clean_ratio < 0.8:
            score -= (1.0 - clean_ratio)
            issues.append("noisy_ocr")

        final_score = max(0.0, min(1.0, round(score, 2)))
        is_degraded = final_score < self.quality_threshold

        if not issues:
            issues.append("none")

        return final_score, is_degraded, issues

    def process_document(self, ocr_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline method. Intercepts the payload, runs classification,
        and securely attaches metadata.
        """
        result = ocr_payload.copy()

        text = result.get("text", "")
        file_name = result.get("file_name", "")
        file_path = result.get("file_path", "")
        text_length = result.get("text_length", 0)

        try:
            doc_type, dept, conf = self._run_inference(text, file_name, file_path)

            q_score, is_degraded, issues = self._assess_quality(text, text_length)

            result["document_type"] = doc_type
            result["department_category"] = dept
            result["quality_score"] = q_score
            result["is_degraded"] = is_degraded
            result["detected_issues"] = issues
            result["triage_confidence"] = conf
            result["status"] = "success"

        except Exception as e:
            result["status"] = "failed"
            result["error_message"] = f"Triage Error: {str(e)}"
            result["document_type"] = "unknown"
            result["department_category"] = "general"
            result["quality_score"] = 0.0
            result["is_degraded"] = True
            result["detected_issues"] = ["triage_failure"]
            result["triage_confidence"] = 0.0

        return result
