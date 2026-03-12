"""
OCR Service — Extracts text from images and scanned PDFs using Tesseract.
Runs conditionally only when the parser detects insufficient text.
"""
from pathlib import Path
from typing import Dict, Any
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class OCRService:
    """
    Uses Tesseract OCR to extract text from images and scanned documents.
    Activates only when needs_ocr=True (text below MIN_TEXT_THRESHOLD).
    """

    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self._available = None

    def _check_available(self) -> bool:
        """Check if Tesseract + Pillow are available."""
        if self._available is not None:
            return self._available
        try:
            import pytesseract
            from PIL import Image
            # Quick check that tesseract binary exists
            pytesseract.get_tesseract_version()
            self._available = True
            logger.info("Tesseract OCR available")
        except Exception as e:
            self._available = False
            logger.warning(f"Tesseract OCR not available: {e}. Image text extraction disabled.")
        return self._available

    def _run_ocr_on_image(self, file_path: Path) -> str:
        """Extract text from a single image file."""
        import pytesseract
        from PIL import Image

        img = Image.open(file_path)
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, lang=self.lang)
        return text.strip()

    def _run_ocr_on_pdf(self, file_path: Path) -> str:
        """Extract text from scanned PDF by rendering pages to images."""
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(str(file_path))
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to image at 200 DPI for good OCR quality
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img, lang=self.lang)
            if text.strip():
                all_text.append(text.strip())

        doc.close()
        return "\n\n".join(all_text)

    def process_document(self, parser_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline method. Checks needs_ocr flag and extracts text if required.
        """
        result = parser_payload.copy()

        if not result.get("needs_ocr", False):
            return result

        if not self._check_available():
            # OCR not available — pass through with warning
            result["detected_issues"] = result.get("detected_issues", [])
            if "ocr_unavailable" not in result.get("detected_issues", []):
                issues = list(result.get("detected_issues", []))
                issues.append("ocr_unavailable")
                result["detected_issues"] = issues
            return result

        file_path = Path(result["file_path"])
        ext = result.get("file_extension", "").lower()

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File {file_path} not found for OCR.")

            extracted_text = ""

            if ext == ".pdf":
                extracted_text = self._run_ocr_on_pdf(file_path)
            elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".webp"}:
                extracted_text = self._run_ocr_on_image(file_path)
            else:
                logger.warning(f"Unsupported OCR format: {ext}")
                return result

            if extracted_text:
                existing_text = result.get("text", "")
                combined = (existing_text + "\n" + extracted_text).strip() if existing_text else extracted_text
                result["text"] = combined
                result["text_length"] = len(combined)
                result["word_count"] = len(combined.split())
                result["has_text"] = True
                result["needs_ocr"] = False
                result["status"] = "success"

                # Update pages_data for image files
                if ext != ".pdf" and not result.get("pages_data"):
                    result["pages_data"] = [{"page_number": 1, "text": combined}]

                logger.info(f"OCR extracted {len(combined)} chars from {file_path.name}")

        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            result["status"] = "failed"
            result["error_message"] = f"OCR Error: {str(e)}"

        return result
