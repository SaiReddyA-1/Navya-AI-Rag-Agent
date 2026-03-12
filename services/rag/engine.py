"""
RAG Engine — Orchestrates the full query pipeline.
Handles 3 query types:
  1. Analytics queries (how many, count, list all) → metadata aggregation
  2. Retrieval queries (show me, find, what is) → vector search + LLM
  3. Follow-up queries → uses conversation history for context
"""
import re
from typing import Dict, Any, List, Optional
from groq import Groq
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)

# Patterns that indicate analytics/count queries (not retrieval)
ANALYTICS_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bhow much\b",
    r"\blist all\b", r"\bshow all\b", r"\ball the\b",
    r"\bsummar(y|ize)\b", r"\boverview\b", r"\bstatistics\b",
    r"\bbreakdown\b", r"\bdistribution\b",
]


class RAGEngine:
    """
    Smart RAG orchestrator with query routing, conversation memory,
    and analytics-aware responses.
    """
    def __init__(self, retriever_client: Any, reranker_client: Any, analytics_engine: Any = None):
        self.retriever = retriever_client
        self.reranker = reranker_client
        self.analytics = analytics_engine
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        self.model = settings.GROQ_MODEL_NAME
        logger.info(f"RAG Engine online. LLM: {self.model} via Groq")

    def _is_analytics_query(self, query: str) -> bool:
        """Detect if user is asking a count/summary question."""
        q = query.lower()
        return any(re.search(p, q) for p in ANALYTICS_PATTERNS)

    def _build_analytics_context(self, query: str) -> str:
        """Build a metadata summary for analytics queries."""
        if not self.analytics:
            return ""

        summary = self.analytics.get_repository_summary()
        by_type = self.analytics.count_by_document_type()
        by_dept = self.analytics.count_by_department()
        degradation = self.analytics.get_degradation_report()
        doc_list = self.analytics.get_document_list()

        # Build a structured context from metadata
        lines = [
            "=== REPOSITORY METADATA (COMPLETE) ===",
            f"Total unique documents: {summary['total_documents_ingested']}",
            f"Total searchable chunks: {summary['total_searchable_chunks']}",
            f"Average quality score: {summary['repository_health_score']}",
            "",
            "--- Documents by Type ---",
        ]
        for dtype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {dtype}: {count} documents")

        lines.append("")
        lines.append("--- Documents by Department ---")
        for dept, count in sorted(by_dept.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {dept}: {count} documents")

        lines.append("")
        lines.append(f"--- Quality Report ---")
        lines.append(f"  Degraded documents: {degradation['degraded_count']} ({degradation['degraded_percentage']}%)")
        if degradation['common_issues']:
            lines.append(f"  Common issues: {', '.join(degradation['common_issues'].keys())}")

        # Include file listing (grouped by type)
        lines.append("")
        lines.append("--- Complete File Listing ---")
        by_type_files: Dict[str, List[str]] = {}
        for doc in doc_list:
            dt = doc.get("document_type", "unknown")
            by_type_files.setdefault(dt, []).append(doc["file_name"])

        for dtype, files in sorted(by_type_files.items()):
            lines.append(f"\n  [{dtype.upper()}] ({len(files)} files):")
            for f in sorted(files):
                lines.append(f"    - {f}")

        return "\n".join(lines)

    def _build_strict_context(self, top_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks with citations for the LLM."""
        blocks = []
        for i, chunk in enumerate(top_chunks):
            file_name = chunk.get("file_name", "Unknown")
            doc_type = chunk.get("document_type", "Unknown")
            dept = chunk.get("department_category", "Unknown")
            quality = chunk.get("quality_score", 0.0)
            text = chunk.get("text", "")

            block = (
                f"--- Source [{i+1}] ---\n"
                f"File: {file_name} | Type: {doc_type} | Dept: {dept} | Quality: {quality:.0%}\n"
                f"Content:\n{text}\n---"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def _call_llm(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Call Groq API with full conversation context."""
        if not self.groq_client:
            return "Groq API key not configured. Please set GROQ_API_KEY in .env file."

        try:
            all_messages = [{"role": "system", "content": system_prompt}] + messages

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API failed: {e}")
            return f"LLM generation failed: {str(e)}"

    def execute_rag(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        Builds a UNIFIED context from all available sources so the LLM
        can handle any combination of questions in a single message.
        """
        chat_history = chat_history or []

        # ── Gather all context layers ─────────────────────────────────
        context_parts = []
        all_sources = []
        query_types = []

        # LAYER 1: If a specific file is mentioned, get its full content
        target_file = self._detect_specific_file(user_query, chat_history)
        if target_file:
            doc_context, doc_sources = self._get_full_doc_context(target_file)
            if doc_context:
                context_parts.append(f"=== FULL DOCUMENT: {target_file} ===\n{doc_context}")
                all_sources.extend(doc_sources)
                query_types.append("full_document")

        # LAYER 2: Always include analytics metadata for count/summary questions
        if self.analytics:
            analytics_context = self._build_analytics_context(user_query)
            context_parts.append(f"=== REPOSITORY ANALYTICS ===\n{analytics_context}")
            query_types.append("analytics")

        # LAYER 3: Semantic retrieval for content-based questions
        effective_query = self._expand_followup(user_query, chat_history)
        chunks = self.retriever.retrieve_context(effective_query, top_k=settings.TOP_K_RETRIEVAL)

        if chunks:
            best_chunks = self.reranker.rerank(
                query=effective_query,
                retrieved_chunks=chunks,
                top_k=settings.TOP_K_RERANK,
            )
            retrieval_context = self._build_strict_context(best_chunks)
            context_parts.append(f"=== RETRIEVED DOCUMENTS ===\n{retrieval_context}")
            query_types.append("retrieval")

            for c in best_chunks:
                all_sources.append({
                    "document_id": c.get("document_id", ""),
                    "file_name": c.get("file_name", ""),
                    "document_type": c.get("document_type", ""),
                    "department": c.get("department_category", ""),
                    "quality_score": c.get("quality_score", 0.0),
                    "folder_path": c.get("folder_path", c.get("file_path", "")),
                    "page_number": c.get("page_number", 1),
                })

        # ── Build unified prompt ──────────────────────────────────────
        if not context_parts:
            return {
                "answer": "I could not find any information matching your request.",
                "sources": [], "confidence_score": 0.0,
                "retrieved_chunks": [], "query_type": "none",
            }

        full_context = "\n\n".join(context_parts)

        system_prompt = (
            "You are an enterprise document intelligence assistant.\n"
            "You have access to THREE types of information:\n"
            "1. FULL DOCUMENT content — complete text of specifically requested files\n"
            "2. REPOSITORY ANALYTICS — metadata counts, file listings, quality stats\n"
            "3. RETRIEVED DOCUMENTS — semantically relevant chunks from vector search\n\n"
            "RULES:\n"
            "- Answer ALL questions in the user's message, even if there are multiple.\n"
            "- For count/summary questions, use the REPOSITORY ANALYTICS section.\n"
            "- For specific file questions, use the FULL DOCUMENT section.\n"
            "- For content questions, use the RETRIEVED DOCUMENTS section.\n"
            "- ALWAYS cite sources with exact file names.\n"
            "- Be precise with numbers — use exact counts from the metadata.\n"
            "- Present data in structured format — tables, bullet points, clear sections.\n"
            "- If information is not available, say so clearly for that specific question.\n\n"
            f"{full_context}"
        )

        messages = self._build_conversation_messages(chat_history, user_query)
        answer = self._call_llm(system_prompt, messages)

        # Deduplicate sources by file_name
        seen = set()
        unique_sources = []
        for s in all_sources:
            fname = s.get("file_name", "")
            if fname not in seen:
                seen.add(fname)
                unique_sources.append(s)

        avg_conf = sum(s.get("quality_score", 0.8) for s in unique_sources) / max(len(unique_sources), 1)

        return {
            "answer": answer,
            "sources": unique_sources,
            "confidence_score": round(avg_conf, 2),
            "retrieved_chunks": chunks if chunks else [],
            "query_type": "+".join(query_types),
        }

    def _build_conversation_messages(
        self, chat_history: List[Dict[str, str]], current_query: str
    ) -> List[Dict[str, str]]:
        """Build message list with conversation history for context continuity."""
        messages = []

        # Include last 4 exchanges for context (8 messages: 4 user + 4 assistant)
        recent = chat_history[-8:] if len(chat_history) > 8 else chat_history
        for msg in recent:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": current_query})
        return messages

    def _expand_followup(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        If the current query is a short follow-up (like 'show me that invoice' or
        'explain more'), expand it using context from the last assistant response.
        """
        if not chat_history or len(query.split()) > 8:
            return query

        # Check if this looks like a follow-up
        followup_indicators = [
            "this", "that", "it", "those", "the same", "more about",
            "explain", "details", "tell me more", "elaborate", "which one",
        ]

        q_lower = query.lower()
        is_followup = any(ind in q_lower for ind in followup_indicators)

        if not is_followup:
            return query

        # Find the last assistant message and extract key terms
        for msg in reversed(chat_history):
            if msg["role"] == "assistant":
                content = msg["content"]
                file_mentions = re.findall(r'[\w_.-]+\.(?:pdf|docx|csv|txt|xlsx)', content)
                if file_mentions:
                    expanded = f"{query} (referring to: {', '.join(file_mentions[:3])})"
                    logger.info(f"Expanded follow-up: '{query}' → '{expanded}'")
                    return expanded
                break

        return query

    def _detect_specific_file(self, query: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Detect if user is asking about a specific file.
        Returns file_name if found, else None.
        """
        # Explicit filename in query
        file_match = re.findall(r'[\w_.-]+\.(?:pdf|docx|csv|txt|xlsx|json)', query, re.IGNORECASE)
        if file_match:
            logger.info(f"Specific file in query: {file_match[0]}")
            return file_match[0]

        # Follow-up about previously mentioned file
        followup_words = [
            "this invoice", "that invoice", "this file", "that file",
            "this document", "that document", "more about it",
            "tell me more", "more details", "elaborate", "explain more",
            "can you tell more", "clarify", "show more",
        ]
        q_lower = query.lower()

        if not any(w in q_lower for w in followup_words):
            return None
        if not chat_history:
            return None

        for msg in reversed(chat_history):
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                files = re.findall(r'[\w_.-]+\.(?:pdf|docx|csv|txt|xlsx)', content)
                if files:
                    logger.info(f"Follow-up → full doc: {files[0]}")
                    return files[0]
                sources = msg.get("sources", [])
                if sources:
                    fname = sources[0].get("file_name", "")
                    if fname:
                        logger.info(f"Follow-up → full doc from sources: {fname}")
                        return fname
                break

        return None

    def _get_full_doc_context(self, file_name: str) -> tuple:
        """
        Retrieve ALL chunks of a document and return as context string + sources.
        Does NOT call LLM — just gathers the context.
        Returns: (context_string, sources_list)
        """
        logger.info(f"Full document retrieval: {file_name}")

        db = self.retriever.db_client
        doc_id = db.find_document_by_filename(file_name)

        if not doc_id:
            return "", []

        all_chunks = db.get_document_chunks(doc_id)
        if not all_chunks:
            return "", []

        meta = all_chunks[0]

        # Build page-by-page content
        lines = [
            f"File: {file_name}",
            f"Type: {meta.get('document_type', 'unknown')} | Dept: {meta.get('department_category', 'unknown')}",
            f"Folder: {meta.get('folder_path', '')} | Pages: {meta.get('total_pages', 1)} | Quality: {meta.get('quality_score', 0):.0%}",
            f"Chunks: {len(all_chunks)}",
            "",
        ]
        for chunk in all_chunks:
            page = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            lines.append(f"[Page {page}]\n{text}\n")

        context = "\n".join(lines)

        source = {
            "document_id": doc_id,
            "file_name": file_name,
            "document_type": meta.get("document_type", ""),
            "department": meta.get("department_category", ""),
            "quality_score": meta.get("quality_score", 0.0),
            "folder_path": meta.get("folder_path", ""),
            "page_number": meta.get("total_pages", 1),
        }

        return context, [source]
