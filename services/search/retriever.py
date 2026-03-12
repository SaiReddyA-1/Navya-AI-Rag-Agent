import re
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
from services.search.database import HybridSearchClient
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class QueryRetriever:
    """
    Online Pipeline Stage 1.
    Parses user queries, extracts metadata filters, vectorizes the semantic intent
    using the SAME embedding model as ingestion, and calls HybridSearchClient.
    """
    def __init__(self, db_client: HybridSearchClient, model: SentenceTransformer = None):
        self.db_client = db_client
        if model is not None:
            self.model = model
        else:
            logger.info(f"Loading query embedding model: {settings.EMBEDDING_MODEL_NAME}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    def _parse_query_intent(self, raw_query: str) -> Dict[str, Any]:
        """
        Splits user query into semantic text + hard metadata filters.
        """
        filters: Dict[str, Any] = {}
        cleaned_query = raw_query.lower()

        # Detect document_type
        type_map = {
            "invoices": "invoice", "invoice": "invoice",
            "receipts": "receipt", "receipt": "receipt",
            "contracts": "contract", "contract": "contract",
            "agreements": "agreement", "agreement": "agreement",
            "reports": "report", "report": "report",
            "policies": "policy", "policy": "policy",
            "purchase orders": "purchase_order", "purchase order": "purchase_order",
            "shipping orders": "shipping_order", "shipping order": "shipping_order",
            "inventory": "inventory_report", "stock": "inventory_report",
            "inventory reports": "inventory_report",
            "memos": "memo", "memo": "memo",
            "manuals": "manual", "manual": "manual",
            "proposals": "proposal", "proposal": "proposal",
            "specifications": "specification", "spec": "specification",
            "resumes": "resume", "resume": "resume", "cv": "resume",
            "certificates": "certificate", "certificate": "certificate",
            "presentations": "presentation", "presentation": "presentation",
        }
        for phrase, doc_type in type_map.items():
            if phrase in cleaned_query:
                filters["document_type"] = doc_type
                cleaned_query = cleaned_query.replace(phrase, "")
                break

        # Detect department_category
        for dept in ["finance", "legal", "hr", "operations", "engineering", "marketing", "sales"]:
            if dept in cleaned_query:
                filters["department_category"] = dept
                cleaned_query = cleaned_query.replace(dept, "")

        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

        if len(cleaned_query) < 3:
            cleaned_query = raw_query

        return {"cleaned_query": cleaned_query.strip(), "metadata_filters": filters}

    def _embed_query(self, cleaned_query: str) -> List[float]:
        """Embeds query using the same SentenceTransformer model as ingestion."""
        return self.model.encode(cleaned_query, normalize_embeddings=True).tolist()

    def retrieve_context(self, raw_query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Main retrieval execution flow."""
        top_k = top_k or settings.TOP_K_RETRIEVAL

        intent = self._parse_query_intent(raw_query)
        semantic_query = intent["cleaned_query"]
        db_filters = intent["metadata_filters"]

        logger.info(f"Search: [Query: '{semantic_query}'] [Filters: {db_filters}]")

        query_vector = self._embed_query(semantic_query)

        retrieved_chunks = self.db_client.hybrid_search(
            query_vector=query_vector,
            metadata_filters=db_filters,
            top_k=top_k
        )

        return retrieved_chunks
