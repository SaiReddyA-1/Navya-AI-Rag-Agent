from typing import Dict, Any, List
from sentence_transformers import CrossEncoder
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class CrossEncoderReranker:
    """
    Uses a real CrossEncoder model to precisely score query-chunk pairs
    and isolate the most relevant chunks for the LLM.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        logger.info(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Scores each chunk against the query using the CrossEncoder
        and returns only the top_k most relevant.
        """
        top_k = top_k or settings.TOP_K_RERANK

        if not retrieved_chunks:
            return []

        try:
            pairs = [[query, chunk.get("text", "")] for chunk in retrieved_chunks]

            scores = self.model.predict(pairs)

            for i, chunk in enumerate(retrieved_chunks):
                chunk["reranker_score"] = float(scores[i])

            best_chunks = sorted(retrieved_chunks, key=lambda x: x["reranker_score"], reverse=True)
            return best_chunks[:top_k]

        except Exception as e:
            logger.error(f"Reranker failed: {e}. Falling back to vector ordering.")
            return retrieved_chunks[:top_k]
