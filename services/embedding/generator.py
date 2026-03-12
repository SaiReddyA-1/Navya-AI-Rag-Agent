from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class EmbeddingGenerator:
    """
    Transforms human-readable chunk text into dense vector embeddings
    for semantic search, while preserving inherited metadata.
    """
    def __init__(self, model_name: str = None, batch_size: int = None, model: SentenceTransformer = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        if model is not None:
            self.model = model
        else:
            logger.info(f"Loading embedding model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)

        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedder ready: Model={self.model_name}, Dim={self.embedding_dimension}, Batch={self.batch_size}")

    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main pipeline method. Batch-embeds chunk texts and appends the
        'embedding' array to each chunk object.
        """
        if not chunks:
            return []

        try:
            texts = [chunk.get("text", "") for chunk in chunks]

            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 50
            ).tolist()

            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]

            logger.info(f"Embedded {len(chunks)} chunks successfully")
            return chunks

        except Exception as e:
            logger.error(f"Embedding Generation Failed: {str(e)}")
            return chunks
