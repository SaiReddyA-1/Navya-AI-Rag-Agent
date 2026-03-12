"""
Startup Ingestion Script — Runs the full offline pipeline on startup.
Pipeline: Scan → Parse → OCR → Triage → Chunk → Embed → Upload to OpenSearch
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.logger import setup_enterprise_logger
from services.ingestion.connectors import LocalSystemConnector
from services.processing.parser import DocumentParser
from services.processing.ocr import OCRService
from services.processing.triage import TriageService
from services.embedding.chunker import AdvancedChunkingEngine
from services.embedding.generator import EmbeddingGenerator
from services.search.database import HybridSearchClient

logger = setup_enterprise_logger("ingest")


def run_ingestion():
    start = time.time()
    logger.info("=" * 60)
    logger.info("STARTING OFFLINE INGESTION PIPELINE")
    logger.info(f"Data directory: {settings.DATA_DIR}")
    logger.info("=" * 60)

    # Initialize pipeline stages
    connector = LocalSystemConnector(settings.DATA_DIR)
    parser = DocumentParser()
    ocr = OCRService()
    triage = TriageService()
    chunker = AdvancedChunkingEngine()

    logger.info("Loading embedding model (first run downloads ~1.3GB)...")
    embedder = EmbeddingGenerator()

    logger.info("Connecting to database...")
    db = HybridSearchClient()
    logger.info(f"Database mode: {db.mode}")

    # Check if data already exists
    existing = db.get_total_count()
    if existing > 0:
        logger.info(f"Database already contains {existing} chunks. Skipping ingestion.")
        return db

    # Run pipeline
    total_docs = 0
    total_chunks = 0
    failed_docs = 0

    for payload in connector.scan_repository():
        total_docs += 1
        file_name = payload["file_name"]

        try:
            parsed = parser.parse_document(payload)
            if parsed.get("status") == "failed":
                logger.warning(f"SKIP (parse failed): {file_name}")
                failed_docs += 1
                continue

            ocr_result = ocr.process_document(parsed)
            triaged = triage.process_document(ocr_result)
            chunks = chunker.process_document(triaged)

            if not chunks:
                logger.warning(f"SKIP (no chunks): {file_name}")
                failed_docs += 1
                continue

            embedded = embedder.process_chunks(chunks)
            result = db.upload_chunks(embedded)
            total_chunks += result["inserted"]

            if total_docs % 25 == 0:
                logger.info(f"Progress: {total_docs} docs, {total_chunks} chunks indexed")

        except Exception as e:
            logger.error(f"FAILED: {file_name} — {e}")
            failed_docs += 1

    elapsed = round(time.time() - start, 1)
    logger.info("=" * 60)
    logger.info(f"INGESTION COMPLETE — {elapsed}s")
    logger.info(f"  Documents: {total_docs} scanned, {failed_docs} failed")
    logger.info(f"  Chunks: {total_chunks} indexed")
    logger.info(f"  DB mode: {db.mode}")
    logger.info("=" * 60)

    return db


if __name__ == "__main__":
    run_ingestion()
