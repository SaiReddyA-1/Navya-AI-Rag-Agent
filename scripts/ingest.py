"""
Startup Ingestion Script — Runs the full offline pipeline on startup.
Pipeline: Scan → Parse → OCR → Triage → Chunk → Embed → Upload to OpenSearch

Uses FileManifest for incremental processing:
- NEW files → full pipeline
- MODIFIED files → delete old chunks, re-process
- DELETED files → remove chunks from OpenSearch
- UNCHANGED files → skip entirely
"""
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.logger import setup_enterprise_logger
from services.ingestion.connectors import LocalSystemConnector, FileManifest
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

    # Load or create file manifest
    manifest = FileManifest()
    manifest_is_new = len(manifest.get_files_dict()) == 0

    # First-run migration: if OpenSearch has data but manifest is empty,
    # rebuild manifest from existing payloads to avoid re-processing
    existing_count = db.get_total_count()
    if manifest_is_new and existing_count > 0:
        logger.info(f"Manifest is empty but DB has {existing_count} chunks. Migrating...")
        all_payloads = db.get_all_payloads()
        manifest.build_from_opensearch(all_payloads)
        manifest.save()
        logger.info("Manifest migration complete.")

    # Scan filesystem
    scanned = list(connector.scan_repository())
    logger.info(f"Scanned {len(scanned)} files from filesystem.")

    # Compute diff against manifest
    diff = manifest.compute_diff(scanned)
    logger.info(
        f"Diff: {len(diff['new'])} new, {len(diff['modified'])} modified, "
        f"{len(diff['deleted'])} deleted, {len(diff['unchanged'])} unchanged"
    )

    # Counters
    added_docs = 0
    updated_docs = 0
    removed_docs = 0
    total_chunks = 0
    failed_docs = 0

    # ── Handle DELETED files ──────────────────────────────────────────
    for entry in diff["deleted"]:
        old_doc_id = entry.get("document_id")
        rel_path = entry.get("relative_path", entry.get("file_name", ""))
        if old_doc_id:
            result = db.delete_document(old_doc_id)
            deleted_count = result.get("deleted_chunks", 0)
            logger.info(f"REMOVED: {entry.get('file_name', rel_path)} ({deleted_count} chunks)")
        manifest.remove_entry(rel_path)
        removed_docs += 1

    # ── Handle MODIFIED files (delete old, then re-process) ───────────
    to_process = []
    for payload in diff["modified"]:
        rel_path = payload.get("relative_path", payload["file_name"])
        old_doc_id = manifest.get_document_id(rel_path)
        if old_doc_id:
            db.delete_document(old_doc_id)
            logger.info(f"UPDATING: {payload['file_name']} (deleted old chunks)")
        to_process.append(payload)
        updated_docs += 1

    # ── Handle NEW files ──────────────────────────────────────────────
    to_process.extend(diff["new"])

    # ── Process all new + modified files through the pipeline ─────────
    for payload in to_process:
        file_name = payload["file_name"]
        rel_path = payload.get("relative_path", file_name)

        try:
            parsed = parser.parse_document(payload)
            if parsed.get("status") == "failed":
                logger.warning(f"SKIP (parse failed): {file_name}")
                manifest.mark_failed(rel_path, "parse_failed")
                failed_docs += 1
                continue

            ocr_result = ocr.process_document(parsed)
            triaged = triage.process_document(ocr_result)
            chunks = chunker.process_document(triaged)

            if not chunks:
                logger.warning(f"SKIP (no chunks): {file_name}")
                manifest.mark_failed(rel_path, "no_chunks")
                failed_docs += 1
                continue

            embedded = embedder.process_chunks(chunks)
            result = db.upload_chunks(embedded)
            total_chunks += result["inserted"]

            if payload not in diff["modified"]:
                added_docs += 1

            # Update manifest with full metadata
            manifest.update_entry(rel_path, {
                "document_id": payload["document_id"],
                "file_path": payload["file_path"],
                "relative_path": rel_path,
                "parent_folder": payload.get("parent_folder", ""),
                "file_name": file_name,
                "file_extension": payload["file_extension"],
                "file_size_bytes": payload["file_size_bytes"],
                "content_hash": payload["document_id"],
                "modified_time": payload["modified_time"],
                "status": "processed",
                "document_type": triaged.get("document_type", "unknown"),
                "department_category": triaged.get("department_category", "general"),
                "quality_score": triaged.get("quality_score", 0.0),
                "triage_confidence": triaged.get("triage_confidence", 0.0),
                "chunk_count": len(chunks),
                "last_processed": datetime.utcnow().isoformat() + "Z",
                "error_message": None,
            })

            processed = added_docs + updated_docs
            if processed % 25 == 0 and processed > 0:
                logger.info(f"Progress: {processed} docs, {total_chunks} chunks indexed")
                manifest.save()  # Periodic save

        except Exception as e:
            logger.error(f"FAILED: {file_name} — {e}")
            manifest.mark_failed(rel_path, str(e))
            failed_docs += 1

    # Save manifest
    manifest.save()

    elapsed = round(time.time() - start, 1)
    logger.info("=" * 60)
    logger.info(f"INGESTION COMPLETE — {elapsed}s")
    logger.info(f"  New: {added_docs}, Updated: {updated_docs}, Removed: {removed_docs}")
    logger.info(f"  Unchanged: {len(diff['unchanged'])}, Failed: {failed_docs}")
    logger.info(f"  Chunks indexed: {total_chunks}")
    logger.info(f"  DB mode: {db.mode}")
    logger.info("=" * 60)

    return db


if __name__ == "__main__":
    run_ingestion()
