"""
Advanced Chunking Engine — Splits documents into chunks with full metadata inheritance.
For PDFs, uses page-aware chunking so each chunk knows its page number.
"""
import re
from pathlib import Path
from typing import Dict, Any, List
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class AdvancedChunkingEngine:
    """
    Page-aware recursive chunking with full metadata inheritance.
    Every chunk carries: page_number, folder_path, repository, triage metadata, etc.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        logger.info(f"Initialized Chunker: Size={self.chunk_size}, Overlap={self.chunk_overlap}")

    def _generate_chunk_id(self, doc_id: str, index: int) -> str:
        return f"{doc_id[:16]}_c{index}"

    def _recursive_split(self, text: str) -> List[str]:
        """Recursive character splitter: paragraphs → sentences → spaces → hard cut."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separators = ["\n\n", "\n", ". ", " "]

        for sep in separators:
            if sep in text:
                splits = text.split(sep)
                chunks = []
                current_chunk = ""

                for split in splits:
                    piece = split + sep if split else ""
                    if len(current_chunk) + len(piece) <= self.chunk_size:
                        current_chunk += piece
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        if len(piece) > self.chunk_size:
                            if sep == " ":
                                chunks.extend(
                                    piece[i:i+self.chunk_size]
                                    for i in range(0, len(piece), self.chunk_size)
                                )
                                current_chunk = ""
                            else:
                                current_chunk = piece
                        else:
                            current_chunk = piece

                if current_chunk:
                    chunks.append(current_chunk.strip())

                if all(len(c) <= self.chunk_size for c in chunks):
                    return self._apply_overlap(chunks)

        return self._apply_overlap(
            [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        )

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i-1]
            overlap = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            overlapped.append(overlap + " " + chunks[i])
        return overlapped

    def _extract_folder_path(self, file_path: str) -> str:
        """Extract the folder category from the file path."""
        p = Path(file_path)
        # Get the parent folder relative to Data/Rag
        parts = p.parts
        for i, part in enumerate(parts):
            if part == "Rag" and i + 1 < len(parts):
                return "/" + "/".join(parts[i+1:-1]) + "/"
        return "/" + p.parent.name + "/"

    def _extract_repository(self, file_path: str) -> str:
        """Extract repository name from path."""
        p = Path(file_path)
        parts = p.parts
        for i, part in enumerate(parts):
            if part == "Data":
                return parts[i + 1] if i + 1 < len(parts) else "local"
        return "local"

    def process_document(self, triage_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main chunking method. Uses page-aware splitting for PDFs.
        Every chunk inherits ALL metadata from upstream pipeline stages.
        """
        if triage_payload.get("status") == "failed" or not triage_payload.get("text", "").strip():
            return []

        doc_id = triage_payload["document_id"]
        file_path = triage_payload.get("file_path", "")
        file_name = triage_payload.get("file_name", "")
        total_pages = triage_payload.get("total_pages", 1)
        pages_data = triage_payload.get("pages_data", [])
        folder_path = self._extract_folder_path(file_path)
        repository = self._extract_repository(file_path)

        # Build the shared metadata that every chunk inherits
        shared_meta = {
            "document_id": doc_id,
            "file_path": file_path,
            "file_name": file_name,
            "file_extension": triage_payload.get("file_extension", ""),
            "folder_path": folder_path,
            "repository": repository,
            "total_pages": total_pages,
            "word_count": triage_payload.get("word_count", 0),
            # Triage metadata
            "document_type": triage_payload.get("document_type", "unknown"),
            "department_category": triage_payload.get("department_category", "general"),
            "quality_score": triage_payload.get("quality_score", 0.0),
            "is_degraded": triage_payload.get("is_degraded", True),
            "detected_issues": triage_payload.get("detected_issues", []),
            "triage_confidence": triage_payload.get("triage_confidence", 0.0),
            "recommended_action": self._get_recommended_action(triage_payload),
            # File metadata
            "file_size_bytes": triage_payload.get("file_size_bytes", 0),
            "created_time": triage_payload.get("created_time", ""),
            "modified_time": triage_payload.get("modified_time", ""),
            "ingested_time": triage_payload.get("ingested_time", ""),
        }

        final_chunks = []
        chunk_index = 0

        # Page-aware chunking: if we have per-page data, chunk each page separately
        if pages_data and len(pages_data) > 1:
            for page_info in pages_data:
                page_num = page_info.get("page_number", 1)
                page_text = page_info.get("text", "").strip()
                if not page_text:
                    continue

                page_chunks = self._recursive_split(page_text)
                for sub_idx, chunk_text in enumerate(page_chunks):
                    if not chunk_text.strip():
                        continue
                    chunk_record = {
                        "chunk_id": self._generate_chunk_id(doc_id, chunk_index),
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "page_number": page_num,
                        "total_chunks_in_doc": 0,  # filled below
                        **shared_meta,
                    }
                    final_chunks.append(chunk_record)
                    chunk_index += 1
        else:
            # Single-page or non-PDF: chunk the full text
            text = triage_payload["text"]
            text_chunks = self._recursive_split(text)
            for idx, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue
                chunk_record = {
                    "chunk_id": self._generate_chunk_id(doc_id, idx),
                    "text": chunk_text,
                    "chunk_index": idx,
                    "page_number": 1,
                    "total_chunks_in_doc": 0,
                    **shared_meta,
                }
                final_chunks.append(chunk_record)
                chunk_index += 1

        # Fill in total chunk count per doc
        for c in final_chunks:
            c["total_chunks_in_doc"] = len(final_chunks)

        return final_chunks

    def _get_recommended_action(self, payload: Dict[str, Any]) -> str:
        """Determine recommended action based on triage results."""
        q = payload.get("quality_score", 1.0)
        issues = payload.get("detected_issues", [])

        if q >= 0.8 and "none" in issues:
            return "none"
        elif q >= 0.6:
            return "review"
        elif any(i in issues for i in ["blurred", "noisy_ocr"]):
            return "rescan"
        else:
            return "enhance"
