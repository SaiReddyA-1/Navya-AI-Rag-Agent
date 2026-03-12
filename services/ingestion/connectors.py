import os
import json
import hashlib
from typing import Iterator, Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


# Supported files based on parser and OCR capabilities
SUPPORTED_EXTENSIONS = set(settings.SUPPORTED_EXTENSIONS)


class FileManifest:
    """
    Persistent JSON manifest that tracks every file's state in Data/Rag/.
    Enables incremental processing: only new/modified files are processed,
    deleted files are cleaned up from OpenSearch.
    """

    def __init__(self, manifest_path: str = None):
        self.manifest_path = Path(manifest_path or settings.MANIFEST_FILE)
        self._data = self.load()

    def load(self) -> dict:
        """Read manifest from disk. Returns empty structure if missing/corrupted."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "files" in data:
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Manifest corrupted, starting fresh: {e}")
        return {"version": 1, "files": {}, "last_updated": ""}

    def save(self):
        """Atomic write: write to .tmp then os.replace() to avoid corruption."""
        self._data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp_path, self.manifest_path)

    def compute_diff(self, scanned_files: List[Dict[str, Any]]) -> dict:
        """
        Compare filesystem scan against manifest.
        Returns: {new: [...], modified: [...], deleted: [...], unchanged: [...]}
        """
        manifest_files = self._data["files"]
        scanned_by_path = {}
        for payload in scanned_files:
            rel = payload.get("relative_path", payload["file_name"])
            scanned_by_path[rel] = payload

        new, modified, unchanged = [], [], []

        for rel_path, payload in scanned_by_path.items():
            if rel_path not in manifest_files:
                new.append(payload)
            elif manifest_files[rel_path].get("content_hash") != payload["document_id"]:
                modified.append(payload)
            else:
                unchanged.append(payload)

        # Files in manifest but no longer on disk
        deleted = []
        for rel_path, entry in manifest_files.items():
            if rel_path not in scanned_by_path:
                deleted.append(entry)

        return {"new": new, "modified": modified, "deleted": deleted, "unchanged": unchanged}

    def update_entry(self, relative_path: str, entry: dict):
        """Add or update a file entry in the manifest."""
        self._data["files"][relative_path] = entry

    def remove_entry(self, relative_path: str):
        """Remove a file entry from the manifest."""
        self._data["files"].pop(relative_path, None)

    def get_document_id(self, relative_path: str) -> Optional[str]:
        """Look up old document_id for a file (needed for delete-before-reprocess)."""
        entry = self._data["files"].get(relative_path)
        return entry.get("document_id") if entry else None

    def mark_failed(self, relative_path: str, error: str):
        """Mark a file as failed with error message."""
        if relative_path in self._data["files"]:
            self._data["files"][relative_path]["status"] = "failed"
            self._data["files"][relative_path]["error_message"] = error
        else:
            self._data["files"][relative_path] = {
                "status": "failed",
                "error_message": error,
                "last_processed": datetime.utcnow().isoformat() + "Z",
            }

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Return all manifest entries as a list."""
        return list(self._data["files"].values())

    def get_files_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the raw files dict for direct lookup."""
        return self._data["files"]

    def build_from_opensearch(self, payloads: List[Dict[str, Any]]):
        """
        One-time migration: build manifest from existing OpenSearch data.
        Groups chunks by document_id to reconstruct file-level entries.
        """
        seen = {}
        for chunk in payloads:
            doc_id = chunk.get("document_id")
            if not doc_id or doc_id in seen:
                if doc_id in seen:
                    seen[doc_id]["chunk_count"] = seen[doc_id].get("chunk_count", 0) + 1
                continue
            file_path = chunk.get("file_path", "")
            file_name = chunk.get("file_name", "")
            # Derive relative path from file_path
            try:
                rel = str(Path(file_path).relative_to(Path(settings.DATA_DIR)))
            except (ValueError, TypeError):
                rel = file_name

            seen[doc_id] = {
                "document_id": doc_id,
                "file_path": file_path,
                "relative_path": rel,
                "parent_folder": str(Path(rel).parent) if str(Path(rel).parent) != "." else "",
                "file_name": file_name,
                "file_extension": chunk.get("file_extension", ""),
                "file_size_bytes": chunk.get("file_size_bytes", 0),
                "content_hash": doc_id,
                "modified_time": chunk.get("modified_time", ""),
                "status": "processed",
                "document_type": chunk.get("document_type", "unknown"),
                "department_category": chunk.get("department_category", "general"),
                "quality_score": chunk.get("quality_score", 0.0),
                "triage_confidence": chunk.get("triage_confidence", 0.0),
                "chunk_count": 1,
                "last_processed": chunk.get("ingested_time", ""),
                "error_message": None,
            }

        for doc_id, entry in seen.items():
            self._data["files"][entry["relative_path"]] = entry
        logger.info(f"Manifest migration: rebuilt {len(seen)} entries from OpenSearch")


class LocalSystemConnector:
    """
    Scans a local directory of unsorted enterprise files.
    Yields standardized initial metadata payloads for the Parser.
    """

    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)

    def _generate_document_id(self, file_path: Path) -> str:
        """Helper: Creates a unique SHA-256 hash based on the file content."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            return hashlib.sha256(str(file_path).encode()).hexdigest()

    def _get_file_metadata(self, file_path: Path) -> dict:
        """Helper: Uses the OS to extract basic facts about the file."""
        stat = file_path.stat()
        return {
            "file_size_bytes": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
        }

    # Files to skip during ingestion (reference datasets, not enterprise documents)
    SKIP_FILES = {"company-document-text.csv"}

    def _is_supported_file(self, file_path: Path) -> bool:
        """Rejects hidden files, skip-listed files, and unsupported extensions."""
        if file_path.name.startswith("."):
            return False
        if file_path.name in self.SKIP_FILES:
            return False
        extension = file_path.suffix.lower()
        return extension in SUPPORTED_EXTENSIONS

    def scan_repository(self) -> Iterator[Dict[str, Any]]:
        """
        MAIN METHOD: Traverses directory recursively, filtering unsupported files,
        reading metadata, hashing files, and yielding standard schema payloads.
        """
        if not self.base_directory.exists():
            logger.info(f"Directory {self.base_directory} not found.")
            return

        for file_path in self.base_directory.rglob("*"):
            if not file_path.is_file():
                continue

            if not self._is_supported_file(file_path):
                continue

            sys_meta = self._get_file_metadata(file_path)
            doc_id = self._generate_document_id(file_path)

            # Compute relative path and parent folder for manifest tracking
            try:
                relative = file_path.relative_to(self.base_directory)
            except ValueError:
                relative = Path(file_path.name)

            rel_path = str(relative)
            parent_folder = str(relative.parent) if str(relative.parent) != "." else ""

            payload = {
                "document_id": doc_id,
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "file_size_bytes": sys_meta["file_size_bytes"],
                "created_time": sys_meta["created_time"],
                "modified_time": sys_meta["modified_time"],
                "ingested_time": datetime.utcnow().isoformat() + "Z",
                "relative_path": rel_path,
                "parent_folder": parent_folder,
            }

            logger.debug(f"Yielding payload for {payload['file_name']}")
            yield payload
