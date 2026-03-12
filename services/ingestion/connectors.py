import os
import hashlib
from typing import Iterator, Dict, Any
from datetime import datetime
from pathlib import Path
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


# Supported files based on parser and OCR capabilities
SUPPORTED_EXTENSIONS = set(settings.SUPPORTED_EXTENSIONS)

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
                # Read and update hash in chunks to avoid blowing up memory for huge files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            # Fallback for unreadable files
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

        # Recursive scan over everything inside the base directory
        for file_path in self.base_directory.rglob("*"):
            if not file_path.is_file():
                continue

            if not self._is_supported_file(file_path):
                continue

            # Extract System Facts
            sys_meta = self._get_file_metadata(file_path)
            
            # Generate ID (SHA-256 on bytes)
            doc_id = self._generate_document_id(file_path)
            
            # Formulate Strict Payload Schema
            payload = {
                "document_id": doc_id,
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "file_size_bytes": sys_meta["file_size_bytes"],
                "created_time": sys_meta["created_time"],
                "modified_time": sys_meta["modified_time"],
                "ingested_time": datetime.utcnow().isoformat() + "Z"
            }
            
            # Yield exactly one file at a time safely to the pipeline
            logger.debug(f"Yielding payload for {payload['file_name']}")
            yield payload
