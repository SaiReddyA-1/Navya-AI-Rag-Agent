"""
Document Parser — Extracts text from enterprise file formats.
Produces per-page text for PDFs to enable page-level chunking.
"""
import os
import json
from typing import Dict, Any, List
from pathlib import Path
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class DocumentParser:
    """
    Extracts raw text from documents. For PDFs, preserves per-page text
    so downstream chunking can maintain page number references.
    """
    def __init__(self, min_text_threshold: int = None):
        if min_text_threshold is None:
            min_text_threshold = settings.MIN_TEXT_THRESHOLD
        self.min_text_threshold = min_text_threshold

    def _parse_pdf(self, file_path: Path) -> dict:
        """Extract per-page text from PDF using PyMuPDF."""
        import fitz
        doc = fitz.open(str(file_path))
        pages = []
        images = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            pages.append({"page_number": page_num + 1, "text": page_text})
            if page.get_images():
                images.append({"page": page_num + 1})
        doc.close()
        full_text = "\n".join(p["text"] for p in pages)
        return {
            "text": full_text,
            "pages_data": pages,
            "total_pages": len(pages),
            "images": images,
            "tables": [],
        }

    def _parse_docx(self, file_path: Path) -> dict:
        from docx import Document as DocxDoc
        doc = DocxDoc(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs]
        full_text = "\n".join(paragraphs)
        tables = []
        for table in doc.tables:
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            tables.append(rows)
        return {
            "text": full_text,
            "pages_data": [{"page_number": 1, "text": full_text}],
            "total_pages": max(1, len(paragraphs) // 40),
            "images": [],
            "tables": tables,
        }

    def _parse_txt(self, file_path: Path) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        return {
            "text": text,
            "pages_data": [{"page_number": 1, "text": text}],
            "total_pages": 1,
            "images": [],
            "tables": [],
        }

    def _parse_excel(self, file_path: Path) -> dict:
        import pandas as pd
        ext = file_path.suffix.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(file_path, encoding="utf-8", errors="replace")
            else:
                df = pd.read_excel(file_path)
            text = df.to_string(index=False)
            tables = [df.columns.tolist()] + df.values.tolist()
            return {
                "text": text,
                "pages_data": [{"page_number": 1, "text": text}],
                "total_pages": 1,
                "images": [],
                "tables": [tables],
            }
        except Exception as e:
            logger.warning(f"Excel/CSV parse failed for {file_path}: {e}")
            return {"text": "", "pages_data": [], "total_pages": 1, "images": [], "tables": []}

    def _parse_image(self, file_path: Path) -> dict:
        return {
            "text": "",
            "pages_data": [],
            "total_pages": 1,
            "images": [{"path": str(file_path)}],
            "tables": [],
        }

    def _parse_json_xml(self, file_path: Path, is_json: bool = True) -> dict:
        text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if is_json:
                    data = json.load(f)
                    text = json.dumps(data, indent=2)
                else:
                    text = f.read()
        except Exception:
            pass
        return {
            "text": text,
            "pages_data": [{"page_number": 1, "text": text}],
            "total_pages": 1,
            "images": [],
            "tables": [],
        }

    def parse_document(self, ingestion_payload: dict) -> dict:
        """Main pipeline method. Extracts text and per-page data."""
        result = ingestion_payload.copy()
        result.update({
            "text": "",
            "pages": 0,
            "total_pages": 0,
            "pages_data": [],
            "text_length": 0,
            "word_count": 0,
            "has_text": False,
            "needs_ocr": False,
            "images": [],
            "tables": [],
            "status": "success",
            "error_message": None,
        })

        file_path = Path(ingestion_payload["file_path"])
        ext = result["file_extension"]

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File {file_path} not found.")

            parsed_data = {}

            if ext == ".pdf":
                parsed_data = self._parse_pdf(file_path)
            elif ext == ".docx":
                parsed_data = self._parse_docx(file_path)
            elif ext == ".txt":
                parsed_data = self._parse_txt(file_path)
            elif ext in {".xlsx", ".xls", ".csv"}:
                parsed_data = self._parse_excel(file_path)
            elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}:
                parsed_data = self._parse_image(file_path)
            elif ext == ".json":
                parsed_data = self._parse_json_xml(file_path, is_json=True)
            elif ext == ".xml":
                parsed_data = self._parse_json_xml(file_path, is_json=False)
            else:
                parsed_data = {"text": "", "pages_data": [], "total_pages": 1, "images": [], "tables": []}

            text = parsed_data.get("text", "")
            text_length = len(text.strip())
            word_count = len(text.split()) if text.strip() else 0

            result.update({
                "text": text,
                "pages": parsed_data.get("total_pages", 1),
                "total_pages": parsed_data.get("total_pages", 1),
                "pages_data": parsed_data.get("pages_data", []),
                "text_length": text_length,
                "word_count": word_count,
                "has_text": text_length > 0,
                "needs_ocr": text_length < self.min_text_threshold,
                "images": parsed_data.get("images", []),
                "tables": parsed_data.get("tables", []),
            })

        except Exception as e:
            result["status"] = "failed"
            result["error_message"] = str(e)
            result["needs_ocr"] = True

        return result
