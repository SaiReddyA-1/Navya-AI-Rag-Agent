"""
HybridSearchClient — OpenSearch-backed vector + metadata storage.
Handles index creation, bulk upload, hybrid search, and metadata queries.
Falls back to in-memory mode if OpenSearch is unavailable.
"""
import time
import uuid
import numpy as np
from typing import Dict, Any, List, Optional
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger(__name__)


class HybridSearchClient:
    """
    Production hybrid search using OpenSearch for vector similarity + metadata filtering.
    Automatically falls back to in-memory mode for development/testing.
    """

    def __init__(self, index_name: str = None, embedding_dim: int = None):
        self.index_name = index_name or settings.VECTOR_INDEX_NAME
        self.embedding_dim = embedding_dim or settings.EMBEDDING_DIMENSIONS
        self.os_client = None
        self.mode = "memory"  # "opensearch" or "memory"

        # In-memory fallback stores
        self._db_vectors: Dict[str, List[float]] = {}
        self._db_payloads: Dict[str, Dict[str, Any]] = {}

        self._connect_opensearch()

    def _connect_opensearch(self):
        """Attempt connection to OpenSearch with retry."""
        try:
            from opensearchpy import OpenSearch
            host = settings.OPENSEARCH_HOST
            port = settings.OPENSEARCH_PORT

            self.os_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                timeout=30,
            )

            # Wait for cluster to be ready
            for attempt in range(5):
                try:
                    info = self.os_client.info()
                    logger.info(f"OpenSearch connected: {info['version']['distribution']} {info['version']['number']}")
                    self.mode = "opensearch"
                    self._ensure_index()
                    return
                except Exception:
                    logger.info(f"Waiting for OpenSearch... (attempt {attempt + 1}/5)")
                    time.sleep(3)

            raise ConnectionError("OpenSearch not ready after retries")

        except Exception as e:
            logger.warning(f"OpenSearch unavailable ({e}). Using in-memory mode.")
            self.os_client = None
            self.mode = "memory"

    def _ensure_index(self):
        """Create the vector index if it doesn't exist."""
        if self.os_client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists")
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 128, "m": 16},
                        },
                    },
                    "text": {"type": "text", "analyzer": "standard"},
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "file_extension": {"type": "keyword"},
                    "folder_path": {"type": "keyword"},
                    "repository": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "department_category": {"type": "keyword"},
                    "quality_score": {"type": "float"},
                    "is_degraded": {"type": "boolean"},
                    "detected_issues": {"type": "keyword"},
                    "triage_confidence": {"type": "float"},
                    "recommended_action": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "page_number": {"type": "integer"},
                    "total_pages": {"type": "integer"},
                    "total_chunks_in_doc": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "file_size_bytes": {"type": "long"},
                    "created_time": {"type": "keyword"},
                    "modified_time": {"type": "keyword"},
                    "ingested_time": {"type": "keyword"},
                }
            },
        }
        self.os_client.indices.create(index=self.index_name, body=index_body)
        logger.info(f"Created index '{self.index_name}' with HNSW vector config")

    # ── Upload ────────────────────────────────────────────────────────────

    def upload_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert chunks with embeddings into the DB."""
        if self.mode == "opensearch":
            return self._os_upload(chunks)
        return self._mem_upload(chunks)

    def _os_upload(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk upload to OpenSearch."""
        from opensearchpy.helpers import bulk

        actions = []
        for chunk in chunks:
            if "embedding" not in chunk:
                continue
            chunk_id = chunk.get("chunk_id") or str(uuid.uuid4())
            doc = {k: v for k, v in chunk.items()}
            actions.append({"_index": self.index_name, "_id": chunk_id, "_source": doc})

        if not actions:
            return {"status": "success", "inserted": 0, "failed": 0}

        success, errors = bulk(self.os_client, actions, raise_on_error=False)
        failed = len(errors) if isinstance(errors, list) else 0
        logger.info(f"OpenSearch bulk upload: {success} ok, {failed} failed")
        # Refresh index for immediate searchability
        self.os_client.indices.refresh(index=self.index_name)
        return {"status": "success", "inserted": success, "failed": failed}

    def _mem_upload(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """In-memory fallback upload."""
        success_count = 0
        for chunk in chunks:
            if "embedding" not in chunk:
                continue
            chunk_id = chunk.get("chunk_id") or str(uuid.uuid4())
            self._db_vectors[chunk_id] = chunk["embedding"]
            self._db_payloads[chunk_id] = {k: v for k, v in chunk.items() if k != "embedding"}
            success_count += 1
        return {"status": "success", "inserted": success_count, "failed": 0}

    # ── Search ────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query_vector: List[float],
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Vector similarity + metadata filter search."""
        if self.mode == "opensearch":
            return self._os_search(query_vector, metadata_filters, top_k)
        return self._mem_search(query_vector, metadata_filters, top_k)

    def _os_search(
        self, query_vector: List[float], metadata_filters: Optional[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """OpenSearch k-NN search with metadata filters."""
        must_clauses = []
        for key, val in (metadata_filters or {}).items():
            must_clauses.append({"term": {key: val}})

        query_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k,
                                }
                            }
                        }
                    ],
                }
            }
            if must_clauses
            else {
                "size": top_k,
                "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}},
            },
        }

        # Simplified: if no metadata filters, use pure knn
        if not must_clauses:
            query_body = {
                "size": top_k,
                "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}},
            }
        else:
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "filter": [{"term": {k: v}} for k, v in metadata_filters.items()],
                        "must": [
                            {"knn": {"embedding": {"vector": query_vector, "k": top_k}}}
                        ],
                    }
                },
            }

        try:
            resp = self.os_client.search(index=self.index_name, body=query_body)
            results = []
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                chunk = {k: v for k, v in src.items() if k != "embedding"}
                chunk["score"] = hit["_score"]
                results.append(chunk)
            return results
        except Exception as e:
            logger.error(f"OpenSearch search failed: {e}")
            return []

    def _mem_search(
        self, query_vector: List[float], metadata_filters: Optional[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """In-memory cosine similarity search."""
        results = []
        metadata_filters = metadata_filters or {}
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        for chunk_id, payload in self._db_payloads.items():
            if not all(payload.get(k) == v for k, v in metadata_filters.items()):
                continue
            db_vec = self._db_vectors.get(chunk_id)
            if not db_vec:
                continue
            d_vec = np.array(db_vec, dtype=np.float32)
            d_norm = np.linalg.norm(d_vec)
            if d_norm == 0:
                continue
            score = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
            chunk = payload.copy()
            chunk["score"] = score
            results.append(chunk)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── Analytics / Metadata Queries ──────────────────────────────────────

    def count_by_filter(self, metadata_filters: Dict[str, Any]) -> int:
        """Pure metadata count query for dashboards."""
        if self.mode == "opensearch":
            return self._os_count(metadata_filters)
        return sum(
            1
            for p in self._db_payloads.values()
            if all(p.get(k) == v for k, v in metadata_filters.items())
        )

    def _os_count(self, metadata_filters: Dict[str, Any]) -> int:
        must = [{"term": {k: v}} for k, v in metadata_filters.items()]
        body = {"query": {"bool": {"must": must}}}
        try:
            return self.os_client.count(index=self.index_name, body=body)["count"]
        except Exception:
            return 0

    def get_all_payloads(self) -> List[Dict[str, Any]]:
        """Return all stored chunk metadata (for analytics)."""
        if self.mode == "opensearch":
            return self._os_get_all()
        return list(self._db_payloads.values())

    def _os_get_all(self) -> List[Dict[str, Any]]:
        """Scroll through all documents in OpenSearch."""
        results = []
        try:
            body = {"query": {"match_all": {}}, "size": 500}
            resp = self.os_client.search(
                index=self.index_name, body=body, scroll="2m"
            )
            scroll_id = resp.get("_scroll_id")
            hits = resp["hits"]["hits"]

            while hits:
                for hit in hits:
                    src = hit["_source"]
                    results.append({k: v for k, v in src.items() if k != "embedding"})
                resp = self.os_client.scroll(scroll_id=scroll_id, scroll="2m")
                hits = resp["hits"]["hits"]

            if scroll_id:
                self.os_client.clear_scroll(scroll_id=scroll_id)
        except Exception as e:
            logger.error(f"OpenSearch scroll failed: {e}")
        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """Return index-level stats for pipeline monitor."""
        if self.mode == "opensearch":
            try:
                stats = self.os_client.indices.stats(index=self.index_name)
                idx = stats["indices"].get(self.index_name, {})
                primaries = idx.get("primaries", {})
                return {
                    "mode": "opensearch",
                    "index_name": self.index_name,
                    "total_chunks": primaries.get("docs", {}).get("count", 0),
                    "store_size": primaries.get("store", {}).get("size_in_bytes", 0),
                    "search_count": primaries.get("search", {}).get("query_total", 0),
                }
            except Exception:
                return {"mode": "opensearch", "error": "stats unavailable"}
        return {
            "mode": "memory",
            "index_name": self.index_name,
            "total_chunks": len(self._db_payloads),
            "store_size": 0,
            "search_count": 0,
        }

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete all chunks belonging to a document."""
        if self.mode == "opensearch":
            return self._os_delete_doc(document_id)
        return self._mem_delete_doc(document_id)

    def _os_delete_doc(self, document_id: str) -> Dict[str, Any]:
        try:
            body = {"query": {"term": {"document_id": document_id}}}
            resp = self.os_client.delete_by_query(index=self.index_name, body=body)
            self.os_client.indices.refresh(index=self.index_name)
            deleted = resp.get("deleted", 0)
            return {"status": "success", "deleted_chunks": deleted}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _mem_delete_doc(self, document_id: str) -> Dict[str, Any]:
        to_delete = [
            cid
            for cid, p in self._db_payloads.items()
            if p.get("document_id") == document_id
        ]
        for cid in to_delete:
            self._db_payloads.pop(cid, None)
            self._db_vectors.pop(cid, None)
        return {"status": "success", "deleted_chunks": len(to_delete)}

    def get_total_count(self) -> int:
        """Quick total document count."""
        if self.mode == "opensearch":
            try:
                return self.os_client.count(index=self.index_name, body={"query": {"match_all": {}}})["count"]
            except Exception:
                return 0
        return len(self._db_payloads)

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve ALL chunks of a specific document, ordered by chunk_index."""
        if self.mode == "opensearch":
            return self._os_get_doc_chunks(document_id)
        return self._mem_get_doc_chunks(document_id)

    def _os_get_doc_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        try:
            body = {
                "query": {"term": {"document_id": document_id}},
                "size": 500,
                "sort": [{"chunk_index": {"order": "asc"}}],
            }
            resp = self.os_client.search(index=self.index_name, body=body)
            results = []
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                results.append({k: v for k, v in src.items() if k != "embedding"})
            return results
        except Exception as e:
            logger.error(f"get_document_chunks failed: {e}")
            return []

    def _mem_get_doc_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        chunks = [
            p.copy()
            for p in self._db_payloads.values()
            if p.get("document_id") == document_id
        ]
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        return chunks

    def find_documents_by_metadata(
        self, filters: Dict[str, Any], max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for documents by arbitrary metadata fields.
        Supports wildcard matching for file_path and folder_path.
        Returns unique documents (deduplicated by document_id).
        """
        if self.mode == "opensearch":
            return self._os_find_by_metadata(filters, max_results)
        return self._mem_find_by_metadata(filters, max_results)

    def _os_find_by_metadata(self, filters: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        try:
            must_clauses = []
            wildcard_fields = {"file_path", "folder_path", "file_name"}
            for key, val in filters.items():
                if key in wildcard_fields and "*" not in str(val):
                    must_clauses.append({"wildcard": {key: f"*{val}*"}})
                elif key in wildcard_fields:
                    must_clauses.append({"wildcard": {key: val}})
                else:
                    must_clauses.append({"term": {key: val}})

            body = {
                "query": {"bool": {"must": must_clauses}},
                "size": max_results * 3,  # over-fetch to deduplicate
            }
            resp = self.os_client.search(index=self.index_name, body=body)
            seen_ids = set()
            results = []
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                doc_id = src.get("document_id")
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                results.append({k: v for k, v in src.items() if k != "embedding"})
                if len(results) >= max_results:
                    break
            return results
        except Exception as e:
            logger.error(f"find_documents_by_metadata failed: {e}")
            return []

    def _mem_find_by_metadata(self, filters: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        seen_ids = set()
        results = []
        wildcard_fields = {"file_path", "folder_path", "file_name"}
        for payload in self._db_payloads.values():
            match = True
            for key, val in filters.items():
                pval = str(payload.get(key, ""))
                if key in wildcard_fields:
                    if str(val).replace("*", "") not in pval:
                        match = False
                        break
                elif payload.get(key) != val:
                    match = False
                    break
            if match:
                doc_id = payload.get("document_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(payload.copy())
                    if len(results) >= max_results:
                        break
        return results

    def find_document_by_filename(self, file_name: str) -> Optional[str]:
        """Find document_id by file name (for follow-up queries)."""
        if self.mode == "opensearch":
            try:
                body = {"query": {"term": {"file_name": file_name}}, "size": 1}
                resp = self.os_client.search(index=self.index_name, body=body)
                hits = resp["hits"]["hits"]
                if hits:
                    return hits[0]["_source"].get("document_id")
            except Exception:
                pass
            return None
        else:
            for p in self._db_payloads.values():
                if p.get("file_name") == file_name:
                    return p.get("document_id")
            return None
