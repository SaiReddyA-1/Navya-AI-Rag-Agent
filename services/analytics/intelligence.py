"""
Repository Intelligence Engine — analytics on metadata without vector math.
Powers the Intelligence Dashboard with aggregate queries.
"""
from typing import Dict, Any, List
from collections import defaultdict
from services.search.database import HybridSearchClient


class RepositoryIntelligenceEngine:
    """
    Computes analytics on metadata for repository health and compliance.
    """

    def __init__(self, db_client: HybridSearchClient):
        self.db = db_client

    def _get_all_payloads(self) -> List[Dict[str, Any]]:
        """Fetch all chunk payloads from DB."""
        return self.db.get_all_payloads()

    def _get_unique_documents(self, payloads: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Deduplicate chunks to get unique documents."""
        if payloads is None:
            payloads = self._get_all_payloads()
        seen = set()
        unique = []
        for p in payloads:
            doc_id = p.get("document_id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique.append(p)
        return unique

    def get_repository_summary(self) -> Dict[str, Any]:
        """High-level metrics for dashboard."""
        payloads = self._get_all_payloads()
        unique_docs = self._get_unique_documents(payloads)
        return {
            "total_documents_ingested": len(unique_docs),
            "total_searchable_chunks": len(payloads),
            "repository_health_score": self._avg_quality(unique_docs),
        }

    def _avg_quality(self, docs: List[Dict[str, Any]]) -> float:
        if not docs:
            return 0.0
        total = sum(d.get("quality_score", 0.0) for d in docs)
        return round(total / len(docs), 2)

    def count_by_department(self) -> Dict[str, int]:
        """GROUP BY department_category."""
        docs = self._get_unique_documents()
        counts = defaultdict(int)
        for d in docs:
            counts[d.get("department_category", "unknown")] += 1
        return dict(counts)

    def count_by_document_type(self) -> Dict[str, int]:
        """GROUP BY document_type."""
        docs = self._get_unique_documents()
        counts = defaultdict(int)
        for d in docs:
            counts[d.get("document_type", "unknown")] += 1
        return dict(counts)

    def get_degradation_report(self) -> Dict[str, Any]:
        """Stats on documents failing quality threshold."""
        docs = self._get_unique_documents()
        total = len(docs)
        if total == 0:
            return {"degraded_count": 0, "degraded_percentage": 0.0, "common_issues": {}}

        degraded = [d for d in docs if d.get("is_degraded", False)]
        issue_counts = defaultdict(int)
        for d in degraded:
            for issue in d.get("detected_issues", []):
                if issue != "none":
                    issue_counts[issue] += 1

        return {
            "degraded_count": len(degraded),
            "degraded_percentage": round((len(degraded) / total) * 100, 1),
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)),
        }

    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all unique documents with metadata for Data Manager."""
        docs = self._get_unique_documents()
        result = []
        # Count chunks per document
        all_payloads = self._get_all_payloads()
        chunk_counts = defaultdict(int)
        for p in all_payloads:
            chunk_counts[p.get("document_id", "")] += 1

        for d in docs:
            doc_id = d.get("document_id", "")
            result.append({
                "document_id": doc_id,
                "file_name": d.get("file_name", ""),
                "file_path": d.get("file_path", ""),
                "file_extension": d.get("file_extension", ""),
                "document_type": d.get("document_type", "unknown"),
                "department": d.get("department_category", "general"),
                "quality_score": d.get("quality_score", 0.0),
                "is_degraded": d.get("is_degraded", False),
                "detected_issues": d.get("detected_issues", []),
                "triage_confidence": d.get("triage_confidence", 0.0),
                "chunk_count": chunk_counts.get(doc_id, 0),
                "ingested_time": d.get("ingested_time", ""),
            })
        return result

    def get_pipeline_state(self) -> Dict[str, Any]:
        """Compute pipeline stage statistics for Pipeline Monitor."""
        payloads = self._get_all_payloads()
        docs = self._get_unique_documents(payloads)

        # Count by status
        ext_counts = defaultdict(int)
        type_counts = defaultdict(int)
        dept_counts = defaultdict(int)
        quality_buckets = {"excellent": 0, "good": 0, "fair": 0, "degraded": 0}

        for d in docs:
            ext_counts[d.get("file_extension", "unknown")] += 1
            type_counts[d.get("document_type", "unknown")] += 1
            dept_counts[d.get("department_category", "unknown")] += 1
            q = d.get("quality_score", 0.0)
            if q >= 0.9:
                quality_buckets["excellent"] += 1
            elif q >= 0.7:
                quality_buckets["good"] += 1
            elif q >= 0.5:
                quality_buckets["fair"] += 1
            else:
                quality_buckets["degraded"] += 1

        return {
            "total_documents": len(docs),
            "total_chunks": len(payloads),
            "avg_chunks_per_doc": round(len(payloads) / len(docs), 1) if docs else 0,
            "file_types": dict(ext_counts),
            "document_types": dict(type_counts),
            "departments": dict(dept_counts),
            "quality_distribution": quality_buckets,
            "db_stats": self.db.get_index_stats(),
        }
