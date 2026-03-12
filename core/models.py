from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Metadata related to an ingested document."""
    file_name: str
    file_path: str
    folder_path: str
    repository: str
    document_type: Optional[str] = "unknown"
    owner: Optional[str] = None
    department: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    
    # Fields from Parser/OCR
    page_count: Optional[int] = None
    text_length: Optional[int] = None
    has_text: Optional[bool] = None
    has_images: Optional[bool] = None
    has_tables: Optional[bool] = None
    
    # Fields populated by the Triage AI
    quality_score: Optional[float] = None
    is_degraded: Optional[bool] = None
    detected_issues: Optional[List[str]] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    department_category: Optional[str] = None
    triage_confidence: Optional[float] = None

class Document(BaseModel):
    """A fully extracted document entity."""
    document_id: str
    text: str
    metadata: DocumentMetadata
    tables: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    
class ChunkMetadata(BaseModel):
    """Metadata retained at the chunk level."""
    document_id: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    folder_path: str
    repository: str
    document_type: Optional[str] = None
    department_category: Optional[str] = None
    quality_score: Optional[float] = None
    is_degraded: Optional[bool] = None

class Chunk(BaseModel):
    """A smaller parsed section of a document. Ready for vectorization."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

class QueryRequest(BaseModel):
    """Incoming user search request."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 10

class RetrievedChunk(BaseModel):
    """A chunk retrieved from the search index."""
    chunk: Chunk
    score: float # Similarity / BM25 score

class AnswerSource(BaseModel):
    """Citation mapping used in LLM generation."""
    document_id: str
    file_name: str
    page_number: Optional[int] = None
    folder_path: str

class RAGResponse(BaseModel):
    """Final output to the user."""
    answer: str
    sources: List[AnswerSource]
    confidence_score: float
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
