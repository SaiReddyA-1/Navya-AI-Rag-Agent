import json
import re
from typing import Dict, Any
from core.models import QueryRequest

class QueryParser:
    """
    Parses natural language into structured search parameters (intent + metadata filters).
    In a full production scenario, this uses an LLM call or fine-tuned NER model.
    """
    
    def parse_query(self, user_text: str) -> QueryRequest:
        text_lower = user_text.lower()
        filters: Dict[str, Any] = {}
        
        # 1. Simple heuristic keyword extraction (Simulating LLM NER extraction)
        if "finance" in text_lower or "financial" in text_lower:
            filters["document_type"] = "financial"
        elif "legal" in text_lower or "contract" in text_lower:
            filters["document_type"] = "legal"
            
        if "degraded" in text_lower or "bad quality" in text_lower:
             filters["max_quality"] = 0.5 # We want items with quality < 0.5
             
        # Simulate extraction of repository intent
        if "sharepoint" in text_lower:
            filters["repository"] = "sharepoint"
            
        print(f"[QueryParser] Parsed query to filters: {filters}")
        
        # The cleaned query for vector search:
        cleaned_query = re.sub(r'(in sharepoint|financial documents|finance|degraded|bad quality)', '', text_lower).strip()
        if not cleaned_query: 
            cleaned_query = user_text
            
        return QueryRequest(
            query=cleaned_query,
            filters=filters if filters else None,
            top_k=20 # Fetch 20, rerank to 5
        )
