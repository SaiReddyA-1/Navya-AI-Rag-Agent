FROM python:3.11-slim

WORKDIR /app

# System dependencies (curl + Tesseract OCR for image text extraction)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer)
# Install CPU-only PyTorch first (~200MB instead of ~915MB GPU version)
# This prevents sentence-transformers from pulling the full CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download ML models during build so startup is fast
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-large-en-v1.5'); \
print('Downloading reranker model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached.')"

# Copy application code
COPY . .

EXPOSE 8501

# Default: run ingestion then start Streamlit
CMD ["sh", "-c", "python scripts/ingest.py && streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0"]
