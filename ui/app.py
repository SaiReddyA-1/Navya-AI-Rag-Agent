"""
NavAI RAG — Enterprise Document Intelligence Platform
Multi-page Streamlit Dashboard with persistent chat history.
"""
import streamlit as st
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
from services.search.database import HybridSearchClient
from services.search.retriever import QueryRetriever
from services.rag.reranker import CrossEncoderReranker
from services.rag.engine import RAGEngine
from services.analytics.intelligence import RepositoryIntelligenceEngine
from services.chat.history import ChatHistoryManager
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger("ui")

st.set_page_config(
    page_title="NavAI RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetric"] label { font-size: 12px !important; color: #6c757d !important; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 700 !important; color: #212529 !important; }

    /* Pipeline stages */
    .pipeline-stage {
        background: #ffffff;
        padding: 16px 20px;
        border-radius: 10px;
        border-left: 5px solid #228be6;
        margin-bottom: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border: 1px solid #dee2e6;
    }
    .pipeline-stage-active { border-left-color: #2b8a3e; }

    /* Badges */
    .status-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600;
    }
    .badge-green { background: #d3f9d8; color: #2b8a3e; }
    .badge-red { background: #ffe3e3; color: #c92a2a; }
    .badge-yellow { background: #fff3bf; color: #e67700; }

    /* Sidebar styling */
    .sidebar-card {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
        border: 1px solid #ced4da;
    }
    .sidebar-label {
        color: #6c757d; font-size: 10px; text-transform: uppercase;
        letter-spacing: 1.2px; margin-bottom: 2px; font-weight: 500;
    }
    .sidebar-value { font-weight: 700; font-size: 15px; }
    .sidebar-value-green { color: #2b8a3e; }
    .sidebar-value-blue { color: #1971c2; }
    .sidebar-value-dark { color: #212529; font-size: 13px; }

    /* Hero metrics */
    .hero-metric {
        text-align: center; padding: 20px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px; border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .hero-metric .value { font-size: 36px; font-weight: 800; color: #212529; }
    .hero-metric .label { font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    /* Section headers */
    .section-divider {
        border: none; border-top: 2px solid #e9ecef; margin: 24px 0 16px 0;
    }

    /* Dataframe improvements */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Chat controls */
    .chat-controls {
        background: #f8f9fa; padding: 10px 16px; border-radius: 8px;
        border: 1px solid #e9ecef; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Backend Initialization ────────────────────────────────────────────────
@st.cache_resource
def load_backend():
    """Initialize all pipeline components once."""
    db = HybridSearchClient()
    embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    retriever = QueryRetriever(db, model=embed_model)
    reranker = CrossEncoderReranker()
    analytics = RepositoryIntelligenceEngine(db)
    rag_engine = RAGEngine(retriever, reranker, analytics_engine=analytics)
    chat_mgr = ChatHistoryManager()
    return db, embed_model, retriever, reranker, rag_engine, analytics, chat_mgr


def get_backend():
    return load_backend()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1: AI Chat & Retrieval
# ══════════════════════════════════════════════════════════════════════════
def page_chat():
    st.header("AI Chat & Retrieval")
    st.caption("Ask questions against your enterprise document repository. Answers are grounded with citations.")

    _, _, _, _, rag_engine, _, chat_mgr = get_backend()

    # ── Initialize session state ──────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "chat_loaded" not in st.session_state:
        # Auto-load most recent conversation on first visit
        conversations = chat_mgr.load_conversations()
        if conversations:
            latest = conversations[-1]
            st.session_state.messages = latest["messages"]
            st.session_state.conversation_id = latest["id"]
        st.session_state.chat_loaded = True

    # ── Conversation Controls ─────────────────────────────────────────
    conversations = chat_mgr.load_conversations()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        if conversations:
            options = ["Current"] + [
                f"{c['updated'][:10]} — {c['preview'][:45]}..."
                for c in reversed(conversations)
            ]
            selected_idx = st.selectbox(
                "History",
                range(len(options)),
                format_func=lambda i: options[i],
                label_visibility="collapsed",
            )
            if selected_idx > 0:
                conv = list(reversed(conversations))[selected_idx - 1]
                if st.session_state.conversation_id != conv["id"]:
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = conv["messages"]
                    st.rerun()
    with col2:
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
    with col3:
        if conversations and st.button("Clear All", use_container_width=True):
            chat_mgr.clear_all()
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    # ── Render chat history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} Sources"):
                    for i, s in enumerate(msg["sources"]):
                        st.caption(
                            f"**[{i+1}]** {s['file_name']} · "
                            f"{s.get('document_type','')} · "
                            f"{s.get('department','')} · "
                            f"Quality: {s.get('quality_score',0):.0%}"
                        )

    # ── Input ─────────────────────────────────────────────────────────
    if query := st.chat_input("E.g., 'Show me finance invoices' or 'What purchase orders exist?'"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving → Reranking → Generating..."):
                resp = rag_engine.execute_rag(
                    query,
                    chat_history=st.session_state.messages,
                )

            st.markdown(resp["answer"])

            conf = resp.get("confidence_score", 0.0)
            sources = resp.get("sources", [])
            query_type = resp.get("query_type", "retrieval")

            # Confidence badge
            badge_parts = [f"Confidence: {conf:.0%}"]
            if "analytics" in query_type:
                badge_parts.append("Analytics")
            if "full_document" in query_type:
                badge_parts.append("Full Document")
            if "retrieval" in query_type:
                badge_parts.append("Semantic Search")
            badge_text = " · ".join(badge_parts)

            if conf >= 0.8:
                st.success(badge_text)
            elif conf >= 0.5:
                st.warning(badge_text)
            else:
                st.error(f"Low {badge_text}")

            if sources:
                with st.expander(f"📚 {len(sources)} Sources"):
                    for i, s in enumerate(sources):
                        st.caption(
                            f"**[{i+1}]** {s['file_name']} · "
                            f"{s.get('document_type','')} · "
                            f"{s.get('department','')} · "
                            f"Quality: {s.get('quality_score',0):.0%}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": resp["answer"],
                "sources": sources,
                "confidence": conf,
            })

            # Auto-save conversation
            st.session_state.conversation_id = chat_mgr.save_conversation(
                st.session_state.messages,
                st.session_state.conversation_id,
            )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2: Data Manager
# ══════════════════════════════════════════════════════════════════════════
def page_data_manager():
    st.header("Data Manager")
    st.caption("View, search, and manage documents in the RAG knowledge base.")

    db, embed_model, _, _, _, analytics, _ = get_backend()

    # ── Summary Bar ───────────────────────────────────────────────────
    docs = analytics.get_document_list()
    type_counts = analytics.count_by_document_type()
    dept_counts = analytics.count_by_department()
    total_chunks = sum(d["chunk_count"] for d in docs)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", len(docs))
    col2.metric("Chunks", f"{total_chunks:,}")
    col3.metric("Types", len(type_counts))
    col4.metric("Departments", len(dept_counts))

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_term = st.text_input("Search documents", placeholder="Filter by filename...")
    with col2:
        filter_type = st.selectbox("Filter by type", ["All"] + sorted(list(type_counts.keys())))
    with col3:
        st.write("")
        refresh = st.button("Refresh", use_container_width=True)

    # ── Filtered Documents ────────────────────────────────────────────
    filtered = docs
    if search_term:
        filtered = [d for d in filtered if search_term.lower() in d["file_name"].lower()]
    if filter_type != "All":
        filtered = [d for d in filtered if d["document_type"] == filter_type]

    if not filtered:
        st.info("No documents found matching your criteria.")
        return

    df = pd.DataFrame(filtered)
    display_cols = [
        "file_name", "file_path", "document_type", "department", "quality_score",
        "is_degraded", "chunk_count", "triage_confidence", "file_extension"
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    # Column config for better display
    column_config = {
        "file_name": st.column_config.TextColumn("File Name", width="medium"),
        "file_path": st.column_config.TextColumn("File Path", width="large"),
        "document_type": st.column_config.TextColumn("Type", width="small"),
        "department": st.column_config.TextColumn("Department", width="small"),
        "quality_score": st.column_config.ProgressColumn(
            "Quality", min_value=0, max_value=1, format="%.0f%%",
        ),
        "is_degraded": st.column_config.CheckboxColumn("Degraded", width="small"),
        "chunk_count": st.column_config.NumberColumn("Chunks", width="small"),
        "triage_confidence": st.column_config.ProgressColumn(
            "Triage Conf.", min_value=0, max_value=1, format="%.0f%%",
        ),
        "file_extension": st.column_config.TextColumn("Ext", width="small"),
    }

    st.dataframe(
        df[display_cols],
        column_config=column_config,
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    st.caption(f"Showing {len(filtered)} of {len(docs)} documents · {sum(d['chunk_count'] for d in filtered):,} chunks")

    # ── Delete Documents ──────────────────────────────────────────────
    st.divider()
    st.subheader("Delete Documents")
    st.warning("This will permanently remove selected documents and all their chunks from the index.")

    del_options = {f"{d['file_name']} ({d['document_id'][:12]}...)": d['document_id'] for d in filtered}
    selected_docs = st.multiselect("Select documents to delete", list(del_options.keys()))

    if selected_docs:
        st.info(f"{len(selected_docs)} document(s) selected for deletion.")
        if st.button(f"Delete {len(selected_docs)} Document(s)", type="primary"):
            total_deleted = 0
            for label in selected_docs:
                doc_id = del_options[label]
                result = db.delete_document(doc_id)
                if result["status"] == "success":
                    total_deleted += result["deleted_chunks"]
            st.success(f"Deleted {total_deleted} chunks from {len(selected_docs)} documents.")
            st.cache_resource.clear()
            st.rerun()

    # ── Add Documents ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Add New Documents")

    uploaded_files = st.file_uploader(
        "Upload documents to ingest into the RAG pipeline",
        type=["pdf", "txt", "csv", "docx", "json", "xlsx", "xls", "xml",
              "jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button(f"Ingest {len(uploaded_files)} Document(s)", type="primary"):
        progress = st.progress(0, text="Starting pipeline...")
        for i, uploaded in enumerate(uploaded_files):
            progress.progress(
                (i) / len(uploaded_files),
                text=f"Processing {uploaded.name} ({i+1}/{len(uploaded_files)})..."
            )
            _ingest_uploaded_file(uploaded, db, embed_model)
        progress.progress(1.0, text="All documents ingested!")
        st.success(f"Successfully ingested {len(uploaded_files)} document(s).")
        st.cache_resource.clear()
        st.rerun()


def _ingest_uploaded_file(uploaded_file, db, embed_model):
    """Run the full offline pipeline on an uploaded file.
    Saves the file to Data/Rag/uploads/ so it persists across restarts.
    """
    import hashlib
    from services.processing.parser import DocumentParser
    from services.processing.ocr import OCRService
    from services.processing.triage import TriageService
    from services.embedding.chunker import AdvancedChunkingEngine
    from services.embedding.generator import EmbeddingGenerator

    # Save to persistent uploads directory (Docker-mounted volume)
    upload_dir = Path(settings.DATA_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / uploaded_file.name

    # If file with same name exists, add suffix to avoid overwrite
    if save_path.exists():
        stem = save_path.stem
        ext = save_path.suffix
        counter = 1
        while save_path.exists():
            save_path = upload_dir / f"{stem}_{counter}{ext}"
            counter += 1

    file_bytes = uploaded_file.read()
    save_path.write_bytes(file_bytes)

    doc_id = hashlib.sha256(file_bytes).hexdigest()

    payload = {
        "document_id": doc_id,
        "file_path": str(save_path),
        "file_name": uploaded_file.name,
        "file_extension": Path(uploaded_file.name).suffix.lower(),
        "file_size_bytes": len(file_bytes),
        "created_time": datetime.utcnow().isoformat() + "Z",
        "modified_time": datetime.utcnow().isoformat() + "Z",
        "ingested_time": datetime.utcnow().isoformat() + "Z",
    }

    parsed = DocumentParser().parse_document(payload)
    ocr_result = OCRService().process_document(parsed)
    triaged = TriageService().process_document(ocr_result)
    chunks = AdvancedChunkingEngine().process_document(triaged)

    if chunks:
        embedder = EmbeddingGenerator(model=embed_model)
        embedded = embedder.process_chunks(chunks)
        db.upload_chunks(embedded)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3: Pipeline Monitor
# ══════════════════════════════════════════════════════════════════════════
def page_pipeline_monitor():
    st.header("Pipeline Monitor")
    st.caption("End-to-end pipeline state, database health, and processing statistics.")

    _, _, _, _, _, analytics, _ = get_backend()
    db = get_backend()[0]

    state = analytics.get_pipeline_state()
    db_stats = state["db_stats"]

    # ── Pipeline Flow ─────────────────────────────────────────────────
    st.subheader("Pipeline Architecture")

    stages = [
        ("Repository Scanner", f"{state['total_documents']} documents discovered", "Scans Data/Rag/ recursively for supported files", "📂"),
        ("Document Parser", f"{', '.join(f'{v} {k}' for k, v in state['file_types'].items())}", "Extracts text from PDF, DOCX, CSV, TXT, images", "📄"),
        ("OCR Engine", "Conditional — runs on low-text documents", "Text extraction for scanned/image docs", "🔍"),
        ("Triage AI", f"{len(state['document_types'])} types · {len(state['departments'])} departments", "Classifies type, department, quality score", "🏷️"),
        ("Chunking Engine", f"{state['total_chunks']:,} chunks · {state['avg_chunks_per_doc']} avg/doc", f"Recursive split: {settings.CHUNK_SIZE} chars, {settings.CHUNK_OVERLAP} overlap", "✂️"),
        ("Embedding Generator", f"{settings.EMBEDDING_MODEL_NAME.split('/')[-1]} ({settings.EMBEDDING_DIMENSIONS}D)", "Dense vector encoding with batch processing", "🧬"),
        ("Vector Index", f"{db_stats.get('total_chunks', state['total_chunks']):,} indexed · {db_stats['mode'].upper()}", "HNSW cosine similarity + keyword metadata", "🗄️"),
    ]

    for i, (name, metric, desc, icon) in enumerate(stages):
        col_status, col_info, col_metric = st.columns([0.5, 3, 2])
        with col_status:
            st.markdown(f"<div style='text-align:center; padding-top:12px; font-size:24px;'>{icon}</div>", unsafe_allow_html=True)
        with col_info:
            st.markdown(f"""
            <div class="pipeline-stage pipeline-stage-active">
                <strong style="color: #212529;">Stage {i+1}: {name}</strong><br/>
                <span style="color: #495057; font-size: 13px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_metric:
            st.markdown(f"""
            <div style="padding: 14px; text-align: center; background: #ffffff; border-radius: 10px; border: 1px solid #dee2e6; box-shadow: 0 1px 4px rgba(0,0,0,0.08);">
                <span style="font-size: 14px; font-weight: 600; color: #1971c2;">{metric}</span>
            </div>
            """, unsafe_allow_html=True)
        if i < len(stages) - 1:
            st.markdown("<div style='text-align: left; color: #adb5bd; font-size: 18px; padding-left: 22px;'>│</div>", unsafe_allow_html=True)

    st.divider()

    # ── Database State ────────────────────────────────────────────────
    st.subheader("Database State")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mode", db_stats["mode"].upper())
    col2.metric("Index", db_stats.get("index_name", "N/A"))
    col3.metric("Total Chunks", f"{db_stats.get('total_chunks', state['total_chunks']):,}")
    if db_stats.get("store_size"):
        size_mb = round(db_stats["store_size"] / (1024 * 1024), 2)
        col4.metric("Store Size", f"{size_mb} MB")
    else:
        col4.metric("Searches", db_stats.get("search_count", 0))

    st.divider()

    # ── Classification Results ────────────────────────────────────────
    st.subheader("Triage Classification Results")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Document Types**")
        if state["document_types"]:
            df_types = pd.DataFrame(
                {"Type": list(state["document_types"].keys()),
                 "Count": list(state["document_types"].values())}
            ).set_index("Type")
            st.bar_chart(df_types)
        else:
            st.info("No data")

    with col_b:
        st.write("**Departments**")
        if state["departments"]:
            df_dept = pd.DataFrame(
                {"Department": list(state["departments"].keys()),
                 "Count": list(state["departments"].values())}
            ).set_index("Department")
            st.bar_chart(df_dept)
        else:
            st.info("No data")

    st.divider()

    # ── Quality Distribution ──────────────────────────────────────────
    st.subheader("Quality Distribution")

    q = state["quality_distribution"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Excellent (≥90%)", q["excellent"])
    col2.metric("Good (70-89%)", q["good"])
    col3.metric("Fair (50-69%)", q["fair"])
    col4.metric("Degraded (<50%)", q["degraded"])

    st.divider()

    # ── Config Summary ────────────────────────────────────────────────
    st.subheader("Active Configuration")

    config_data = {
        "Setting": [
            "Embedding Model", "Embedding Dimensions", "Reranker Model",
            "Chunk Size", "Chunk Overlap", "Top-K Retrieval", "Top-K Rerank",
            "LLM Provider", "LLM Model", "Quality Threshold", "Data Directory",
        ],
        "Value": [
            settings.EMBEDDING_MODEL_NAME, str(settings.EMBEDDING_DIMENSIONS),
            settings.RERANKER_MODEL_NAME, str(settings.CHUNK_SIZE),
            str(settings.CHUNK_OVERLAP), str(settings.TOP_K_RETRIEVAL),
            str(settings.TOP_K_RERANK), "Groq API", settings.GROQ_MODEL_NAME,
            str(settings.TRIAGE_QUALITY_THRESHOLD), settings.DATA_DIR,
        ],
    }
    st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4: Intelligence Dashboard
# ══════════════════════════════════════════════════════════════════════════
def page_intelligence():
    st.header("Repository Intelligence Dashboard")
    st.caption("Enterprise analytics on document quality, compliance, and risk.")

    _, _, _, _, _, analytics, _ = get_backend()

    # ── Hero Metrics ──────────────────────────────────────────────────
    summary = analytics.get_repository_summary()
    health = summary["repository_health_score"]

    # Color the health score
    if isinstance(health, float) and health > 0:
        health_display = f"{health:.0%}"
        if health >= 0.8:
            health_color = "#40c057"
        elif health >= 0.6:
            health_color = "#e67700"
        else:
            health_color = "#c92a2a"
    else:
        health_display = "N/A"
        health_color = "#6c757d"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="hero-metric">
            <div class="value">{summary['total_documents_ingested']}</div>
            <div class="label">Total Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="hero-metric">
            <div class="value">{summary['total_searchable_chunks']:,}</div>
            <div class="label">Searchable Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="hero-metric">
            <div class="value" style="color: {health_color};">{health_display}</div>
            <div class="label">Repository Health</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # ── Distributions ─────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Documents by Department")
        dept = analytics.count_by_department()
        if dept:
            df_dept = pd.DataFrame(
                {"Department": list(dept.keys()), "Count": list(dept.values())}
            ).set_index("Department")
            st.bar_chart(df_dept)
        else:
            st.info("No data")

    with col_b:
        st.subheader("Documents by Type")
        types = analytics.count_by_document_type()
        if types:
            df_types = pd.DataFrame(
                {"Type": list(types.keys()), "Count": list(types.values())}
            ).set_index("Type")
            st.bar_chart(df_types)
        else:
            st.info("No data")

    st.divider()

    # ── Degradation Report ────────────────────────────────────────────
    st.subheader("Degradation & Risk Report")

    risk = analytics.get_degradation_report()
    pct = risk["degraded_percentage"]

    col1, col2 = st.columns(2)
    col1.metric("Degraded Documents", risk["degraded_count"])
    col2.metric("Degradation Rate", f"{pct}%")

    if pct > 10:
        st.error(f"ALERT: {pct}% of documents are below quality threshold ({settings.TRIAGE_QUALITY_THRESHOLD}). Recommend rescanning affected files.")
    elif pct > 0:
        st.warning(f"{pct}% of documents are below quality threshold.")
    else:
        st.success("All documents passing quality checks.")

    if risk["common_issues"]:
        st.write("**Common Issues:**")
        for issue, count in risk["common_issues"].items():
            st.write(f"- `{issue}`: {count} documents")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN — Sidebar Navigation
# ══════════════════════════════════════════════════════════════════════════
def main():
    db = get_backend()[0]

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 8px 0 4px 0;">
            <span style="font-size: 28px; font-weight: 800; color: #228be6;">Nav</span><span style="font-size: 28px; font-weight: 800; color: #212529;">AI</span>
            <span style="font-size: 28px; font-weight: 800; color: #868e96;"> RAG</span>
            <div style="font-size: 11px; color: #868e96; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;">Enterprise Document Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🤖 AI Chat",
                "📁 Data Manager",
                "⚙️ Pipeline Monitor",
                "📊 Intelligence",
            ],
        )

        st.divider()

        # ── Live System Stats ─────────────────────────────────────────
        stats = db.get_index_stats()
        total_chunks = stats.get("total_chunks", 0)
        db_mode = stats["mode"].upper()
        embed_name = settings.EMBEDDING_MODEL_NAME.split("/")[-1]
        llm_model = settings.GROQ_MODEL_NAME

        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Vector Database</div>
            <div class="sidebar-value sidebar-value-green">{db_mode}</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-label">Indexed Chunks</div>
            <div class="sidebar-value sidebar-value-blue">{total_chunks:,}</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-label">Embedding Model</div>
            <div class="sidebar-value sidebar-value-dark">{embed_name}</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-label">LLM Provider</div>
            <div class="sidebar-value sidebar-value-dark">Groq API</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-label">LLM Model</div>
            <div class="sidebar-value sidebar-value-dark">{llm_model}</div>
        </div>
        <div style="text-align: center; padding: 10px 0 4px 0;">
            <span style="background: #2b8a3e; color: white; padding: 4px 14px; border-radius: 12px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">v2.0 Production</span>
        </div>
        """, unsafe_allow_html=True)

    # Route pages
    if page == "🤖 AI Chat":
        page_chat()
    elif page == "📁 Data Manager":
        page_data_manager()
    elif page == "⚙️ Pipeline Monitor":
        page_pipeline_monitor()
    elif page == "📊 Intelligence":
        page_intelligence()


if __name__ == "__main__":
    main()
