"""PolyMind — Streamlit Demo App.

A beautiful, interactive demo showcasing PolyMind's multimodal
knowledge assistant capabilities.
"""


import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────
API_URL = "http://localhost:8000"
PAGE_TITLE = "PolyMind — Multimodal Knowledge Assistant"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .score-good { color: #28a745; font-weight: bold; }
    .score-warn { color: #ffc107; font-weight: bold; }
    .score-bad { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 PolyMind</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Self-evaluating, multimodal, multi-agent knowledge assistant</div>', unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Context")
    st.caption("Provide files for context-aware responses")

    audio_file = st.file_uploader(
        "🎙️ Audio (MP3/WAV)",
        type=["mp3", "wav"],
        help="Upload audio for transcription and analysis"
    )

    image_file = st.file_uploader(
        "🖼️ Image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        help="Upload image for visual question answering"
    )

    doc_file = st.file_uploader(
        "📄 Document (PDF/CSV)",
        type=["pdf", "csv"],
        help="Upload document for analysis"
    )

    st.divider()
    st.header("📊 System Info")

    # Check API health
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API: Online (v{health.get('version', '?')})")
    except Exception:
        st.error("API: Offline")

    st.divider()
    st.caption("PolyMind v0.7.0 | Groq LLM | Qdrant | LangGraph")

# ── Chat Interface ──────────────────────────────────────
query = st.chat_input("Ask anything about your documents...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Process query
    with st.spinner("🔍 Running agent graph..."):
        files = {}
        data = {"question": query, "user_id": "demo"}

        if audio_file:
            files["audio_file"] = audio_file
        if image_file:
            files["image_file"] = image_file
        if doc_file:
            files["doc_file"] = doc_file

        try:
            response = requests.post(
                f"{API_URL}/query/",
                data=data,
                files=files,
                timeout=30,
            ).json()
        except Exception as e:
            response = {
                "answer": f"Error connecting to API: {e}",
                "modality": "error",
                "confidence": 0.0,
                "citations": [],
                "critic_scores": {},
                "retry_count": 0,
                "processing_time_ms": 0,
            }

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response.get("answer", "No answer generated"))

        # Show metadata
        col1, col2, col3 = st.columns(3)

        with col1:
            modality = response.get("modality", "text")
            st.metric("Modality", modality.upper())

        with col2:
            confidence = response.get("confidence", 0)
            st.metric("Confidence", f"{confidence:.0%}")

        with col3:
            time_ms = response.get("processing_time_ms", 0)
            st.metric("Latency", f"{time_ms:.0f}ms")

        # Show critic scores
        scores = response.get("critic_scores", {})
        if scores:
            with st.expander("📊 Critic Scores", expanded=True):
                cols = st.columns(min(len(scores), 3))
                for i, (metric, data) in enumerate(scores.items()):
                    with cols[i % 3]:
                        if isinstance(data, dict):
                            score = data.get("score", 0)
                            passed = data.get("passed", False)
                        else:
                            score = float(data) if data else 0
                            passed = True

                        color = "🟢" if passed else "🔴"
                        st.metric(
                            label=f"{color} {metric.replace('_', ' ').title()}",
                            value=f"{score:.2f}",
                        )

        # Show citations
        citations = response.get("citations", [])
        if citations:
            with st.expander(f"📚 Sources ({len(citations)})"):
                for c in citations[:5]:
                    source = c.get("source", "unknown")
                    score = c.get("score", 0)
                    st.caption(f"• {source} (relevance: {score:.3f})")

        # Show retry info
        retries = response.get("retry_count", 0)
        if retries > 0:
            st.info(f"🔄 {retries} retry(ies) performed")

# ── Footer ──────────────────────────────────────────────
st.divider()
st.markdown(
    "**PolyMind** — Built with LangGraph, Groq, Qdrant, and RAGAS. "
    "Self-evaluating multi-agent knowledge assistant.",
    help="Phase 8 of 8 complete. 168+ tests, CI-gated eval harness.",
)
