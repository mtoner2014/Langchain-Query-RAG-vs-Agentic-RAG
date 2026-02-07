"""
MedlinePlus Healthcare RAG Assistant — Streamlit UI
A medically-themed interface with Agentic and Non-Agentic RAG modes.
"""

import re
import streamlit as st
import time
from rag_medlineplus import MedlinePlusRAG
from agentic_rag_medlineplus import AgenticMedlinePlusRAG
from image_processor import MedicalImageAnalyzer, validate_upload

# ── Crisis / Emergency Guardrails ────────────────────────────────────────────

EMERGENCY_NUMBERS: dict[str, str] = {
    "Canada": "911",
    "United States": "911",
    "United Kingdom": "999",
    "Australia": "000",
    "France": "15 / 112",
    "Germany": "112",
    "India": "112",
    "Japan": "119",
    "Mexico": "911",
    "Brazil": "192",
    "South Korea": "119",
    "Philippines": "911",
}

_SUICIDE_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life",
    "self-harm", "self harm", "want to die", "don't want to live",
    "hurting myself", "cutting myself",
]

_EMERGENCY_KEYWORDS = [
    "heart attack", "can't breathe", "choking", "overdose",
    "severe bleeding", "unconscious", "seizure", "anaphylaxis",
    "stroke symptoms",
]

_SUICIDE_RE = re.compile(
    "|".join(re.escape(k) for k in _SUICIDE_KEYWORDS), re.IGNORECASE
)
_EMERGENCY_RE = re.compile(
    "|".join(re.escape(k) for k in _EMERGENCY_KEYWORDS), re.IGNORECASE
)


def detect_crisis(query: str) -> str | None:
    """Return ``'suicide'``, ``'emergency'``, or ``None``."""
    if _SUICIDE_RE.search(query):
        return "suicide"
    if _EMERGENCY_RE.search(query):
        return "emergency"
    return None


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedlinePlus Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --med-primary: #0C6E87;
    --med-primary-light: #0E8BA6;
    --med-accent: #00B4D8;
    --med-bg: #F0F7FA;
    --med-card: #FFFFFF;
    --med-text: #1A2B3C;
    --med-text-muted: #5A6F80;
    --med-border: #D5E5ED;
    --med-success: #10B981;
    --med-warning: #F59E0B;
    --med-danger: #EF4444;
    --med-gradient: linear-gradient(135deg, #0C6E87 0%, #00B4D8 100%);
}

/* ── Global ── */
.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Header banner ── */
.med-header {
    background: var(--med-gradient);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.med-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.med-header::after {
    content: '';
    position: absolute;
    bottom: -60%;
    right: 10%;
    width: 200px;
    height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.med-header h1 {
    margin: 0 0 0.3rem 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.med-header p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.95rem;
    font-weight: 300;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0F7FA 0%, #E3EFF5 100%);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
    color: var(--med-primary);
}

/* ── Mode badge ── */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.mode-agentic {
    background: rgba(16,185,129,0.12);
    color: #059669;
    border: 1px solid rgba(16,185,129,0.25);
}
.mode-nonagentic {
    background: rgba(12,110,135,0.10);
    color: #0C6E87;
    border: 1px solid rgba(12,110,135,0.20);
}

/* ── Chat bubbles ── */
.chat-user {
    background: var(--med-gradient);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.75rem 0;
    max-width: 85%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(12,110,135,0.15);
}
.chat-assistant {
    background: var(--med-card);
    color: var(--med-text);
    padding: 1.25rem 1.5rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.75rem 0;
    max-width: 90%;
    font-size: 0.93rem;
    line-height: 1.65;
    border: 1px solid var(--med-border);
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.chat-assistant strong {
    color: var(--med-primary);
}

/* ── Image upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #B0C4CE;
    border-radius: 12px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--med-accent);
}

/* ── Disclaimer card ── */
.med-disclaimer {
    background: linear-gradient(135deg, #FFF7ED 0%, #FEF3C7 100%);
    border-left: 4px solid var(--med-warning);
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.82rem;
    color: #92400E;
    line-height: 1.5;
    margin-bottom: 1rem;
}

/* ── Info cards in sidebar ── */
.info-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    border: 1px solid var(--med-border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.info-card h4 {
    margin: 0 0 0.4rem 0;
    color: var(--med-primary);
    font-size: 0.85rem;
    font-weight: 600;
}
.info-card p {
    margin: 0;
    color: var(--med-text-muted);
    font-size: 0.8rem;
    line-height: 1.5;
}

/* ── Pulse dot animation ── */
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--med-success);
    animation: pulse 1.5s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* ── Stat pill ── */
.stat-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.stat-pill {
    background: rgba(12,110,135,0.08);
    color: var(--med-primary);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ── Streamlit overrides ── */
.stChatInput > div {
    border-radius: 14px !important;
    border: 2px solid var(--med-border) !important;
}
.stChatInput > div:focus-within {
    border-color: var(--med-accent) !important;
    box-shadow: 0 0 0 3px rgba(0,180,216,0.12) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--med-border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--med-primary-light); }

/* ── Crisis banners ── */
.crisis-banner {
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    line-height: 1.6;
    font-size: 0.93rem;
}
.crisis-banner strong { display: block; font-size: 1rem; margin-bottom: 0.3rem; }
.crisis-suicide {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border-left: 5px solid #F59E0B;
    color: #78350F;
}
.crisis-emergency {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    border-left: 5px solid #EF4444;
    color: #7F1D1D;
}

/* ── Welcome cards ── */
.welcome-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 1rem;
}
.welcome-card {
    background: white;
    border: 1px solid var(--med-border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    cursor: default;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.welcome-card:hover {
    border-color: var(--med-accent);
    box-shadow: 0 2px 8px rgba(0,180,216,0.10);
}
.welcome-card .icon { font-size: 1.3rem; margin-bottom: 0.3rem; }
.welcome-card .title {
    font-weight: 600;
    color: var(--med-text);
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
}
.welcome-card .desc {
    color: var(--med-text-muted);
    font-size: 0.78rem;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Non-Agentic RAG"
if "rag_non_agentic" not in st.session_state:
    st.session_state.rag_non_agentic = None
if "rag_agentic" not in st.session_state:
    st.session_state.rag_agentic = None
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "image_analyzer" not in st.session_state:
    st.session_state.image_analyzer = None


def get_image_analyzer():
    """Lazily initialise and return the image analyzer."""
    if st.session_state.image_analyzer is None:
        st.session_state.image_analyzer = MedicalImageAnalyzer()
    return st.session_state.image_analyzer


def get_rag_instance():
    """Lazily initialise and return the selected RAG backend."""
    if st.session_state.rag_mode == "Agentic RAG":
        if st.session_state.rag_agentic is None:
            st.session_state.rag_agentic = AgenticMedlinePlusRAG()
        return st.session_state.rag_agentic
    else:
        if st.session_state.rag_non_agentic is None:
            st.session_state.rag_non_agentic = MedlinePlusRAG()
        return st.session_state.rag_non_agentic


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.2rem;">
        <span style="font-size:2.5rem;">🩺</span>
        <h2 style="margin:0.3rem 0 0 0; color:#0C6E87; font-size:1.15rem;">Health Assistant</h2>
        <p style="color:#5A6F80; font-size:0.78rem; margin:0;">Powered by MedlinePlus</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Mode selector
    st.markdown("##### Retrieval Mode")
    mode = st.radio(
        "Choose RAG mode",
        options=["Non-Agentic RAG", "Agentic RAG"],
        index=0 if st.session_state.rag_mode == "Non-Agentic RAG" else 1,
        label_visibility="collapsed",
        help="Choose how the system retrieves and processes medical information.",
    )

    if mode != st.session_state.rag_mode:
        st.session_state.rag_mode = mode
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()

    # Mode description cards
    if st.session_state.rag_mode == "Agentic RAG":
        st.markdown("""
        <div class="info-card">
            <h4>Agentic RAG</h4>
            <p>An AI agent that <strong>reasons</strong> about your query and
            intelligently selects specialised tools — topic search,
            symptom lookup, or treatment info — before synthesising
            a response. Maintains conversation context.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="stat-row">
            <span class="stat-pill">3 specialised tools</span>
            <span class="stat-pill">Conversation memory</span>
            <span class="stat-pill">ReAct reasoning</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card">
            <h4>Non-Agentic RAG</h4>
            <p>A direct retrieval pipeline that scrapes MedlinePlus,
            splits content into chunks, filters by FAISS similarity,
            and feeds the most relevant context to the LLM.
            Fast and straightforward.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="stat-row">
            <span class="stat-pill">FAISS vector search</span>
            <span class="stat-pill">Top-3 chunks</span>
            <span class="stat-pill">Single pass</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Country selector (for emergency numbers)
    st.markdown("##### Your Country")
    selected_country = st.selectbox(
        "Select your country",
        options=list(EMERGENCY_NUMBERS.keys()),
        index=0,  # defaults to Canada
        label_visibility="collapsed",
    )
    st.session_state.user_country = selected_country

    st.markdown("---")

    # Session stats
    st.markdown("##### Session")
    st.markdown(f"""
    <div class="info-card">
        <p><span class="pulse-dot"></span> <strong>{st.session_state.query_count}</strong> queries this session</p>
    </div>
    """, unsafe_allow_html=True)

    # Clear chat
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        if st.session_state.rag_agentic:
            st.session_state.rag_agentic.clear_history()
        st.rerun()

    st.markdown("---")

    # Image upload info
    st.markdown("""
    <div class="info-card">
        <h4 style="margin:0 0 0.4rem 0; font-size:0.9rem; color:#0C6E87;">
            📷 Image Analysis
        </h4>
        <p style="margin:0; font-size:0.78rem; color:#5A6F80;">
            Upload lab reports, prescription labels, or symptom photos.
            The AI will extract medical information and search MedlinePlus
            for relevant context.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # Disclaimer
    st.markdown("""
    <div class="med-disclaimer">
        <strong>Medical Disclaimer</strong><br>
        This tool provides general health information sourced from
        MedlinePlus and is <em>not</em> a substitute for professional
        medical advice, diagnosis, or treatment. Always consult a
        qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)


# ── Main area ────────────────────────────────────────────────────────────────

# Header
is_agentic = st.session_state.rag_mode == "Agentic RAG"
badge_class = "mode-agentic" if is_agentic else "mode-nonagentic"
badge_label = "Agentic" if is_agentic else "Non-Agentic"
badge_icon = "🤖" if is_agentic else "📄"

st.markdown(f"""
<div class="med-header">
    <h1>MedlinePlus Health Assistant</h1>
    <p>Evidence-based health information at your fingertips</p>
    <div style="margin-top:0.8rem;">
        <span class="mode-badge {badge_class}">{badge_icon} {badge_label} RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome state — show example cards when no messages
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; color:#5A6F80; margin:1.5rem 0 0.5rem 0; font-size:0.95rem;">
        Ask a health-related question to get started
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        ("💊", "Medications & Treatments", "What are the common treatments for type 2 diabetes?"),
        ("🫀", "Conditions & Diseases", "Tell me about hypertension and its risk factors"),
        ("🩻", "Symptoms & Diagnosis", "What could cause persistent chest pain?"),
        ("🏥", "Prevention & Wellness", "How can I lower my cholesterol naturally?"),
    ]
    col1, col2 = st.columns(2)
    for idx, (icon, title, query) in enumerate(suggestions):
        with (col1 if idx % 2 == 0 else col2):
            if st.button(f"{icon} **{title}**\n\n{query}", key=f"suggestion_{idx}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()

# Chat history
for msg in st.session_state.messages:
    css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🩺"):
        st.markdown(msg["content"])

# Image upload
uploaded_file = st.file_uploader(
    "Upload a medical image (lab report, prescription, symptom photo)",
    type=["png", "jpg", "jpeg", "webp", "gif"],
    key="image_upload",
    help="Upload an image of a lab report, prescription label, or symptom photo for analysis.",
)

upload_error = None
if uploaded_file is not None:
    upload_error = validate_upload(uploaded_file.name, uploaded_file.size)
    if upload_error:
        st.error(upload_error)

has_image = uploaded_file is not None and upload_error is None

# Chat input
prompt = st.chat_input("Ask a health question — or upload an image above")

# Handle suggestion button clicks
if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

if prompt or has_image:
    # Build display message
    if has_image and prompt:
        user_display = f"[Image uploaded] {prompt}"
    elif has_image:
        user_display = "[Uploaded medical image for analysis]"
    else:
        user_display = prompt

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_display})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_display)
        if has_image:
            st.image(uploaded_file, caption="Uploaded image", width=300)

    # Crisis / emergency guardrail (on text input)
    if prompt:
        crisis_type = detect_crisis(prompt)
        if crisis_type == "suicide":
            st.markdown("""
            <div class="crisis-banner crisis-suicide">
                <strong>⚠️ Crisis Resources</strong>
                If you or someone you know is in crisis, please reach out immediately.<br>
                📞 <strong>988 Suicide &amp; Crisis Lifeline</strong> — call or text <strong>988</strong><br>
                📞 <strong>Crisis Services Canada</strong> — <strong>1-833-456-4566</strong>
            </div>
            """, unsafe_allow_html=True)
        elif crisis_type == "emergency":
            emergency_num = EMERGENCY_NUMBERS.get(
                st.session_state.get("user_country", "Canada"), "911"
            )
            st.markdown(f"""
            <div class="crisis-banner crisis-emergency">
                <strong>🚨 Medical Emergency</strong>
                If this is a medical emergency, call <strong>{emergency_num}</strong> immediately.
            </div>
            """, unsafe_allow_html=True)

    # Generate response
    with st.chat_message("assistant", avatar="🩺"):
        try:
            rag = get_rag_instance()

            if has_image:
                # Step 1: Analyze the uploaded image
                with st.spinner("Analyzing uploaded image..."):
                    analyzer = get_image_analyzer()
                    image_bytes = uploaded_file.getvalue()
                    mime_type = uploaded_file.type or "image/jpeg"
                    image_result = analyzer.analyze_image(
                        image_bytes, mime_type, user_text=prompt or ""
                    )

                st.markdown("**Image Analysis:**")
                st.info(image_result["summary"])

                if image_result["image_type"] == "other":
                    st.warning(
                        "This image does not appear to contain recognizable "
                        "medical content. Results may be limited."
                    )

                st.markdown(
                    '<div class="med-disclaimer">'
                    "<strong>Image Analysis Disclaimer:</strong> "
                    "AI image analysis is not a substitute for professional "
                    "medical interpretation. Lab results, prescriptions, and "
                    "symptoms should always be reviewed by a qualified "
                    "healthcare provider.</div>",
                    unsafe_allow_html=True,
                )

                # Step 2: Use extracted query for MedlinePlus retrieval
                search_query = image_result["search_query"]
                if search_query:
                    with st.spinner(
                        "Agent is reasoning and searching MedlinePlus..."
                        if is_agentic
                        else "Searching MedlinePlus for related information..."
                    ):
                        start = time.time()
                        response = rag.query(search_query)
                        elapsed = time.time() - start

                    st.markdown("**Related Health Information from MedlinePlus:**")
                    st.markdown(response)
                    st.caption(
                        f"Response generated in {elapsed:.1f}s via "
                        f"**{st.session_state.rag_mode}**"
                    )
                else:
                    response = "Could not extract a search query from the image."
                    st.warning(response)
                    elapsed = 0

                full_response = (
                    f"**Image Analysis:** {image_result['summary']}\n\n"
                    f"**Extracted Terms:** {', '.join(image_result['medical_terms'])}\n\n"
                    f"**Related Health Information:**\n{response}"
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                st.session_state.query_count += 1

            else:
                # Text-only: existing behavior
                with st.spinner(
                    "Agent is reasoning and searching MedlinePlus..."
                    if is_agentic
                    else "Searching MedlinePlus and generating response..."
                ):
                    start = time.time()
                    response = rag.query(prompt)
                    elapsed = time.time() - start

                    st.markdown(response)
                    st.caption(
                        f"Response generated in {elapsed:.1f}s via "
                        f"**{st.session_state.rag_mode}**"
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.session_state.query_count += 1

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
