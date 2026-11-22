import os
import time
import streamlit as st
import pandas as pd

# Load .env variables natively
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

from src.rag import build_documents, build_faiss_index, stream_rag

st.set_page_config(
    page_title="CountryWise AI - Sovereign Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System matching ZAY-G style mockup
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Gradient Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #e0e7ff 0%, #f8fafc 45%, #ede9fe 100%) !important;
    }

    /* Fixed top spacing to prevent tab cut-off */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* Ensure tabs are clean, visible, and fully elevated */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(10px) !important;
        padding: 6px 8px !important;
        border-radius: 16px !important;
        gap: 8px !important;
        border: 1px solid rgba(226, 232, 240, 0.9) !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
        color: #475569 !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.12) !important;
        color: #4f46e5 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e1b4b;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }

    /* Primary Action Sidebar Button (New Chat) */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 16px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
        margin-top: 4px;
        margin-bottom: 8px;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    .nav-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 14px;
        margin-bottom: 6px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #edf2f7;
        color: #334155;
        font-size: 0.86rem;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    .history-header {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #64748b;
        text-transform: uppercase;
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
    }

    .history-item {
        font-size: 0.82rem;
        color: #475569;
        padding: 7px 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 8px;
        margin-bottom: 4px;
    }

    /* Center Hero Stage */
    .hero-stage {
        text-align: center;
        padding: 24px 10px 10px 10px;
    }
    .hero-icon {
        font-size: 4rem;
        margin-bottom: 8px;
        filter: drop-shadow(0 10px 15px rgba(99, 102, 241, 0.25));
        animation: float 3s ease-in-out infinite alternate;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        100% { transform: translateY(-8px); }
    }
    .hero-headline {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 24px;
        max-width: 620px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Prompt Suggestion Buttons inside Main Tabs */
    .stTabs div[data-testid="stButton"] button {
        border-radius: 24px !important;
        padding: 8px 18px !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
        transition: all 0.2s ease !important;
    }
    .stTabs div[data-testid="stButton"] button:hover {
        border-color: #6366f1 !important;
        color: #4f46e5 !important;
        background: #f5f3ff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.18) !important;
    }

    /* Chat Bubble Styling */
    .user-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 14px;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        font-size: 0.92rem;
        max-width: 78%;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.22);
    }
    .assistant-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 16px;
    }
    .chat-bubble-assistant {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #1e293b;
        padding: 18px 22px;
        border-radius: 18px 18px 18px 4px;
        font-size: 0.94rem;
        line-height: 1.6;
        width: 100%;
        box-shadow: 0 4px 18px rgba(0,0,0,0.04);
    }

    /* Inline Badges */
    .chip-bar {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .chip {
        padding: 3px 8px;
        border-radius: 6px;
        background: #f1f5f9;
        color: #64748b;
    }
    .chip-purple {
        background: #ede9fe;
        color: #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="font-size:1.6rem;">🌐</span> CountryWise AI
    </div>
    <div style="font-size:0.78rem; color:#64748b; margin-bottom:14px;">Encyclopedic Sovereign Intelligence</div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Knowledge Base Engine:",
        ["Enhanced (Fused Multi-API)", "Baseline (Original WHR)"],
        index=0
    )

    data_path = "data/fused_dataset.csv" if mode.startswith("Enhanced") else "data/original_dataset.csv"

    @st.cache_resource
    def load_rag_pipeline(path):
        df = pd.read_csv(path)
        docs = build_documents(df)
        index = build_faiss_index(docs)
        return df, docs, index

    try:
        df, docs, index = load_rag_pipeline(data_path)
    except Exception as e:
        st.error(f"Cannot load dataset: {e}")
        st.stop()

    st.write("")
    # Primary Action: New Chat Button
    if st.button("✏️  New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.run_prompt = None
        st.rerun()

    st.write("")
    # Navigation Items matching reference mockup
    st.markdown("""
    <div class="nav-pill">📚 <span>Sovereign Library</span></div>
    <div class="nav-pill">🏛️ <span>Global Leaderboard</span></div>
    <div class="nav-pill">📂 <span>Saved Profiles</span></div>
    """, unsafe_allow_html=True)

    # History Drawer
    st.markdown('<div class="history-header">Recent Inquiries ↗</div>', unsafe_allow_html=True)
    if "chat_history" in st.session_state and st.session_state.chat_history:
        recent_queries = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
        for q in reversed(recent_queries[-5:]):
            st.markdown(f'<div class="history-item"><span>↗ {q[:25]}...</span></div>', unsafe_allow_html=True)
    else:
        st.caption("No queries recorded yet.")

# ----------------- MAIN TABS -----------------
tab_chat, tab_compare, tab_eda, tab_data = st.tabs([
    "💬 Knowledge Chat",
    "⚖️ Bilateral Compare",
    "📊 Empirical EDA",
    "📁 Dataset Explorer"
])

# ----------------- TAB 1: CHAT -----------------
with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "run_prompt" not in st.session_state:
        st.session_state.run_prompt = None

    # Welcome Stage (Shows when chat is empty)
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="hero-stage">
            <div class="hero-icon">🌍</div>
            <div class="hero-headline">How can I help you explore today?</div>
            <div class="hero-sub">Ask deep questions about any sovereign country's history, governance, climate, economy, or social wellbeing.</div>
        </div>
        """, unsafe_allow_html=True)

    # Prompt Suggestions Pills
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("🏆 Top 5 Happiest", use_container_width=True):
        st.session_state.run_prompt = "Which are the top 5 happiest nations in the world and what are their scores?"
    if p2.button("🇫🇮 Finland Deep Dive", use_container_width=True):
        st.session_state.run_prompt = "Tell me about the history, origins, and governance of Finland."
    if p3.button("🇮🇳 India Climate & Econ", use_container_width=True):
        st.session_state.run_prompt = "What is India's capital city, annual mean temperature, and happiness rank?"
    if p4.button("🛡️ Out-of-Scope Test", use_container_width=True):
        st.session_state.run_prompt = "What is the stock ticker for Apple?"

    st.write("")

    # Chat Transcript Loop
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-container"><div class="chat-bubble-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-container"><div class="chat-bubble-assistant">{msg["content"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="chip-bar" style="margin-top:-6px; margin-bottom:12px;">
                <span class="chip">⏱️ {msg.get('latency', 0.0)}s</span>
                <span class="chip chip-purple">{msg.get('mode', mode)}</span>
            </div>
            """, unsafe_allow_html=True)
            if "context" in msg and msg["context"]:
                with st.expander(f"View Retrieved Context Chunks ({len(msg['context'])})"):
                    for i, c in enumerate(msg["context"], 1):
                        st.caption(f"Chunk #{i} | Cosine Match: {c['score']:.3f}")
                        st.code(c["text"], language="text")
        st.write("")

    # Chat Input Box
    typed_q = st.chat_input("Chat here (e.g. History of Japan, Climate of Norway, GDP of India)...")
    active_prompt = typed_q or st.session_state.run_prompt

    if active_prompt:
        st.session_state.run_prompt = None
        st.session_state.chat_history.append({"role": "user", "content": active_prompt})
        st.markdown(f'<div class="user-container"><div class="chat-bubble-user">{active_prompt}</div></div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            start_t = time.time()
            stream_gen, retrieved = stream_rag(active_prompt, index, docs)

            def token_gen():
                for chunk in stream_gen:
                    delta = chunk.choices[0].delta.content or ""
                    yield delta

            answer_text = st.write_stream(token_gen())
            elapsed = round(time.time() - start_t, 2)

            st.markdown(f"""
            <div class="chip-bar">
                <span class="chip">⏱️ {elapsed}s</span>
                <span class="chip chip-purple">{mode}</span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"View Retrieved Context Chunks ({len(retrieved)})"):
                for i, c in enumerate(retrieved, 1):
                    st.caption(f"Chunk #{i} | Cosine Match: {c['score']:.3f}")
                    st.code(c["text"], language="text")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_text,
                "latency": elapsed,
                "mode": mode,
                "context": retrieved
            })
            st.rerun()

# ----------------- TAB 2: BILATERAL COMPARE -----------------
with tab_compare:
    st.subheader("Bilateral Sovereign Comparison")
    st.caption("Select two nations to compare their quantitative metrics and geopolitical indicators.")

    c_list = sorted(df["country"].unique().tolist())
    ca, cb = st.columns(2)
    with ca:
        n1 = st.selectbox("Select Country A:", c_list, index=c_list.index("Finland") if "Finland" in c_list else 0)
    with cb:
        n2 = st.selectbox("Select Country B:", c_list, index=c_list.index("India") if "India" in c_list else 1)

    if n1 and n2:
        r1 = df[df["country"] == n1].iloc[0]
        r2 = df[df["country"] == n2].iloc[0]

        fields = [
            ("Happiness Rank", "happiness_rank"),
            ("Happiness Score", "happiness_score"),
            ("Capital City", "capital"),
            ("Region", "region"),
            ("Annual Mean Temp (°C)", "annual_mean_temp_c"),
            ("World Bank Life Expectancy", "wb_life_expectancy"),
            ("Log GDP Per Capita", "log_gdp_per_capita"),
            ("Social Support Index", "social_support"),
            ("Official Currency", "currency_code"),
            ("USD Exchange Rate", "exchange_rate_usd")
        ]

        data_rows = []
        for label, col in fields:
            v1 = r1.get(col, "N/A")
            v2 = r2.get(col, "N/A")
            if isinstance(v1, float): v1 = f"{v1:.2f}"
            if isinstance(v2, float): v2 = f"{v2:.2f}"
            data_rows.append({"Indicator": label, n1: v1, n2: v2})

        st.dataframe(pd.DataFrame(data_rows).set_index("Indicator"), use_container_width=True)

# ----------------- TAB 3: STATISTICAL EDA -----------------
with tab_eda:
    st.subheader("Statistical Empirical Evidence")
    st.caption("Correlation metrics and distribution curves across 156 sovereign states.")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("report/figures/fig1_correlation_matrix.png"):
            st.image("report/figures/fig1_correlation_matrix.png", caption="Fig 1: Correlation Matrix", use_container_width=True)
        if os.path.exists("report/figures/fig2_happiness_vs_temp.png"):
            st.image("report/figures/fig2_happiness_vs_temp.png", caption="Fig 2: Happiness vs Annual Temperature", use_container_width=True)
    with col2:
        if os.path.exists("report/figures/fig3_happiness_vs_life_expectancy.png"):
            st.image("report/figures/fig3_happiness_vs_life_expectancy.png", caption="Fig 3: Happiness vs Life Expectancy", use_container_width=True)
        if os.path.exists("report/figures/fig4_happiness_by_region.png"):
            st.image("report/figures/fig4_happiness_by_region.png", caption="Fig 4: Regional Happiness Distribution", use_container_width=True)

# ----------------- TAB 4: DATA EXPLORER -----------------
with tab_data:
    st.subheader("Sovereign Knowledge Base Explorer")
    query_filter = st.text_input("Filter nations by name:")
    filtered = df[df["country"].str.contains(query_filter, case=False, na=False)] if query_filter else df
    st.dataframe(filtered, use_container_width=True, height=450)