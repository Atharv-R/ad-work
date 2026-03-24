# dashboard/app.py

"""
Ad-Work Dashboard
=================
Main entry point for the Streamlit application.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path so we can import adwork
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# --- Page Config (must be first Streamlit call) ---
st.set_page_config(
    page_title="Ad-Work",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Custom CSS for clean look ---
st.markdown("""
<style>
    /* Tighter padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Ad-Work")
    st.caption("Agentic Ad Optimization")
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigate",
        options=[
            "📊 Overview",
            "📈 Forecasts",
            "🎯 Recommendations",
            "🔍 Competitors",
            "📤 Upload Data",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )
    
    st.divider()
    
    # System status
    st.caption("System Status")
    
    # Check LLM connection
    try:
        from adwork.config import settings
        provider = settings.llm_provider
        has_key = bool(settings.groq_api_key if provider == "groq" else settings.openai_api_key)
        
        if has_key:
            st.success(f"LLM: {provider.upper()} ✓", icon="🤖")
        else:
            st.warning(f"LLM: No API key", icon="⚠️")
    except Exception:
        st.error("Config error", icon="❌")
    
    # Check DB connection
    try:
        from adwork.db.connection import get_db
        db = get_db()
        st.success("Database: Connected ✓", icon="💾")
    except Exception:
        st.error("Database: Error", icon="❌")


# --- Main Content ---

if page == "📊 Overview":
    st.title("📊 Campaign Overview")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Spend",
            value="$12,450",
            delta="-$320 vs last week",
        )
    with col2:
        st.metric(
            label="ROAS",
            value="3.2x",
            delta="0.4x",
        )
    with col3:
        st.metric(
            label="Conversions",
            value="1,847",
            delta="12%",
        )
    with col4:
        st.metric(
            label="Active Campaigns",
            value="8",
            delta="2 new",
        )
    
    st.divider()
    
    # Placeholder charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Spend by Channel")
        st.info("📊 Chart will appear once data is loaded (Phase 1)")
        
    with col_right:
        st.subheader("Predicted vs Actual ROAS")
        st.info("📈 Chart will appear once forecasting is built (Phase 2)")
    
    st.divider()
    
    # Recent recommendations preview
    st.subheader("Recent Recommendations")
    
    # Sample recommendation cards
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        with st.container(border=True):
            st.markdown("#### ⬆️ Increase Google Search Bid +12%")
            st.markdown(
                "*Demand forecast shows +18% next week. "
                "CTR trending up over 14 days.*"
            )
            st.markdown("**Confidence:** :green[HIGH]")
            col_a, col_b = st.columns(2)
            with col_a:
                st.button("✅ Apply", key="apply_1", use_container_width=True)
            with col_b:
                st.button("❌ Dismiss", key="dismiss_1", use_container_width=True)
    
    with rec_col2:
        with st.container(border=True):
            st.markdown("#### 🔄 Rotate Creative on Meta Campaign")
            st.markdown(
                "*Ad fatigue detected: CTR dropped 23% over 7 days. "
                "Recommend fresh creative from variant pool.*"
            )
            st.markdown("**Confidence:** :orange[MEDIUM]")
            col_a, col_b = st.columns(2)
            with col_a:
                st.button("✅ Apply", key="apply_2", use_container_width=True)
            with col_b:
                st.button("❌ Dismiss", key="dismiss_2", use_container_width=True)


elif page == "📈 Forecasts":
    st.title("📈 Demand Forecasts")
    st.info("🔮 Forecasting module will be built in Phase 2. Stay tuned!")
    

elif page == "🎯 Recommendations":
    st.title("🎯 Optimization Recommendations")
    st.info("🎯 Recommendations engine will be built in Phase 4-5.")


elif page == "🔍 Competitors":
    st.title("🔍 Competitor Intelligence")
    st.info("🔍 Competitor monitoring will be built in Phase 6.")


elif page == "📤 Upload Data":
    st.title("📤 Upload Campaign Data")
    
    st.markdown("""
    Upload your campaign performance data as a CSV file.  
    Supported exports: **Google Ads**, **Meta Ads**, **Amazon Ads**
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Export your campaign report from your ad platform and upload it here.",
    )
    
    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.markdown("---")
        st.markdown("**Don't have data yet?** Use our sample datasets:")
        if st.button("Load Sample Google Ads Data"):
            st.info("Sample data will be available after Phase 1.")


elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    from adwork.config import settings
    
    st.subheader("Current Configuration")
    
    config_data = {
        "LLM Provider": settings.llm_provider.upper(),
        "LLM Model": "Llama 3.3 70B" if settings.llm_provider == "groq" else "GPT-4o-mini",
        "Database Path": settings.duckdb_path,
        "Forecast Horizon": f"{settings.forecast_horizon_days} days",
        "Optimization Interval": f"Every {settings.optimization_interval_hours} hours",
    }
    
    for key, value in config_data.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**{key}**")
        with col2:
            st.code(value)
    
    st.divider()
    
    # LLM Test
    st.subheader("Test LLM Connection")
    
    if st.button("🧪 Send Test Message"):
        with st.spinner("Calling LLM..."):
            try:
                from adwork.agent.llm_client import get_llm_client
                
                llm = get_llm_client()
                response = llm.complete(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful advertising optimization assistant. Be concise.",
                        },
                        {
                            "role": "user", 
                            "content": "In one sentence, what's the most important metric for evaluating ad campaign performance?",
                        },
                    ]
                )
                
                st.success(f"Response from {response.provider}/{response.model}:")
                st.markdown(f"> {response.content}")
                st.caption(f"Tokens used: {response.usage}")
                
            except Exception as e:
                st.error(f"LLM connection failed: {e}")