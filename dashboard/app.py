# dashboard/app.py

"""
Ad-Work Dashboard
=================
Main entry point for the Streamlit application.

Run with:
    uv run streamlit run dashboard/app.py

    Or on Windows:
    .venv\\Scripts\\streamlit run dashboard/app.py
"""

# dashboard/app.py

"""
Ad-Work Dashboard
=================
Run with:
    uv run streamlit run dashboard/app.py
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

# --- Path setup (BEFORE any adwork imports) ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

# --- Page Config (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="Ad-Work",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
    }
    section[data-testid="stSidebar"] { background-color: #fafafa; }
</style>
""", unsafe_allow_html=True)


# --- Safe imports with error catching ---
# On Streamlit Cloud, if any import crashes, the app dies silently.
# Wrapping in try/except lets us show the actual error in the UI.

try:
    from adwork.config import settings
    config_loaded = True
    config_error = None
except Exception as e:
    config_loaded = False
    config_error = str(e)

try:
    from adwork.db.connection import get_db
    from adwork.db.queries import (
        get_date_range,
        get_kpis_comparison,
        get_daily_spend_by_platform,
        get_daily_roas,
        get_campaign_summary,
        get_trends,
        has_data,
    )
    db = get_db()
    db_connected = True
    db_error = None
except Exception as e:
    db_connected = False
    db_error = str(e)

try:
    from components.charts import (
        spend_by_platform_chart,
        roas_trend_chart,
        campaign_performance_table,
        trends_chart,
        platform_pie_chart,
    )
    from components.cards import (
        recommendation_card,
        metric_card_row,
        no_data_message,
    )
    components_loaded = True
    components_error = None
except Exception as e:
    components_loaded = False
    components_error = str(e)


# --- Show import errors if anything failed ---
if not config_loaded or not db_connected or not components_loaded:
    st.title("⚠️ Ad-Work — Startup Error")
    if not config_loaded:
        st.error(f"**Config failed to load:** {config_error}")
    if not db_connected:
        st.error(f"**Database connection failed:** {db_error}")
    if not components_loaded:
        st.error(f"**Dashboard components failed to load:** {components_error}")
    st.info(
        "If you're seeing this on Streamlit Cloud, check that your secrets "
        "are configured correctly in the app settings."
    )
    st.stop()


# --- Auto-seed if database is empty ---
data_loaded = has_data()

if not data_loaded:
    try:
        # Import seed logic directly (no subprocess)
        from scripts_inline import seed_demo_inline
        seed_demo_inline.run_seed()
        data_loaded = has_data()
        if data_loaded:
            st.rerun()
    except Exception:
        pass  # Seeding failed — user will see the "no data" message


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Ad-Work")
    st.caption("Agentic Ad Optimization")
    st.divider()

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
    st.caption("System Status")

    # LLM status
    provider = settings.llm_provider
    has_key = bool(
        settings.groq_api_key if provider == "groq" else settings.openai_api_key
    )
    if has_key:
        st.success(f"LLM: {provider.upper()} ✓", icon="🤖")
    else:
        st.warning("LLM: No API key", icon="⚠️")

    # DB status
    st.success("Database: Connected ✓", icon="💾")

    # Data status
    if data_loaded:
        date_range = get_date_range()
        if date_range:
            st.success(f"Data: {date_range[0]} → {date_range[1]}", icon="📅")
    else:
        st.warning("Data: No campaigns loaded", icon="📭")

    # Date filter
    if data_loaded:
        date_range = get_date_range()
        if date_range:
            st.divider()
            st.caption("Date Filter")
            filter_start = st.date_input(
                "From", value=date_range[0],
                min_value=date_range[0], max_value=date_range[1],
            )
            filter_end = st.date_input(
                "To", value=date_range[1],
                min_value=date_range[0], max_value=date_range[1],
            )
        else:
            filter_start, filter_end = None, None
    else:
        filter_start, filter_end = None, None


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Campaign Overview")

    if not data_loaded:
        no_data_message()
    else:
        comparison = get_kpis_comparison(filter_start, filter_end)
        metric_card_row(comparison["kpis"], comparison["deltas"])

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            spend_df = get_daily_spend_by_platform(filter_start, filter_end)
            st.plotly_chart(spend_by_platform_chart(spend_df), use_container_width=True)

        with col_right:
            roas_df = get_daily_roas(filter_start, filter_end)
            st.plotly_chart(roas_trend_chart(roas_df), use_container_width=True)

        col_pie, col_future = st.columns(2)

        with col_pie:
            st.plotly_chart(platform_pie_chart(spend_df), use_container_width=True)

        with col_future:
            st.markdown("#### 🔮 Predicted vs Actual")
            st.info("Forecast comparison will appear after Phase 2.")

        st.divider()

        st.subheader("Campaign Performance")
        summary_df = get_campaign_summary(filter_start, filter_end)
        formatted = campaign_performance_table(summary_df)
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Recent Recommendations")
        st.caption("🤖 AI-generated recommendations appear after Phase 5. Samples below.")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            recommendation_card(
                action_type="bid_adjustment",
                title="Increase Google Search (Laptops) bid +12%",
                reasoning="CTR trending up over 14 days. Google Trends shows rising "
                          "search interest for 'laptop'. Forecast predicts +18% demand next week.",
                confidence="high",
                campaign_name="Google - Search (Laptops)",
                card_key="sample_1",
            )

        with rec_col2:
            recommendation_card(
                action_type="creative_rotation",
                title="Rotate creative on Meta Interest campaign",
                reasoning="Ad fatigue detected: CTR dropped from 0.6% to 0.4% over 21 days. "
                          "Recommend fresh creative or audience refresh.",
                confidence="medium",
                campaign_name="Meta - Prospecting (Interest)",
                card_key="sample_2",
            )


elif page == "📈 Forecasts":
    st.title("📈 Demand Forecasts & Trends")

    if not data_loaded:
        no_data_message()
    else:
        st.subheader("📉 Google Trends — Market Signals")
        trends_df = get_trends()

        if not trends_df.empty:
            st.plotly_chart(trends_chart(trends_df), use_container_width=True)
            st.caption(
                "Google Trends data shows relative search interest (0–100). "
                "These signals feed into the demand forecasting model in Phase 2."
            )
        else:
            st.info("No Google Trends data loaded. Trends are optional — the dashboard works without them.")

        st.divider()
        st.subheader("🔮 Demand Forecasts")
        st.info("📈 Time-series forecasting (Prophet) will be built in Phase 2.")

        st.subheader("📅 Seasonality Calendar")
        st.info("Heatmap calendar showing when to push budget harder. Coming in Phase 2.")


elif page == "🎯 Recommendations":
    st.title("🎯 Optimization Recommendations")

    if not data_loaded:
        no_data_message()
    else:
        st.info("🤖 The LLM agent optimization loop will be built in Phases 4–5.")
        st.divider()
        st.subheader("Example Recommendations (Preview)")

        examples = [
            {
                "action_type": "bid_adjustment",
                "title": "Increase Google Shopping bid +8%",
                "reasoning": "Shopping ROAS is 4.2x (above 3.0x target). Bandit model suggests higher bid captures additional profitable clicks.",
                "confidence": "high",
                "campaign_name": "Google - Shopping",
            },
            {
                "action_type": "budget_reallocation",
                "title": "Shift $50/day from Meta Interest → Meta Remarketing",
                "reasoning": "Meta Interest ROAS declined to 0.9x (below break-even). Remarketing maintains 3.8x ROAS with room to scale.",
                "confidence": "high",
                "campaign_name": "Meta - Prospecting (Interest)",
            },
            {
                "action_type": "creative_rotation",
                "title": "Refresh ad creative for Meta Lookalike",
                "reasoning": "CTR dropped 15% over 3 weeks while impressions remained stable. Creative refresh typically recovers 50–70% of lost CTR.",
                "confidence": "medium",
                "campaign_name": "Meta - Prospecting (Lookalike)",
            },
            {
                "action_type": "bid_adjustment",
                "title": "Decrease Amazon Auto bid -10%",
                "reasoning": "Auto-targeting ACOS is 28%, above the 22% target. Manual campaign performs better for the same keywords.",
                "confidence": "medium",
                "campaign_name": "Amazon - Sponsored Products (Auto)",
            },
        ]

        for i in range(0, len(examples), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(examples):
                    with col:
                        recommendation_card(**examples[i + j], card_key=f"example_{i + j}")


elif page == "🔍 Competitors":
    st.title("🔍 Competitor Intelligence")
    st.info("🔍 Competitor monitoring will be built in Phase 6.")


elif page == "📤 Upload Data":
    st.title("📤 Upload Campaign Data")

    st.markdown("""
    Upload your campaign performance CSV.  
    **Supported:** Google Ads, Meta Ads, Amazon Ads exports, or Ad-Work internal format.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded: {uploaded_file.name} — {len(df)} rows, {len(df.columns)} columns")

        from adwork.data.schemas import detect_platform_from_columns
        detected = detect_platform_from_columns(df.columns.tolist())

        platform_labels = {
            "google": "🔵 Google Ads",
            "meta": "🔷 Meta Ads",
            "amazon": "🟠 Amazon Ads",
            "unknown": "📄 Internal / Unknown",
        }
        st.info(f"**Detected platform:** {platform_labels.get(detected.value, detected.value)}")

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.divider()
        if st.button("🚀 Import into Ad-Work", type="primary", use_container_width=True):
            with st.spinner("Importing..."):
                from adwork.data.ingestion import ingest_csv
                result = ingest_csv(df)

            if result["status"] == "success":
                st.success(f"✅ {result['rows_loaded']} rows loaded across {len(result['campaigns_found'])} campaigns.")
                st.balloons()
            elif result["status"] == "partial":
                st.warning(f"⚠️ {result['rows_loaded']}/{result['rows_total']} rows loaded.")
            else:
                st.error("❌ Import failed.")
                for err in result["errors"][:5]:
                    st.text(err)
    else:
        st.markdown("---")
        st.subheader("📁 No data yet?")
        st.markdown(
            "Run the seed script locally:\n\n"
            "```bash\nuv run python scripts/seed_demo.py\n```\n\n"
            "Or on Windows:\n"
            "```\n.venv\\Scripts\\python scripts\\seed_demo.py\n```"
        )


elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    st.subheader("Configuration")
    config_data = {
        "LLM Provider": settings.llm_provider.upper(),
        "LLM Model": "Llama 3.3 70B" if settings.llm_provider == "groq" else "GPT-4o-mini",
        "Database Path": settings.duckdb_path,
        "Forecast Horizon": f"{settings.forecast_horizon_days} days",
        "Optimization Interval": f"Every {settings.optimization_interval_hours} hours",
    }
    for key, value in config_data.items():
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"**{key}**")
        with c2:
            st.code(value)

    st.divider()

    st.subheader("Database Statistics")
    try:
        stats = {
            "Campaigns": db.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0],
            "Daily Metrics": db.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0],
            "Recommendations": db.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0],
            "Forecasts": db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0],
            "Trend Points": db.execute("SELECT COUNT(*) FROM search_trends").fetchone()[0],
        }
        for key, value in stats.items():
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**{key}**")
            with c2:
                st.code(f"{value:,}")
    except Exception as e:
        st.error(f"Error: {e}")

    st.divider()

    st.subheader("Test LLM Connection")
    if st.button("🧪 Send Test Message"):
        with st.spinner("Calling LLM..."):
            try:
                from adwork.agent.llm_client import get_llm_client
                llm = get_llm_client()
                response = llm.complete(messages=[
                    {"role": "system", "content": "You are an ad optimization assistant. Be concise."},
                    {"role": "user", "content": "In one sentence, what's the most important metric for ad campaigns?"},
                ])
                st.success(f"✅ {response.provider}/{response.model}:")
                st.markdown(f"> {response.content}")
                st.caption(f"Tokens: {response.usage}")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

    st.divider()
    st.subheader("⚠️ Danger Zone")
    if st.button("🗑️ Reset Database"):
        from adwork.db.connection import reset_db
        reset_db()
        st.warning("Database reset. Run seed script to reload.")
        st.rerun()