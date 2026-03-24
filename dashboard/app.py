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

import streamlit as st
import sys
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Add src/ and dashboard/ to path
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

# --- Imports (after path setup) ---
from adwork.config import settings
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


# --- Initialize DB connection ---
try:
    db = get_db()
    db_connected = True
except Exception:
    db_connected = False

# --- Auto-seed if database is empty (for Streamlit Cloud) ---
if db_connected and not has_data():
    import subprocess
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "seed_demo.py")],
            check=True,
            capture_output=True,
        )
        st.rerun()
    except Exception:
        pass  # Seed failed — user will see the "no data" message


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
    try:
        provider = settings.llm_provider
        has_key = bool(
            settings.groq_api_key if provider == "groq" else settings.openai_api_key
        )
        if has_key:
            st.success(f"LLM: {provider.upper()} ✓", icon="🤖")
        else:
            st.warning("LLM: No API key", icon="⚠️")
    except Exception:
        st.error("Config error", icon="❌")

    # DB status
    if db_connected:
        st.success("Database: Connected ✓", icon="💾")
    else:
        st.error("Database: Error", icon="❌")

    # Data status
    data_loaded = has_data() if db_connected else False
    if data_loaded:
        date_range = get_date_range()
        if date_range:
            st.success(f"Data: {date_range[0]} → {date_range[1]}", icon="📅")
    else:
        st.warning("Data: No campaigns loaded", icon="📭")

    # Date filter (only when data exists)
    if data_loaded and date_range:
        st.divider()
        st.caption("Date Filter")
        filter_start = st.date_input("From", value=date_range[0], min_value=date_range[0], max_value=date_range[1])
        filter_end = st.date_input("To", value=date_range[1], min_value=date_range[0], max_value=date_range[1])
    else:
        filter_start = None
        filter_end = None


# ─────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Campaign Overview")

    if not data_loaded:
        no_data_message()
    else:
        # KPI Cards
        comparison = get_kpis_comparison(filter_start, filter_end)
        metric_card_row(comparison["kpis"], comparison["deltas"])

        st.divider()

        # Charts row
        col_left, col_right = st.columns(2)

        with col_left:
            spend_df = get_daily_spend_by_platform(filter_start, filter_end)
            st.plotly_chart(
                spend_by_platform_chart(spend_df),
                use_container_width=True,
            )

        with col_right:
            roas_df = get_daily_roas(filter_start, filter_end)
            st.plotly_chart(
                roas_trend_chart(roas_df),
                use_container_width=True,
            )

        # Spend distribution pie + space for future chart
        col_pie, col_future = st.columns(2)

        with col_pie:
            st.plotly_chart(
                platform_pie_chart(spend_df),
                use_container_width=True,
            )

        with col_future:
            # Placeholder for predicted vs actual (Phase 2)
            st.markdown("#### 🔮 Predicted vs Actual")
            st.info("Forecast comparison will appear after Phase 2 (demand forecasting).")

        st.divider()

        # Campaign performance table
        st.subheader("Campaign Performance")
        summary_df = get_campaign_summary(filter_start, filter_end)
        formatted = campaign_performance_table(summary_df)
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        st.divider()

        # Sample recommendations (static for now, agent builds these in Phase 5)
        st.subheader("Recent Recommendations")
        st.caption("🤖 AI-generated recommendations will appear after Phase 5. Samples shown below.")

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
                          "This campaign shows declining ROAS. Recommend fresh creative or audience refresh.",
                confidence="medium",
                campaign_name="Meta - Prospecting (Interest)",
                card_key="sample_2",
            )


elif page == "📈 Forecasts":
    st.title("📈 Demand Forecasts & Trends")

    if not data_loaded:
        no_data_message()
    else:
        # Google Trends section
        st.subheader("📉 Google Trends — Market Signals")

        trends_df = get_trends()

        if not trends_df.empty:
            st.plotly_chart(
                trends_chart(trends_df),
                use_container_width=True,
            )

            st.caption(
                "Google Trends data shows relative search interest (0–100) over time. "
                "These signals feed into the demand forecasting model in Phase 2."
            )
        else:
            st.info(
                "No Google Trends data loaded yet. Run the seed script to fetch trends:\n\n"
                "```\nuv run python scripts/seed_demo.py\n```"
            )

        st.divider()

        # Placeholder for Prophet forecasts
        st.subheader("🔮 Demand Forecasts")
        st.info(
            "📈 Time-series forecasting (Prophet) will be built in Phase 2.\n\n"
            "The model will predict clicks, conversions, and spend 14 days ahead "
            "with confidence intervals, using historical campaign data + Google Trends as features."
        )

        # Placeholder for seasonality calendar
        st.subheader("📅 Seasonality Calendar")
        st.info(
            "A heatmap calendar showing when to push budget harder based on "
            "historical patterns and upcoming events. Coming in Phase 2."
        )


elif page == "🎯 Recommendations":
    st.title("🎯 Optimization Recommendations")

    if not data_loaded:
        no_data_message()
    else:
        st.info(
            "🤖 The LLM agent optimization loop will be built in Phases 4–5.\n\n"
            "It will run daily: analyze performance → check forecasts → "
            "run the contextual bandit optimizer → generate plain-English recommendations."
        )

        st.divider()

        # Show some static example recommendations
        st.subheader("Example Recommendations (Preview)")

        examples = [
            {
                "action_type": "bid_adjustment",
                "title": "Increase Google Shopping bid +8%",
                "reasoning": "Shopping campaign ROAS is 4.2x (above 3.0x target). "
                             "Demand forecast shows stable volume. Bandit model suggests "
                             "higher bid will capture additional profitable clicks.",
                "confidence": "high",
                "campaign_name": "Google - Shopping",
            },
            {
                "action_type": "budget_reallocation",
                "title": "Shift $50/day from Meta Interest → Meta Remarketing",
                "reasoning": "Meta Interest campaign ROAS declined to 0.9x (below break-even). "
                             "Meta Remarketing maintains 3.8x ROAS with room to scale. "
                             "Reallocating budget improves portfolio-level returns.",
                "confidence": "high",
                "campaign_name": "Meta - Prospecting (Interest)",
            },
            {
                "action_type": "creative_rotation",
                "title": "Refresh ad creative for Meta Lookalike",
                "reasoning": "Audience fatigue signals: CTR dropped 15% over 3 weeks while "
                             "impressions remained stable. Creative refresh typically recovers "
                             "50–70% of lost CTR within the first week.",
                "confidence": "medium",
                "campaign_name": "Meta - Prospecting (Lookalike)",
            },
            {
                "action_type": "bid_adjustment",
                "title": "Decrease Amazon Auto bid -10%",
                "reasoning": "Auto-targeting ACOS is 28%, above the 22% target. "
                             "Manual campaign performs better for the same keywords. "
                             "Reducing auto bid frees budget for manual targeting.",
                "confidence": "medium",
                "campaign_name": "Amazon - Sponsored Products (Auto)",
            },
        ]

        for i in range(0, len(examples), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(examples):
                    ex = examples[i + j]
                    with col:
                        recommendation_card(
                            **ex,
                            card_key=f"example_{i + j}",
                        )


elif page == "🔍 Competitors":
    st.title("🔍 Competitor Intelligence")
    st.info("🔍 Competitor monitoring will be built in Phase 6 using Meta Ad Library API and Google Ads Transparency Center.")


elif page == "📤 Upload Data":
    st.title("📤 Upload Campaign Data")

    st.markdown("""
    Upload your campaign performance data as a CSV file.  
    **Supported formats:** Google Ads, Meta Ads, Amazon Ads exports, or Ad-Work internal format.
    
    The system will auto-detect the platform and normalize the data.
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Export your campaign report from your ad platform and upload it here.",
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ Loaded file: {uploaded_file.name} — {len(df)} rows, {len(df.columns)} columns")

        # Auto-detect platform
        from adwork.data.schemas import detect_platform_from_columns
        detected = detect_platform_from_columns(df.columns.tolist())

        platform_labels = {
            "google": "🔵 Google Ads",
            "meta": "🔷 Meta Ads",
            "amazon": "🟠 Amazon Ads",
            "unknown": "📄 Internal / Unknown Format",
        }
        st.info(f"**Detected platform:** {platform_labels.get(detected.value, detected.value)}")

        # Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # Column check
        st.subheader("Columns Found")
        st.code(", ".join(df.columns.tolist()))

        # Import button
        st.divider()
        if st.button("🚀 Import into Ad-Work", type="primary", use_container_width=True):
            with st.spinner("Validating and importing..."):
                from adwork.data.ingestion import ingest_csv

                result = ingest_csv(df)

            if result["status"] == "success":
                st.success(
                    f"✅ Import successful! {result['rows_loaded']} rows loaded "
                    f"across {len(result['campaigns_found'])} campaigns."
                )
                st.balloons()
            elif result["status"] == "partial":
                st.warning(
                    f"⚠️ Partial import: {result['rows_loaded']}/{result['rows_total']} rows loaded. "
                    f"{result['rows_skipped']} rows skipped."
                )
                if result["errors"]:
                    with st.expander("View errors"):
                        for err in result["errors"]:
                            st.text(err)
            else:
                st.error("❌ Import failed.")
                if result["errors"]:
                    for err in result["errors"][:5]:
                        st.text(err)

    else:
        st.markdown("---")
        st.subheader("📁 Don't have data yet?")
        st.markdown(
            "Run the seed script to load sample data for 10 campaigns across 3 platforms:\n\n"
            "```bash\nuv run python scripts/seed_demo.py\n```\n\n"
            "Or on Windows:\n"
            "```\n.venv\\Scripts\\python scripts\\seed_demo.py\n```"
        )

        # Show sample files if they exist
        sample_dir = ROOT / "data" / "sample_campaigns"
        if sample_dir.exists():
            csv_files = list(sample_dir.glob("*.csv"))
            if csv_files:
                st.markdown("**Available sample files:**")
                for f in csv_files:
                    size_kb = f.stat().st_size / 1024
                    st.markdown(f"- `{f.name}` ({size_kb:.0f} KB)")


elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

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

    # Database stats
    st.subheader("Database Statistics")

    if db_connected:
        try:
            stats = {
                "Campaigns": db.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0],
                "Daily Metric Rows": db.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0],
                "Recommendations": db.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0],
                "Forecasts": db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0],
                "Trend Data Points": db.execute("SELECT COUNT(*) FROM search_trends").fetchone()[0],
            }
            for key, value in stats.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{key}**")
                with col2:
                    st.code(f"{value:,}")
        except Exception as e:
            st.error(f"Error reading database stats: {e}")

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

                st.success(f"✅ Response from {response.provider}/{response.model}:")
                st.markdown(f"> {response.content}")
                st.caption(f"Tokens used: {response.usage}")

            except Exception as e:
                st.error(f"❌ LLM connection failed: {e}")

    st.divider()

    # Reset database
    st.subheader("⚠️ Danger Zone")
    if st.button("🗑️ Reset Database", type="secondary"):
        from adwork.db.connection import reset_db
        reset_db()
        st.warning("Database has been reset. Run the seed script to reload sample data.")
        st.rerun()