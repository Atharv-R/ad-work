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

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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
        get_all_campaigns,
        get_campaign_summary,
        get_daily_metrics_for_campaign,
        get_daily_roas,
        get_daily_spend_by_platform,
        get_date_range,
        get_forecast_summary,
        get_kpis_comparison,
        get_trends,
        has_data,
        has_forecasts,
        store_forecast_results,
    )
    db = get_db()
    db_connected = True
    db_error = None
except Exception as e:
    db_connected = False
    db_error = str(e)

from adwork.data.competitors import (
    ingest_competitor_csv,
    seed_competitor_data,
)
from adwork.db.queries import (
    clear_competitor_data,
    get_competitor_ads,
    get_competitor_advertisers,
    has_competitor_data,
    store_cluster_results,
)
from adwork.models.competitor_nlp import CompetitorAnalyzer

try:
    from components.cards import (
        metric_card_row,
        no_data_message,
        recommendation_card,
    )
    from components.charts import (
        campaign_performance_table,
        competitor_cluster_bars,
        competitor_cluster_scatter,
        competitor_platform_breakdown,
        competitor_strategy_heatmap,
        platform_pie_chart,
        roas_trend_chart,
        spend_by_platform_chart,
        trends_chart,
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

# --- Helper for matching campaigns to trends keywords ---
def _match_campaign_trends(campaign_name: str, trends_df) -> pd.DataFrame | None:
    """Match a campaign to relevant Google Trends keywords."""
    if trends_df is None or trends_df.empty:
        return None

    name_lower = campaign_name.lower()
    keyword_matches = {
        "laptop": ["laptop"],
        "headphone": ["headphones"],
        "monitor": ["computer monitor"],
        "earbud": ["wireless earbuds"],
        "keyboard": ["gaming keyboard"],
    }

    matched = []
    for trigger, keywords in keyword_matches.items():
        if trigger in name_lower:
            matched.extend(keywords)

    if not matched:
        matched = trends_df["keyword"].unique().tolist()[:3]

    filtered = trends_df[trends_df["keyword"].isin(matched)]
    return filtered if not filtered.empty else None


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
    st.title("📈 Demand Forecasts")

    if not data_loaded:
        no_data_message()
    else:
        # --- Campaign and metric selectors ---
        campaigns_df = get_all_campaigns() if db_connected else pd.DataFrame()

        col_sel1, col_sel2, col_sel3 = st.columns([3, 2, 2])

        with col_sel1:
            campaign_options = dict(zip(
                campaigns_df["campaign_id"],
                campaigns_df["campaign_name"],
            )) if not campaigns_df.empty else {}

            selected_campaign_id = st.selectbox(
                "Campaign",
                options=list(campaign_options.keys()),
                format_func=lambda x: campaign_options.get(x, x),
            )

        with col_sel2:
            selected_metric = st.selectbox(
                "Metric",
                options=["clicks", "conversions", "spend", "revenue"],
            )

        with col_sel3:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button(
                "🔮 Generate Forecast", type="primary", use_container_width=True,
            )

        st.divider()

        # --- Generate forecast on button click ---
        if generate_btn and selected_campaign_id:
            with st.spinner(f"Training forecast model for {campaign_options.get(selected_campaign_id, '')}..."):
                try:
                    from adwork.db.queries import (
                        get_daily_metrics_for_campaign,
                        store_forecast_results,
                    )
                    from adwork.models.forecaster import DemandForecaster

                    hist = get_daily_metrics_for_campaign(selected_campaign_id)
                    trends_data = get_trends()

                    # Match trends to campaign
                    campaign_name = campaign_options.get(selected_campaign_id, "")
                    matched_trends = _match_campaign_trends(campaign_name, trends_data)

                    forecaster = DemandForecaster()
                    result = forecaster.run(
                        historical_df=hist,
                        metric=selected_metric,
                        campaign_id=selected_campaign_id,
                        trends_df=matched_trends,
                    )

                    # Store in DB
                    store_forecast_results(
                        selected_campaign_id, selected_metric, result["forecast"],
                    )

                    # Save to session state so we can display it
                    st.session_state["forecast_result"] = result
                    st.success(
                        f"✅ Forecast generated using **{result['engine'].title()}** engine | "
                        f"MAPE: {result['evaluation']['mape']:.1f}%"
                    )

                except Exception as e:
                    st.error(f"❌ Forecast failed: {e}")
                    st.session_state.pop("forecast_result", None)

        # --- Display results ---
        result = st.session_state.get("forecast_result")

        if result and result.get("campaign_id") == selected_campaign_id and result.get("metric") == selected_metric:

            # Import chart functions
            from components.charts import (
                backtest_chart,
                evaluation_metrics_display,
                forecast_chart,
                seasonality_heatmap,
                weekly_seasonality_chart,
            )

            # Evaluation metrics row
            st.subheader("📊 Model Evaluation (Backtest)")
            eval_display = evaluation_metrics_display(result["evaluation"])

            mcols = st.columns(4)
            for i, (name, info) in enumerate(eval_display.items()):
                with mcols[i]:
                    icon = "✅" if info["good"] else "⚠️"
                    st.metric(label=f"{icon} {name}", value=info["value"], help=info["help"])

            st.divider()

            # Forecast chart
            st.subheader("🔮 Forecast")
            st.plotly_chart(
                forecast_chart(result["forecast"], selected_metric),
                use_container_width=True,
            )

            # Backtest chart
            col_bt, col_season = st.columns(2)

            with col_bt:
                st.plotly_chart(
                    backtest_chart(result["backtest"], selected_metric),
                    use_container_width=True,
                )

            with col_season:
                st.plotly_chart(
                    weekly_seasonality_chart(result.get("components", {})),
                    use_container_width=True,
                )

            # Seasonality heatmap
            st.subheader("📅 Seasonality Calendar")
            hist_for_heatmap = get_daily_metrics_for_campaign(selected_campaign_id)
            st.plotly_chart(
                seasonality_heatmap(hist_for_heatmap, selected_metric),
                use_container_width=True,
            )

        else:
            # No forecast generated yet — show stored forecasts or prompt
            from adwork.db.queries import get_forecast_summary, has_forecasts

            if has_forecasts():
                st.subheader("📋 Stored Forecasts")
                st.caption("Click **Generate Forecast** above to create a new forecast with full evaluation.")
                summary = get_forecast_summary()
                st.dataframe(summary, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "👆 Select a campaign and metric above, then click **Generate Forecast** "
                    "to train a Prophet model and see predictions with evaluation metrics."
                )

        # --- Google Trends section ---
        st.divider()
        st.subheader("📉 Google Trends — Market Signals")
        trends_data = get_trends()

        if not trends_data.empty:
            from components.charts import trends_chart
            st.plotly_chart(trends_chart(trends_data), use_container_width=True)
            st.caption(
                "Search interest data feeds into the forecasting model as an external regressor, "
                "improving predictions by capturing market demand shifts."
            )
        else:
            st.info("No Google Trends data loaded. Trends are optional — forecasting works without them.")



elif page == "🎯 Recommendations":
    st.title("🎯 Optimization Recommendations")

    if not data_loaded:
        no_data_message()
    else:
        # Check for agent output
        agent_output_path = ROOT / "data" / "processed" / "agent_output.json"
        sim_path = ROOT / "data" / "processed" / "simulation_results.json"
        alloc_path = ROOT / "data" / "processed" / "allocation_results.json"

        # Run Agent button
        col_run, col_clear = st.columns([3, 1])

        with col_run:
            if st.button("🤖 Run Optimization Agent", type="primary", use_container_width=True):
                with st.spinner("Running LangGraph agent pipeline..."):
                    try:
                        from adwork.pipeline.daily_loop import run_daily_optimization
                        result = run_daily_optimization()

                        # Also run simulation for charts
                        import json as json_lib

                        from adwork.models.bandit import run_bandit_simulation

                        demo_camp = db.execute(
                            "SELECT * FROM campaigns WHERE platform='google' LIMIT 1"
                        ).df()
                        if not demo_camp.empty:
                            cid = demo_camp.iloc[0]["campaign_id"]
                            dm = db.execute(
                                f"SELECT * FROM daily_metrics WHERE campaign_id='{cid}' ORDER BY date"
                            ).df().tail(30)
                            if not dm.empty:
                                t_i = dm["impressions"].sum()
                                t_cl = dm["clicks"].sum()
                                t_co = dm["conversions"].sum()
                                t_sp = dm["spend"].sum()
                                t_re = dm["revenue"].sum()
                                sim = run_bandit_simulation(
                                    base_metrics={
                                        "base_impressions": t_i / len(dm),
                                        "base_ctr": t_cl / t_i if t_i > 0 else 0.03,
                                        "base_cpc": t_sp / t_cl if t_cl > 0 else 1.5,
                                        "base_conv_rate": t_co / t_cl if t_cl > 0 else 0.03,
                                        "base_aov": t_re / t_co if t_co > 0 else 200,
                                    },
                                    n_rounds=90,
                                )
                                proc = ROOT / "data" / "processed"
                                proc.mkdir(parents=True, exist_ok=True)
                                with open(proc / "simulation_results.json", "w") as f:
                                    json_lib.dump({
                                        "strategies": [s.model_dump() for s in sim["strategies"]],
                                        "oracle": sim["oracle"],
                                        "n_rounds": sim["n_rounds"],
                                        "ts_beliefs": sim["ts_beliefs"],
                                    }, f, indent=2)

                        n_recs = len(result.get("final_recommendations", []))
                        st.success(f"✅ Agent complete: {n_recs} recommendations generated")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Agent failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        with col_clear:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear", use_container_width=True):
                from adwork.db.queries import clear_recommendations
                clear_recommendations()
                for f in (ROOT / "data" / "processed").glob("*.json"):
                    f.unlink(missing_ok=True)
                st.rerun()

        # ── Display Agent Output ──
        if agent_output_path.exists():
            with open(agent_output_path) as f:
                agent_data = json.load(f)

            # Daily Summary
            summary = agent_data.get("daily_summary", "")
            if summary:
                st.divider()
                health = agent_data.get("portfolio_health", "normal")
                health_emoji = "🚨" if health == "critical" else "✅"
                st.subheader(f"{health_emoji} Daily Brief")
                st.markdown(summary)

            # Agent thought process
            agent_log = agent_data.get("agent_log", [])
            if agent_log:
                with st.expander("🤖 Agent Thought Process", expanded=False):
                    for entry in agent_log:
                        st.markdown(f"- {entry}")
                    errors = agent_data.get("errors", [])
                    if errors:
                        st.markdown("**Errors:**")
                        for err in errors:
                            st.markdown(f"- ⚠️ {err}")
                    provider = agent_data.get("llm_provider", "unknown")
                    ts = agent_data.get("run_timestamp", "")[:19]
                    st.caption(f"LLM: {provider} | Run: {ts}")

            # Recommendations
            recs = agent_data.get("final_recommendations", [])
            if recs:
                st.divider()
                st.subheader(f"📋 Recommendations ({len(recs)})")

                sorted_recs = sorted(recs, key=lambda r: r.get("priority", 99))
                cols = st.columns(2)
                for idx, rec in enumerate(sorted_recs):
                    with cols[idx % 2]:
                        action_type = rec.get("action_type", "bid_adjustment")
                        title = f"#{rec.get('priority', '?')} {rec.get('action_summary', 'Action')}"

                        recommendation_card(
                            action_type=action_type,
                            title=title,
                            reasoning=rec.get("reasoning", ""),
                            confidence=rec.get("confidence", "medium"),
                            campaign_name=rec.get("campaign_name", ""),
                            card_key=f"agent_{idx}",
                        )

        else:
            st.info(
                "👆 Click **Run Optimization Agent** to execute the full LangGraph pipeline:\n\n"
                "1. Gather campaign data\n"
                "2. LLM analyzes performance\n"
                "3. Route by portfolio health\n"
                "4. Run Thompson Sampling optimization\n"
                "5. LLM synthesizes recommendations with reasoning"
            )
        # ── Simulation Charts ──
        if sim_path.exists():
            st.divider()
            st.subheader("🧪 Bandit Simulation (Validation)")

            with open(sim_path) as f:
                sim_data = json.load(f)

            from components.charts import (
                arm_selection_chart,
                beliefs_chart,
                budget_allocation_chart,
                regret_curve_chart,
            )

            col_r, col_a = st.columns(2)
            with col_r:
                st.plotly_chart(regret_curve_chart(sim_data), use_container_width=True)
            with col_a:
                st.plotly_chart(arm_selection_chart(sim_data), use_container_width=True)

            ts_beliefs = sim_data.get("ts_beliefs", {})
            if ts_beliefs:
                st.plotly_chart(beliefs_chart(ts_beliefs), use_container_width=True)

            # Summary table
            st.caption("**Simulation Summary (90 rounds)**")
            sim_summary = []
            for s in sim_data.get("strategies", []):
                final_regret = s["cumulative_regret"][-1] if s.get("cumulative_regret") else 0
                sim_summary.append({
                    "Strategy": s["strategy"],
                    "Total Reward": f"${s['total_reward']:,.0f}",
                    "Final Regret": f"${final_regret:,.0f}",
                })
            oracle = sim_data.get("oracle", {})
            sim_summary.append({
                "Strategy": "Oracle (best possible)",
                "Total Reward": f"${oracle.get('total_reward', 0):,.0f}",
                "Final Regret": "$0",
            })
            st.dataframe(pd.DataFrame(sim_summary), use_container_width=True, hide_index=True)

            # Budget allocation chart
            if alloc_path.exists():
                with open(alloc_path) as f:
                    alloc_data = json.load(f)

                st.divider()
                st.subheader("💰 Budget Allocation")
                st.plotly_chart(
                    budget_allocation_chart(
                        alloc_data.get("current", {}),
                        alloc_data.get("recommended", {}),
                    ),
                    use_container_width=True,
                )

# ── 🔍 Competitors page ─────────────────────────────────────────
elif page == "🔍 Competitors":
    st.header("🔍 Competitor Intelligence")

    

    # ── Sidebar controls for this page ──────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Competitor Data")

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("🎲 Load Demo Data", use_container_width=True):
            seed_competitor_data()
            st.rerun()
    with col_s2:
        if st.button("🗑️ Clear Data", use_container_width=True):
            clear_competitor_data()
            st.rerun()

    # ── Upload section ──────────────────────────────────────────
    with st.expander("📤 Upload Competitor CSV", expanded=not has_competitor_data()):
        st.markdown(
            "Upload a CSV with at least an **ad copy** column. "
            "Accepted column names: `ad_copy`, `copy`, `body`, `text`, `description`, `ad_text`."
        )
        uploaded = st.file_uploader(
            "Choose CSV", type=["csv"], key="comp_upload"
        )
        if uploaded is not None:
            try:
                upload_df = pd.read_csv(uploaded)
                st.dataframe(upload_df.head(), use_container_width=True)
                if st.button("Import Competitor Ads"):
                    ads = ingest_competitor_csv(upload_df)
                    st.success(f"Imported {len(ads)} competitor ads")
                    st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    if not has_competitor_data():
        st.info("No competitor data yet. Load demo data or upload a CSV above.")
        st.stop()

    # ── Load data ───────────────────────────────────────────────
    all_ads_df = get_competitor_ads()
    n_ads = len(all_ads_df)
    n_advertisers = all_ads_df["advertiser_name"].nunique()
    n_platforms = all_ads_df["platform"].nunique()

    # ── KPI row ─────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Ads", n_ads)
    k2.metric("Advertisers", n_advertisers)
    k3.metric("Platforms", n_platforms)
    k4.metric("Categories", all_ads_df["category"].nunique())

    st.markdown("---")

    # ── Run analysis ────────────────────────────────────────────
    n_clusters = st.slider("Number of clusters", 3, 8, 5)

    if st.button("🔬 Run Cluster Analysis", type="primary", use_container_width=True):
        with st.spinner("Clustering competitor ads..."):
            analyzer = CompetitorAnalyzer(n_clusters=n_clusters)
            results = analyzer.analyze(all_ads_df)

            # Store in session state so charts persist
            st.session_state["comp_results"] = results

            # Persist cluster results to DB
            from datetime import date
            store_cluster_results(results["clusters"], str(date.today()))

        st.success(
            f"Clustered {n_ads} ads into {len(results['clusters'])} strategies"
        )

    # ── Display results ─────────────────────────────────────────
    if "comp_results" not in st.session_state:
        st.info("Click **Run Cluster Analysis** above to analyse competitor ads.")
        st.stop()

    results = st.session_state["comp_results"]
    clusters = results["clusters"]
    ads_df = results["ads_df"]
    strategy_matrix = results["strategy_matrix"]

    # ── Charts ──────────────────────────────────────────────────
    tab_scatter, tab_strategy, tab_explore = st.tabs([
        "🗺️ Cluster Map", "📊 Strategy Analysis", "🔎 Explore Ads"
    ])

    with tab_scatter:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(
                competitor_cluster_scatter(ads_df),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                competitor_cluster_bars(clusters),
                use_container_width=True,
            )

        # Cluster detail expanders
        for c in clusters:
            with st.expander(f"**{c['label']}** — {c['n_ads']} ads"):
                st.markdown(f"**Top terms:** {', '.join(c['top_terms'][:8])}")
                cluster_ads = ads_df[ads_df["cluster"] == c["cluster_id"]]
                for _, row in cluster_ads.head(3).iterrows():
                    st.markdown(
                        f"> **{row.get('advertiser_name', '')}** "
                        f"({row.get('platform', '')}): "
                        f"*{row.get('headline', '')}*\n> {row['ad_copy']}"
                    )

    with tab_strategy:

        st.plotly_chart(
            competitor_strategy_heatmap(strategy_matrix),
            use_container_width=True,
        )
        st.plotly_chart(
            competitor_platform_breakdown(ads_df),
            use_container_width=True,
        )

        # Key insight callout
        if not strategy_matrix.empty:
            dominant = strategy_matrix.idxmax(axis=1)
            st.markdown("#### Key Observations")
            for adv, strat in dominant.items():
                count = strategy_matrix.loc[adv, strat]
                st.markdown(f"- **{adv}** leans toward **{strat}** ({count} ads)")

    with tab_explore:
        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            adv_filter = st.selectbox(
                "Advertiser",
                ["All"] + get_competitor_advertisers(),
            )
        with f2:
            plat_filter = st.selectbox(
                "Platform",
                ["All"] + sorted(all_ads_df["platform"].unique().tolist()),
            )
        with f3:
            cluster_filter = st.selectbox(
                "Strategy",
                ["All"] + [c["label"] for c in clusters],
            )

        filtered = ads_df.copy()
        if adv_filter != "All":
            filtered = filtered[filtered["advertiser_name"] == adv_filter]
        if plat_filter != "All":
            filtered = filtered[filtered["platform"] == plat_filter]
        if cluster_filter != "All":
            filtered = filtered[filtered["cluster_label"] == cluster_filter]

        st.markdown(f"**Showing {len(filtered)} ads**")

        for _, row in filtered.iterrows():
            with st.container():
                st.markdown(
                    f"**{row.get('headline', 'No headline')}** · "
                    f"`{row.get('advertiser_name', '')}` · "
                    f"`{row.get('platform', '')}` · "
                    f"🏷️ {row.get('cluster_label', 'Unclustered')}"
                )
                st.caption(row["ad_copy"])
                if row.get("cta"):
                    st.markdown(f"CTA: **{row['cta']}**")
                st.markdown("---")

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

    # ── CTR Model Info ──
    st.subheader("🧠 CTR Prediction Model")

    try:
        from adwork.models.registry import ctr_model_exists, get_ctr_metadata

        if ctr_model_exists():
            meta = get_ctr_metadata()
            if meta:
                ev = meta.get("evaluation", {})

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("AUC-ROC", f"{ev.get('auc_roc', 0):.4f}")
                with m2:
                    st.metric("Log Loss", f"{ev.get('log_loss', 0):.4f}")
                with m3:
                    st.metric("PR-AUC", f"{ev.get('pr_auc', 0):.4f}")
                with m4:
                    st.metric("Cal. Error", f"{ev.get('calibration_error', 0):.4f}")

                # Details
                with st.expander("Model Details"):
                    det_col1, det_col2 = st.columns(2)
                    with det_col1:
                        st.markdown(f"**Trained:** {meta.get('trained_at', 'Unknown')[:19]}")
                        st.markdown(f"**Data:** {meta.get('data_source', 'Unknown')}")
                        st.markdown(f"**Sample Size:** {meta.get('sample_size', 'Unknown'):,}")
                    with det_col2:
                        st.markdown(f"**Best Iteration:** {meta.get('best_iteration', 'Unknown')}")
                        st.markdown(f"**Test Size:** {ev.get('test_size', 'Unknown'):,}")
                        st.markdown(f"**Base Rate:** {ev.get('base_rate', 0):.4f}")

                # Feature importance
                feat_imp = ev.get("feature_importance", {})
                if feat_imp:
                    with st.expander("Feature Importance (Top 15)"):
                        imp_df = pd.DataFrame(
                            list(feat_imp.items()),
                            columns=["Feature", "Importance (Gain)"],
                        )
                        st.bar_chart(imp_df.set_index("Feature"), height=350)

                # Calibration curve
                cal = ev.get("calibration_curve", {})
                if cal.get("mean_predicted") and cal.get("fraction_positive"):
                    with st.expander("Calibration Curve"):
                        import plotly.graph_objects as go

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=cal["mean_predicted"],
                            y=cal["fraction_positive"],
                            mode="lines+markers",
                            name="Model",
                            line=dict(color="#4F8BF9", width=2),
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode="lines", name="Perfect Calibration",
                            line=dict(color="gray", dash="dash"),
                        ))
                        fig.update_layout(
                            title="Calibration: Predicted vs Actual Click Rate",
                            xaxis_title="Mean Predicted Probability",
                            yaxis_title="Fraction of Positives",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No CTR model trained yet. Train it locally:\n\n"
                "```bash\n"
                "uv run python scripts/train_ctr.py            # Real Criteo data\n"
                "uv run python scripts/train_ctr.py --synthetic # Quick test\n"
                "```"
            )

    except Exception as e:
        st.warning(f"Could not load model info: {e}")

    st.divider()

    # ── Database Statistics ──
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

    # ── LLM Test ──
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
            except Exception as e:
                st.error(f"❌ Failed: {e}")

    st.divider()
    st.subheader("⚠️ Danger Zone")
    if st.button("🗑️ Reset Database"):
        from adwork.db.connection import reset_db
        reset_db()
        st.warning("Database reset. Run seed script to reload.")
        st.rerun()