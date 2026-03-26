# dashboard/components/charts.py

"""
Plotly chart builders for the Ad-Work dashboard.
Each function returns a Plotly figure ready for st.plotly_chart().
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def spend_by_platform_chart(df: pd.DataFrame) -> go.Figure:
    """
    Stacked area chart of daily spend by platform.
    
    Args:
        df: DataFrame with columns [date, platform, spend]
    """
    if df.empty:
        return _empty_chart("No spend data available")

    df = df.copy()
    df["platform"] = df["platform"].str.capitalize()

    fig = px.area(
        df,
        x="date",
        y="spend",
        color="platform",
        title="Daily Spend by Platform",
        labels={"spend": "Spend ($)", "date": "Date", "platform": "Platform"},
        color_discrete_map={
            "Google": "#4285F4",
            "Meta": "#1877F2",
            "Amazon": "#FF9900",
        },
    )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
    )

    return fig


def roas_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of daily ROAS with a reference line at 1.0 (break-even).
    
    Args:
        df: DataFrame with columns [date, roas]
    """
    if df.empty:
        return _empty_chart("No ROAS data available")

    fig = go.Figure()

    # ROAS line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["roas"],
        mode="lines",
        name="ROAS",
        line=dict(color="#4F8BF9", width=2),
        hovertemplate="Date: %{x}<br>ROAS: %{y:.2f}x<extra></extra>",
    ))

    # 7-day moving average
    if len(df) >= 7:
        df = df.copy()
        df["roas_ma7"] = df["roas"].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["roas_ma7"],
            mode="lines",
            name="7-day avg",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
        ))

    # Break-even reference line
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="gray",
        annotation_text="Break-even (1.0x)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Daily ROAS Trend",
        yaxis_title="ROAS (x)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
    )

    return fig


def campaign_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format campaign summary DataFrame for display.
    Returns a formatted DataFrame ready for st.dataframe().
    """
    if df.empty:
        return df

    df = df.copy()

    format_map = {
        "impressions": lambda x: f"{x:,.0f}",
        "clicks": lambda x: f"{x:,.0f}",
        "ctr": lambda x: f"{x:.2%}",
        "spend": lambda x: f"${x:,.2f}",
        "avg_cpc": lambda x: f"${x:.2f}",
        "conversions": lambda x: f"{x:,.0f}",
        "conv_rate": lambda x: f"{x:.2%}",
        "revenue": lambda x: f"${x:,.2f}",
        "roas": lambda x: f"{x:.2f}x",
    }

    for col, fmt in format_map.items():
        if col in df.columns:
            df[col] = df[col].apply(fmt)

    display_names = {
        "campaign_name": "Campaign",
        "platform": "Platform",
        "impressions": "Impressions",
        "clicks": "Clicks",
        "ctr": "CTR",
        "spend": "Spend",
        "avg_cpc": "Avg CPC",
        "conversions": "Conv.",
        "conv_rate": "Conv Rate",
        "revenue": "Revenue",
        "roas": "ROAS",
    }

    df = df.rename(columns=display_names)
    if "Platform" in df.columns:
        df["Platform"] = df["Platform"].str.capitalize()

    return df


def trends_chart(df: pd.DataFrame) -> go.Figure:
    """
    Multi-line chart of Google Trends interest over time.
    
    Args:
        df: DataFrame with columns [date, keyword, interest]
    """
    if df.empty:
        return _empty_chart("No trends data available")

    df = df.copy()
    df["keyword"] = df["keyword"].str.title()

    fig = px.line(
        df,
        x="date",
        y="interest",
        color="keyword",
        title="Google Trends — Search Interest Over Time",
        labels={
            "interest": "Search Interest (0–100)",
            "date": "Date",
            "keyword": "Keyword",
        },
    )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
    )

    return fig


def daily_metrics_chart(df: pd.DataFrame, metric: str = "spend") -> go.Figure:
    """
    Single campaign daily metric over time.
    Used on drill-down views.
    
    Args:
        df: DataFrame with columns [date, <metric>]
        metric: Which column to plot — 'spend', 'clicks', 'conversions', 'revenue'
    """
    if df.empty or metric not in df.columns:
        return _empty_chart(f"No {metric} data available")

    display_names = {
        "spend": "Spend ($)",
        "clicks": "Clicks",
        "conversions": "Conversions",
        "revenue": "Revenue ($)",
        "impressions": "Impressions",
        "ctr": "CTR",
        "roas": "ROAS",
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[metric],
        mode="lines+markers",
        name=display_names.get(metric, metric),
        line=dict(color="#4F8BF9", width=2),
        marker=dict(size=4),
    ))

    # Add 7-day moving average if enough data
    if len(df) >= 7:
        ma = df[metric].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=ma,
            mode="lines",
            name="7-day avg",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
        ))

    fig.update_layout(
        title=f"Daily {display_names.get(metric, metric)}",
        yaxis_title=display_names.get(metric, metric),
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
    )

    return fig


def platform_pie_chart(df: pd.DataFrame) -> go.Figure:
    """
    Pie chart showing spend distribution across platforms.
    
    Args:
        df: DataFrame with columns [platform, spend]
    """
    if df.empty:
        return _empty_chart("No data available")

    # Aggregate by platform
    platform_spend = df.groupby("platform")["spend"].sum().reset_index()
    platform_spend["platform"] = platform_spend["platform"].str.capitalize()

    fig = px.pie(
        platform_spend,
        values="spend",
        names="platform",
        title="Spend Distribution by Platform",
        color="platform",
        color_discrete_map={
            "Google": "#4285F4",
            "Meta": "#1877F2",
            "Amazon": "#FF9900",
        },
        hole=0.4,  # Donut chart
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
    )

    return fig


def forecast_chart(forecast_df: pd.DataFrame, metric: str = "clicks") -> go.Figure:
    """
    The signature forecast visualization:
    - Historical actuals as solid blue line
    - Forecast as dashed red line
    - Confidence interval as shaded area
    """
    if forecast_df.empty:
        return _empty_chart("No forecast data")

    df = forecast_df.copy()
    hist = df[~df["is_forecast"]]
    fwd = df[df["is_forecast"]]

    fig = go.Figure()

    # Historical actuals
    if not hist.empty and hist["actual"].notna().any():
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["actual"],
            mode="lines", name="Actual",
            line=dict(color="#4F8BF9", width=2),
        ))

    # Historical fitted values
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["predicted"],
            mode="lines", name="Fitted",
            line=dict(color="#4F8BF9", width=1, dash="dot"),
            opacity=0.5,
        ))

    # Forward forecast
    if not fwd.empty:
        connect_date = hist["date"].iloc[-1] if not hist.empty else fwd["date"].iloc[0]
        connect_val = hist["predicted"].iloc[-1] if not hist.empty else fwd["predicted"].iloc[0]

        fwd_dates = pd.concat([pd.Series([connect_date]), fwd["date"]], ignore_index=True)
        fwd_preds = pd.concat([pd.Series([connect_val]), fwd["predicted"]], ignore_index=True)
        fwd_lower = pd.concat([pd.Series([connect_val]), fwd["lower"]], ignore_index=True)
        fwd_upper = pd.concat([pd.Series([connect_val]), fwd["upper"]], ignore_index=True)

        # Confidence interval shading
        fig.add_trace(go.Scatter(
            x=pd.concat([fwd_dates, fwd_dates[::-1]]),
            y=pd.concat([fwd_upper, fwd_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(79, 139, 249, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Confidence",
            showlegend=True,
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fwd_dates, y=fwd_preds,
            mode="lines+markers", name="Forecast",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            marker=dict(size=5),
        ))

    # Vertical divider at forecast start
    if not hist.empty and not fwd.empty:
        cutoff = hist["date"].iloc[-1]

        # Use add_shape + add_annotation separately
        # (add_vline with annotation_text breaks on date axes)
        fig.add_shape(
            type="line",
            x0=cutoff, x1=cutoff,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="gray", width=1, dash="dot"),
            opacity=0.5,
        )
        fig.add_annotation(
            x=cutoff,
            y=1.0,
            yref="paper",
            text="Forecast →",
            showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor="left",
            yanchor="bottom",
        )

    metric_labels = {
        "clicks": "Clicks", "conversions": "Conversions",
        "spend": "Spend ($)", "revenue": "Revenue ($)",
    }

    fig.update_layout(
        title=f"{metric_labels.get(metric, metric)} — Actual vs Forecast",
        yaxis_title=metric_labels.get(metric, metric),
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
    )

    return fig


def backtest_chart(backtest_df: pd.DataFrame, metric: str = "clicks") -> go.Figure:
    """
    Backtest comparison: actual vs predicted in held-out test period.
    Shows how well the model would have performed.
    """
    if backtest_df.empty:
        return _empty_chart("No backtest data")

    fig = go.Figure()

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([backtest_df["date"], backtest_df["date"][::-1]]),
        y=pd.concat([backtest_df["upper"], backtest_df["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(79, 139, 249, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% CI",
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=backtest_df["date"], y=backtest_df["actual"],
        mode="lines+markers", name="Actual",
        line=dict(color="#4F8BF9", width=2),
        marker=dict(size=6),
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=backtest_df["date"], y=backtest_df["predicted"],
        mode="lines+markers", name="Predicted",
        line=dict(color="#FF6B6B", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))

    metric_labels = {
        "clicks": "Clicks", "conversions": "Conversions",
        "spend": "Spend ($)", "revenue": "Revenue ($)",
    }

    fig.update_layout(
        title=f"Backtest — {metric_labels.get(metric, metric)} (Held-Out Period)",
        yaxis_title=metric_labels.get(metric, metric),
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
    )

    return fig


def weekly_seasonality_chart(components: dict) -> go.Figure:
    """
    Bar chart showing the weekly seasonality effect by day of week.
    Positive bars = above average, negative = below average.
    """
    weekly = components.get("weekly")
    if not weekly:
        return _empty_chart("No weekly seasonality data")

    days = list(weekly.keys())
    values = list(weekly.values())
    colors = ["#4CAF50" if v >= 0 else "#FF5252" for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=days,
        y=values,
        marker_color=colors,
        text=[f"{v:+.1f}" for v in values],
        textposition="outside",
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)

    fig.update_layout(
        title="Weekly Seasonality Effect",
        yaxis_title="Effect on Metric",
        xaxis_title="Day of Week",
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        showlegend=False,
    )

    return fig


def seasonality_heatmap(historical_df: pd.DataFrame, metric: str = "clicks") -> go.Figure:
    """
    Calendar-style heatmap: week × day-of-week.
    Shows performance patterns across the entire date range.
    """
    if historical_df.empty or metric not in historical_df.columns:
        return _empty_chart("No data for heatmap")

    df = historical_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.day_name()
    df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    pivot = df.pivot_table(
        index="dow", columns="week_start", values=metric, aggfunc="sum",
    )

    # Reindex rows to correct day order
    pivot = pivot.reindex(day_order)

    # Format week labels
    week_labels = [d.strftime("%b %d") for d in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=week_labels,
        y=day_order,
        colorscale="Blues",
        hovertemplate="Week of %{x}<br>%{y}<br>Value: %{z:,.0f}<extra></extra>",
    ))

    metric_labels = {
        "clicks": "Clicks", "conversions": "Conversions",
        "spend": "Spend ($)", "revenue": "Revenue ($)",
    }

    fig.update_layout(
        title=f"Seasonality Heatmap — {metric_labels.get(metric, metric)} by Week & Day",
        xaxis_title="Week",
        yaxis_title="",
        margin=dict(l=80, r=20, t=50, b=40),
        height=350,
    )

    return fig

def forecast_vs_actual_chart(df: pd.DataFrame, metric: str = "clicks") -> go.Figure:
    """Scatter plot: predicted vs actual with perfect-prediction line."""
    if df.empty:
        return _empty_chart("No forecast vs actual data — run forecasts first")

    fig = go.Figure()

    # Perfect prediction line
    min_val = min(df["actual_value"].min(), df["predicted_value"].min())
    max_val = max(df["actual_value"].max(), df["predicted_value"].max())
    padding = (max_val - min_val) * 0.05
    line_range = [min_val - padding, max_val + padding]

    fig.add_trace(go.Scatter(
        x=line_range,
        y=line_range,
        mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Perfect Prediction",
        showlegend=True,
    ))

    # Points coloured by campaign
    campaigns = df["campaign_id"].unique()
    colors = px.colors.qualitative.Set2
    for i, cid in enumerate(campaigns):
        subset = df[df["campaign_id"] == cid]
        fig.add_trace(go.Scatter(
            x=subset["actual_value"],
            y=subset["predicted_value"],
            mode="markers",
            name=cid,
            marker=dict(size=7, color=colors[i % len(colors)], opacity=0.7),
            hovertemplate=(
                "Actual: %{x:.0f}<br>"
                "Predicted: %{y:.0f}<br>"
                "Date: %{customdata[0]}"
                "<extra>%{customdata[1]}</extra>"
            ),
            customdata=subset[["date", "campaign_id"]].values,
        ))

    fig.update_layout(
        title=f"Predicted vs Actual — {metric.title()}",
        xaxis_title=f"Actual {metric.title()}",
        yaxis_title=f"Predicted {metric.title()}",
        height=450,
        template="plotly_white",
    )
    return fig

def evaluation_metrics_display(evaluation: dict) -> dict:
    """
    Format evaluation metrics for display.
    Returns a dict ready for rendering as metric cards.
    """
    return {
        "MAPE": {
            "value": f"{evaluation.get('mape', 0):.1f}%",
            "help": "Mean Absolute Percentage Error — lower is better",
            "good": evaluation.get("mape", 100) < 20,
        },
        "MAE": {
            "value": f"{evaluation.get('mae', 0):,.1f}",
            "help": "Mean Absolute Error — average prediction miss in raw units",
            "good": True,
        },
        "RMSE": {
            "value": f"{evaluation.get('rmse', 0):,.1f}",
            "help": "Root Mean Squared Error — penalizes large misses",
            "good": True,
        },
        "Coverage": {
            "value": f"{evaluation.get('coverage', 0):.0%}",
            "help": "% of actuals within the 80% confidence interval",
            "good": evaluation.get("coverage", 0) >= 0.70,
        },
    }


def regret_curve_chart(simulation_data: dict) -> go.Figure:
    """
    Cumulative regret comparison across bandit strategies.
    The key chart proving Thompson Sampling works.
    """
    strategies = simulation_data.get("strategies", [])
    if not strategies:
        return _empty_chart("No simulation data")

    fig = go.Figure()

    colors = {
        "Thompson Sampling": "#4F8BF9",
        "Epsilon-Greedy (ε=0.1)": "#FF9900",
        "Uniform Random": "#FF5252",
    }

    for s in strategies:
        name = s["strategy"] if isinstance(s, dict) else s.strategy
        regret = s["cumulative_regret"] if isinstance(s, dict) else s.cumulative_regret

        fig.add_trace(go.Scatter(
            x=list(range(1, len(regret) + 1)),
            y=regret,
            mode="lines",
            name=name,
            line=dict(color=colors.get(name, "#888"), width=2),
        ))

    fig.update_layout(
        title="Cumulative Regret by Strategy",
        xaxis_title="Round (Day)",
        yaxis_title="Cumulative Regret ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
    )

    fig.add_annotation(
        text="Lower regret = better strategy",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=11, color="gray"),
    )

    return fig


def arm_selection_chart(simulation_data: dict) -> go.Figure:
    """
    Bar chart showing how often each bid level was selected per strategy.
    Thompson Sampling should concentrate on the best arm over time.
    """
    strategies = simulation_data.get("strategies", [])
    if not strategies:
        return _empty_chart("No simulation data")

    fig = go.Figure()

    colors = {
        "Thompson Sampling": "#4F8BF9",
        "Epsilon-Greedy (ε=0.1)": "#FF9900",
        "Uniform Random": "#FF5252",
    }

    for s in strategies:
        name = s["strategy"] if isinstance(s, dict) else s.strategy
        counts = s["arm_counts"] if isinstance(s, dict) else s.arm_counts

        sorted_arms = sorted(counts.keys(), key=lambda x: float(x))

        fig.add_trace(go.Bar(
            x=[f"{float(a):.2f}x" for a in sorted_arms],
            y=[counts[a] for a in sorted_arms],
            name=name,
            marker_color=colors.get(name, "#888"),
        ))

    fig.update_layout(
        title="Bid Level Selection Frequency",
        xaxis_title="Bid Multiplier",
        yaxis_title="Times Selected",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
    )

    oracle_arm = simulation_data.get("oracle", {}).get("best_arm")
    if oracle_arm:
        fig.add_annotation(
            text=f"Oracle best: {float(oracle_arm):.2f}x",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=11, color="green"),
            xanchor="right",
        )

    return fig


def beliefs_chart(beliefs: dict) -> go.Figure:
    """
    Visualize the bandit's posterior beliefs about each arm.
    Shows mean ± uncertainty for each bid level.
    """
    if not beliefs:
        return _empty_chart("No belief data")

    arms = sorted(beliefs.keys(), key=lambda x: float(x))
    means = [beliefs[a]["mean"] for a in arms]
    uncertainties = [beliefs[a]["uncertainty"] for a in arms]
    n_obs = [beliefs[a].get("n_obs", 0) for a in arms]

    labels = [f"{float(a):.2f}x" for a in arms]

    fig = go.Figure()

    # Uncertainty bars
    fig.add_trace(go.Bar(
        x=labels,
        y=means,
        error_y=dict(type="data", array=uncertainties, visible=True),
        marker_color=["#4F8BF9" if m == max(means) else "#B0C4DE" for m in means],
        text=[f"n={n}" for n in n_obs],
        textposition="outside",
    ))

    fig.update_layout(
        title="Bandit Posterior Beliefs (Mean ± Uncertainty)",
        xaxis_title="Bid Multiplier",
        yaxis_title="Expected Profit ($)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        showlegend=False,
    )

    return fig


def budget_allocation_chart(current: dict, recommended: dict) -> go.Figure:
    """
    Side-by-side comparison of current vs recommended budget allocation.
    """
    if not current or not recommended:
        return _empty_chart("No allocation data")

    # Shorten campaign names for display
    def shorten(name):
        return name.replace("Google - ", "G: ").replace("Meta - ", "M: ").replace("Amazon - ", "A: ")

    campaigns = list(current.keys())
    short_names = [shorten(c) for c in campaigns]
    current_vals = [current[c] for c in campaigns]
    recommended_vals = [recommended.get(c, 0) for c in campaigns]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=short_names,
        y=current_vals,
        name="Current",
        marker_color="#B0C4DE",
    ))

    fig.add_trace(go.Bar(
        x=short_names,
        y=recommended_vals,
        name="Recommended",
        marker_color="#4F8BF9",
    ))

    fig.update_layout(
        title="Budget Allocation: Current vs Recommended ($/day)",
        xaxis_title="Campaign",
        yaxis_title="Daily Budget ($)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=40),
        height=400,
    )

    return fig


def _empty_chart(message: str) -> go.Figure:
    """Return a blank chart with a centered message."""
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig

# ── Competitor charts ────────────────────────────────────────────────

def competitor_cluster_scatter(ads_df: pd.DataFrame) -> go.Figure:
    """2D PCA scatter plot coloured by cluster label."""
    fig = go.Figure()

    if "cluster_label" not in ads_df.columns:
        return _empty_chart("Run competitor analysis first")

    colors = px.colors.qualitative.Set2
    labels = ads_df["cluster_label"].unique()

    for i, label in enumerate(sorted(labels)):
        mask = ads_df["cluster_label"] == label
        subset = ads_df[mask]
        fig.add_trace(go.Scatter(
            x=subset["x"],
            y=subset["y"],
            mode="markers",
            name=label,
            marker=dict(size=10, color=colors[i % len(colors)], opacity=0.8),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br><br>"
                "<i>%{customdata[2]:.60s}...</i>"
                "<extra></extra>"
            ),
            customdata=subset[["advertiser_name", "category", "ad_copy"]].values,
        ))

    fig.update_layout(
        title="Competitor Ad Clusters (PCA Projection)",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        height=500,
        legend_title="Strategy",
        template="plotly_white",
    )
    return fig


def competitor_strategy_heatmap(strategy_matrix: pd.DataFrame) -> go.Figure:
    """Advertiser × strategy heatmap."""
    if strategy_matrix.empty:
        return _empty_chart("No strategy data")

    fig = go.Figure(go.Heatmap(
        z=strategy_matrix.values,
        x=strategy_matrix.columns.tolist(),
        y=strategy_matrix.index.tolist(),
        colorscale="Blues",
        text=strategy_matrix.values,
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> → %{x}<br>Ads: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Advertiser Strategy Mix",
        xaxis_title="Strategy Cluster",
        yaxis_title="Advertiser",
        height=max(350, len(strategy_matrix) * 45),
        template="plotly_white",
    )
    return fig


def competitor_cluster_bars(clusters: list[dict]) -> go.Figure:
    """Bar chart of ads per cluster with top terms annotation."""
    if not clusters:
        return _empty_chart("No clusters")

    labels = [c["label"] for c in clusters]
    counts = [c["n_ads"] for c in clusters]
    terms = [", ".join(c["top_terms"][:5]) for c in clusters]

    colors = px.colors.qualitative.Set2

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker_color=[colors[i % len(colors)] for i in range(len(labels))],
        hovertemplate="<b>%{x}</b><br>Ads: %{y}<br>Top terms: %{customdata}<extra></extra>",
        customdata=terms,
    ))
    fig.update_layout(
        title="Ads per Strategy Cluster",
        xaxis_title="Strategy",
        yaxis_title="Number of Ads",
        height=400,
        template="plotly_white",
    )
    return fig


def competitor_platform_breakdown(ads_df: pd.DataFrame) -> go.Figure:
    """Stacked bar: platform distribution per advertiser."""
    if ads_df.empty:
        return _empty_chart("No competitor data")

    ct = ads_df.groupby(["advertiser_name", "platform"]).size().unstack(fill_value=0)
    platform_colors = {"google": "#4285F4", "meta": "#1877F2", "amazon": "#FF9900"}

    fig = go.Figure()
    for platform in ct.columns:
        fig.add_trace(go.Bar(
            x=ct.index,
            y=ct[platform],
            name=platform.title(),
            marker_color=platform_colors.get(platform, "#999"),
        ))

    fig.update_layout(
        barmode="stack",
        title="Platform Mix by Advertiser",
        xaxis_title="Advertiser",
        yaxis_title="Number of Ads",
        height=400,
        template="plotly_white",
    )
    return fig