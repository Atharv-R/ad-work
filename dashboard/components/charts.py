# dashboard/components/charts.py

"""
Plotly chart builders for the Ad-Work dashboard.
Each function returns a Plotly figure ready for st.plotly_chart().
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


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