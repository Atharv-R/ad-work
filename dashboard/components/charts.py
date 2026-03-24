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