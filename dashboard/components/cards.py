# dashboard/components/cards.py

"""
Reusable UI card components for the dashboard.
"""

import streamlit as st


# Map action types to emoji and display name
ACTION_DISPLAY = {
    "bid_adjustment": ("⬆️", "Bid Adjustment"),
    "budget_reallocation": ("💰", "Budget Reallocation"),
    "creative_rotation": ("🔄", "Creative Rotation"),
    "pause_campaign": ("⏸️", "Pause Campaign"),
}

# Map confidence to Streamlit color syntax
CONFIDENCE_COLORS = {
    "high": ":green[HIGH]",
    "medium": ":orange[MEDIUM]",
    "low": ":red[LOW]",
}


def recommendation_card(
    action_type: str,
    title: str,
    reasoning: str,
    confidence: str,
    campaign_name: str | None = None,
    card_key: str = "0",
) -> None:
    """
    Render a single recommendation card.
    
    Args:
        action_type: One of 'bid_adjustment', 'budget_reallocation', etc.
        title: Short action description
        reasoning: Plain English explanation
        confidence: 'high', 'medium', or 'low'
        campaign_name: Optional campaign name
        card_key: Unique key for Streamlit buttons
    """
    emoji, type_label = ACTION_DISPLAY.get(action_type, ("📋", action_type))
    conf_display = CONFIDENCE_COLORS.get(confidence, confidence)

    with st.container(border=True):
        # Header
        st.markdown(f"#### {emoji} {title}")

        if campaign_name:
            st.caption(f"📂 {campaign_name}")

        # Reasoning
        st.markdown(f"*{reasoning}*")

        # Footer: confidence + actions
        col_conf, col_apply, col_dismiss = st.columns([2, 1, 1])

        with col_conf:
            st.markdown(f"**Confidence:** {conf_display}")
        with col_apply:
            st.button("✅ Apply", key=f"apply_{card_key}", use_container_width=True)
        with col_dismiss:
            st.button("❌ Dismiss", key=f"dismiss_{card_key}", use_container_width=True)


def metric_card_row(kpis: dict, deltas: dict) -> None:
    """
    Render the top-level KPI metric cards.
    
    Args:
        kpis: Dict from queries.get_kpis()
        deltas: Dict from queries.get_kpis_comparison()["deltas"]
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Spend",
            value=f"${kpis['total_spend']:,.0f}",
            delta=deltas.get("spend"),
        )
    with col2:
        st.metric(
            label="ROAS",
            value=f"{kpis['overall_roas']:.2f}x",
            delta=deltas.get("roas"),
        )
    with col3:
        st.metric(
            label="Conversions",
            value=f"{kpis['total_conversions']:,}",
            delta=deltas.get("conversions"),
        )
    with col4:
        st.metric(
            label="Active Campaigns",
            value=kpis["active_campaigns"],
            delta=deltas.get("campaigns"),
        )


def no_data_message() -> None:
    """Show a friendly message when no data is loaded yet."""
    st.markdown("---")
    st.markdown(
        """
        ### 👋 Welcome to Ad-Work!
        
        No campaign data loaded yet. You have two options:
        
        **Option 1: Load sample data** (recommended for first time)
        ```bash
        uv run python scripts/seed_demo.py
        ```
        Or on Windows:
        ```
        .venv\\Scripts\\python scripts\\seed_demo.py
        ```
        Then refresh this page.
        
        **Option 2: Upload your own data**  
        Go to **📤 Upload Data** in the sidebar and upload a CSV export 
        from Google Ads, Meta Ads, or Amazon Ads.
        """
    )