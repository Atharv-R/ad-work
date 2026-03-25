# src/adwork/agent/prompts.py

"""
LLM Prompt Templates
====================
Structured prompts that produce parseable JSON output.
Designed to work reliably with both Llama 3.3 70B (Groq) and GPT-4o-mini.
"""

SYSTEM_PROMPT = """You are Ad-Work, an AI advertising optimization analyst.
You analyze campaign performance data, demand forecasts, and optimization signals
to generate actionable recommendations with plain-English reasoning.

Rules:
- Always cite specific numbers (ROAS, spend, conversion counts)
- Name specific campaigns in your analysis
- Explain your reasoning step by step
- Confidence: HIGH = strong signal + sufficient data, MEDIUM = moderate signal, LOW = weak/uncertain signal
- Always respond with valid JSON only. No markdown, no text outside JSON."""


PERFORMANCE_ANALYSIS_PROMPT = """Analyze these advertising campaigns and assess portfolio health.

{campaign_summary}

Respond with this exact JSON structure:
{{
    "portfolio_health": "critical" or "normal",
    "analysis": "2-3 sentence overall assessment",
    "top_performers": [
        {{"campaign": "name", "reason": "why it's performing well"}}
    ],
    "underperformers": [
        {{"campaign": "name", "reason": "why it's underperforming", "severity": "high" or "medium"}}
    ],
    "key_trends": [
        {{"campaign": "name", "direction": "improving" or "declining" or "stable", "detail": "what's happening"}}
    ]
}}

Mark portfolio_health as "critical" ONLY if any campaign has ROAS below 1.0x AND is declining. Otherwise mark "normal"."""


FORECAST_ANALYSIS_PROMPT = """Review these demand signals and forecasts alongside campaign performance.

Campaign Performance:
{campaign_summary}

Forecast & Trend Data:
{forecast_summary}

Respond with this exact JSON structure:
{{
    "forecast_insights": "2-3 sentence summary of demand outlook",
    "opportunities": [
        {{"campaign": "name", "opportunity": "what to do and why based on forecast"}}
    ],
    "risks": [
        {{"campaign": "name", "risk": "what could go wrong"}}
    ]
}}"""


SYNTHESIS_PROMPT = """You are generating the final daily optimization recommendations.

Performance Analysis:
{performance_analysis}

Forecast Insights:
{forecast_insights}

Bid Optimization Results (from Thompson Sampling):
{bid_recommendations}

Budget Allocation Results:
{budget_shifts}

Combine all signals into a prioritized set of recommendations and a daily executive summary.
Each recommendation should have clear reasoning that references the data.

Respond with this exact JSON structure:
{{
    "recommendations": [
        {{
            "campaign_id": "id",
            "campaign_name": "name",
            "action_type": "bid_adjustment" or "budget_reallocation" or "creative_rotation" or "pause_campaign",
            "action_summary": "short action description",
            "reasoning": "2-3 sentences explaining WHY, citing specific numbers",
            "confidence": "high" or "medium" or "low",
            "priority": 1
        }}
    ],
    "daily_summary": "A 2-3 paragraph executive brief summarizing the portfolio status, key actions taken, and outlook. Write as if briefing a marketing director."
}}

Prioritize by impact. Include at most 8 recommendations. Number priorities from 1 (highest) to N."""


SYNTHESIS_CRITICAL_PROMPT = """URGENT: Portfolio has critical issues requiring immediate attention.

Performance Analysis:
{performance_analysis}

Bid Optimization Results (from Thompson Sampling):
{bid_recommendations}

Budget Allocation Results:
{budget_shifts}

Generate URGENT recommendations focused on stopping losses and reallocating budget from failing campaigns.

Respond with this exact JSON structure:
{{
    "recommendations": [
        {{
            "campaign_id": "id",
            "campaign_name": "name",
            "action_type": "bid_adjustment" or "budget_reallocation" or "creative_rotation" or "pause_campaign",
            "action_summary": "short urgent action",
            "reasoning": "why this is urgent, cite numbers",
            "confidence": "high" or "medium" or "low",
            "priority": 1
        }}
    ],
    "daily_summary": "URGENT executive brief. Lead with the problems, then the actions being taken."
}}"""