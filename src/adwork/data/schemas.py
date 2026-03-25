# src/adwork/data/schemas.py

"""
Data schemas for Ad-Work.

Every piece of data in the system flows through these models.
They validate types, enforce constraints, and document the expected format.
"""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field, field_validator

# --- Enums ---

class Platform(str, Enum):
    """Supported advertising platforms."""
    GOOGLE = "google"
    META = "meta"
    AMAZON = "amazon"
    UNKNOWN = "unknown"


class CampaignStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class ActionType(str, Enum):
    BID_ADJUSTMENT = "bid_adjustment"
    BUDGET_REALLOCATION = "budget_reallocation"
    CREATIVE_ROTATION = "creative_rotation"
    PAUSE_CAMPAIGN = "pause_campaign"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- Core Models ---

class Campaign(BaseModel):
    """A single advertising campaign."""
    campaign_id: str
    campaign_name: str
    platform: Platform
    status: CampaignStatus = CampaignStatus.ACTIVE
    daily_budget: float = Field(ge=0, description="Daily budget in USD")

    @field_validator("campaign_id")
    @classmethod
    def campaign_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("campaign_id cannot be empty")
        return v.strip()


class DailyMetrics(BaseModel):
    """One day of performance data for one campaign."""
    campaign_id: str
    date: date
    impressions: int = Field(ge=0, default=0)
    clicks: int = Field(ge=0, default=0)
    conversions: int = Field(ge=0, default=0)
    spend: float = Field(ge=0, default=0.0)
    revenue: float = Field(ge=0, default=0.0)

    @computed_field
    @property
    def ctr(self) -> float:
        """Click-through rate."""
        if self.impressions == 0:
            return 0.0
        return round(self.clicks / self.impressions, 6)

    @computed_field
    @property
    def cpc(self) -> float:
        """Cost per click."""
        if self.clicks == 0:
            return 0.0
        return round(self.spend / self.clicks, 4)

    @computed_field
    @property
    def roas(self) -> float:
        """Return on ad spend."""
        if self.spend == 0:
            return 0.0
        return round(self.revenue / self.spend, 4)

    @computed_field
    @property
    def conversion_rate(self) -> float:
        """Conversion rate from clicks."""
        if self.clicks == 0:
            return 0.0
        return round(self.conversions / self.clicks, 6)


class Recommendation(BaseModel):
    """An optimization recommendation from the agent."""
    campaign_id: str | None = None
    action_type: ActionType
    action_detail: str = Field(description="JSON string with specific action parameters")
    reasoning: str = Field(description="Plain English explanation")
    confidence: Confidence
    status: str = "pending"
    llm_provider: str | None = None
    llm_model: str | None = None
    created_at: datetime | None = None


class SearchTrend(BaseModel):
    """Google Trends data point."""
    keyword: str
    date: date
    interest: int = Field(ge=0, le=100, description="Google Trends interest score 0-100")


# --- CSV Column Mappings ---
# These map platform-specific column names to our internal schema.
# Used by the ingestion module to normalize uploads.

GOOGLE_ADS_COLUMNS = {
    "Campaign": "campaign_name",
    "Campaign ID": "campaign_id",
    "Day": "date",
    "Impressions": "impressions",
    "Clicks": "clicks",
    "Cost": "spend",
    "Conversions": "conversions",
    "Conv. value": "revenue",
    "Avg. CPC": "_cpc",           # Computed, but kept for validation
    "CTR": "_ctr",                 # Computed, but kept for validation
}

META_ADS_COLUMNS = {
    "Campaign name": "campaign_name",
    "Campaign ID": "campaign_id",
    "Day": "date",
    "Impressions": "impressions",
    "Link clicks": "clicks",
    "Amount spent (USD)": "spend",
    "Purchases": "conversions",
    "Purchase conversion value": "revenue",
}

AMAZON_ADS_COLUMNS = {
    "Campaign Name": "campaign_name",
    "Campaign ID": "campaign_id",
    "Date": "date",
    "Impressions": "impressions",
    "Clicks": "clicks",
    "Spend": "spend",
    "Orders": "conversions",
    "Sales": "revenue",
}

# Internal format (what our sample data and DuckDB use)
INTERNAL_COLUMNS = {
    "campaign_id", "campaign_name", "platform", "date",
    "impressions", "clicks", "conversions", "spend", "revenue",
}


def detect_platform_from_columns(columns: list[str]) -> Platform:
    """
    Auto-detect which ad platform a CSV came from based on column names.
    
    Args:
        columns: List of column names from the CSV header
        
    Returns:
        Detected Platform enum value
    """
    col_set = set(columns)

    # Check for platform-specific marker columns
    if "Amount spent (USD)" in col_set or "Link clicks" in col_set:
        return Platform.META
    if "Orders" in col_set and "ACOS" in col_set or "Campaign Name" in col_set and "Spend" in col_set and "Sales" in col_set:
        return Platform.AMAZON
    if "Conv. value" in col_set or "Avg. CPC" in col_set:
        return Platform.GOOGLE
    
    # Check if it's already in our internal format
    if "platform" in col_set and "campaign_id" in col_set:
        return Platform.UNKNOWN  # Internal format, platform is in the data
    
    return Platform.UNKNOWN