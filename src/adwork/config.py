# src/adwork/config.py

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings. Reads from .env file automatically.
    """
    
    # LLM
    llm_provider: str = Field(default="groq", description="LLM provider: 'groq' or 'openai'")
    groq_api_key: str = Field(default="", description="Groq API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    
    # Database
    duckdb_path: str = Field(default="data/adwork.duckdb", description="Path to DuckDB database file")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Model defaults
    forecast_horizon_days: int = Field(default=14, description="How many days ahead to forecast")
    optimization_interval_hours: int = Field(default=6, description="How often the optimization loop runs")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
    
    @property
    def duckdb_full_path(self) -> Path:
        """Ensure the database directory exists and return full path."""
        path = Path(self.duckdb_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Singleton — import this everywhere
settings = Settings()