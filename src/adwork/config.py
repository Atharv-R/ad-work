# src/adwork/config.py

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


def _load_streamlit_secrets():
    """
    Bridge Streamlit Cloud secrets into environment variables.
    
    Streamlit Cloud stores secrets in st.secrets (a TOML file).
    Pydantic-settings reads from environment variables.
    This function copies secrets → env vars so everything works
    in both local dev (.env file) and cloud (Streamlit secrets).
    """
    try:
        import streamlit as st
        
        for key, value in st.secrets.items():
            # Only set if not already in environment
            # (local .env takes priority during development)
            if key not in os.environ:
                os.environ[key] = str(value)
    except Exception:
        # Not running in Streamlit, or no secrets configured
        # That's fine — pydantic-settings will read from .env
        pass


# Call this before Settings() reads environment
_load_streamlit_secrets()


class Settings(BaseSettings):
    """
    Application settings. 
    
    Reads from (in priority order):
    1. Environment variables (set explicitly)
    2. Streamlit Cloud secrets (bridged to env vars above)
    3. .env file (local development)
    4. Default values
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
    "env_file_encoding": "utf-8",
    "extra": "ignore",          # Don't crash on extra env vars
    }
    
    @property
    def duckdb_full_path(self) -> Path:
        """Ensure the database directory exists and return full path."""
        path = Path(self.duckdb_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Singleton — import this everywhere
try:
    settings = Settings()
except Exception:
    # .env file might not exist (Streamlit Cloud)
    # Secrets were already loaded into env vars by _load_streamlit_secrets()
    settings = Settings(_env_file=None)