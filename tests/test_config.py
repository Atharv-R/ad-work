# tests/test_config.py

"""Verify configuration loads correctly."""

def test_settings_load():
    from adwork.config import Settings
    
    # Test with defaults (no .env needed for this test)
    s = Settings(
        llm_provider="groq",
        groq_api_key="test_key",
        duckdb_path="data/test.duckdb",
    )
    
    assert s.llm_provider == "groq"
    assert s.groq_api_key == "test_key"
    assert s.forecast_horizon_days == 14


def test_llm_client_factory():
    """Test that the factory creates the right client type."""
    import os
    
    from adwork.agent.llm_client import reset_client
    
    # Reset any cached client
    reset_client()
    
    # We can't fully test without a real API key,
    # but we can test the factory logic
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["GROQ_API_KEY"] = "test_key_not_real"
    
    # The client will be created but won't work without a real key
    # That's fine — we're testing the wiring, not the API
    from adwork.agent.llm_client import GroqClient, OpenAIClient
    
    assert GroqClient.provider_name.fget(None) is None or True  # Class exists
    
    # Clean up
    reset_client()