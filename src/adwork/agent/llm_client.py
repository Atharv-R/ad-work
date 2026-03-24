# src/adwork/agent/llm_client.py

"""
LLM Client Abstraction Layer
=============================
Swap between Groq (free) and OpenAI (paid) with an environment variable.

Usage:
    from adwork.agent.llm_client import get_llm_client
    
    llm = get_llm_client()
    response = llm.complete(messages=[{"role": "user", "content": "Hello"}])
    
Design decision: This abstraction means every other module in the codebase
is LLM-agnostic. Switch providers by changing LLM_PROVIDER in .env.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import json

from adwork.config import settings


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: dict = {}


class LLMClient(ABC):
    """Abstract base class for LLM providers."""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
    
    @abstractmethod
    def _raw_complete(self, messages: list[dict], temperature: float = 0.1) -> LLMResponse:
        """Provider-specific completion call."""
        ...
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"LLM call failed, retrying (attempt {retry_state.attempt_number})..."
        ),
    )
    def complete(self, messages: list[dict], temperature: float = 0.1) -> LLMResponse:
        """
        Send messages to LLM with automatic retry logic.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: 0.0 = deterministic, 1.0 = creative
            
        Returns:
            LLMResponse with content, model info, and usage stats
        """
        logger.debug(f"LLM call to {self.provider_name}/{self.model_name} | {len(messages)} messages")
        response = self._raw_complete(messages, temperature)
        logger.debug(f"LLM response: {len(response.content)} chars | usage: {response.usage}")
        return response
    
    def complete_json(self, messages: list[dict], temperature: float = 0.1) -> dict:
        """
        Get a JSON response from the LLM.
        Adds instruction to return JSON and parses the result.
        """
        # Inject JSON instruction into the last user message
        enhanced_messages = messages.copy()
        last_msg = enhanced_messages[-1].copy()
        last_msg["content"] += "\n\nRespond with valid JSON only. No markdown, no explanation outside the JSON."
        enhanced_messages[-1] = last_msg
        
        response = self.complete(enhanced_messages, temperature)
        
        # Extract JSON from response (handle markdown code blocks)
        content = response.content.strip()
        if content.startswith("```"):
            # Remove markdown code block wrapping
            lines = content.split("\n")
            # Drop first line (```json) and last line (```)
            content = "\n".join(lines[1:-1])
        
        return json.loads(content)
    
    def complete_pydantic(
        self, 
        messages: list[dict], 
        response_model: type[BaseModel], 
        temperature: float = 0.1,
    ) -> BaseModel:
        """
        Get a structured response parsed into a Pydantic model.
        
        Args:
            messages: Chat messages
            response_model: Pydantic model class to parse into
            temperature: LLM temperature
            
        Returns:
            Instance of response_model
        """
        # Add schema to prompt so the LLM knows the expected structure
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        
        enhanced_messages = messages.copy()
        last_msg = enhanced_messages[-1].copy()
        last_msg["content"] += (
            f"\n\nRespond with valid JSON matching this schema:\n{schema_str}"
            f"\n\nReturn ONLY the JSON object. No markdown, no explanation."
        )
        enhanced_messages[-1] = last_msg
        
        data = self.complete_json(enhanced_messages, temperature)
        return response_model.model_validate(data)


class GroqClient(LLMClient):
    """Groq API client (free tier, Llama 3.3 70B)."""
    
    def __init__(self):
        from groq import Groq
        
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        self.client = Groq(api_key=settings.groq_api_key)
        self._model = "llama-3.3-70b-versatile"
        logger.info(f"Initialized Groq client with model {self._model}")
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def _raw_complete(self, messages: list[dict], temperature: float = 0.1) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._model,
            provider="groq",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )


class OpenAIClient(LLMClient):
    """OpenAI API client (paid, GPT-4o-mini)."""
    
    def __init__(self):
        from openai import OpenAI
        
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        self.client = OpenAI(api_key=settings.openai_api_key)
        self._model = "gpt-4o-mini"
        logger.info(f"Initialized OpenAI client with model {self._model}")
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def _raw_complete(self, messages: list[dict], temperature: float = 0.1) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )


# --- Factory ---

_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """
    Factory function — returns the configured LLM client.
    
    Reads LLM_PROVIDER from environment/settings.
    Caches the instance (singleton pattern).
    
    Usage:
        llm = get_llm_client()
        response = llm.complete([{"role": "user", "content": "Hello"}])
    """
    global _client_instance
    
    if _client_instance is None:
        provider = settings.llm_provider.lower()
        
        if provider == "groq":
            _client_instance = GroqClient()
        elif provider == "openai":
            _client_instance = OpenAIClient()
        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. Use 'groq' or 'openai'."
            )
    
    return _client_instance


def reset_client() -> None:
    """Reset the cached client. Useful for testing or provider switching."""
    global _client_instance
    _client_instance = None