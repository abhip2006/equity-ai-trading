"""
Base LLM Client Interface

Abstract base class and factory for LLM providers.
Provides unified interface for all LLM interactions in the trading battle system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """Unified message format across all providers"""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Unified response format across all providers"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    raw_response: Optional[Any] = None  # Original provider response


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    def __init__(self, message: str, provider: str, model: str):
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}/{model}] {message}")


class RateLimitError(LLMError):
    """Rate limit exceeded"""
    pass


class AuthenticationError(LLMError):
    """Authentication failed"""
    pass


class InvalidRequestError(LLMError):
    """Invalid request parameters"""
    pass


class LLMClient(ABC):
    """
    Abstract base class for all LLM providers.

    Each provider adapter must implement this interface.
    """

    def __init__(self, model: str, api_key: str, **kwargs):
        """
        Initialize LLM client

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4")
            api_key: API key for authentication
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion from messages

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: On provider errors
            RateLimitError: When rate limited
            AuthenticationError: On auth failure
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        messages: List[LLMMessage],
        response_schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a structured JSON response

        Args:
            messages: List of conversation messages
            response_schema: JSON schema for response validation
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with JSON content

        Raises:
            LLMError: On provider errors or JSON parsing failure
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g., 'openai', 'anthropic')"""
        pass

    def _normalize_temperature(self, temperature: float) -> float:
        """Ensure temperature is in valid range [0.0, 1.0]"""
        return max(0.0, min(1.0, temperature))


class LLMClientFactory:
    """
    Factory for creating LLM client instances.

    Automatically selects the appropriate provider adapter based on provider name.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register_provider(cls, provider_name: str, client_class: type):
        """Register a provider adapter"""
        cls._registry[provider_name.lower()] = client_class
        logger.debug(f"Registered LLM provider: {provider_name}")

    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        api_key: str,
        **kwargs
    ) -> LLMClient:
        """
        Create an LLM client instance

        Args:
            provider: Provider name (openai, anthropic, xai, deepseek, google, openrouter)
            model: Model identifier
            api_key: API key for authentication
            **kwargs: Provider-specific configuration

        Returns:
            LLMClient instance

        Raises:
            ValueError: If provider not supported
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Available providers: {available}"
            )

        client_class = cls._registry[provider_lower]
        logger.info(f"Creating LLM client: {provider}/{model}")

        return client_class(model=model, api_key=api_key, **kwargs)


def get_llm_client(
    provider: str,
    model: str,
    api_key: str,
    **kwargs
) -> LLMClient:
    """
    Convenience function to create an LLM client

    Args:
        provider: Provider name (openai, anthropic, xai, deepseek, google, openrouter)
        model: Model identifier
        api_key: API key
        **kwargs: Provider-specific configuration

    Returns:
        LLMClient instance
    """
    return LLMClientFactory.create(provider, model, api_key, **kwargs)
