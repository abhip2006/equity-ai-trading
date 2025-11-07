"""
LLM Abstraction Layer

Unified interface for multiple LLM providers in the trading battle system.

Usage:
    from active_trader_llm.llm import get_llm_client, LLMMessage

    # Create a client
    client = get_llm_client(
        provider="openai",
        model="gpt-4o",
        api_key="your-key-here"
    )

    # Generate completion
    messages = [
        LLMMessage(role="system", content="You are a trading assistant."),
        LLMMessage(role="user", content="Analyze AAPL...")
    ]
    response = client.generate(messages, temperature=0.3, max_tokens=1000)
    print(response.content)

    # Generate structured JSON
    response = client.generate_structured(
        messages,
        response_schema={},
        temperature=0.3
    )
    import json
    trade_plan = json.loads(response.content)

Supported Providers:
    - openai: GPT-4, GPT-4-Turbo, GPT-3.5-Turbo
    - anthropic: Claude Sonnet, Claude Opus, Claude Haiku
    - xai: Grok models
    - deepseek: DeepSeek models
    - google: Gemini Pro, Gemini Flash
    - openrouter: Qwen, LLaMA, Mistral, and others
"""

# Import core classes
from .llm_client import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    LLMClientFactory,
    get_llm_client
)

# Import all providers (triggers registration)
import active_trader_llm.llm.providers

__all__ = [
    # Core classes
    'LLMClient',
    'LLMMessage',
    'LLMResponse',

    # Exceptions
    'LLMError',
    'RateLimitError',
    'AuthenticationError',
    'InvalidRequestError',

    # Factory
    'LLMClientFactory',
    'get_llm_client'
]

__version__ = '1.0.0'
