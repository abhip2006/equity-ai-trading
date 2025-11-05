"""
LLM Provider Adapters

Auto-imports all provider adapters to register them with the factory.
"""

# Import all adapters to trigger registration
from active_trader_llm.llm.providers.openai_adapter import OpenAIAdapter
from active_trader_llm.llm.providers.anthropic_adapter import AnthropicAdapter
from active_trader_llm.llm.providers.xai_adapter import XAIAdapter
from active_trader_llm.llm.providers.deepseek_adapter import DeepSeekAdapter
from active_trader_llm.llm.providers.openrouter_adapter import OpenRouterAdapter

# Google adapter is optional (requires google-generativeai package)
try:
    from active_trader_llm.llm.providers.google_adapter import GoogleAdapter
except ImportError:
    GoogleAdapter = None

__all__ = [
    'OpenAIAdapter',
    'AnthropicAdapter',
    'XAIAdapter',
    'DeepSeekAdapter',
    'GoogleAdapter',
    'OpenRouterAdapter'
]
