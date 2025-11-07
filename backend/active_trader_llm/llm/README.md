# LLM Abstraction Layer

A unified interface for multiple LLM providers in the trading battle system. Allows seamless switching between OpenAI, Anthropic, xAI, DeepSeek, Google, and OpenRouter models.

## Features

- **Unified Interface**: Single API for all providers
- **Provider Flexibility**: Switch providers without changing code
- **Error Handling**: Consistent error handling across providers
- **Retry Logic**: Automatic retries with exponential backoff
- **Token Tracking**: Unified usage tracking across all providers
- **Structured Outputs**: JSON generation support for all providers

## Supported Providers

| Provider | Models | API Type | Notes |
|----------|--------|----------|-------|
| **OpenAI** | GPT-4, GPT-4-Turbo, GPT-3.5-Turbo | Native SDK | Best JSON mode support |
| **Anthropic** | Claude Sonnet, Opus, Haiku | Native SDK | Excellent reasoning |
| **xAI** | Grok models | HTTP (OpenAI-compatible) | Real-time data access |
| **DeepSeek** | DeepSeek models | HTTP (OpenAI-compatible) | Cost-efficient |
| **Google** | Gemini Pro, Gemini Flash | Native SDK | Fast and multimodal |
| **OpenRouter** | Qwen, LLaMA, Mistral, etc. | HTTP (OpenAI-compatible) | Access to many models |

## Installation

```bash
# Core dependencies (already in requirements.txt)
pip install openai anthropic requests

# Optional: For Google Gemini
pip install google-generativeai
```

## Quick Start

```python
from active_trader_llm.llm import get_llm_client, LLMMessage

# Create a client
client = get_llm_client(
    provider="openai",
    model="gpt-4o",
    api_key="sk-..."
)

# Generate completion
messages = [
    LLMMessage(role="system", content="You are a trading assistant."),
    LLMMessage(role="user", content="Analyze AAPL stock...")
]

response = client.generate(messages, temperature=0.3, max_tokens=1000)
print(response.content)
print(f"Tokens used: {response.usage}")
```

## API Reference

### Core Classes

#### `LLMClient` (Abstract Base)
Base class for all provider adapters.

**Methods:**
- `generate(messages, temperature, max_tokens, **kwargs) -> LLMResponse`
  - Generate a text completion
- `generate_structured(messages, response_schema, temperature, max_tokens, **kwargs) -> LLMResponse`
  - Generate a structured JSON response

**Properties:**
- `provider_name: str` - Provider identifier

#### `LLMMessage`
Unified message format.

```python
LLMMessage(role="system|user|assistant", content="text")
```

#### `LLMResponse`
Unified response format.

**Attributes:**
- `content: str` - Generated text or JSON
- `model: str` - Model used
- `provider: str` - Provider name
- `usage: Dict[str, int]` - Token usage (prompt_tokens, completion_tokens, total_tokens)
- `raw_response: Any` - Original provider response

### Factory Functions

#### `get_llm_client(provider, model, api_key, **kwargs)`
Create an LLM client instance.

**Arguments:**
- `provider: str` - Provider name (openai, anthropic, xai, deepseek, google, openrouter)
- `model: str` - Model identifier
- `api_key: str` - API key for authentication
- `**kwargs` - Provider-specific configuration (timeout, max_retries, etc.)

**Returns:** `LLMClient` instance

**Example:**
```python
client = get_llm_client(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=120.0,
    max_retries=5
)
```

### Exception Classes

- `LLMError` - Base exception for all LLM errors
- `RateLimitError` - Rate limit exceeded
- `AuthenticationError` - Authentication failed
- `InvalidRequestError` - Invalid request parameters

All exceptions include provider and model information.

## Usage Examples

### Basic Trading Decision

```python
from active_trader_llm.llm import get_llm_client, LLMMessage

client = get_llm_client("openai", "gpt-3.5-turbo", api_key="...")

messages = [
    LLMMessage(role="system", content="You are a stock trader."),
    LLMMessage(role="user", content="""
Analyze AAPL:
- Price: $175.50
- RSI: 65
- Volume: Above average

Recommendation?
""")
]

response = client.generate(messages, temperature=0.3)
print(response.content)
```

### Structured Trade Plan

```python
import json

system_prompt = """You are a trader. Output JSON:
{
    "action": "buy|sell|hold",
    "confidence": 0.0-1.0,
    "rationale": "text"
}
"""

messages = [
    LLMMessage(role="system", content=system_prompt),
    LLMMessage(role="user", content="Analyze TSLA...")
]

response = client.generate_structured(messages, response_schema={})
trade_plan = json.loads(response.content)
print(f"Action: {trade_plan['action']}, Confidence: {trade_plan['confidence']}")
```

### Multi-Provider Comparison

```python
providers = [
    ("openai", "gpt-4o", os.getenv("OPENAI_API_KEY")),
    ("anthropic", "claude-3-5-sonnet-20241022", os.getenv("ANTHROPIC_API_KEY")),
    ("xai", "grok-beta", os.getenv("XAI_API_KEY"))
]

for provider, model, api_key in providers:
    client = get_llm_client(provider, model, api_key)
    response = client.generate(messages, temperature=0.5)
    print(f"{provider}: {response.content}")
```

### Error Handling

```python
from active_trader_llm.llm import (
    get_llm_client,
    RateLimitError,
    AuthenticationError,
    LLMError
)

try:
    client = get_llm_client("openai", "gpt-4o", api_key="...")
    response = client.generate(messages)
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Implement backoff logic
except AuthenticationError as e:
    print(f"Auth failed: {e}")
    # Check API key
except LLMError as e:
    print(f"LLM error: {e}")
    # Fallback to another provider
```

## Provider-Specific Notes

### OpenAI
- Supports native JSON mode via `response_format`
- Best for structured outputs
- Models: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

### Anthropic
- System message handled separately
- Excellent for complex reasoning
- Models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`

### xAI (Grok)
- OpenAI-compatible HTTP API
- Real-time data access (web search)
- Endpoint: `https://api.x.ai/v1`

### DeepSeek
- OpenAI-compatible HTTP API
- Cost-efficient option
- Endpoint: `https://api.deepseek.com/v1`

### Google (Gemini)
- Requires `google-generativeai` package
- Fast and supports multimodal inputs
- Models: `gemini-pro`, `gemini-flash`

### OpenRouter
- Access to many models (Qwen, LLaMA, Mistral, etc.)
- OpenAI-compatible HTTP API
- Models: `qwen/qwen-2.5-72b-instruct`, `meta-llama/llama-3.1-70b-instruct`, etc.

## Testing

Run the test suite to validate all configured providers:

```bash
# Test all providers
python -m active_trader_llm.llm.test_llm_abstraction

# Run usage examples
python -m active_trader_llm.llm.example_usage
```

## Integration with Trading System

### Current Pattern (trader_agent.py)

```python
# OLD: Direct OpenAI usage
from openai import OpenAI
client = OpenAI(api_key=api_key)
response = client.chat.completions.create(model="gpt-4o", messages=...)
```

### New Pattern (with abstraction)

```python
# NEW: Provider-agnostic
from active_trader_llm.llm import get_llm_client, LLMMessage

client = get_llm_client(
    provider=config['llm']['provider'],  # From config
    model=config['llm']['model'],
    api_key=config['llm']['api_key']
)

messages = [LLMMessage(role="system", content=system_prompt), ...]
response = client.generate(messages, temperature=0.3)
```

### Battle System Integration

The battle system can now easily compare different models:

```python
# Battle configuration
traders = [
    {"name": "GPT-4", "provider": "openai", "model": "gpt-4o"},
    {"name": "Claude", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    {"name": "Grok", "provider": "xai", "model": "grok-beta"},
    {"name": "Qwen", "provider": "openrouter", "model": "qwen/qwen-2.5-72b-instruct"}
]

# Each trader gets same data, makes decision
for trader in traders:
    client = get_llm_client(
        provider=trader['provider'],
        model=trader['model'],
        api_key=get_api_key(trader['provider'])
    )

    response = client.generate_structured(messages, response_schema={})
    trade_plan = json.loads(response.content)

    # Execute trade, track performance
    execute_trade(trader['name'], trade_plan)
```

## Configuration

Add to `config.yaml`:

```yaml
# LLM configuration
llm:
  provider: openai  # openai | anthropic | xai | deepseek | google | openrouter
  model: gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}  # Use environment variable
  temperature: 0.3
  max_tokens: 2000
  timeout: 60.0
  max_retries: 3

# Battle system: Multiple models competing
battle:
  enabled: true
  traders:
    - name: GPT-4
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}

    - name: Claude
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      api_key: ${ANTHROPIC_API_KEY}

    - name: Grok
      provider: xai
      model: grok-beta
      api_key: ${XAI_API_KEY}
```

## Performance Considerations

- **OpenAI**: Fastest API, best JSON support, moderate cost
- **Anthropic**: Slower but excellent reasoning, higher cost
- **xAI**: Variable speed, access to real-time data
- **DeepSeek**: Slowest but very cheap
- **Google**: Very fast, good for high-throughput
- **OpenRouter**: Speed varies by underlying model

## Cost Optimization

```python
# Use cheaper models for pre-filtering
cheap_client = get_llm_client("openai", "gpt-3.5-turbo", api_key)

# Use expensive models for final decisions
premium_client = get_llm_client("anthropic", "claude-3-5-sonnet-20241022", api_key)

# Pre-filter with cheap model
response = cheap_client.generate(pre_filter_messages)
if should_analyze_deeper(response):
    # Deep analysis with premium model
    response = premium_client.generate(detailed_messages)
```

## Roadmap

- [ ] Add support for streaming responses
- [ ] Implement automatic fallback (if primary provider fails, try secondary)
- [ ] Add caching layer to reduce API calls
- [ ] Support for function calling (OpenAI tools)
- [ ] Add more providers (Mistral, Cohere, etc.)

## License

Part of the equity-ai-trading project.
