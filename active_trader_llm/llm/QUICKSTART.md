# Quick Start Guide - LLM Abstraction Layer

Get started with the LLM abstraction layer in 5 minutes.

## Installation

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt
```

## Setup API Keys

Create or update `.env` file in project root:

```bash
# At minimum, set one provider
OPENAI_API_KEY=sk-...

# Optional: For battle system
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=...
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
```

## Test Your Setup

```bash
# Test all configured providers
python -m active_trader_llm.llm.test_llm_abstraction
```

Expected output:
```
✓ PASS    | openai          | gpt-3.5-turbo
✓ PASS    | anthropic       | claude-3-5-sonnet-20241022
⊘ Skipped | xai             | grok-beta (XAI_API_KEY not set)
...
```

## Basic Usage

### 1. Single Provider (OpenAI)

```python
import os
from active_trader_llm.llm import get_llm_client, LLMMessage

# Create client
client = get_llm_client(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Generate completion
messages = [
    LLMMessage(role="system", content="You are a stock trader."),
    LLMMessage(role="user", content="Analyze AAPL at $175. Buy or sell?")
]

response = client.generate(messages, temperature=0.3, max_tokens=200)
print(response.content)
print(f"Tokens: {response.usage}")
```

### 2. Structured JSON Output

```python
import json

# Same client as above
system_prompt = """You are a trader. Output JSON only:
{
    "action": "buy|sell|hold",
    "confidence": 0.0-1.0,
    "rationale": "text"
}
"""

messages = [
    LLMMessage(role="system", content=system_prompt),
    LLMMessage(role="user", content="AAPL is at $175, RSI 65. Decision?")
]

response = client.generate_structured(messages, response_schema={})
trade_plan = json.loads(response.content)

print(f"Action: {trade_plan['action']}")
print(f"Confidence: {trade_plan['confidence']}")
print(f"Rationale: {trade_plan['rationale']}")
```

### 3. Compare Multiple Providers

```python
providers = [
    ("openai", "gpt-3.5-turbo", os.getenv("OPENAI_API_KEY")),
    ("anthropic", "claude-3-5-sonnet-20241022", os.getenv("ANTHROPIC_API_KEY")),
]

for provider, model, api_key in providers:
    if not api_key:
        print(f"Skipping {provider} (no API key)")
        continue

    client = get_llm_client(provider, model, api_key)
    response = client.generate(messages, temperature=0.3)

    print(f"\n{provider}/{model}:")
    print(response.content)
```

## Run Examples

```bash
# Run all usage examples
python -m active_trader_llm.llm.example_usage
```

Examples include:
1. Basic trade decision
2. Structured trade plan
3. Multi-provider comparison
4. Battle system integration

## Common Issues

### Import Error
```python
ImportError: No module named 'active_trader_llm.llm'
```

**Solution:** Make sure you're in the project root directory.

### API Key Error
```
AuthenticationError: [openai/gpt-3.5-turbo] Invalid API key
```

**Solution:** Check `.env` file has correct API key.

### Google Package Missing
```
ImportError: google-generativeai package not installed
```

**Solution:** Install optional dependency:
```bash
pip install google-generativeai
```

## Next Steps

1. **Read Full Documentation**: See `README.md`
2. **Integration Guide**: See `INTEGRATION_GUIDE.md`
3. **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
4. **Configure Battle System**: Update `config.yaml`

## Provider-Specific Notes

### OpenAI
```python
client = get_llm_client("openai", "gpt-4o", os.getenv("OPENAI_API_KEY"))
```
Models: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

### Anthropic
```python
client = get_llm_client("anthropic", "claude-3-5-sonnet-20241022", os.getenv("ANTHROPIC_API_KEY"))
```
Models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`

### xAI (Grok)
```python
client = get_llm_client("xai", "grok-beta", os.getenv("XAI_API_KEY"))
```
Models: `grok-beta`

### DeepSeek
```python
client = get_llm_client("deepseek", "deepseek-chat", os.getenv("DEEPSEEK_API_KEY"))
```
Models: `deepseek-chat`, `deepseek-coder`

### Google
```python
client = get_llm_client("google", "gemini-pro", os.getenv("GOOGLE_API_KEY"))
```
Models: `gemini-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`

### OpenRouter
```python
client = get_llm_client("openrouter", "qwen/qwen-2.5-72b-instruct", os.getenv("OPENROUTER_API_KEY"))
```
Models: `qwen/qwen-2.5-72b-instruct`, `meta-llama/llama-3.1-70b-instruct`, and 100+ more

## Error Handling

```python
from active_trader_llm.llm import (
    get_llm_client,
    RateLimitError,
    AuthenticationError,
    LLMError
)

try:
    client = get_llm_client("openai", "gpt-4o", api_key)
    response = client.generate(messages)
except RateLimitError:
    print("Rate limited - waiting...")
    time.sleep(60)
except AuthenticationError:
    print("Invalid API key")
except LLMError as e:
    print(f"LLM error: {e}")
```

## Support

- Check logs for detailed error messages
- Run test suite: `python -m active_trader_llm.llm.test_llm_abstraction`
- Review examples: `python -m active_trader_llm.llm.example_usage`
- Read full docs: `active_trader_llm/llm/README.md`

## Ready for Battle System?

See `INTEGRATION_GUIDE.md` for step-by-step integration into the trading system.
