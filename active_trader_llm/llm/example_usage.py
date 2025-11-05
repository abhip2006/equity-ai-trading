"""
Example: Using LLM Abstraction Layer for Trading

Demonstrates how to use the unified LLM interface for trading decisions.
"""

import os
import json
from active_trader_llm.llm import get_llm_client, LLMMessage


def example_basic_trade_decision():
    """Example: Basic trade decision using any provider"""
    print("Example 1: Basic Trade Decision\n" + "="*50)

    # Create client (can swap provider without changing code)
    client = get_llm_client(
        provider="openai",  # Try: anthropic, xai, deepseek, google, openrouter
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Prepare trading prompt
    messages = [
        LLMMessage(
            role="system",
            content="You are a professional stock trader. Analyze technical data and provide trade recommendations."
        ),
        LLMMessage(
            role="user",
            content="""
Analyze AAPL:
- Current Price: $175.50
- 5 EMA: $174.20
- 20 EMA: $172.80
- 50 SMA: $170.00
- RSI: 65
- Volume: 2.5M shares (avg: 2.0M)

Should I buy, sell, or hold? Provide brief reasoning.
"""
        )
    ]

    # Generate response
    response = client.generate(messages, temperature=0.3, max_tokens=500)

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Decision: {response.content}")
    print(f"Tokens used: {response.usage}\n")


def example_structured_trade_plan():
    """Example: Generate structured trade plan JSON"""
    print("Example 2: Structured Trade Plan\n" + "="*50)

    client = get_llm_client(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Trading system prompt (matches current trader_agent.py)
    system_prompt = """You are an active equity trader. Generate trade plans in JSON format.

Output JSON with:
{
    "action": "open_long|open_short|hold|pass",
    "symbol": "TICKER",
    "entry": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "position_size_pct": 0.0-1.0,
    "confidence": 0.0-1.0,
    "rationale": "Brief explanation"
}
"""

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(
            role="user",
            content="""
AAPL Technical Data:
- Current Price: $175.50
- 5 EMA: $176.00 (bullish)
- 20 EMA: $174.00
- 50 SMA: $170.00 (above - uptrend)
- RSI: 58 (neutral)
- Volume: 3.2M (above average - confirming)

Generate trade plan for AAPL.
"""
        )
    ]

    # Generate structured JSON
    response = client.generate_structured(
        messages,
        response_schema={},
        temperature=0.3,
        max_tokens=500
    )

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}\n")

    # Parse and display trade plan
    trade_plan = json.loads(response.content)
    print("Trade Plan:")
    print(json.dumps(trade_plan, indent=2))
    print()


def example_multi_provider_comparison():
    """Example: Compare responses from multiple providers"""
    print("Example 3: Multi-Provider Comparison\n" + "="*50)

    # Same prompt for all providers
    messages = [
        LLMMessage(
            role="system",
            content="You are a stock analyst. Provide brief market outlook in one sentence."
        ),
        LLMMessage(
            role="user",
            content="What's your outlook on tech stocks this week?"
        )
    ]

    # Test multiple providers
    providers = [
        ("openai", "gpt-3.5-turbo", os.getenv("OPENAI_API_KEY")),
        ("anthropic", "claude-3-5-sonnet-20241022", os.getenv("ANTHROPIC_API_KEY")),
    ]

    for provider, model, api_key in providers:
        if not api_key:
            print(f"⊘ {provider}/{model} - API key not set\n")
            continue

        try:
            client = get_llm_client(provider=provider, model=model, api_key=api_key)
            response = client.generate(messages, temperature=0.5, max_tokens=100)

            print(f"{provider}/{model}:")
            print(f"  Response: {response.content}")
            print(f"  Tokens: {response.usage}")
            print()

        except Exception as e:
            print(f"✗ {provider}/{model} failed: {e}\n")


def example_battle_system_integration():
    """Example: How battle system would use this abstraction"""
    print("Example 4: Battle System Integration\n" + "="*50)

    # Battle configuration (from config file)
    battle_config = {
        "traders": [
            {"name": "GPT-4", "provider": "openai", "model": "gpt-4o"},
            {"name": "Claude", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            {"name": "Grok", "provider": "xai", "model": "grok-beta"},
        ]
    }

    # Simulate battle: each trader makes a decision
    market_data = {
        "symbol": "AAPL",
        "price": 175.50,
        "indicators": {"ema_5": 176.00, "rsi": 58}
    }

    system_prompt = "You are a trader. Decide: buy, sell, or hold. One word only."
    user_prompt = f"AAPL is at ${market_data['price']}, RSI={market_data['indicators']['rsi']}. Decision?"

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt)
    ]

    print("Battle Round - All traders decide on AAPL:\n")

    for trader_config in battle_config["traders"]:
        provider = trader_config["provider"]
        model = trader_config["model"]
        api_key = os.getenv(f"{provider.upper()}_API_KEY")

        if not api_key:
            print(f"{trader_config['name']:10} | SKIP (no API key)")
            continue

        try:
            client = get_llm_client(provider=provider, model=model, api_key=api_key)
            response = client.generate(messages, temperature=0.3, max_tokens=50)

            decision = response.content.strip().upper()
            print(f"{trader_config['name']:10} | {decision}")

        except Exception as e:
            print(f"{trader_config['name']:10} | ERROR: {e}")

    print()


if __name__ == "__main__":
    # Run all examples
    try:
        example_basic_trade_decision()
        example_structured_trade_plan()
        example_multi_provider_comparison()
        example_battle_system_integration()

        print("\n✓ All examples completed!")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
