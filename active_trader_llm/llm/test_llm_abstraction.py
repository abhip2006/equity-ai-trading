"""
Test script for LLM abstraction layer

Validates that all providers work with the unified interface.
Run this to ensure your API keys are configured correctly.

Usage:
    python -m active_trader_llm.llm.test_llm_abstraction
"""

import os
import json
import logging
from typing import Dict

from active_trader_llm.llm import (
    get_llm_client,
    LLMMessage,
    LLMResponse,
    LLMError,
    RateLimitError,
    AuthenticationError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_provider(
    provider: str,
    model: str,
    api_key: str,
    test_structured: bool = True
) -> Dict[str, bool]:
    """
    Test a single provider with both generate and generate_structured

    Returns:
        Dict with test results
    """
    results = {
        "provider": provider,
        "model": model,
        "connection": False,
        "generate": False,
        "generate_structured": False,
        "error": None
    }

    try:
        # Create client
        client = get_llm_client(provider=provider, model=model, api_key=api_key)
        results["connection"] = True
        logger.info(f"✓ [{provider}/{model}] Connection successful")

        # Test 1: Basic generation
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Say 'Hello, world!' and nothing else.")
        ]

        response = client.generate(messages, temperature=0.0, max_tokens=50)
        logger.info(f"✓ [{provider}/{model}] Generate response: {response.content[:100]}")

        if response.content:
            results["generate"] = True

        if response.usage:
            logger.info(f"  Token usage: {response.usage}")

        # Test 2: Structured JSON generation
        if test_structured:
            messages = [
                LLMMessage(
                    role="system",
                    content="You are a trading assistant. Respond with valid JSON only."
                ),
                LLMMessage(
                    role="user",
                    content='Generate a trade recommendation for AAPL. Return JSON: {"symbol": "AAPL", "action": "buy", "confidence": 0.8}'
                )
            ]

            response = client.generate_structured(
                messages,
                response_schema={},
                temperature=0.0,
                max_tokens=200
            )

            logger.info(f"✓ [{provider}/{model}] Structured response: {response.content[:100]}")

            # Validate JSON
            try:
                data = json.loads(response.content)
                logger.info(f"  Parsed JSON: {data}")
                results["generate_structured"] = True
            except json.JSONDecodeError as e:
                logger.error(f"✗ [{provider}/{model}] Invalid JSON: {e}")
                results["error"] = f"Invalid JSON: {e}"

    except AuthenticationError as e:
        logger.error(f"✗ [{provider}/{model}] Authentication failed: {e}")
        results["error"] = f"Auth error: {e}"

    except RateLimitError as e:
        logger.error(f"✗ [{provider}/{model}] Rate limited: {e}")
        results["error"] = f"Rate limit: {e}"

    except LLMError as e:
        logger.error(f"✗ [{provider}/{model}] LLM error: {e}")
        results["error"] = str(e)

    except Exception as e:
        logger.error(f"✗ [{provider}/{model}] Unexpected error: {e}")
        results["error"] = f"Unexpected: {e}"

    return results


def main():
    """Run tests for all configured providers"""
    print("\n" + "="*80)
    print("LLM Abstraction Layer - Provider Tests")
    print("="*80 + "\n")

    # Define providers to test
    # Format: (provider, model, env_var_for_key)
    providers_to_test = [
        ("openai", "gpt-3.5-turbo", "OPENAI_API_KEY"),
        ("openai", "gpt-4o", "OPENAI_API_KEY"),
        ("anthropic", "claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY"),
        ("xai", "grok-beta", "XAI_API_KEY"),
        ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),
        ("google", "gemini-pro", "GOOGLE_API_KEY"),
        ("openrouter", "qwen/qwen-2.5-72b-instruct", "OPENROUTER_API_KEY"),
    ]

    all_results = []

    for provider, model, env_var in providers_to_test:
        api_key = os.getenv(env_var)

        if not api_key:
            logger.warning(f"⊘ [{provider}/{model}] Skipped - {env_var} not set")
            all_results.append({
                "provider": provider,
                "model": model,
                "connection": False,
                "generate": False,
                "generate_structured": False,
                "error": f"{env_var} not set"
            })
            continue

        logger.info(f"\n--- Testing {provider}/{model} ---")
        result = test_provider(provider, model, api_key)
        all_results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80 + "\n")

    for result in all_results:
        provider = result["provider"]
        model = result["model"]
        status = "✓ PASS" if result["generate"] and result["generate_structured"] else "✗ FAIL"

        print(f"{status:8} | {provider:15} | {model:30}")

        if result["error"]:
            print(f"         | Error: {result['error']}")

    # Count successes
    total = len(all_results)
    passed = sum(1 for r in all_results if r["generate"] and r["generate_structured"])

    print(f"\n{passed}/{total} providers passed all tests")

    if passed == total:
        print("\n✓ All configured providers working correctly!")
        return 0
    else:
        print("\n⚠ Some providers failed. Check logs above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
