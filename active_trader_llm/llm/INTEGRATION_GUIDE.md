# Integration Guide: LLM Abstraction Layer

This guide shows how to integrate the LLM abstraction layer into the existing trading system components.

## Current State Analysis

### Files Using OpenAI Directly

1. **`active_trader_llm/trader/trader_agent.py`** (Primary)
   - Direct OpenAI client instantiation
   - Uses `chat.completions.create()`
   - Needs full refactor to use abstraction

2. **Other potential locations**
   - Scanner components
   - Analyst agents
   - Research agents

## Migration Strategy

### Phase 1: Replace TraderAgent (Priority)

**File:** `active_trader_llm/trader/trader_agent.py`

**Current Pattern:**
```python
from openai import OpenAI

class TraderAgent:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key, timeout=60.0)
        self.model = model

    def decide(self, ...):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[...],
            temperature=0.6,
            max_tokens=900
        )
        content = response.choices[0].message.content.strip()
```

**New Pattern:**
```python
from active_trader_llm.llm import get_llm_client, LLMMessage

class TraderAgent:
    def __init__(self, provider="openai", model="gpt-3.5-turbo", api_key=None):
        self.client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            timeout=60.0
        )
        self.model = model
        self.provider = provider

    def decide(self, ...):
        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt)
        ]

        response = self.client.generate(
            messages,
            temperature=0.6,
            max_tokens=900
        )
        content = response.content
```

### Phase 2: Update Configuration

**File:** `config.yaml`

**Add battle system configuration:**
```yaml
# Single LLM configuration (backward compatible)
llm:
  provider: openai
  model: gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}
  temperature: 0.3
  max_tokens: 2000

# Battle system: Multiple models competing
battle:
  enabled: false  # Set to true to enable battle mode
  traders:
    - name: GPT-4
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}
      weight: 1.0  # Equal weighting

    - name: Claude-Sonnet
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      api_key: ${ANTHROPIC_API_KEY}
      weight: 1.0

    - name: Grok
      provider: xai
      model: grok-beta
      api_key: ${XAI_API_KEY}
      weight: 1.0

    - name: Qwen
      provider: openrouter
      model: qwen/qwen-2.5-72b-instruct
      api_key: ${OPENROUTER_API_KEY}
      weight: 1.0
```

### Phase 3: Environment Variables

**File:** `.env`

Add API keys for all providers:
```bash
# Existing
OPENAI_API_KEY=sk-...

# New providers for battle system
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=...
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
```

## Detailed Migration Steps

### Step 1: Update trader_agent.py

```python
# active_trader_llm/trader/trader_agent.py

from active_trader_llm.llm import get_llm_client, LLMMessage, LLMError, RateLimitError

class TraderAgent:
    """
    Synthesizes multi-agent signals into concrete trade plans.
    Now supports multiple LLM providers via abstraction layer.
    """

    SYSTEM_PROMPT = """..."""  # Keep existing prompt

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize trader agent

        Args:
            provider: LLM provider (openai, anthropic, xai, deepseek, google, openrouter)
            model: Model identifier
            api_key: API key for authentication
            validation_config: Trade plan validation configuration
            enable_validation: Whether to enable trade plan validation
        """
        self.provider = provider
        self.model = model

        # Create LLM client
        self.client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            timeout=60.0,
            max_retries=3
        )

        self.enable_validation = enable_validation
        self.validator = TradePlanValidator(validation_config) if enable_validation else None

    def decide(self, symbol, analyst_outputs, researcher_outputs, risk_params, **kwargs):
        """Generate trade plan (updated to use abstraction)"""

        prompt = self._build_decision_prompt(
            symbol, analyst_outputs, researcher_outputs,
            risk_params, **kwargs
        )

        # Convert to LLMMessage format
        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt)
        ]

        try:
            # Generate response using abstraction
            response = self.client.generate(
                messages,
                temperature=0.6,
                max_tokens=900
            )

            content = response.content

            # Log token usage
            if response.usage:
                logger.debug(
                    f"LLM usage [{self.provider}/{self.model}]: "
                    f"{response.usage['total_tokens']} tokens"
                )

            # Rest of existing parsing logic...
            # (Keep all JSON parsing, validation, etc.)

        except RateLimitError as e:
            logger.error(f"Rate limit exceeded for {self.provider}: {e}")
            return None

        except LLMError as e:
            logger.error(f"LLM error: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
```

### Step 2: Update main.py Initialization

```python
# active_trader_llm/main.py

def create_trader_agent(config: Dict) -> TraderAgent:
    """Create trader agent from config"""

    llm_config = config.get('llm', {})

    return TraderAgent(
        provider=llm_config.get('provider', 'openai'),
        model=llm_config.get('model', 'gpt-3.5-turbo'),
        api_key=os.getenv(llm_config.get('api_key', 'OPENAI_API_KEY')),
        validation_config=validation_config,
        enable_validation=config.get('trade_validation', {}).get('enabled', True)
    )
```

### Step 3: Create Battle System (New Feature)

**File:** `active_trader_llm/battle/battle_system.py`

```python
"""
Battle System: Multiple LLMs compete on same market data
"""

from typing import List, Dict
from active_trader_llm.llm import get_llm_client, LLMMessage
from active_trader_llm.trader.trader_agent import TraderAgent, TradePlan

class BattleSystem:
    """Manages competition between multiple LLM traders"""

    def __init__(self, battle_config: Dict):
        self.traders = []

        for trader_cfg in battle_config.get('traders', []):
            agent = TraderAgent(
                provider=trader_cfg['provider'],
                model=trader_cfg['model'],
                api_key=os.getenv(trader_cfg.get('api_key', f"{trader_cfg['provider'].upper()}_API_KEY"))
            )

            self.traders.append({
                'name': trader_cfg['name'],
                'agent': agent,
                'weight': trader_cfg.get('weight', 1.0),
                'performance': {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
            })

    def generate_decisions(self, symbol, analyst_outputs, **kwargs) -> List[TradePlan]:
        """All traders make decisions on same data"""

        plans = []

        for trader in self.traders:
            try:
                plan = trader['agent'].decide(
                    symbol, analyst_outputs, **kwargs
                )

                if plan and plan.action in ['open_long', 'open_short']:
                    plan.trader_name = trader['name']  # Add trader identification
                    plans.append(plan)

            except Exception as e:
                logger.error(f"Trader {trader['name']} failed: {e}")

        return plans

    def select_best_plan(self, plans: List[TradePlan]) -> Optional[TradePlan]:
        """
        Select best plan based on:
        1. Confidence
        2. Historical performance (weight)
        3. Risk/reward ratio
        """

        if not plans:
            return None

        # Weighted scoring
        scored_plans = []
        for plan in plans:
            trader = next(t for t in self.traders if t['name'] == plan.trader_name)
            weight = trader['weight']

            score = (
                plan.confidence * 0.4 +
                weight * 0.3 +
                min(plan.risk_reward_ratio / 3.0, 1.0) * 0.3
            )

            scored_plans.append((score, plan))

        # Return highest scoring plan
        scored_plans.sort(reverse=True, key=lambda x: x[0])
        return scored_plans[0][1]
```

## Testing Migration

### 1. Test Single Provider

```bash
# Update config.yaml to use new provider
llm:
  provider: openai
  model: gpt-3.5-turbo

# Run backtest
python -m active_trader_llm.main backtest --config config.yaml
```

### 2. Test Multiple Providers

```bash
# Enable battle mode
battle:
  enabled: true

# Run with battle system
python -m active_trader_llm.main backtest --config config.yaml --battle
```

### 3. Validate All Providers

```bash
# Run provider tests
python -m active_trader_llm.llm.test_llm_abstraction
```

## Rollback Plan

If migration causes issues:

1. **Keep old imports available**
   ```python
   # Fallback to direct OpenAI
   try:
       from active_trader_llm.llm import get_llm_client
   except ImportError:
       from openai import OpenAI
   ```

2. **Feature flag in config**
   ```yaml
   llm:
     use_abstraction: false  # Disable new system
   ```

3. **Git revert**
   ```bash
   git revert <migration-commit>
   ```

## Performance Benchmarks

Run benchmarks to compare providers:

```bash
python -m active_trader_llm.llm.benchmark_providers
```

Expected results:
- **OpenAI GPT-4**: High quality, moderate speed, moderate cost
- **Anthropic Claude**: Highest quality, slower, highest cost
- **xAI Grok**: Good quality, variable speed, moderate cost
- **DeepSeek**: Good quality, slowest, lowest cost
- **Google Gemini**: Good quality, fastest, low cost

## Monitoring

Add logging to track provider performance:

```python
logger.info(f"Decision made by {self.provider}/{self.model}")
logger.info(f"Tokens used: {response.usage}")
logger.info(f"Latency: {response.latency_ms}ms")
```

## Next Steps

After successful migration:

1. Run multi-provider backtests
2. Compare performance metrics
3. Tune battle system weights
4. Optimize cost vs. performance
5. Deploy to paper trading

## Support

Issues? Check:
- API keys are set correctly in `.env`
- Provider adapters are imported in `__init__.py`
- Config file has correct provider names
- Test suite passes: `python -m active_trader_llm.llm.test_llm_abstraction`
