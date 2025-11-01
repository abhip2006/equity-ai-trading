"""
Risk Manager: Validates and adjusts trade plans under portfolio constraints.
"""

import logging
from typing import Dict, List, Optional
from pydantic import BaseModel
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskDecision(BaseModel):
    """Risk manager decision schema"""
    approved: bool
    modifications: Dict = {}
    reason: str


class PortfolioState(BaseModel):
    """Current portfolio state"""
    cash: float
    equity: float
    positions: List[Dict] = []
    daily_pnl: float = 0.0


class RiskManager:
    """
    Validates trade plans against portfolio constraints.

    IMPORTANT: Position sizing limits have been removed to allow the LLM agent
    to determine appropriate position sizes based on market conditions and performance.

    Optional safety rails (can be disabled via risk_params):
    - enforce_daily_drawdown: Emergency stop on excessive daily losses (default: True)
    - emergency_drawdown_limit: Max daily loss % before halting (default: 0.20)

    All other constraints (max_position_pct, max_concurrent_positions,
    sector concentration, min_risk_reward) have been removed.
    """

    def __init__(self, risk_params: Dict):
        """
        Initialize risk manager.

        Args:
            risk_params: Dict with optional safety parameters
                - enforce_daily_drawdown: Enable emergency drawdown stop (default: True)
                - emergency_drawdown_limit: Max daily loss % (default: 0.20)
        """
        self.risk_params = risk_params
        logger.info(f"RiskManager initialized with params: {risk_params}")
        logger.info("Position sizing limits DISABLED - agent decides based on performance")

    def evaluate(
        self,
        trade_plan: 'TradePlan',
        portfolio_state: PortfolioState
    ) -> RiskDecision:
        """
        Evaluate trade plan with minimal interference.

        Position sizing limits have been removed - the agent decides based on performance.
        Only emergency safety rail: optional daily drawdown circuit breaker.

        Args:
            trade_plan: Proposed trade plan from agent
            portfolio_state: Current portfolio state

        Returns:
            RiskDecision with approval status (modifications removed)
        """
        checks = []
        approved = True

        # Log what the agent requested (for observability)
        current_positions = len(portfolio_state.positions)
        requested_size = trade_plan.position_size_pct
        rr_ratio = trade_plan.risk_reward_ratio

        logger.info(
            f"{trade_plan.symbol} agent request: "
            f"size={requested_size:.1%}, RR={rr_ratio:.2f}, "
            f"positions={current_positions}"
        )

        checks.append(f"Agent requested: {requested_size:.1%} position size")
        checks.append(f"Risk/reward: {rr_ratio:.2f}")
        checks.append(f"Current positions: {current_positions}")

        # ONLY EMERGENCY SAFETY RAIL: Daily drawdown circuit breaker (optional)
        enforce_dd = self.risk_params.get('enforce_daily_drawdown', True)

        if enforce_dd:
            emergency_dd_limit = self.risk_params.get('emergency_drawdown_limit', 0.20)
            current_dd_pct = abs(portfolio_state.daily_pnl) / portfolio_state.equity if portfolio_state.equity > 0 else 0

            if current_dd_pct >= emergency_dd_limit:
                logger.error(
                    f"EMERGENCY STOP: Daily drawdown {current_dd_pct:.1%} >= {emergency_dd_limit:.1%}"
                )
                return RiskDecision(
                    approved=False,
                    reason=f"Emergency drawdown circuit breaker triggered ({current_dd_pct:.1%} >= {emergency_dd_limit:.1%})"
                )

            checks.append(f"Daily drawdown: {current_dd_pct:.1%} < {emergency_dd_limit:.1%} (emergency limit)")

        # APPROVED - Agent's decision passes through unchanged
        reason = "; ".join(checks)
        logger.info(f"Trade approved for {trade_plan.symbol}: {reason}")

        return RiskDecision(
            approved=approved,
            modifications={},  # No modifications - agent decides
            reason=reason
        )


    def batch_evaluate(
        self,
        trade_plans: List['TradePlan'],
        portfolio_state: PortfolioState
    ) -> List[tuple['TradePlan', RiskDecision]]:
        """
        Evaluate multiple trade plans sequentially.

        Since position limits are removed, all plans pass through unless
        emergency drawdown circuit breaker is triggered.

        Args:
            trade_plans: List of trade plans to evaluate
            portfolio_state: Current portfolio state

        Returns:
            List of (trade_plan, risk_decision) tuples
        """
        results = []

        for plan in trade_plans:
            decision = self.evaluate(plan, portfolio_state)
            results.append((plan, decision))

            # Update portfolio state simulation for next evaluation
            if decision.approved:
                portfolio_state.positions.append({
                    'symbol': plan.symbol,
                    'size_pct': plan.position_size_pct  # Use agent's requested size (no modifications)
                })

        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from active_trader_llm.trader.trader_agent import TradePlan

    # Sample trade plan - agent decides 7% position size
    plan = TradePlan(
        symbol="AAPL",
        strategy="momentum_breakout",
        direction="long",
        entry=175.0,
        stop_loss=172.0,
        take_profit=180.0,
        position_size_pct=0.07,  # Agent's decision - will be approved
        time_horizon="3d",
        rationale=["Strong momentum", "Breakout confirmed"],
        risk_reward_ratio=1.67
    )

    # Sample portfolio
    portfolio = PortfolioState(
        cash=50000,
        equity=100000,
        positions=[
            {'symbol': 'MSFT', 'size_pct': 0.05, 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'size_pct': 0.04, 'sector': 'Technology'}
        ],
        daily_pnl=-500  # Small loss today (-0.5%, well below emergency threshold)
    )

    # Risk parameters - only emergency safety rail
    risk_params = {
        'enforce_daily_drawdown': True,  # Optional emergency circuit breaker
        'emergency_drawdown_limit': 0.20  # 20% daily loss triggers halt
    }

    manager = RiskManager(risk_params)
    decision = manager.evaluate(plan, portfolio)

    print(f"\nRisk Decision:")
    print(f"  Approved: {decision.approved}")
    print(f"  Modifications: {decision.modifications}")  # Should be empty
    print(f"  Reason: {decision.reason}")
    print(f"\nAgent's 7% position size was approved (no limits enforced)")
