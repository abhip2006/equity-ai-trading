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
    Validates trade plans against risk parameters and portfolio constraints.

    Implements:
    - Position sizing limits
    - Concurrent position limits
    - Sector concentration limits
    - Daily drawdown cutoffs
    - Correlation checks (optional)
    """

    def __init__(self, risk_params: Dict):
        """
        Initialize risk manager.

        Args:
            risk_params: Dict with risk parameters
                - max_position_pct: Max % of portfolio per position
                - max_concurrent_positions: Max number of open positions
                - max_daily_drawdown: Max daily loss as % of portfolio
                - max_sector_concentration: Max % in single sector (optional)
        """
        self.risk_params = risk_params
        logger.info(f"RiskManager initialized with params: {risk_params}")

    def evaluate(
        self,
        trade_plan: 'TradePlan',
        portfolio_state: PortfolioState
    ) -> RiskDecision:
        """
        Evaluate and potentially modify trade plan.

        Args:
            trade_plan: Proposed trade plan
            portfolio_state: Current portfolio state

        Returns:
            RiskDecision with approval status and any modifications
        """
        checks = []
        modifications = {}
        approved = True

        # 1. Check concurrent positions limit
        current_positions = len(portfolio_state.positions)
        max_positions = self.risk_params.get('max_concurrent_positions', 8)

        if current_positions >= max_positions:
            return RiskDecision(
                approved=False,
                reason=f"Max concurrent positions reached ({current_positions}/{max_positions})"
            )

        checks.append(f"Concurrent positions: {current_positions}/{max_positions} ✓")

        # 2. Check position size
        max_position_pct = self.risk_params.get('max_position_pct', 0.05)
        requested_size = trade_plan.position_size_pct

        if requested_size > max_position_pct:
            modifications['position_size_pct'] = max_position_pct
            checks.append(f"Position size reduced: {requested_size:.1%} -> {max_position_pct:.1%}")
        else:
            checks.append(f"Position size OK: {requested_size:.1%} <= {max_position_pct:.1%} ✓")

        # 3. Check daily drawdown
        max_daily_dd = self.risk_params.get('max_daily_drawdown', 0.10)
        current_dd_pct = abs(portfolio_state.daily_pnl) / portfolio_state.equity if portfolio_state.equity > 0 else 0

        if current_dd_pct >= max_daily_dd:
            return RiskDecision(
                approved=False,
                reason=f"Daily drawdown limit exceeded ({current_dd_pct:.1%} >= {max_daily_dd:.1%})"
            )

        checks.append(f"Daily drawdown: {current_dd_pct:.1%} < {max_daily_dd:.1%} ✓")

        # 4. Check sector concentration (if enabled)
        max_sector_concentration = self.risk_params.get('max_sector_concentration')
        if max_sector_concentration:
            sector_exposure = self._calculate_sector_exposure(portfolio_state, trade_plan)

            if sector_exposure > max_sector_concentration:
                # Could reduce size instead of rejecting
                size_reduction = max_sector_concentration / sector_exposure
                adjusted_size = trade_plan.position_size_pct * size_reduction

                modifications['position_size_pct'] = min(
                    adjusted_size,
                    modifications.get('position_size_pct', max_position_pct)
                )

                checks.append(f"Sector concentration reduced to {max_sector_concentration:.1%}")
            else:
                checks.append(f"Sector concentration OK: {sector_exposure:.1%} <= {max_sector_concentration:.1%} ✓")

        # 5. Check minimum risk/reward
        min_rr = self.risk_params.get('min_risk_reward', 1.5)
        if trade_plan.risk_reward_ratio < min_rr:
            return RiskDecision(
                approved=False,
                reason=f"Risk/reward too low ({trade_plan.risk_reward_ratio:.2f} < {min_rr:.2f})"
            )

        checks.append(f"Risk/reward: {trade_plan.risk_reward_ratio:.2f} >= {min_rr:.2f} ✓")

        # Compile decision
        reason = "; ".join(checks)

        if modifications:
            reason = f"Approved with modifications: {modifications}. " + reason

        logger.info(f"Risk evaluation for {trade_plan.symbol}: {reason}")

        return RiskDecision(
            approved=approved,
            modifications=modifications,
            reason=reason
        )

    def _get_sector(self, symbol: str) -> str:
        """
        Get sector for a symbol using yfinance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Sector name or 'Unknown' if lookup fails
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            sector = ticker.info.get('sector', 'Unknown')

            # Handle ETFs and indices that don't have sector
            if sector == 'Unknown' or sector is None:
                # Common index/ETF classifications
                index_etf_map = {
                    'SPY': 'Index', 'QQQ': 'Technology Index', 'DIA': 'Index',
                    'IWM': 'Index', 'VTI': 'Index', 'VOO': 'Index',
                    'XLF': 'Financials', 'XLE': 'Energy', 'XLK': 'Technology',
                    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Consumer Staples',
                    'XLY': 'Consumer Discretionary', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
                    'XLB': 'Materials', 'XLC': 'Communication Services'
                }
                sector = index_etf_map.get(symbol, 'Unknown')

            logger.debug(f"Sector lookup for {symbol}: {sector}")
            return sector

        except Exception as e:
            logger.warning(f"Failed to lookup sector for {symbol}: {e}")
            return 'Unknown'

    def _calculate_sector_exposure(self, portfolio_state: PortfolioState, trade_plan: 'TradePlan') -> float:
        """
        Calculate sector exposure including proposed trade.

        Uses live sector data from yfinance to ensure proper diversification.
        """
        # Get actual sector for the proposed trade (LIVE DATA)
        sector = self._get_sector(trade_plan.symbol)

        if sector == 'Unknown':
            logger.warning(f"Unknown sector for {trade_plan.symbol}, cannot enforce sector limits")
            return 0.0  # Don't count unknown sectors in concentration

        # Calculate current exposure in this sector
        sector_exposure = sum(
            pos['size_pct']
            for pos in portfolio_state.positions
            if pos.get('sector') == sector
        )

        # Add proposed trade
        sector_exposure += trade_plan.position_size_pct

        logger.debug(f"Sector '{sector}' exposure: {sector_exposure:.1%} (including proposed {trade_plan.position_size_pct:.1%})")

        return sector_exposure

    def batch_evaluate(
        self,
        trade_plans: List['TradePlan'],
        portfolio_state: PortfolioState
    ) -> List[tuple['TradePlan', RiskDecision]]:
        """
        Evaluate multiple trade plans.

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
                    'size_pct': decision.modifications.get('position_size_pct', plan.position_size_pct)
                })

        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from active_trader_llm.trader.trader_agent import TradePlan

    # Sample trade plan
    plan = TradePlan(
        symbol="AAPL",
        strategy="momentum_breakout",
        direction="long",
        entry=175.0,
        stop_loss=172.0,
        take_profit=180.0,
        position_size_pct=0.07,  # Requesting 7% (over limit)
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
        daily_pnl=-500  # Small loss today
    )

    # Risk parameters
    risk_params = {
        'max_position_pct': 0.05,
        'max_concurrent_positions': 8,
        'max_daily_drawdown': 0.10,
        'max_sector_concentration': 0.30,
        'min_risk_reward': 1.5
    }

    manager = RiskManager(risk_params)
    decision = manager.evaluate(plan, portfolio)

    print(f"\nRisk Decision:")
    print(f"  Approved: {decision.approved}")
    print(f"  Modifications: {decision.modifications}")
    print(f"  Reason: {decision.reason}")
