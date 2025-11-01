"""
Strategy Monitor: Tracks performance and triggers strategy switching.

Implements rolling window monitoring and strategy degradation detection.
"""

import logging
from typing import Dict, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PerformanceMetrics(BaseModel):
    """Rolling performance metrics"""
    strategy_name: str
    regime: str
    window_size: int
    win_rate: float
    net_pnl: float
    avg_return: float
    max_drawdown: float
    sharpe_proxy: float
    avg_rr: float
    num_trades: int


class SwitchRecommendation(BaseModel):
    """Strategy switch recommendation"""
    current_strategy: str
    recommended_strategy: str
    reason: str
    confidence: float
    metrics: Dict


class StrategyMonitor:
    """
    Monitors strategy performance and triggers switches when degradation detected.

    Based on PRD degradation rules:
    - Window of last N trades
    - Min win rate threshold
    - Min net P/L threshold
    - Hysteresis to prevent frequent switching
    """

    def __init__(
        self,
        memory_manager: 'MemoryManager',
        switching_config: Dict
    ):
        """
        Initialize strategy monitor.

        Args:
            memory_manager: MemoryManager instance
            switching_config: Dict with:
                - window_trades: Number of trades to evaluate (default 30)
                - min_win_rate: Minimum acceptable win rate (default 0.35)
                - min_net_pl: Minimum acceptable net P/L (default 0.0)
                - hysteresis_trades_before_switch: Cooldown period (default 10)
        """
        self.memory = memory_manager
        self.config = switching_config
        self.switch_history = {}  # Track last switch time per strategy

        logger.info(f"StrategyMonitor initialized with config: {switching_config}")

    def compute_metrics(
        self,
        strategy_name: str,
        regime: str,
        window_trades: Optional[int] = None
    ) -> PerformanceMetrics:
        """
        Compute rolling performance metrics for a strategy.

        Args:
            strategy_name: Strategy to evaluate
            regime: Current market regime
            window_trades: Window size (uses config default if not specified)

        Returns:
            PerformanceMetrics with calculated statistics
        """
        window = window_trades or self.config.get('window_trades', 30)

        # Get recent trades for this strategy
        recent_trades = self.memory.get_recent_trades(strategy=strategy_name, limit=window)

        if not recent_trades:
            logger.warning(f"No trades found for {strategy_name}")
            return PerformanceMetrics(
                strategy_name=strategy_name,
                regime=regime,
                window_size=0,
                win_rate=0.0,
                net_pnl=0.0,
                avg_return=0.0,
                max_drawdown=0.0,
                sharpe_proxy=0.0,
                avg_rr=0.0,
                num_trades=0
            )

        # Calculate metrics
        num_trades = len(recent_trades)
        wins = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
        win_rate = wins / num_trades if num_trades > 0 else 0.0

        pnls = [t.pnl for t in recent_trades if t.pnl is not None]
        net_pnl = sum(pnls)
        avg_return = net_pnl / num_trades if num_trades > 0 else 0.0

        # Calculate max drawdown (simplified)
        cumulative_pnl = 0
        peak = 0
        max_dd = 0

        for pnl in pnls:
            cumulative_pnl += pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_dd = max(max_dd, drawdown)

        # Sharpe proxy (returns / std dev)
        if len(pnls) > 1:
            import numpy as np
            std_dev = np.std(pnls)
            sharpe_proxy = avg_return / std_dev if std_dev > 0 else 0.0
        else:
            sharpe_proxy = 0.0

        # Average risk/reward
        rrs = [t.context_snapshot.get('rr', 0) for t in recent_trades if t.context_snapshot]
        avg_rr = sum(rrs) / len(rrs) if rrs else 0.0

        metrics = PerformanceMetrics(
            strategy_name=strategy_name,
            regime=regime,
            window_size=window,
            win_rate=win_rate,
            net_pnl=net_pnl,
            avg_return=avg_return,
            max_drawdown=max_dd,
            sharpe_proxy=sharpe_proxy,
            avg_rr=avg_rr,
            num_trades=num_trades
        )

        logger.info(f"{strategy_name} metrics: WR={win_rate:.1%}, Net P/L=${net_pnl:.2f}, Sharpe={sharpe_proxy:.2f}")

        return metrics

    def check_degradation(
        self,
        strategy_name: str,
        regime: str
    ) -> tuple[bool, str]:
        """
        Check if strategy has degraded below thresholds.

        Args:
            strategy_name: Strategy to check
            regime: Current regime

        Returns:
            Tuple of (is_degraded, reason)
        """
        metrics = self.compute_metrics(strategy_name, regime)

        # Check if enough trades for evaluation
        if metrics.num_trades < self.config.get('window_trades', 30):
            return False, f"Insufficient trades ({metrics.num_trades})"

        # Check win rate
        min_wr = self.config.get('min_win_rate', 0.35)
        if metrics.win_rate < min_wr:
            return True, f"Win rate below threshold ({metrics.win_rate:.1%} < {min_wr:.1%})"

        # Check net P/L
        min_pnl = self.config.get('min_net_pl', 0.0)
        if metrics.net_pnl < min_pnl:
            return True, f"Net P/L below threshold (${metrics.net_pnl:.2f} < ${min_pnl:.2f})"

        return False, "Performance acceptable"

    def select_best_strategy(
        self,
        available_strategies: List[str],
        current_regime: str
    ) -> str:
        """
        Select best performing strategy for current regime.

        Args:
            available_strategies: List of strategy names
            current_regime: Current market regime

        Returns:
            Best strategy name
        """
        best_strategy = None
        best_score = float('-inf')

        for strategy in available_strategies:
            # Get strategy stats for this regime
            stats_list = self.memory.get_strategy_stats(
                strategy_name=strategy,
                regime=current_regime
            )

            if not stats_list:
                # No history - default low score
                score = 0.0
            else:
                stats = stats_list[0]

                # Score = weighted combination of metrics using configurable weights
                win_rate_weight = self.config.get('win_rate_weight', 50.0)
                avg_return_weight = self.config.get('avg_return_weight', 0.5)
                avg_return_cap = self.config.get('avg_return_cap', 100.0)
                avg_rr_weight = self.config.get('avg_rr_weight', 10.0)

                score = (
                    stats.win_rate * win_rate_weight +
                    min(stats.avg_return, avg_return_cap) * avg_return_weight +
                    stats.avg_rr * avg_rr_weight
                )

                # Penalize recent switches (hysteresis)
                last_switch = self.switch_history.get(strategy)
                if last_switch:
                    trades_since_switch = stats.total_trades - last_switch.get('trades_at_switch', 0)
                    hysteresis = self.config.get('hysteresis_trades_before_switch', 10)

                    if trades_since_switch < hysteresis:
                        penalty = (hysteresis - trades_since_switch) * 5
                        score -= penalty
                        logger.debug(f"{strategy} penalty: {penalty} (switched {trades_since_switch} trades ago)")

            logger.info(f"{strategy} score for {current_regime}: {score:.2f}")

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy or available_strategies[0]

    def monitor_and_switch(
        self,
        current_strategy: str,
        available_strategies: List[str],
        current_regime: str
    ) -> Optional[SwitchRecommendation]:
        """
        Monitor current strategy and recommend switch if needed.

        Args:
            current_strategy: Currently active strategy
            available_strategies: Available strategies to switch to
            current_regime: Current market regime

        Returns:
            SwitchRecommendation if switch warranted, None otherwise
        """
        # Check if current strategy has degraded
        is_degraded, reason = self.check_degradation(current_strategy, current_regime)

        if not is_degraded:
            logger.info(f"{current_strategy} performing adequately: {reason}")
            return None

        logger.warning(f"{current_strategy} degraded: {reason}")

        # Select best alternative strategy
        best_strategy = self.select_best_strategy(available_strategies, current_regime)

        if best_strategy == current_strategy:
            logger.info(f"{current_strategy} still best option despite degradation")
            return None

        # Get metrics for both strategies
        current_metrics = self.compute_metrics(current_strategy, current_regime)
        new_metrics = self.compute_metrics(best_strategy, current_regime)

        # Calculate confidence based on performance gap
        performance_gap = new_metrics.net_pnl - current_metrics.net_pnl
        confidence = min(0.5 + performance_gap / 1000, 1.0)  # Cap at 1.0

        recommendation = SwitchRecommendation(
            current_strategy=current_strategy,
            recommended_strategy=best_strategy,
            reason=f"{reason}. {best_strategy} has better recent performance.",
            confidence=confidence,
            metrics={
                'current_win_rate': current_metrics.win_rate,
                'new_win_rate': new_metrics.win_rate,
                'current_pnl': current_metrics.net_pnl,
                'new_pnl': new_metrics.net_pnl
            }
        )

        # Record switch
        self.switch_history[best_strategy] = {
            'timestamp': datetime.now().isoformat(),
            'from_strategy': current_strategy,
            'trades_at_switch': new_metrics.num_trades
        }

        logger.info(f"SWITCH RECOMMENDED: {current_strategy} -> {best_strategy} (confidence: {confidence:.2f})")

        return recommendation


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    from active_trader_llm.memory.memory_manager import MemoryManager, TradeMemory

    memory = MemoryManager("data/test_memory.db")

    # Add some sample trades
    strategies = ['momentum_breakout', 'mean_reversion']
    for i in range(40):
        trade = TradeMemory(
            trade_id=f"trade_{i:03d}",
            timestamp=datetime.now().isoformat(),
            symbol="AAPL",
            strategy=strategies[i % 2],
            direction="long",
            entry_price=175.0,
            exit_price=175.0 + (i % 10 - 5),
            pnl=(i % 10 - 5) * 100,
            outcome="win" if i % 10 > 5 else "loss",
            context_snapshot={'regime': 'trending_bull', 'rr': 1.5 + (i % 10) / 10}
        )
        memory.add_trade(trade)

        memory.update_strategy_stats(
            trade.strategy,
            "trending_bull",
            {'pnl': trade.pnl, 'rr': 1.5 + (i % 10) / 10}
        )

    # Monitor
    config = {
        'window_trades': 20,
        'min_win_rate': 0.45,
        'min_net_pl': 100.0,
        'hysteresis_trades_before_switch': 5
    }

    monitor = StrategyMonitor(memory, config)

    recommendation = monitor.monitor_and_switch(
        'momentum_breakout',
        ['momentum_breakout', 'mean_reversion'],
        'trending_bull'
    )

    if recommendation:
        print(f"\n=== SWITCH RECOMMENDATION ===")
        print(f"From: {recommendation.current_strategy}")
        print(f"To: {recommendation.recommended_strategy}")
        print(f"Reason: {recommendation.reason}")
        print(f"Confidence: {recommendation.confidence:.2f}")
    else:
        print("\nNo switch recommended")
