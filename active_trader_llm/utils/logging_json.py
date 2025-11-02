"""
JSON logging utilities for decisions and outcomes.

All trading decisions, executions, and outcomes are logged in structured JSON format
for transparency, analysis, and debugging.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class JSONLogger:
    """
    Structured JSON logging for trading decisions and outcomes.

    Writes newline-delimited JSON (JSONL) format for easy parsing and analysis.
    """

    def __init__(self, log_path: str = "logs/trade_log.jsonl"):
        """
        Initialize JSON logger.

        Args:
            log_path: Path to JSONL log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"JSONLogger initialized: {self.log_path}")

    def _write_entry(self, entry: Dict[str, Any]):
        """Write a single JSON entry to log file"""
        with open(self.log_path, 'a') as f:
            json.dump(entry, f)
            f.write('\n')

    def log_decision(
        self,
        trade_id: str,
        timestamp: str,
        symbol: str,
        analyst_outputs: Dict,
        researcher_outputs: Dict,
        trade_plan: Dict,
        risk_decision: Dict
    ):
        """
        Log a trading decision with full context.

        Args:
            trade_id: Unique trade identifier
            timestamp: Decision timestamp
            symbol: Stock symbol
            analyst_outputs: All analyst outputs
            researcher_outputs: Bull/bear debate
            trade_plan: Proposed trade plan
            risk_decision: Risk manager decision
        """
        entry = {
            'type': 'decision',
            'trade_id': trade_id,
            'timestamp': timestamp,
            'symbol': symbol,
            'analyst_outputs': analyst_outputs,
            'researcher_outputs': researcher_outputs,
            'trade_plan': trade_plan,
            'risk_decision': risk_decision
        }

        self._write_entry(entry)
        logger.info(f"Logged decision for {symbol} (trade_id: {trade_id})")

    def log_execution(
        self,
        trade_id: str,
        timestamp: str,
        symbol: str,
        direction: str,
        filled_price: float,
        filled_qty: float,
        slippage: float,
        execution_method: str = "paper",
        broker_order_id: str = None,
        order_status: str = None
    ):
        """
        Log trade execution details.

        Args:
            trade_id: Trade identifier
            timestamp: Execution timestamp
            symbol: Stock symbol
            direction: long/short
            filled_price: Actual fill price
            filled_qty: Filled quantity
            slippage: Price slippage from expected entry
            execution_method: paper/live/alpaca_paper/alpaca_live
            broker_order_id: Broker's order ID (for live trading)
            order_status: Order status from broker (filled/partial/pending)
        """
        entry = {
            'type': 'execution',
            'trade_id': trade_id,
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'filled_price': filled_price,
            'filled_qty': filled_qty,
            'slippage': slippage,
            'execution_method': execution_method
        }

        # Add broker-specific fields if provided
        if broker_order_id:
            entry['broker_order_id'] = broker_order_id
        if order_status:
            entry['order_status'] = order_status

        self._write_entry(entry)
        logger.info(f"Logged execution for {trade_id}: {direction} {symbol} @ {filled_price}")

    def log_outcome(
        self,
        trade_id: str,
        timestamp: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        return_pct: float,
        duration_hours: float,
        exit_reason: str,
        lessons: list
    ):
        """
        Log trade outcome and learnings.

        Args:
            trade_id: Trade identifier
            timestamp: Exit timestamp
            symbol: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss in dollars
            return_pct: Return percentage
            duration_hours: Trade duration in hours
            exit_reason: Why trade was closed (target/stop/manual)
            lessons: List of lessons learned
        """
        entry = {
            'type': 'outcome',
            'trade_id': trade_id,
            'timestamp': timestamp,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'return_pct': return_pct,
            'duration_hours': duration_hours,
            'exit_reason': exit_reason,
            'lessons': lessons
        }

        self._write_entry(entry)
        logger.info(f"Logged outcome for {trade_id}: {exit_reason}, P/L ${pnl:.2f}")

    def log_strategy_switch(
        self,
        timestamp: str,
        old_strategy: str,
        new_strategy: str,
        regime: str,
        reason: str,
        metrics: Dict
    ):
        """
        Log strategy switching events.

        Args:
            timestamp: Switch timestamp
            old_strategy: Previous strategy
            new_strategy: New strategy
            regime: Current market regime
            reason: Reason for switch
            metrics: Performance metrics leading to switch
        """
        entry = {
            'type': 'strategy_switch',
            'timestamp': timestamp,
            'old_strategy': old_strategy,
            'new_strategy': new_strategy,
            'regime': regime,
            'reason': reason,
            'metrics': metrics
        }

        self._write_entry(entry)
        logger.info(f"Logged strategy switch: {old_strategy} -> {new_strategy}")

    def log_error(
        self,
        timestamp: str,
        component: str,
        error_type: str,
        error_message: str,
        context: Dict
    ):
        """
        Log errors and exceptions.

        Args:
            timestamp: Error timestamp
            component: Component where error occurred
            error_type: Error type/class
            error_message: Error message
            context: Additional context
        """
        entry = {
            'type': 'error',
            'timestamp': timestamp,
            'component': component,
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        }

        self._write_entry(entry)
        logger.error(f"Logged error in {component}: {error_message}")

    def read_logs(self, entry_type: str = None, limit: int = 100) -> list:
        """
        Read log entries from file.

        Args:
            entry_type: Filter by entry type (decision/execution/outcome/etc)
            limit: Maximum entries to return

        Returns:
            List of log entries
        """
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)

                    if entry_type is None or entry.get('type') == entry_type:
                        entries.append(entry)

                        if len(entries) >= limit:
                            break

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse log line: {line[:100]}")

        return entries


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    json_logger = JSONLogger("logs/test_trade_log.jsonl")

    # Log a decision
    json_logger.log_decision(
        trade_id="trade_001",
        timestamp=datetime.now().isoformat(),
        symbol="AAPL",
        analyst_outputs={
            'technical': {'signal': 'long', 'confidence': 0.7}
        },
        researcher_outputs={
            'bull': {'thesis': ['Momentum strong'], 'confidence': 0.75},
            'bear': {'thesis': ['Near resistance'], 'confidence': 0.45}
        },
        trade_plan={
            'strategy': 'momentum_breakout',
            'direction': 'long',
            'entry': 175.0
        },
        risk_decision={
            'approved': True,
            'modifications': {}
        }
    )

    # Log execution
    json_logger.log_execution(
        trade_id="trade_001",
        timestamp=datetime.now().isoformat(),
        symbol="AAPL",
        direction="long",
        filled_price=175.05,
        filled_qty=100,
        slippage=0.05
    )

    # Log outcome
    json_logger.log_outcome(
        trade_id="trade_001",
        timestamp=datetime.now().isoformat(),
        symbol="AAPL",
        entry_price=175.05,
        exit_price=178.20,
        pnl=315.0,
        return_pct=1.8,
        duration_hours=48,
        exit_reason="take_profit",
        lessons=["Momentum strategy worked well in trending_bull regime"]
    )

    # Read logs
    decisions = json_logger.read_logs(entry_type='decision')
    print(f"\nRead {len(decisions)} decision entries")
