"""
Raw Metrics Analyzer: LLM analyzes raw stock metrics without any interpretation.

NO THRESHOLDS, NO FILTERING - just raw data sent to LLM for analysis.
LLM decides which stocks are interesting based on raw metrics.
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

from .market_aggregator import StockMetrics

logger = logging.getLogger(__name__)


class StockPick(BaseModel):
    """LLM's pick of a stock with reasoning"""
    symbol: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    reasoning: str = Field(..., description="Why this stock is interesting")
    key_metrics: List[str] = Field(..., description="Key metrics that stood out")


class BatchAnalysisResult(BaseModel):
    """Result of analyzing a batch of stocks"""
    picked_symbols: List[str] = Field(..., description="Symbols the LLM picked")
    stock_picks: Dict[str, StockPick] = Field(..., description="Detailed picks")
    batch_context: str = Field(..., description="Overall context of this batch")


class RawMetricsAnalyzer:
    """
    LLM analyzes raw stock metrics without any pre-filtering or thresholds.

    The LLM receives ONLY raw calculated metrics and decides what's interesting.
    NO interpretation, NO thresholds - pure LLM autonomy.
    """

    SYSTEM_PROMPT = """You are a quantitative stock analyst reviewing raw market data.

You will receive calculated metrics for stocks. Your task is to identify which stocks
are interesting for trading based SOLELY on the raw metrics provided.

IMPORTANT:
- You receive PRE-CALCULATED metrics (not raw prices)
- You interpret these metrics to find trading opportunities
- NO thresholds are enforced - you decide what's interesting
- Consider the relative context - what stands out in THIS batch?
- Be selective - only pick stocks with genuinely interesting setups

Return ONLY valid JSON matching this schema, NO prose:
{
    "picked_symbols": ["SYMBOL1", "SYMBOL2", ...],
    "stock_picks": {
        "SYMBOL1": {
            "symbol": "SYMBOL1",
            "confidence": 0.75,
            "reasoning": "Strong momentum with +8.2% 5-day change and 4.5x volume spike",
            "key_metrics": ["5d_momentum", "volume_ratio", "ADR"]
        },
        ...
    },
    "batch_context": "This batch shows strong momentum across tech stocks with elevated volume"
}

GUIDELINES:
- Pick 3-10 stocks per batch (be selective)
- Focus on stocks that stand out from the batch
- Consider momentum, volume, volatility, and liquidity together
- Explain your reasoning clearly"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3
    ):
        """
        Initialize raw metrics analyzer.

        Args:
            api_key: OpenAI/Anthropic API key
            model: Model to use (gpt-4o, gpt-3.5-turbo, etc.)
            temperature: Temperature for LLM (0.3 for consistency)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _format_stock_metrics(self, stock: StockMetrics) -> str:
        """
        Format a single stock's metrics for the LLM.

        Args:
            stock: StockMetrics object

        Returns:
            Formatted string of metrics
        """
        # Format each metric safely
        momentum = f"{stock.price_change_5d_pct:.2f}%" if stock.price_change_5d_pct is not None else "N/A"
        volume_ratio = f"{stock.volume_ratio:.2f}x" if stock.volume_ratio is not None else "N/A"
        dist_52w = f"{stock.distance_from_52w_high_pct:.2f}%" if stock.distance_from_52w_high_pct is not None else "N/A"
        price = f"${stock.current_price:.2f}" if stock.current_price is not None else "N/A"
        liquidity = f"${stock.daily_liquidity/1e9:.2f}B" if stock.daily_liquidity is not None else "N/A"
        adr = f"{stock.adr_percent:.2f}%" if stock.adr_percent is not None else "N/A"
        volume = f"{stock.avg_volume_20d/1e6:.1f}M" if stock.avg_volume_20d is not None else "N/A"

        return f"""
{stock.symbol} ({stock.sector}):
  5-day momentum: {momentum}
  Volume ratio: {volume_ratio} (current vs 20-day avg)
  Position vs MA50: {stock.position_vs_ma50 or 'N/A'}
  Position vs MA200: {stock.position_vs_ma200 or 'N/A'}
  Distance from 52w high: {dist_52w}
  Current price: {price}
  Daily liquidity: {liquidity}
  ADR (avg daily range): {adr}
  Avg volume (20d): {volume} shares
"""

    def _build_batch_prompt(self, batch_metrics: List[StockMetrics]) -> str:
        """
        Build prompt for a batch of stocks with raw metrics.

        Args:
            batch_metrics: List of StockMetrics objects

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze these {len(batch_metrics)} stocks and identify which are interesting for trading.

RAW METRICS (pre-calculated, NOT by you):

"""
        for stock in batch_metrics:
            prompt += self._format_stock_metrics(stock)
            prompt += "\n" + "-"*60 + "\n"

        prompt += """
Based on these RAW metrics, identify which stocks are interesting for trading.
Consider what stands out in THIS batch - momentum, volume, volatility patterns, etc.

Return your analysis as JSON:"""

        return prompt

    def analyze_batch(
        self,
        batch_metrics: List[StockMetrics]
    ) -> Optional[BatchAnalysisResult]:
        """
        Analyze a batch of stocks with raw metrics (1 LLM call).

        Args:
            batch_metrics: List of StockMetrics objects (max 50)

        Returns:
            BatchAnalysisResult or None on error
        """
        if not batch_metrics:
            logger.warning("Empty batch provided")
            return None

        prompt = self._build_batch_prompt(batch_metrics)

        try:
            logger.info(f"Analyzing {len(batch_metrics)} stocks with raw metrics...")

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=3000,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract JSON from response
            content = response.choices[0].message.content

            # Try to parse JSON
            try:
                result_dict = json.loads(content)
                result = BatchAnalysisResult(**result_dict)
                logger.info(f"LLM picked {len(result.picked_symbols)}/{len(batch_metrics)} stocks")
                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {content}")
                return None

        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            return None

    def analyze_all_batches(
        self,
        all_metrics: List[StockMetrics],
        batch_size: int = 50,
        max_batches: Optional[int] = None
    ) -> List[str]:
        """
        Analyze all stocks in batches (multiple LLM calls).

        Args:
            all_metrics: List of all StockMetrics objects
            batch_size: Stocks per batch (default 50)
            max_batches: Maximum batches to process (None = all)

        Returns:
            List of all picked symbols across all batches
        """
        all_picked = []

        # Split into batches
        batches = [
            all_metrics[i:i + batch_size]
            for i in range(0, len(all_metrics), batch_size)
        ]

        # Limit to max_batches if specified
        if max_batches:
            batches = batches[:max_batches]

        logger.info(f"Analyzing {len(all_metrics)} stocks in {len(batches)} batches")
        logger.info(f"Batch size: {batch_size}, Max batches: {max_batches or 'unlimited'}")

        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {len(batch)} stocks")

            result = self.analyze_batch(batch)

            if result:
                all_picked.extend(result.picked_symbols)
                logger.info(f"Batch {i+1}: {len(result.picked_symbols)} picked")
                logger.info(f"Picked: {result.picked_symbols}")
            else:
                logger.warning(f"Batch {i+1} failed, skipping")

        logger.info(f"Total picked stocks across all batches: {len(all_picked)}")
        return all_picked


# Example usage
if __name__ == "__main__":
    import os
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    # Create sample stock metrics
    sample_metrics = [
        StockMetrics(
            symbol="AAPL",
            sector="Technology",
            price_change_5d_pct=4.5,
            volume_ratio=3.2,
            position_vs_ma50="above",
            position_vs_ma200="above",
            distance_from_52w_high_pct=-3.5,
            current_price=175.0,
            avg_volume_20d=50e6,
            daily_liquidity=175.0 * 50e6,
            adr_percent=2.5
        ),
        StockMetrics(
            symbol="TSLA",
            sector="Consumer Discretionary",
            price_change_5d_pct=8.2,
            volume_ratio=4.5,
            position_vs_ma50="above",
            position_vs_ma200="above",
            distance_from_52w_high_pct=-5.2,
            current_price=250.0,
            avg_volume_20d=100e6,
            daily_liquidity=250.0 * 100e6,
            adr_percent=5.8
        ),
        StockMetrics(
            symbol="MSFT",
            sector="Technology",
            price_change_5d_pct=3.8,
            volume_ratio=2.9,
            position_vs_ma50="above",
            position_vs_ma200="above",
            distance_from_52w_high_pct=-6.5,
            current_price=380.0,
            avg_volume_20d=25e6,
            daily_liquidity=380.0 * 25e6,
            adr_percent=2.1
        )
    ]

    analyzer = RawMetricsAnalyzer(api_key=os.getenv('OPENAI_API_KEY'))

    # Analyze batch
    result = analyzer.analyze_batch(sample_metrics)

    if result:
        print(f"\nAnalysis Result:")
        print(f"  Picked: {result.picked_symbols}")
        print(f"  Context: {result.batch_context}")
        for sym, pick in result.stock_picks.items():
            print(f"\n  {sym}:")
            print(f"    Confidence: {pick.confidence:.2f}")
            print(f"    Key metrics: {pick.key_metrics}")
            print(f"    Reasoning: {pick.reasoning}")
