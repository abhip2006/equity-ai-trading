"""
Sentiment Analyst: Analyzes news and social sentiment (optional module).
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SentimentSignal(BaseModel):
    """Sentiment analyst output schema"""
    analyst: str = "Sentiment"
    symbol: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    drivers: List[str]


class SentimentAnalyst:
    """
    Analyzes sentiment from news/social media (optional).

    IMPORTANT: This is currently a stub. Do NOT enable sentiment analysis in production
    until a real data source is configured. This analyzer will refuse to return
    fabricated neutral sentiment to prevent misleading the trading system.

    To implement:
    - Integrate FinBERT for news analysis
    - Connect Reddit/Twitter APIs for social sentiment
    - Use news aggregators (Benzinga, NewsAPI, etc.)
    """

    def __init__(self, enabled: bool = False):
        """
        Initialize sentiment analyst.

        Args:
            enabled: Must be explicitly set to True to enable (not recommended without real data source)
        """
        if enabled:
            logger.warning("SentimentAnalyst enabled but no real data source configured!")
            logger.warning("This analyzer is a STUB and should NOT be used in production")
        logger.info("SentimentAnalyst initialized (stub mode - will not return fabricated data)")

    def analyze(self, symbol: str, sentiment_data: Optional[Dict] = None) -> Optional[SentimentSignal]:
        """
        Analyze sentiment for a symbol.

        Args:
            symbol: Stock symbol
            sentiment_data: Optional pre-fetched sentiment data (REQUIRED for real analysis)

        Returns:
            None - this stub will not fabricate sentiment data
        """
        # STUB: Refuse to return fabricated sentiment
        # In production, integrate with real data sources:
        # - FinBERT for news analysis
        # - Reddit/Twitter APIs for social sentiment
        # - News aggregators (Benzinga, NewsAPI, etc.)

        logger.error(f"Sentiment analysis requested for {symbol} but no real data source configured")
        logger.error("SentimentAnalyst is a stub and will not fabricate neutral sentiment")
        logger.error("Please disable sentiment analysis or implement a real data source")

        # Return None instead of fabricated neutral sentiment
        return None

    def analyze_batch(self, symbols: List[str]) -> Dict[str, SentimentSignal]:
        """Analyze sentiment for multiple symbols"""
        return {symbol: self.analyze(symbol) for symbol in symbols}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyst = SentimentAnalyst()
    signal = analyst.analyze("AAPL")

    print(f"\nSymbol: {signal.symbol}")
    print(f"Sentiment Score: {signal.sentiment_score:.2f}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Drivers: {signal.drivers}")
