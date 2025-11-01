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
    Analyzes sentiment from news/social media (optional, stub for now).

    Can be extended with FinBERT, Reddit/Twitter APIs, or news aggregators.
    """

    def __init__(self):
        """Initialize sentiment analyst"""
        logger.info("SentimentAnalyst initialized (stub mode)")

    def analyze(self, symbol: str, sentiment_data: Optional[Dict] = None) -> SentimentSignal:
        """
        Analyze sentiment for a symbol.

        Args:
            symbol: Stock symbol
            sentiment_data: Optional pre-fetched sentiment data
                Required keys: 'sentiment_score' (float, -1 to 1)
                Optional keys: 'news_sentiment', 'social_sentiment', 'sources'

        Returns:
            SentimentSignal with sentiment score

        Raises:
            ValueError: If sentiment_data is missing (STUB MODE - REQUIRES REAL DATA)
        """
        # CRITICAL: This is a STUB implementation that requires real sentiment data
        # In production, integrate with:
        # - FinBERT for news analysis
        # - Reddit/Twitter APIs for social sentiment
        # - News aggregators (Benzinga, NewsAPI, etc.)

        if not sentiment_data:
            logger.error(f"❌ SENTIMENT ANALYST STUB: No sentiment_data provided for {symbol}. This analyst REQUIRES live data integration.")
            logger.error("   To fix: Integrate FinBERT, NewsAPI, or Reddit sentiment and pass via sentiment_data parameter")
            raise ValueError(f"SentimentAnalyst requires live sentiment_data for {symbol}. Cannot operate in stub mode without data.")

        sentiment_score = sentiment_data.get('sentiment_score')

        # Validate sentiment_score is present and in valid range
        if sentiment_score is None:
            logger.error(f"❌ SENTIMENT ANALYST: 'sentiment_score' key missing from sentiment_data for {symbol}")
            raise ValueError("sentiment_data must include 'sentiment_score' key with value between -1.0 and 1.0")

        if not isinstance(sentiment_score, (int, float)) or sentiment_score < -1.0 or sentiment_score > 1.0:
            logger.error(f"❌ SENTIMENT ANALYST: Invalid sentiment_score {sentiment_score} for {symbol} (must be -1.0 to 1.0)")
            raise ValueError(f"Invalid sentiment_score: {sentiment_score}. Must be float between -1.0 and 1.0")

        # LIVE DATA VALIDATION: Check if sentiment looks like default/fake data
        if sentiment_score == 0.0:
            logger.warning(f"⚠️  SENTIMENT ANALYST: Sentiment=0.0 exactly for {symbol} - this may be default/fake data")

        # Extract drivers or use generic
        drivers = sentiment_data.get('drivers', [])
        if not drivers:
            drivers = [f"sentiment_score: {sentiment_score:.2f}"]
            if 'news_sentiment' in sentiment_data:
                drivers.append(f"news: {sentiment_data['news_sentiment']:.2f}")
            if 'social_sentiment' in sentiment_data:
                drivers.append(f"social: {sentiment_data['social_sentiment']:.2f}")

        # Calculate confidence based on signal strength
        confidence = min(0.3 + abs(sentiment_score) * 0.5, 0.9)

        logger.info(f"Sentiment analysis for {symbol}: score={sentiment_score:.2f}, confidence={confidence:.2f}")

        return SentimentSignal(
            symbol=symbol,
            sentiment_score=sentiment_score,
            confidence=confidence,
            drivers=drivers
        )

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
