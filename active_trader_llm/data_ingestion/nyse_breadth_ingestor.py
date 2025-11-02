"""
NYSE Breadth Data Ingestor: Fetches NYSE market breadth from Unicorn Research.

Unicorn Research provides free NYSE breadth data (EOD) since 2002.
Data includes: Advancing/Declining issues, Volume, New Highs/Lows.

Source: http://unicorn.us.com/trading/allbreadth.csv
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import requests
import csv
from io import StringIO
from pathlib import Path

logger = logging.getLogger(__name__)


class NYSEBreadthIngestor:
    """
    Fetches NYSE market breadth data from Unicorn Research.

    Provides free daily breadth data including:
    - Advancing/Declining issues (since 2002)
    - Advancing/Declining volume (since 2002)
    - New 52-week highs/lows (since 2005)

    NOTE: Data is EOD only (no intraday TICK or TRIN).
    """

    # Unicorn Research data URL
    DATA_URL = "http://unicorn.us.com/trading/allbreadth.csv"

    def __init__(self, cache_dir: str = "data/breadth_cache", cache_duration_hours: int = 24):
        """
        Initialize NYSE breadth ingestor.

        Args:
            cache_dir: Directory to cache breadth data
            cache_duration_hours: Cache duration in hours (default 24h for EOD data)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "nyse_breadth.csv"
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.cache_file.exists():
            return False
        if self._cache is None or self._cache_timestamp is None:
            return False
        return datetime.now() - self._cache_timestamp < self.cache_duration

    def _download_breadth_data(self) -> Optional[str]:
        """
        Download breadth CSV from Unicorn Research.

        Returns:
            CSV content as string, or None if download fails
        """
        try:
            logger.info(f"Downloading NYSE breadth data from {self.DATA_URL}")
            response = requests.get(self.DATA_URL, timeout=30)
            response.raise_for_status()

            # Save to cache
            with open(self.cache_file, 'w') as f:
                f.write(response.text)

            logger.info("NYSE breadth data downloaded successfully")
            return response.text

        except Exception as e:
            logger.error(f"Failed to download NYSE breadth data: {e}")
            return None

    def _parse_breadth_csv(self, csv_content: str) -> Optional[Dict]:
        """
        Parse breadth CSV and extract latest data.

        CSV format from Unicorn Research:
        Date,NYSE Advancing,NYSE Declining,NYSE Unchanged,NYSE Adv Vol,NYSE Decl Vol,...

        Returns:
            Dict with latest breadth data, or None if parsing fails
        """
        try:
            # Parse CSV
            csv_reader = csv.DictReader(StringIO(csv_content))
            rows = list(csv_reader)

            if not rows:
                logger.error("No data in breadth CSV")
                return None

            # Get most recent row (last row)
            latest = rows[-1]

            # Extract values (handle potential missing/empty values)
            def safe_int(value: str) -> Optional[int]:
                try:
                    return int(value.replace(',', '')) if value else None
                except (ValueError, AttributeError):
                    return None

            advancing = safe_int(latest.get('NYSE Advancing', ''))
            declining = safe_int(latest.get('NYSE Declining', ''))
            unchanged = safe_int(latest.get('NYSE Unchanged', ''))
            adv_volume = safe_int(latest.get('NYSE Adv Vol', ''))
            decl_volume = safe_int(latest.get('NYSE Decl Vol', ''))
            new_highs = safe_int(latest.get('NYSE New Highs', ''))
            new_lows = safe_int(latest.get('NYSE New Lows', ''))

            # Calculate ratios (simple math, no interpretation)
            advance_decline_ratio = None
            if advancing is not None and declining is not None and declining > 0:
                advance_decline_ratio = advancing / declining

            up_volume_ratio = None
            if adv_volume is not None and decl_volume is not None:
                total_volume = adv_volume + decl_volume
                if total_volume > 0:
                    up_volume_ratio = adv_volume / total_volume

            breadth_data = {
                'date': latest.get('Date', ''),
                'nyse_advancing': advancing,
                'nyse_declining': declining,
                'nyse_unchanged': unchanged,
                'nyse_advancing_volume': adv_volume,
                'nyse_declining_volume': decl_volume,
                'nyse_new_highs': new_highs,
                'nyse_new_lows': new_lows,
                'advance_decline_ratio': advance_decline_ratio,
                'up_volume_ratio': up_volume_ratio,
            }

            logger.info(f"Parsed breadth data for {breadth_data['date']}")
            logger.info(f"  A/D: {advancing}/{declining} (ratio: {advance_decline_ratio:.2f if advance_decline_ratio else 'N/A'})")

            return breadth_data

        except Exception as e:
            logger.error(f"Failed to parse breadth CSV: {e}")
            return None

    def fetch_latest_breadth(self, use_cache: bool = True) -> Dict[str, Optional[int]]:
        """
        Fetch latest NYSE breadth data.

        Args:
            use_cache: Use cached data if available (default True)

        Returns:
            Dict with breadth data (values may be None if unavailable)
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            logger.info("Using cached NYSE breadth data")
            return self._cache

        # Try to download fresh data
        csv_content = self._download_breadth_data()

        # If download failed, try to load from cache file
        if csv_content is None and self.cache_file.exists():
            logger.warning("Download failed, attempting to use cached file")
            try:
                with open(self.cache_file, 'r') as f:
                    csv_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read cache file: {e}")
                csv_content = None

        # If still no data, return empty dict
        if csv_content is None:
            logger.error("No NYSE breadth data available (download and cache both failed)")
            return {
                'nyse_advancing': None,
                'nyse_declining': None,
                'nyse_unchanged': None,
                'nyse_advancing_volume': None,
                'nyse_declining_volume': None,
                'nyse_new_highs': None,
                'nyse_new_lows': None,
                'advance_decline_ratio': None,
                'up_volume_ratio': None,
            }

        # Parse CSV
        breadth_data = self._parse_breadth_csv(csv_content)

        if breadth_data is None:
            logger.error("Failed to parse NYSE breadth data")
            return {
                'nyse_advancing': None,
                'nyse_declining': None,
                'nyse_unchanged': None,
                'nyse_advancing_volume': None,
                'nyse_declining_volume': None,
                'nyse_new_highs': None,
                'nyse_new_lows': None,
                'advance_decline_ratio': None,
                'up_volume_ratio': None,
            }

        # Update cache
        self._cache = breadth_data
        self._cache_timestamp = datetime.now()

        # Log available fields
        available = sum(1 for v in breadth_data.values() if v is not None and not isinstance(v, str))
        logger.info(f"NYSE breadth data fetched: {available}/9 fields available")

        return breadth_data


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestor = NYSEBreadthIngestor()
    breadth = ingestor.fetch_latest_breadth()

    print("\n=== NYSE Breadth Data ===")
    print(f"Date: {breadth.get('date', 'N/A')}")
    print(f"\nAdvancing/Declining:")
    print(f"  Advancing: {breadth.get('nyse_advancing', 'N/A'):,}" if breadth.get('nyse_advancing') else "  Advancing: N/A")
    print(f"  Declining: {breadth.get('nyse_declining', 'N/A'):,}" if breadth.get('nyse_declining') else "  Declining: N/A")
    print(f"  Unchanged: {breadth.get('nyse_unchanged', 'N/A'):,}" if breadth.get('nyse_unchanged') else "  Unchanged: N/A")
    print(f"  A/D Ratio: {breadth.get('advance_decline_ratio', 'N/A'):.2f}" if breadth.get('advance_decline_ratio') else "  A/D Ratio: N/A")
    print(f"\nVolume:")
    print(f"  Advancing Vol: {breadth.get('nyse_advancing_volume', 'N/A'):,}" if breadth.get('nyse_advancing_volume') else "  Advancing Vol: N/A")
    print(f"  Declining Vol: {breadth.get('nyse_declining_volume', 'N/A'):,}" if breadth.get('nyse_declining_volume') else "  Declining Vol: N/A")
    print(f"  Up Vol Ratio: {breadth.get('up_volume_ratio', 'N/A'):.2f}" if breadth.get('up_volume_ratio') else "  Up Vol Ratio: N/A")
    print(f"\n52-Week:")
    print(f"  New Highs: {breadth.get('nyse_new_highs', 'N/A'):,}" if breadth.get('nyse_new_highs') else "  New Highs: N/A")
    print(f"  New Lows: {breadth.get('nyse_new_lows', 'N/A'):,}" if breadth.get('nyse_new_lows') else "  New Lows: N/A")
