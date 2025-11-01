"""
Scanner Database Manager: Manages TradableUniverse cache and scan results.

Tables:
- TradableUniverse: Cached list of tradable stocks with metadata
- ScanResults: Historical scanner outputs for learning and optimization
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TradableStock(BaseModel):
    """Tradable universe entry"""
    symbol: str
    sector: str
    market_cap: Optional[float] = None
    avg_volume_20d: Optional[float] = None
    last_price: Optional[float] = None
    optionable: bool = True
    updated_at: str


class ScanResult(BaseModel):
    """Scanner result entry"""
    scan_id: str
    timestamp: str
    stage1_guidance: Dict
    filtered_count: int
    final_candidates: List[str]
    execution_time_seconds: float
    llm_calls: int
    total_cost: float


class ScannerDB:
    """
    Manages scanner database for universe caching and scan history.

    - TradableUniverse: Cache of 5000+ stocks to avoid re-fetching
    - ScanResults: Track scanner performance over time
    """

    def __init__(self, db_path: str = "data/scanner.db"):
        """Initialize scanner database"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # TradableUniverse table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tradable_universe (
                symbol TEXT PRIMARY KEY,
                sector TEXT,
                market_cap REAL,
                avg_volume_20d REAL,
                last_price REAL,
                optionable INTEGER,
                updated_at TEXT
            )
        ''')

        # ScanResults table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_results (
                scan_id TEXT PRIMARY KEY,
                timestamp TEXT,
                stage1_guidance TEXT,
                filtered_count INTEGER,
                final_candidates TEXT,
                execution_time_seconds REAL,
                llm_calls INTEGER,
                total_cost REAL,
                created_at TEXT
            )
        ''')

        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_universe_sector ON tradable_universe(sector)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_universe_optionable ON tradable_universe(optionable)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_results(timestamp)')

        conn.commit()
        conn.close()
        logger.info("Scanner database initialized")

    def save_tradable_universe(self, stocks: List[TradableStock]):
        """
        Save or update tradable universe.

        Args:
            stocks: List of TradableStock objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for stock in stocks:
            cursor.execute('''
                INSERT OR REPLACE INTO tradable_universe
                (symbol, sector, market_cap, avg_volume_20d, last_price, optionable, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                stock.symbol,
                stock.sector,
                stock.market_cap,
                stock.avg_volume_20d,
                stock.last_price,
                1 if stock.optionable else 0,
                stock.updated_at
            ))

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(stocks)} stocks to tradable universe")

    def get_tradable_universe(
        self,
        sector: Optional[str] = None,
        optionable_only: bool = True,
        min_volume: Optional[float] = None
    ) -> List[TradableStock]:
        """
        Retrieve tradable universe with optional filters.

        Args:
            sector: Filter by sector
            optionable_only: Only optionable stocks
            min_volume: Minimum 20-day average volume

        Returns:
            List of TradableStock objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM tradable_universe WHERE 1=1"
        params = []

        if sector:
            query += " AND sector = ?"
            params.append(sector)

        if optionable_only:
            query += " AND optionable = 1"

        if min_volume:
            query += " AND avg_volume_20d >= ?"
            params.append(min_volume)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        stocks = []
        for row in rows:
            stocks.append(TradableStock(
                symbol=row[0],
                sector=row[1],
                market_cap=row[2],
                avg_volume_20d=row[3],
                last_price=row[4],
                optionable=bool(row[5]),
                updated_at=row[6]
            ))

        return stocks

    def get_universe_age_hours(self) -> Optional[float]:
        """
        Check how old the cached universe is.

        Returns:
            Age in hours, or None if no universe cached
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT MAX(updated_at) FROM tradable_universe')
        row = cursor.fetchone()
        conn.close()

        if not row or not row[0]:
            return None

        last_update = datetime.fromisoformat(row[0])
        age = (datetime.now() - last_update).total_seconds() / 3600
        return age

    def save_scan_result(self, scan_result: ScanResult):
        """
        Save scan result for historical tracking.

        Args:
            scan_result: ScanResult object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO scan_results
            (scan_id, timestamp, stage1_guidance, filtered_count, final_candidates,
             execution_time_seconds, llm_calls, total_cost, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scan_result.scan_id,
            scan_result.timestamp,
            json.dumps(scan_result.stage1_guidance),
            scan_result.filtered_count,
            json.dumps(scan_result.final_candidates),
            scan_result.execution_time_seconds,
            scan_result.llm_calls,
            scan_result.total_cost,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()
        logger.info(f"Saved scan result {scan_result.scan_id}")

    def get_recent_scans(self, limit: int = 10) -> List[ScanResult]:
        """
        Retrieve recent scan results.

        Args:
            limit: Number of recent scans to retrieve

        Returns:
            List of ScanResult objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM scan_results
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append(ScanResult(
                scan_id=row[0],
                timestamp=row[1],
                stage1_guidance=json.loads(row[2]),
                filtered_count=row[3],
                final_candidates=json.loads(row[4]),
                execution_time_seconds=row[5],
                llm_calls=row[6],
                total_cost=row[7]
            ))

        return results

    def get_scan_statistics(self, days: int = 30) -> Dict:
        """
        Calculate scanner performance statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT
                COUNT(*) as total_scans,
                AVG(execution_time_seconds) as avg_execution_time,
                AVG(llm_calls) as avg_llm_calls,
                SUM(total_cost) as total_cost,
                AVG(filtered_count) as avg_filtered_count
            FROM scan_results
            WHERE timestamp >= ?
        ''', (cutoff,))

        row = cursor.fetchone()
        conn.close()

        return {
            'total_scans': row[0] or 0,
            'avg_execution_time_seconds': row[1] or 0.0,
            'avg_llm_calls': row[2] or 0.0,
            'total_cost_dollars': row[3] or 0.0,
            'avg_candidates_found': row[4] or 0.0
        }

    def clear_universe(self):
        """Clear tradable universe cache (for refresh)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM tradable_universe')
        conn.commit()
        conn.close()
        logger.info("Cleared tradable universe cache")


# Example usage
if __name__ == "__main__":
    # IMPORTANT: This example code seeds SAMPLE DATA for demonstration purposes only.
    # This code is safely guarded by __name__ == "__main__" and will NEVER run in
    # production workflows. It only executes when this file is run directly for testing.
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)

    db = ScannerDB()

    # Save sample tradable universe (EXAMPLE DATA ONLY - NOT FOR PRODUCTION)
    sample_stocks = [
        TradableStock(
            symbol="AAPL",
            sector="Technology",
            market_cap=2.5e12,
            avg_volume_20d=50e6,
            last_price=175.0,
            optionable=True,
            updated_at=datetime.now().isoformat()
        ),
        TradableStock(
            symbol="MSFT",
            sector="Technology",
            market_cap=2.3e12,
            avg_volume_20d=25e6,
            last_price=350.0,
            optionable=True,
            updated_at=datetime.now().isoformat()
        )
    ]

    db.save_tradable_universe(sample_stocks)

    # Retrieve universe
    universe = db.get_tradable_universe(sector="Technology")
    print(f"\nFound {len(universe)} Technology stocks")

    # Check age
    age = db.get_universe_age_hours()
    print(f"Universe cache age: {age:.2f} hours")

    # Save scan result
    scan = ScanResult(
        scan_id="scan_001",
        timestamp=datetime.now().isoformat(),
        stage1_guidance={"market_bias": "bullish", "focus_sectors": ["Technology"]},
        filtered_count=150,
        final_candidates=["AAPL", "MSFT", "NVDA"],
        execution_time_seconds=180.5,
        llm_calls=7,
        total_cost=0.16
    )

    db.save_scan_result(scan)

    # Get statistics
    stats = db.get_scan_statistics(days=30)
    print(f"\nScanner stats: {stats}")
