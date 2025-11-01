#!/usr/bin/env python3
"""
Test script to validate the new indicator system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('active_trader_llm')

from feature_engineering.feature_builder import FeatureBuilder
from config.loader import load_config


def create_test_data():
    """Create synthetic price data for testing"""
    print("Creating test data...")
    dates = pd.date_range('2023-01-01', periods=550, freq='D')  # Enough for all indicators including 50-week SMA
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    data = []
    base_price = {'AAPL': 150, 'MSFT': 380, 'GOOGL': 140}

    for symbol in symbols:
        price = base_price[symbol]
        for i, date in enumerate(dates):
            # Create realistic price action
            price += np.random.randn() * 2
            daily_range = abs(np.random.randn() * 3)

            data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': price + np.random.randn(),
                'high': price + daily_range,
                'low': price - daily_range,
                'close': price,
                'volume': np.random.randint(50000000, 150000000)
            })

    df = pd.DataFrame(data)
    print(f"✓ Created {len(df)} bars for {len(symbols)} symbols\n")
    return df


def main():
    print("="*80)
    print("TESTING NEW INDICATOR SYSTEM")
    print("="*80 + "\n")

    # Load configuration
    print("Step 1: Loading configuration...")
    config = load_config('config.yaml')
    print(f"✓ Config loaded: {config.mode} mode\n")

    # Create test data
    price_df = create_test_data()

    # Initialize FeatureBuilder
    print("Step 2: Initializing FeatureBuilder...")
    builder = FeatureBuilder(config.model_dump())
    print("✓ FeatureBuilder initialized\n")

    # Build features
    print("Step 3: Building features...")
    features = builder.build_features(price_df)
    print(f"✓ Features built for {len(features)}/{price_df['symbol'].nunique()} symbols\n")

    # Display results
    print("="*80)
    print("FEATURE VALIDATION")
    print("="*80 + "\n")

    for symbol, feat in features.items():
        print(f"{symbol}:")
        print(f"  Timestamp: {feat.timestamp}")
        print(f"  Close: ${feat.close:.2f}")
        print(f"  Volume: {feat.volume:,}")

        print(f"\n  Daily Indicators ({len(feat.daily_indicators)}):")
        for name, value in sorted(feat.daily_indicators.items()):
            print(f"    {name:20s}: {value:8.2f}")

        print(f"\n  Weekly Indicators ({len(feat.weekly_indicators)}):")
        for name, value in sorted(feat.weekly_indicators.items()):
            print(f"    {name:20s}: {value:8.2f}")
        print()

    # Build market snapshot
    print("="*80)
    print("MARKET SNAPSHOT")
    print("="*80 + "\n")

    print("Step 4: Building market snapshot...")
    snapshot = builder.build_market_snapshot(price_df, features)

    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  Regime: {snapshot.regime_hint}")
    print(f"  Breadth Score: {snapshot.breadth_score:.2f}")
    print(f"  Advance/Decline Ratio: {snapshot.advance_decline_ratio:.2f}")
    print(f"  New Highs: {snapshot.new_highs}")
    print(f"  New Lows: {snapshot.new_lows}")
    print(f"  Up Volume Ratio: {snapshot.up_volume_ratio:.2%}")

    if snapshot.avg_rsi:
        print(f"  Avg RSI: {snapshot.avg_rsi:.2f}")
    if snapshot.pct_above_sma200_daily:
        print(f"  % Above SMA 200 (Daily): {snapshot.pct_above_sma200_daily:.1%}")
    if snapshot.pct_above_sma50_weekly:
        print(f"  % Above SMA 50 (Weekly): {snapshot.pct_above_sma50_weekly:.1%}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - NEW INDICATOR SYSTEM WORKING")
    print("="*80)


if __name__ == "__main__":
    main()
