#!/usr/bin/env python3
"""
Test script for battle API endpoints
Creates sample data and tests all endpoints
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from battle_system import BattleOrchestrator, ModelConfig, ModelStatus, Position, Trade
from datetime import datetime, timedelta
import uuid

def setup_sample_data():
    """Create sample battle data for testing"""
    print("Setting up sample battle data...")

    db_path = Path(__file__).parent.parent / "data" / "battle.db"
    orchestrator = BattleOrchestrator(db_path)

    # Register sample models
    models = [
        ModelConfig(
            model_id="gpt4",
            display_name="GPT-4 Turbo",
            provider="openai",
            model_name="gpt-4-turbo-preview",
            initial_capital=100000.0,
            status=ModelStatus.ACTIVE,
            created_at=datetime.now().isoformat(),
            metadata={"temperature": 0.3}
        ),
        ModelConfig(
            model_id="claude3",
            display_name="Claude 3 Opus",
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            initial_capital=100000.0,
            status=ModelStatus.ACTIVE,
            created_at=datetime.now().isoformat(),
            metadata={"temperature": 0.3}
        ),
        ModelConfig(
            model_id="gpt35",
            display_name="GPT-3.5 Turbo",
            provider="openai",
            model_name="gpt-3.5-turbo",
            initial_capital=100000.0,
            status=ModelStatus.ACTIVE,
            created_at=datetime.now().isoformat(),
            metadata={"temperature": 0.3}
        ),
    ]

    for model in models:
        orchestrator.register_model(model)
        print(f"  Registered: {model.display_name}")

    # Add sample trades
    conn = orchestrator.db.get_connection()
    cursor = conn.cursor()

    # Sample trades for GPT-4
    trades_gpt4 = [
        ("trade_1", "gpt4", "AAPL", "long", 100, 150.0, 155.0, 500.0,
         (datetime.now() - timedelta(days=5)).isoformat(),
         (datetime.now() - timedelta(days=4)).isoformat(),
         "take_profit", "Strong momentum breakout"),
        ("trade_2", "gpt4", "MSFT", "long", 50, 380.0, 375.0, -250.0,
         (datetime.now() - timedelta(days=3)).isoformat(),
         (datetime.now() - timedelta(days=2)).isoformat(),
         "stop_loss", "Failed breakdown"),
        ("trade_3", "gpt4", "NVDA", "long", 30, 800.0, 820.0, 600.0,
         (datetime.now() - timedelta(days=2)).isoformat(),
         (datetime.now() - timedelta(days=1)).isoformat(),
         "take_profit", "AI sector strength"),
    ]

    # Sample trades for Claude
    trades_claude = [
        ("trade_4", "claude3", "GOOGL", "long", 75, 140.0, 145.0, 375.0,
         (datetime.now() - timedelta(days=5)).isoformat(),
         (datetime.now() - timedelta(days=4)).isoformat(),
         "take_profit", "Revenue beat expectations"),
        ("trade_5", "claude3", "TSLA", "long", 40, 200.0, 195.0, -200.0,
         (datetime.now() - timedelta(days=3)).isoformat(),
         (datetime.now() - timedelta(days=2)).isoformat(),
         "stop_loss", "Production miss"),
        ("trade_6", "claude3", "META", "long", 25, 480.0, 490.0, 250.0,
         (datetime.now() - timedelta(days=2)).isoformat(),
         (datetime.now() - timedelta(days=1)).isoformat(),
         "take_profit", "User growth acceleration"),
        ("trade_7", "claude3", "AMD", "long", 100, 150.0, 155.0, 500.0,
         (datetime.now() - timedelta(days=1)).isoformat(),
         datetime.now().isoformat(),
         "take_profit", "Datacenter demand"),
    ]

    # Sample trades for GPT-3.5
    trades_gpt35 = [
        ("trade_8", "gpt35", "AMZN", "long", 30, 175.0, 172.0, -90.0,
         (datetime.now() - timedelta(days=4)).isoformat(),
         (datetime.now() - timedelta(days=3)).isoformat(),
         "stop_loss", "Weak AWS growth"),
        ("trade_9", "gpt35", "SPY", "long", 50, 450.0, 455.0, 250.0,
         (datetime.now() - timedelta(days=2)).isoformat(),
         (datetime.now() - timedelta(days=1)).isoformat(),
         "take_profit", "Market breadth improving"),
    ]

    all_trades = trades_gpt4 + trades_claude + trades_gpt35

    for trade in all_trades:
        cursor.execute("""
            INSERT INTO battle_trades
            (trade_id, model_id, symbol, direction, shares, entry_price, exit_price,
             realized_pnl, opened_at, closed_at, exit_reason, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, trade)

    # Add some open positions
    positions = [
        ("gpt4", "QQQ", "long", 100, 390.0, 385.0, 400.0, datetime.now().isoformat()),
        ("claude3", "NFLX", "long", 50, 600.0, 590.0, 620.0, datetime.now().isoformat()),
        ("gpt35", "DIS", "long", 75, 95.0, 92.0, 100.0, datetime.now().isoformat()),
    ]

    for pos in positions:
        cursor.execute("""
            INSERT INTO battle_positions
            (model_id, symbol, direction, shares, entry_price, stop_loss, take_profit, opened_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, pos)

    # Add sample decisions for comparison and history
    now = datetime.now()
    decisions = [
        # Recent decisions (for comparison endpoint)
        ("gpt4", "AAPL", now.isoformat(), "BUY", "Strong technical setup", 0.85),
        ("claude3", "AAPL", now.isoformat(), "HOLD", "Wait for pullback", 0.65),
        ("gpt35", "AAPL", now.isoformat(), "BUY", "Momentum positive", 0.75),

        # Historical decisions for gpt4
        ("gpt4", "TSLA", (now - timedelta(hours=2)).isoformat(), "BUY", "Breakout above resistance", 0.90),
        ("gpt4", "MSFT", (now - timedelta(hours=4)).isoformat(), "SELL", "Overbought conditions", 0.78),
        ("gpt4", "GOOGL", (now - timedelta(hours=6)).isoformat(), "HOLD", "Consolidation pattern", 0.60),
        ("gpt4", "NVDA", (now - timedelta(hours=8)).isoformat(), "BUY", "AI sector strength", 0.88),
        ("gpt4", "META", (now - timedelta(days=1)).isoformat(), "SELL", "Weak earnings guidance", 0.82),

        # Historical decisions for claude3
        ("claude3", "TSLA", (now - timedelta(hours=2)).isoformat(), "SELL", "Extended move", 0.75),
        ("claude3", "MSFT", (now - timedelta(hours=4)).isoformat(), "HOLD", "Range-bound", 0.55),
        ("claude3", "GOOGL", (now - timedelta(hours=6)).isoformat(), "BUY", "Support level holding", 0.70),
        ("claude3", "NVDA", (now - timedelta(hours=8)).isoformat(), "BUY", "Chip demand strong", 0.85),

        # Historical decisions for gpt35
        ("gpt35", "TSLA", (now - timedelta(hours=2)).isoformat(), "HOLD", "Mixed signals", 0.50),
        ("gpt35", "MSFT", (now - timedelta(hours=4)).isoformat(), "BUY", "Cloud growth", 0.72),
        ("gpt35", "GOOGL", (now - timedelta(hours=6)).isoformat(), "HOLD", "Neutral outlook", 0.58),
    ]

    for dec in decisions:
        cursor.execute("""
            INSERT INTO battle_decisions
            (model_id, symbol, timestamp, decision, reasoning, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, dec)

    conn.commit()
    conn.close()

    print("\n Sample data created successfully!")
    print(f"  - 3 models registered")
    print(f"  - {len(all_trades)} trades added")
    print(f"  - {len(positions)} open positions")
    print(f"  - {len(decisions)} decisions logged")

    return orchestrator


def test_endpoints():
    """Test all battle endpoints"""
    import requests
    import json

    base_url = "http://localhost:8000"

    print("\n" + "="*60)
    print("Testing Battle API Endpoints")
    print("="*60)

    # Test 1: System Status
    print("\n[1] Testing /api/battle/status")
    try:
        resp = requests.get(f"{base_url}/api/battle/status")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Status: {data['status']}")
            print(f"   Active models: {data['active_models']}")
            print(f"   Total models: {data['total_models']}")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 2: Models List
    print("\n[2] Testing /api/battle/models")
    try:
        resp = requests.get(f"{base_url}/api/battle/models")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Found {data['count']} models:")
            for model in data['models']:
                print(f"     - {model['display_name']} ({model['model_id']})")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 3: Leaderboard
    print("\n[3] Testing /api/battle/leaderboard")
    try:
        resp = requests.get(f"{base_url}/api/battle/leaderboard?timeframe=all")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Leaderboard ({data['timeframe']}):")
            for entry in data['rankings']:
                print(f"     #{entry['rank']} {entry['display_name']}: ${entry['total_pnl']:.2f} ({entry['win_rate']:.1f}% WR)")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 4: Model Positions
    print("\n[4] Testing /api/battle/model/gpt4/positions")
    try:
        resp = requests.get(f"{base_url}/api/battle/model/gpt4/positions")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   GPT-4 has {data['count']} open positions:")
            for pos in data['positions']:
                print(f"     - {pos['symbol']}: {pos['shares']} shares @ ${pos['entry_price']}")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 5: Model Trades
    print("\n[5] Testing /api/battle/model/claude3/trades")
    try:
        resp = requests.get(f"{base_url}/api/battle/model/claude3/trades?limit=5")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Claude 3 has {data['count']} trades:")
            for trade in data['trades'][:3]:
                print(f"     - {trade['symbol']}: ${trade['realized_pnl']:.2f} ({trade['exit_reason']})")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 6: Model Metrics
    print("\n[6] Testing /api/battle/model/gpt4/metrics")
    try:
        resp = requests.get(f"{base_url}/api/battle/model/gpt4/metrics")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   GPT-4 Metrics:")
            print(f"     Total P&L: ${data['total_pnl']:.2f} ({data['pnl_pct']:.2f}%)")
            print(f"     Win Rate: {data['win_rate']:.2f}%")
            print(f"     Sharpe: {data['sharpe_ratio']:.2f}")
            print(f"     Max DD: {data['max_drawdown']:.2f}%")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 7: Equity Curve
    print("\n[7] Testing /api/battle/model/claude3/equity_curve")
    try:
        resp = requests.get(f"{base_url}/api/battle/model/claude3/equity_curve")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Claude 3 equity curve has {len(data['equity_curve'])} points")
            if len(data['equity_curve']) > 0:
                print(f"     Start: ${data['equity_curve'][0]['equity']:.2f}")
                print(f"     End: ${data['equity_curve'][-1]['equity']:.2f}")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 8: Compare Positions
    print("\n[8] Testing /api/battle/compare/positions")
    try:
        resp = requests.get(f"{base_url}/api/battle/compare/positions")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Comparing positions across models:")
            for model_id, positions in data['positions_by_model'].items():
                print(f"     {model_id}: {len(positions)} positions")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 9: Compare Equity Curves
    print("\n[9] Testing /api/battle/compare/equity_curves")
    try:
        resp = requests.get(f"{base_url}/api/battle/compare/equity_curves")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Equity curves for {len(data['equity_curves'])} models")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 10: Invalid Model ID (should 404)
    print("\n[10] Testing 404 handling")
    try:
        resp = requests.get(f"{base_url}/api/battle/model/invalid_model/positions")
        if resp.status_code == 404:
            print(f"    Correctly returned 404 for invalid model")
        else:
            print(f"   ERROR: Expected 404, got {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 11: Model Decisions (new endpoint)
    print("\n[11] Testing /api/battle/model/gpt4/decisions")
    try:
        resp = requests.get(f"{base_url}/api/battle/model/gpt4/decisions?limit=5")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   GPT-4 decisions: {data['total']} returned")
            for dec in data['decisions'][:3]:
                print(f"     - {dec['symbol']}: {dec['decision']} (confidence: {dec['confidence']})")
                print(f"       Reasoning: {dec['reasoning']}")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 12: Model Decisions with Timeframe
    print("\n[12] Testing /api/battle/model/gpt4/decisions?timeframe=12")
    try:
        resp = requests.get(f"{base_url}/api/battle/model/gpt4/decisions?timeframe=12&limit=10")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   GPT-4 decisions (last 12 hours): {data['total']} returned")
        else:
            print(f"   ERROR: {resp.status_code}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "="*60)
    print("API Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Battle API Test Utilities")
    parser.add_argument("--setup", action="store_true", help="Setup sample data")
    parser.add_argument("--test", action="store_true", help="Test API endpoints")

    args = parser.parse_args()

    if args.setup:
        setup_sample_data()

    if args.test:
        print("\nMake sure the backend is running: python backend/main.py")
        input("Press Enter to start tests...")
        test_endpoints()

    if not args.setup and not args.test:
        print("Usage:")
        print("  python test_battle_api.py --setup    # Create sample data")
        print("  python test_battle_api.py --test     # Test endpoints")
        print("  python test_battle_api.py --setup --test  # Both")
