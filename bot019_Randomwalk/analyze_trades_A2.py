#!/usr/bin/env python3
"""
Analyze saved trading data and generate reports
"""

import json
import sys
from datetime import datetime, timezone
from collections import defaultdict

def load_state(state_file="/home/ubuntu/013_2025_polymarket/bot019_Randomwalk/stateA2.json"):
    """Load trading state"""
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
        return None

def load_period_trades(trades_dir="/home/ubuntu/013_2025_polymarket/bot019_Randomwalk/tradesA2"):
    """Load all period trade files"""
    import os
    import glob

    trade_files = glob.glob(f"{trades_dir}/trades_*.json")
    all_periods = []

    for file_path in sorted(trade_files):
        try:
            with open(file_path, 'r') as f:
                period_data = json.load(f)
                all_periods.append(period_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return all_periods

def analyze_period_trades(trades_dir="/home/ubuntu/013_2025_polymarket/bot019_Randomwalk/tradesA2"):
    """Analyze trades from period files"""
    periods = load_period_trades(trades_dir)

    if not periods:
        print("No period trade files found")
        return

    print("=" * 80)
    print("📊 PERIOD-BY-PERIOD ANALYSIS")
    print("=" * 80)
    print(f"\nTotal periods: {len(periods)}")

    total_pnl = 0.0
    winning_periods = 0

    for i, period in enumerate(periods, 1):
        period_start = period.get('period_start', 'N/A')[:19]
        strike = period.get('strike_price', 0)
        final = period.get('final_price', 0)
        movement = period.get('price_movement', 0)
        summary = period.get('summary', {})
        pnl = summary.get('total_pnl', 0)

        total_pnl += pnl
        if pnl > 0:
            winning_periods += 1

        winner = "📈 CALL" if period.get('calls_won') else "📉 PUT" if period.get('puts_won') else "➖ NONE"

        print(f"\nPeriod {i}: {period_start}")
        print(f"  Strike: ${strike:,.2f} → Final: ${final:,.2f} (${movement:+,.2f})")
        print(f"  Winner: {winner}")
        print(f"  Trades: {summary.get('call_trades', 0)} CALL, {summary.get('put_trades', 0)} PUT")
        print(f"  PNL: ${pnl:+.2f} (CALL: ${summary.get('call_pnl', 0):+.2f}, PUT: ${summary.get('put_pnl', 0):+.2f})")

    print(f"\n" + "=" * 80)
    print(f"📊 OVERALL SUMMARY")
    print(f"Total periods: {len(periods)}")
    print(f"Winning periods: {winning_periods} ({winning_periods/len(periods)*100:.1f}%)")
    print(f"Total PNL: ${total_pnl:+.2f}")
    print(f"Average PNL per period: ${total_pnl/len(periods):+.2f}")
    print("=" * 80)

def analyze_trades(state):
    """Analyze trade history"""
    trades = state.get('trades_history', [])

    if not trades:
        print("No trades found in history")
        return

    print("=" * 80)
    print("📊 TRADING HISTORY ANALYSIS")
    print("=" * 80)

    # Overall statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get('won', False))
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_pnl = sum(t.get('pnl', 0) for t in trades)

    # Separate by side
    call_trades = [t for t in trades if t.get('side') == 'CALL']
    put_trades = [t for t in trades if t.get('side') == 'PUT']

    call_pnl = sum(t.get('pnl', 0) for t in call_trades)
    put_pnl = sum(t.get('pnl', 0) for t in put_trades)

    call_wins = sum(1 for t in call_trades if t.get('won', False))
    put_wins = sum(1 for t in put_trades if t.get('won', False))

    print(f"\n📈 OVERALL STATISTICS")
    print(f"Total trades: {total_trades}")
    print(f"Winning trades: {winning_trades} ({win_rate:.1f}%)")
    print(f"Losing trades: {losing_trades}")
    print(f"Total PNL: ${total_pnl:.2f}")

    print(f"\n🔵 CALL TRADES")
    print(f"Count: {len(call_trades)}")
    print(f"Wins: {call_wins} ({call_wins/len(call_trades)*100:.1f}%)" if call_trades else "Count: 0")
    print(f"PNL: ${call_pnl:.2f}")

    print(f"\n🔴 PUT TRADES")
    print(f"Count: {len(put_trades)}")
    print(f"Wins: {put_wins} ({put_wins/len(put_trades)*100:.1f}%)" if put_trades else "Count: 0")
    print(f"PNL: ${put_pnl:.2f}")

    # Daily breakdown
    print(f"\n📅 DAILY BREAKDOWN")
    daily_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})

    for trade in trades:
        date = trade.get('timestamp', '')[:10]  # Get YYYY-MM-DD
        daily_stats[date]['trades'] += 1
        daily_stats[date]['pnl'] += trade.get('pnl', 0)
        if trade.get('won', False):
            daily_stats[date]['wins'] += 1

    for date in sorted(daily_stats.keys()):
        stats = daily_stats[date]
        win_rate_daily = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        print(f"{date}: {stats['trades']} trades, Win rate: {win_rate_daily:.1f}%, PNL: ${stats['pnl']:.2f}")

    print("=" * 80)

def analyze_state_history(state):
    """Analyze hourly state snapshots"""
    history = state.get('state_history', [])

    if not history:
        print("\nNo state history found")
        return

    print("\n" + "=" * 80)
    print("📊 HOURLY STATE HISTORY")
    print("=" * 80)

    print(f"\nTotal snapshots: {len(history)}")

    # Show last 10 snapshots
    print(f"\n🕐 LAST 10 HOURLY SNAPSHOTS")
    for snapshot in history[-10:]:
        timestamp = snapshot.get('timestamp', 'N/A')[:19]  # YYYY-MM-DD HH:MM:SS
        chop = snapshot.get('choppiness', 0)
        btc = snapshot.get('btc_price', 0)
        pnl = snapshot.get('total_pnl', 0)
        call_pos = snapshot.get('call_positions', 0)
        put_pos = snapshot.get('put_positions', 0)

        print(f"{timestamp}: BTC ${btc:,.2f}, Chop: {chop:.1f}, Positions: {call_pos}C/{put_pos}P, PNL: ${pnl:.2f}")

    print("=" * 80)

def show_current_state(state):
    """Show current bot state"""
    print("\n" + "=" * 80)
    print("🤖 CURRENT BOT STATE")
    print("=" * 80)

    last_updated = state.get('last_updated', 'N/A')
    print(f"\nLast updated: {last_updated[:19] if last_updated != 'N/A' else 'N/A'}")

    current_positions = state.get('current_positions', [])
    print(f"Current positions: {len(current_positions)}")

    for i, pos in enumerate(current_positions, 1):
        print(f"  Position {i}: {pos['side']} @ ${pos['entry_price']:.3f}, size={pos['size']}, PNL: ${pos['pnl']:.2f}")

    print(f"\nMax prices:")
    print(f"  CALL: ${state.get('max_call_ask', 0):.3f}")
    print(f"  PUT: ${state.get('max_put_ask', 0):.3f}")

    if state.get('last_market_snapshot'):
        snapshot = state['last_market_snapshot']
        print(f"\nLast market snapshot:")
        print(f"  BTC: ${snapshot.get('btc_price', 0):,.2f}")
        print(f"  CALL ask: ${snapshot.get('call_ask', 0):.3f}")
        print(f"  PUT ask: ${snapshot.get('put_ask', 0):.3f}")
        print(f"  Choppiness: {snapshot.get('choppiness', 0):.1f}")

    print("=" * 80)

if __name__ == "__main__":
    state_file = "/home/ubuntu/013_2025_polymarket/bot019_Randomwalk/stateA2.json"
    trades_dir = "/home/ubuntu/013_2025_polymarket/bot019_Randomwalk/tradesA2"

    if len(sys.argv) > 1:
        state_file = sys.argv[1]

    print(f"Loading state from: {state_file}")
    print(f"Loading trades from: {trades_dir}\n")

    # Analyze period-by-period trades
    analyze_period_trades(trades_dir)

    # Analyze overall state
    state = load_state(state_file)

    if state:
        show_current_state(state)
        analyze_trades(state)
        analyze_state_history(state)
    else:
        print("Failed to load state file")
