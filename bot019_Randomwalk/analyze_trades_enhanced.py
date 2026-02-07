#!/usr/bin/env python3
"""
Analyze saved trading data and generate reports with correlation analysis
"""

import json
import sys
from datetime import datetime, timezone
from collections import defaultdict
import statistics
import pytz

def get_strategy_paths(strategy):
    """Get file paths for the specified strategy"""
    base_dir = "/home/ubuntu/013_2025_polymarket/bot019_Randomwalk"
    
    if strategy.upper() == "RANDOMWALK" or strategy == "":
        trades_dir = f"{base_dir}/trades"
        state_file = f"{base_dir}/state.json"
    else:
        trades_dir = f"{base_dir}/trades{strategy}"
        state_file = f"{base_dir}/state{strategy}.json"
    
    return trades_dir, state_file

def load_state(state_file):
    """Load trading state"""
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
        return None

def load_period_trades(trades_dir):
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

def get_market_period(timestamp_str):
    """
    Classify trade into market period based on UTC+1 time
    EUROPE: 9:30-15:30 UTC+1
    USA: 15:30-22:00 UTC+1
    WE_NIGHT_ASIA: 22:00-9:30 UTC+1 and weekends (Friday 22:00 - Monday 9:30)
    """
    try:
        # Parse timestamp (assuming ISO format with timezone)
        if isinstance(timestamp_str, str):
            dt_utc = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            return "UNKNOWN"
        
        # Convert to UTC+1
        utc_plus_1 = pytz.timezone('Europe/Paris')  # UTC+1 (CET)
        dt_local = dt_utc.astimezone(utc_plus_1)
        
        hour = dt_local.hour
        minute = dt_local.minute
        weekday = dt_local.weekday()  # 0=Monday, 6=Sunday
        
        # Check if weekend
        if weekday == 4 and (hour >= 22):  # Friday 22:00+
            return "WE_NIGHT_ASIA"
        elif weekday == 5 or weekday == 6:  # Saturday or Sunday
            return "WE_NIGHT_ASIA"
        elif weekday == 0 and (hour < 9 or (hour == 9 and minute < 30)):  # Monday before 9:30
            return "WE_NIGHT_ASIA"
        
        # Weekday classification
        time_decimal = hour + minute / 60.0
        
        if 9.5 <= time_decimal < 15.5:  # 9:30 - 15:30
            return "EUROPE"
        elif 15.5 <= time_decimal < 22.0:  # 15:30 - 22:00
            return "USA"
        else:  # 22:00 - 9:30
            return "WE_NIGHT_ASIA"
    
    except Exception as e:
        return "UNKNOWN"

def calculate_correlation(x_values, y_values):
    """Calculate Pearson correlation coefficient"""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    
    n = len(x_values)
    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)
    
    numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
    
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return None
    
    return numerator / denominator

def analyze_pnl_correlations(periods):
    """Analyze correlation between various factors and PNL"""
    print("\n" + "=" * 80)
    print("📊 PNL CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Collect all individual trades from all periods
    all_trades = []
    for period in periods:
        for trade in period.get('trades', []):
            all_trades.append(trade)
    
    if not all_trades:
        print("No trades found for correlation analysis")
        return
    
    # Extract data for correlation
    entry_prices = []
    choppiness_values = []
    time_to_expiry_values = []
    pnl_values = []
    
    for trade in all_trades:
        if 'pnl' in trade and 'entry_price' in trade:
            entry_prices.append(trade['entry_price'])
            choppiness_values.append(trade.get('choppiness', 0))
            time_to_expiry_values.append(trade.get('time_to_expiry_seconds', 0))
            pnl_values.append(trade['pnl'])
    
    print(f"\nTotal trades analyzed: {len(pnl_values)}")
    print(f"Total PNL: ${sum(pnl_values):.2f}")
    print(f"Average PNL per trade: ${statistics.mean(pnl_values):.3f}")
    
    # Calculate correlations
    print("\n📈 OVERALL CORRELATION WITH PNL:")
    print("-" * 80)
    
    corr_entry = calculate_correlation(entry_prices, pnl_values)
    corr_chop = calculate_correlation(choppiness_values, pnl_values)
    corr_time = calculate_correlation(time_to_expiry_values, pnl_values)
    
    print(f"Entry Price vs PNL:        {corr_entry:+.4f}" if corr_entry is not None else "Entry Price vs PNL:        N/A")
    print(f"Choppiness vs PNL:         {corr_chop:+.4f}" if corr_chop is not None else "Choppiness vs PNL:         N/A")
    print(f"Time to Expiry vs PNL:     {corr_time:+.4f}" if corr_time is not None else "Time to Expiry vs PNL:     N/A")
    
    # Interpretation guide
    print("\n💡 INTERPRETATION:")
    print("-" * 80)
    
    def interpret_correlation(name, corr):
        if corr is None:
            print(f"{name}: Insufficient data")
            return
        
        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative"
        
        if abs_corr > 0.7:
            strength = "STRONG"
        elif abs_corr > 0.4:
            strength = "MODERATE"
        elif abs_corr > 0.2:
            strength = "WEAK"
        else:
            strength = "NEGLIGIBLE"
        
        print(f"{name}: {strength} {direction} correlation ({corr:+.4f})")
        
        if abs_corr > 0.4:
            if name == "Entry Price" and corr > 0:
                print(f"  → Higher entry prices tend to have BETTER PNL")
            elif name == "Entry Price" and corr < 0:
                print(f"  → Lower entry prices tend to have BETTER PNL")
            elif name == "Choppiness" and corr > 0:
                print(f"  → Higher choppiness tends to have BETTER PNL")
            elif name == "Choppiness" and corr < 0:
                print(f"  → Lower choppiness (trending markets) tends to have BETTER PNL")
            elif name == "Time to Expiry" and corr > 0:
                print(f"  → Entering earlier tends to have BETTER PNL")
            elif name == "Time to Expiry" and corr < 0:
                print(f"  → Entering closer to expiry tends to have BETTER PNL")
    
    interpret_correlation("Entry Price", corr_entry)
    interpret_correlation("Choppiness", corr_chop)
    interpret_correlation("Time to Expiry", corr_time)
    
    # DAILY ANALYSIS
    print("\n" + "=" * 80)
    print("📅 DAILY BREAKDOWN WITH CORRELATIONS")
    print("=" * 80)
    
    daily_trades = defaultdict(list)
    for trade in all_trades:
        if 'entry_time' in trade:
            date = trade['entry_time'][:10]  # YYYY-MM-DD
            daily_trades[date].append(trade)
    
    for date in sorted(daily_trades.keys()):
        day_trades = daily_trades[date]
        
        if len(day_trades) < 2:
            continue
        
        day_entry = [t['entry_price'] for t in day_trades if 'entry_price' in t and 'pnl' in t]
        day_chop = [t.get('choppiness', 0) for t in day_trades if 'pnl' in t]
        day_time = [t.get('time_to_expiry_seconds', 0) for t in day_trades if 'pnl' in t]
        day_pnl = [t['pnl'] for t in day_trades if 'pnl' in t]
        
        corr_e = calculate_correlation(day_entry, day_pnl)
        corr_c = calculate_correlation(day_chop, day_pnl)
        corr_t = calculate_correlation(day_time, day_pnl)
        
        total_pnl = sum(day_pnl)
        avg_pnl = statistics.mean(day_pnl)
        wins = sum(1 for t in day_trades if t.get('won', False))
        
        print(f"\n📆 {date} ({len(day_trades)} trades, {wins} wins, ${total_pnl:+.2f} total)")
        print(f"  Avg PNL: ${avg_pnl:+.3f}")
        print(f"  Entry vs PNL:  {corr_e:+.4f}" if corr_e else "  Entry vs PNL:  N/A")
        print(f"  Chop vs PNL:   {corr_c:+.4f}" if corr_c else "  Chop vs PNL:   N/A")
        print(f"  Time vs PNL:   {corr_t:+.4f}" if corr_t else "  Time vs PNL:   N/A")
    
    # MARKET PERIOD ANALYSIS
    print("\n" + "=" * 80)
    print("🌍 MARKET PERIOD ANALYSIS")
    print("=" * 80)
    print("\nEUROPE:        9:30-15:30 UTC+1")
    print("USA:           15:30-22:00 UTC+1")
    print("WE_NIGHT_ASIA: 22:00-9:30 UTC+1 + Weekends")
    print("-" * 80)
    
    period_trades = defaultdict(list)
    for trade in all_trades:
        if 'entry_time' in trade:
            period = get_market_period(trade['entry_time'])
            period_trades[period].append(trade)
    
    for period_name in ["EUROPE", "USA", "WE_NIGHT_ASIA"]:
        if period_name not in period_trades or len(period_trades[period_name]) < 2:
            print(f"\n🌐 {period_name}: No sufficient data")
            continue
        
        p_trades = period_trades[period_name]
        
        p_entry = [t['entry_price'] for t in p_trades if 'entry_price' in t and 'pnl' in t]
        p_chop = [t.get('choppiness', 0) for t in p_trades if 'pnl' in t]
        p_time = [t.get('time_to_expiry_seconds', 0) for t in p_trades if 'pnl' in t]
        p_pnl = [t['pnl'] for t in p_trades if 'pnl' in t]
        
        corr_e = calculate_correlation(p_entry, p_pnl)
        corr_c = calculate_correlation(p_chop, p_pnl)
        corr_t = calculate_correlation(p_time, p_pnl)
        
        total_pnl = sum(p_pnl)
        avg_pnl = statistics.mean(p_pnl)
        wins = sum(1 for t in p_trades if t.get('won', False))
        win_rate = wins / len(p_trades) * 100 if p_trades else 0
        
        # Split by side
        call_trades_p = [t for t in p_trades if t.get('side') == 'CALL']
        put_trades_p = [t for t in p_trades if t.get('side') == 'PUT']
        
        print(f"\n🌐 {period_name} ({len(p_trades)} trades)")
        print(f"  Win Rate:      {win_rate:.1f}% ({wins}/{len(p_trades)})")
        print(f"  Total PNL:     ${total_pnl:+.2f}")
        print(f"  Avg PNL:       ${avg_pnl:+.3f}")
        print(f"  CALL/PUT:      {len(call_trades_p)}/{len(put_trades_p)}")
        print(f"  Entry vs PNL:  {corr_e:+.4f}" if corr_e else "  Entry vs PNL:  N/A")
        print(f"  Chop vs PNL:   {corr_c:+.4f}" if corr_c else "  Chop vs PNL:   N/A")
        print(f"  Time vs PNL:   {corr_t:+.4f}" if corr_t else "  Time vs PNL:   N/A")
    
    # Separate analysis by side (CALL vs PUT)
    print("\n" + "=" * 80)
    print("📊 CORRELATION BY SIDE (CALL vs PUT)")
    print("=" * 80)
    
    call_trades = [t for t in all_trades if t.get('side') == 'CALL' and 'pnl' in t]
    put_trades = [t for t in all_trades if t.get('side') == 'PUT' and 'pnl' in t]
    
    for side_name, side_trades in [("CALL", call_trades), ("PUT", put_trades)]:
        if not side_trades:
            continue
        
        print(f"\n🔵 {side_name} TRADES ({len(side_trades)} trades):")
        print("-" * 80)
        
        side_entry = [t['entry_price'] for t in side_trades]
        side_chop = [t.get('choppiness', 0) for t in side_trades]
        side_time = [t.get('time_to_expiry_seconds', 0) for t in side_trades]
        side_pnl = [t['pnl'] for t in side_trades]
        
        corr_e = calculate_correlation(side_entry, side_pnl)
        corr_c = calculate_correlation(side_chop, side_pnl)
        corr_t = calculate_correlation(side_time, side_pnl)
        
        print(f"Entry Price vs PNL:        {corr_e:+.4f}" if corr_e is not None else "Entry Price vs PNL:        N/A")
        print(f"Choppiness vs PNL:         {corr_c:+.4f}" if corr_c is not None else "Choppiness vs PNL:         N/A")
        print(f"Time to Expiry vs PNL:     {corr_t:+.4f}" if corr_t is not None else "Time to Expiry vs PNL:     N/A")
        print(f"Average PNL: ${statistics.mean(side_pnl):.3f}")
    
    # Winning vs Losing trades analysis
    print("\n" + "=" * 80)
    print("📊 WINNING vs LOSING TRADES COMPARISON")
    print("=" * 80)
    
    winning_trades = [t for t in all_trades if t.get('won', False)]
    losing_trades = [t for t in all_trades if not t.get('won', False)]
    
    print(f"\n✅ WINNING TRADES ({len(winning_trades)}):")
    if winning_trades:
        print(f"  Avg Entry Price:     ${statistics.mean([t['entry_price'] for t in winning_trades]):.3f}")
        print(f"  Avg Choppiness:      {statistics.mean([t.get('choppiness', 0) for t in winning_trades]):.2f}")
        print(f"  Avg Time to Expiry:  {statistics.mean([t.get('time_to_expiry_seconds', 0) for t in winning_trades]):.0f}s")
        print(f"  Avg PNL:             ${statistics.mean([t['pnl'] for t in winning_trades]):.3f}")
    
    print(f"\n❌ LOSING TRADES ({len(losing_trades)}):")
    if losing_trades:
        print(f"  Avg Entry Price:     ${statistics.mean([t['entry_price'] for t in losing_trades]):.3f}")
        print(f"  Avg Choppiness:      {statistics.mean([t.get('choppiness', 0) for t in losing_trades]):.2f}")
        print(f"  Avg Time to Expiry:  {statistics.mean([t.get('time_to_expiry_seconds', 0) for t in losing_trades]):.0f}s")
        print(f"  Avg PNL:             ${statistics.mean([t['pnl'] for t in losing_trades]):.3f}")
    
    print("=" * 80)

def analyze_period_trades(trades_dir):
    """Analyze trades from period files"""
    periods = load_period_trades(trades_dir)
    
    if not periods:
        print("No period trade files found")
        return periods
    
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
    
    return periods

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
    # Prompt for strategy
    print("=" * 80)
    print("🎯 STRATEGY SELECTION")
    print("=" * 80)
    print("\nAvailable strategies:")
    print("  - [ENTER] or 'Randomwalk' for default Randomwalk strategy")
    print("  - A2 for Strategy A2")
    print("  - B for Strategy B")
    print("  - C for Strategy C")
    print("  - (or any other custom strategy suffix)")
    
    # Allow command-line override for testing
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        print(f"\nUsing strategy from command line: {strategy}")
    else:
        strategy = input("\nEnter strategy to analyze: ").strip()
    
    trades_dir, state_file = get_strategy_paths(strategy)
    
    print(f"\n📂 Configuration:")
    print(f"Strategy: {strategy if strategy else 'Randomwalk'}")
    print(f"State file: {state_file}")
    print(f"Trades directory: {trades_dir}")
    print()
    
    # Analyze period-by-period trades
    periods = analyze_period_trades(trades_dir)
    
    # Correlation analysis
    if periods:
        analyze_pnl_correlations(periods)
    
    # Analyze overall state
    state = load_state(state_file)
    
    if state:
        show_current_state(state)
        analyze_trades(state)
        analyze_state_history(state)
    else:
        print("Failed to load state file")
