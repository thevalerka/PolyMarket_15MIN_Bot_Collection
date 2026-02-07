#!/usr/bin/env python3
"""
PNL Analyzer for Trading Data
Analyzes trades by market period with correlation analysis
Place this file in the same folder as your trades JSON files
"""

import json
import glob
import os
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import pytz

def get_market_period(timestamp_str):
    """
    Classify trade into market period based on UTC+1 time
    EUROPE: 9:30-15:30 UTC+1
    USA: 15:30-22:00 UTC+1
    WE_NIGHT_ASIA: 22:00-9:30 UTC+1 and weekends (Friday 22:00 - Monday 9:30)
    """
    try:
        # Parse timestamp
        if isinstance(timestamp_str, str):
            # Handle various formats
            if 'T' in timestamp_str:
                dt_utc = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                dt_utc = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
        else:
            return "UNKNOWN"
        
        # Convert to UTC+1 (CET)
        utc_plus_1 = pytz.timezone('Europe/Paris')
        dt_local = dt_utc.astimezone(utc_plus_1)
        
        hour = dt_local.hour
        minute = dt_local.minute
        weekday = dt_local.weekday()  # 0=Monday, 6=Sunday
        
        # Check if weekend
        if weekday == 4 and hour >= 22:  # Friday 22:00+
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
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return "UNKNOWN"

def calculate_time_to_expiry(entry_time_str, exit_time_str):
    """Calculate time to expiry in seconds"""
    try:
        entry = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
        exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
        return (exit_time - entry).total_seconds()
    except:
        return 0

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

def load_all_trades(directory="."):
    """Load all trade JSON files from directory"""
    trade_files = glob.glob(os.path.join(directory, "trades_*.json"))
    
    if not trade_files:
        print(f"No trade files found in {directory}")
        return []
    
    all_trades = []
    
    for file_path in sorted(trade_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                period_id = data.get('period_id', 'unknown')
                
                for trade in data.get('trades', []):
                    # Add period info to each trade
                    trade['period_id'] = period_id
                    trade['file'] = os.path.basename(file_path)
                    
                    # Calculate time to expiry
                    if 'entry_time' in trade and 'exit_time' in trade:
                        trade['time_to_expiry'] = calculate_time_to_expiry(
                            trade['entry_time'], 
                            trade['exit_time']
                        )
                    
                    # Determine market period
                    if 'entry_time' in trade:
                        trade['market_period'] = get_market_period(trade['entry_time'])
                    
                    all_trades.append(trade)
                    
            print(f"✓ Loaded {len(data.get('trades', []))} trades from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
    return all_trades

def analyze_overall(trades):
    """Overall analysis"""
    print("\n" + "=" * 80)
    print("📊 OVERALL ANALYSIS")
    print("=" * 80)
    
    total_trades = len(trades)
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
    breakeven = total_trades - wins - losses
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    call_trades = [t for t in trades if t.get('asset_type') == 'CALL']
    put_trades = [t for t in trades if t.get('asset_type') == 'PUT']
    
    call_pnl = sum(t.get('pnl', 0) for t in call_trades)
    put_pnl = sum(t.get('pnl', 0) for t in put_trades)
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total Trades:     {total_trades}")
    print(f"  Wins:             {wins} ({win_rate:.1f}%)")
    print(f"  Losses:           {losses}")
    print(f"  Breakeven:        {breakeven}")
    print(f"  Total PNL:        ${total_pnl:,.2f}")
    print(f"  Average PNL:      ${avg_pnl:.2f}")
    
    print(f"\n🔵 CALL Trades:     {len(call_trades)} (PNL: ${call_pnl:+,.2f})")
    print(f"🔴 PUT Trades:      {len(put_trades)} (PNL: ${put_pnl:+,.2f})")
    
    # Entry price ranges
    if trades:
        entry_prices = [t.get('entry_price', 0) for t in trades]
        print(f"\n💰 Entry Price Range: ${min(entry_prices):.2f} - ${max(entry_prices):.2f}")
        print(f"   Average Entry:     ${statistics.mean(entry_prices):.2f}")

def analyze_correlations(trades):
    """Correlation analysis"""
    print("\n" + "=" * 80)
    print("📈 CORRELATION ANALYSIS")
    print("=" * 80)
    
    if len(trades) < 2:
        print("Insufficient data for correlation analysis")
        return
    
    # Extract data
    entry_prices = []
    time_to_expiry_values = []
    pnl_values = []
    
    for trade in trades:
        if 'pnl' in trade and 'entry_price' in trade:
            entry_prices.append(trade['entry_price'])
            time_to_expiry_values.append(trade.get('time_to_expiry', 0))
            pnl_values.append(trade['pnl'])
    
    # Calculate correlations
    corr_entry = calculate_correlation(entry_prices, pnl_values)
    corr_time = calculate_correlation(time_to_expiry_values, pnl_values)
    
    print("\n🔗 OVERALL CORRELATIONS WITH PNL:")
    print(f"  Entry Price vs PNL:      {corr_entry:+.4f}" if corr_entry else "  Entry Price vs PNL:      N/A")
    print(f"  Time to Expiry vs PNL:   {corr_time:+.4f}" if corr_time else "  Time to Expiry vs PNL:   N/A")
    
    # Interpretation
    def interpret_corr(name, corr):
        if corr is None:
            return
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            strength = "STRONG"
        elif abs_corr > 0.4:
            strength = "MODERATE"
        elif abs_corr > 0.2:
            strength = "WEAK"
        else:
            strength = "NEGLIGIBLE"
        
        direction = "positive" if corr > 0 else "negative"
        print(f"  → {name}: {strength} {direction} ({corr:+.4f})")
    
    print("\n💡 INTERPRETATION:")
    interpret_corr("Entry Price", corr_entry)
    interpret_corr("Time to Expiry", corr_time)

def analyze_by_market_period(trades):
    """Analysis by market period"""
    print("\n" + "=" * 80)
    print("🌍 MARKET PERIOD ANALYSIS")
    print("=" * 80)
    print("\n⏰ Market Periods (UTC+1):")
    print("  EUROPE:        9:30-15:30")
    print("  USA:           15:30-22:00")
    print("  WE_NIGHT_ASIA: 22:00-9:30 + Weekends")
    
    # Group by market period
    period_trades = defaultdict(list)
    for trade in trades:
        period = trade.get('market_period', 'UNKNOWN')
        period_trades[period].append(trade)
    
    # Analyze each period
    for period_name in ["EUROPE", "USA", "WE_NIGHT_ASIA", "UNKNOWN"]:
        if period_name not in period_trades:
            continue
        
        p_trades = period_trades[period_name]
        
        if not p_trades:
            continue
        
        total = len(p_trades)
        pnl = sum(t.get('pnl', 0) for t in p_trades)
        wins = sum(1 for t in p_trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in p_trades if t.get('pnl', 0) < 0)
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_pnl = pnl / total if total > 0 else 0
        
        call_trades = [t for t in p_trades if t.get('asset_type') == 'CALL']
        put_trades = [t for t in p_trades if t.get('asset_type') == 'PUT']
        
        call_pnl = sum(t.get('pnl', 0) for t in call_trades)
        put_pnl = sum(t.get('pnl', 0) for t in put_trades)
        
        print(f"\n{'─' * 80}")
        print(f"🌐 {period_name}")
        print(f"{'─' * 80}")
        print(f"  Total Trades:     {total}")
        print(f"  Win Rate:         {win_rate:.1f}% ({wins}W / {losses}L)")
        print(f"  Total PNL:        ${pnl:+,.2f}")
        print(f"  Average PNL:      ${avg_pnl:+.2f}")
        print(f"  CALL Trades:      {len(call_trades)} (${call_pnl:+,.2f})")
        print(f"  PUT Trades:       {len(put_trades)} (${put_pnl:+,.2f})")
        
        # Correlations for this period
        if len(p_trades) >= 2:
            p_entry = [t['entry_price'] for t in p_trades if 'entry_price' in t and 'pnl' in t]
            p_time = [t.get('time_to_expiry', 0) for t in p_trades if 'pnl' in t]
            p_pnl = [t['pnl'] for t in p_trades if 'pnl' in t]
            
            corr_e = calculate_correlation(p_entry, p_pnl)
            corr_t = calculate_correlation(p_time, p_pnl)
            
            print(f"\n  📊 Correlations:")
            print(f"    Entry vs PNL:   {corr_e:+.4f}" if corr_e else "    Entry vs PNL:   N/A")
            print(f"    Time vs PNL:    {corr_t:+.4f}" if corr_t else "    Time vs PNL:    N/A")

def analyze_by_day(trades):
    """Daily breakdown"""
    print("\n" + "=" * 80)
    print("📅 DAILY BREAKDOWN")
    print("=" * 80)
    
    daily_trades = defaultdict(list)
    for trade in trades:
        if 'entry_time' in trade:
            date = trade['entry_time'][:10]  # YYYY-MM-DD
            daily_trades[date].append(trade)
    
    for date in sorted(daily_trades.keys()):
        day_trades = daily_trades[date]
        
        total = len(day_trades)
        pnl = sum(t.get('pnl', 0) for t in day_trades)
        wins = sum(1 for t in day_trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in day_trades if t.get('pnl', 0) < 0)
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_pnl = pnl / total if total > 0 else 0
        
        # Market period breakdown
        period_counts = defaultdict(int)
        for t in day_trades:
            period_counts[t.get('market_period', 'UNKNOWN')] += 1
        
        period_str = ", ".join([f"{p}:{c}" for p, c in sorted(period_counts.items())])
        
        print(f"\n📆 {date}")
        print(f"  Trades: {total} ({period_str})")
        print(f"  Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
        print(f"  Total PNL: ${pnl:+,.2f}")
        print(f"  Avg PNL: ${avg_pnl:+.2f}")
        
        # Correlations
        if len(day_trades) >= 2:
            d_entry = [t['entry_price'] for t in day_trades if 'entry_price' in t and 'pnl' in t]
            d_time = [t.get('time_to_expiry', 0) for t in day_trades if 'pnl' in t]
            d_pnl = [t['pnl'] for t in day_trades if 'pnl' in t]
            
            corr_e = calculate_correlation(d_entry, d_pnl)
            corr_t = calculate_correlation(d_time, d_pnl)
            
            print(f"  Correlations: Entry={corr_e:+.4f}" if corr_e else "  Correlations: Entry=N/A", end="")
            print(f", Time={corr_t:+.4f}" if corr_t else ", Time=N/A")

def analyze_by_exit_reason(trades):
    """Analysis by exit reason"""
    print("\n" + "=" * 80)
    print("🚪 EXIT REASON ANALYSIS")
    print("=" * 80)
    
    exit_reasons = defaultdict(list)
    for trade in trades:
        reason = trade.get('exit_reason', 'unknown')
        exit_reasons[reason].append(trade)
    
    for reason in sorted(exit_reasons.keys()):
        r_trades = exit_reasons[reason]
        total = len(r_trades)
        pnl = sum(t.get('pnl', 0) for t in r_trades)
        wins = sum(1 for t in r_trades if t.get('pnl', 0) > 0)
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_pnl = pnl / total if total > 0 else 0
        
        print(f"\n🚪 {reason.upper()}")
        print(f"  Trades: {total}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total PNL: ${pnl:+,.2f}")
        print(f"  Avg PNL: ${avg_pnl:+.2f}")

def print_best_worst_trades(trades, n=5):
    """Show best and worst trades"""
    print("\n" + "=" * 80)
    print(f"🏆 TOP {n} BEST & WORST TRADES")
    print("=" * 80)
    
    sorted_trades = sorted(trades, key=lambda t: t.get('pnl', 0), reverse=True)
    
    print(f"\n✅ TOP {n} BEST TRADES:")
    for i, trade in enumerate(sorted_trades[:n], 1):
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_percent', 0)
        asset = trade.get('asset_type', 'N/A')
        entry = trade.get('entry_price', 0)
        exit_p = trade.get('exit_price', 0)
        entry_time = trade.get('entry_time', 'N/A')[:16]
        period = trade.get('market_period', 'N/A')
        
        print(f"  {i}. ${pnl:+.2f} ({pnl_pct:+.1f}%) - {asset} {entry:.2f}→{exit_p:.2f} @ {entry_time} [{period}]")
    
    print(f"\n❌ TOP {n} WORST TRADES:")
    for i, trade in enumerate(sorted_trades[-n:][::-1], 1):
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_percent', 0)
        asset = trade.get('asset_type', 'N/A')
        entry = trade.get('entry_price', 0)
        exit_p = trade.get('exit_price', 0)
        entry_time = trade.get('entry_time', 'N/A')[:16]
        period = trade.get('market_period', 'N/A')
        
        print(f"  {i}. ${pnl:+.2f} ({pnl_pct:+.1f}%) - {asset} {entry:.2f}→{exit_p:.2f} @ {entry_time} [{period}]")

def main():
    import sys
    
    print("=" * 80)
    print("📊 TRADING PNL ANALYZER")
    print("=" * 80)
    
    # Get directory from command line or use current directory
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"\n📂 Loading trades from: {os.path.abspath(directory)}\n")
    
    # Load all trades
    trades = load_all_trades(directory)
    
    if not trades:
        print("\n❌ No trades found!")
        return
    
    print(f"\n✓ Total trades loaded: {len(trades)}")
    
    # Run analyses
    analyze_overall(trades)
    analyze_correlations(trades)
    analyze_by_market_period(trades)
    analyze_by_day(trades)
    analyze_by_exit_reason(trades)
    print_best_worst_trades(trades)
    
    print("\n" + "=" * 80)
    print("✓ Analysis Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
