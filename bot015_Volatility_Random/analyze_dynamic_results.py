import json
import sys
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import statistics

def load_results(filename: str) -> Dict:
    """Load trading results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)

def analyze_strategy_performance(trades: List[Dict]) -> Dict:
    """Analyze performance metrics by strategy"""
    strategy_stats = defaultdict(lambda: {
        'trades': [],
        'wins': 0,
        'losses': 0,
        'tp_exits': 0,
        'sl_exits': 0,
        'expiry_exits': 0,
        'total_pnl': 0.0,
        'durations': []
    })
    
    for trade in trades:
        strategy = trade['strategy']
        pnl = trade['pnl']
        
        strategy_stats[strategy]['trades'].append(trade)
        strategy_stats[strategy]['total_pnl'] += pnl
        strategy_stats[strategy]['durations'].append(trade.get('duration_seconds', 0))
        
        if pnl > 0:
            strategy_stats[strategy]['wins'] += 1
        else:
            strategy_stats[strategy]['losses'] += 1
        
        exit_reason = trade.get('exit_reason', 'UNKNOWN')
        if exit_reason == 'TP':
            strategy_stats[strategy]['tp_exits'] += 1
        elif exit_reason == 'SL':
            strategy_stats[strategy]['sl_exits'] += 1
        elif exit_reason == 'EXPIRY':
            strategy_stats[strategy]['expiry_exits'] += 1
    
    return strategy_stats

def analyze_exit_patterns(trades: List[Dict]) -> Dict:
    """Analyze exit reason patterns"""
    exit_stats = defaultdict(lambda: {
        'count': 0,
        'total_pnl': 0.0,
        'wins': 0,
        'losses': 0
    })
    
    for trade in trades:
        exit_reason = trade.get('exit_reason', 'UNKNOWN')
        pnl = trade['pnl']
        
        exit_stats[exit_reason]['count'] += 1
        exit_stats[exit_reason]['total_pnl'] += pnl
        
        if pnl > 0:
            exit_stats[exit_reason]['wins'] += 1
        else:
            exit_stats[exit_reason]['losses'] += 1
    
    return exit_stats

def analyze_choppiness_correlation(trades: List[Dict]) -> Dict:
    """Analyze performance correlation with choppiness levels"""
    chop_ranges = {
        'Low (0-30)': [],
        'Medium (30-60)': [],
        'High (60-100)': []
    }
    
    for trade in trades:
        # Check instantaneous choppiness first (for Strategy C)
        chop = trade.get('instantaneous_choppiness')
        
        # If not available, check 60min choppiness
        if chop is None and trade.get('market_conditions'):
            osc_60 = trade['market_conditions'].get('oscillation_60min', {})
            chop = osc_60.get('choppiness_index')
        
        if chop is not None:
            pnl = trade['pnl']
            if chop < 30:
                chop_ranges['Low (0-30)'].append(pnl)
            elif chop < 60:
                chop_ranges['Medium (30-60)'].append(pnl)
            else:
                chop_ranges['High (60-100)'].append(pnl)
    
    return chop_ranges

def analyze_time_based_performance(trades: List[Dict]) -> Dict:
    """Analyze performance by hour of day"""
    hourly_stats = defaultdict(lambda: {'pnl': [], 'count': 0})
    
    for trade in trades:
        timestamp = datetime.fromisoformat(trade['timestamp'])
        hour = timestamp.hour
        
        hourly_stats[hour]['pnl'].append(trade['pnl'])
        hourly_stats[hour]['count'] += 1
    
    return hourly_stats

def print_analysis(results: Dict):
    """Print comprehensive analysis"""
    trades = results.get('trades', [])
    
    if not trades:
        print("No trades found in results file")
        return
    
    print("="*80)
    print(f"TRADING ANALYSIS - {results.get('date', 'N/A')}")
    print("="*80)
    print(f"Last Updated: {results.get('last_updated', 'N/A')}")
    print(f"Total Trades: {len(trades)}")
    print(f"Open Positions: {results.get('open_positions', 0)}")
    print(f"Total PNL: ${results.get('total_pnl', {}).get('ALL', 0):+.4f}")
    print()
    
    # Strategy Performance Analysis
    print("="*80)
    print("STRATEGY PERFORMANCE")
    print("="*80)
    
    strategy_stats = analyze_strategy_performance(trades)
    
    for strategy in ['A', 'A2', 'B', 'C']:
        if strategy not in strategy_stats:
            continue
        
        stats = strategy_stats[strategy]
        total_trades = len(stats['trades'])
        
        if total_trades == 0:
            continue
        
        win_rate = (stats['wins'] / total_trades) * 100
        avg_pnl = stats['total_pnl'] / total_trades
        avg_duration = statistics.mean(stats['durations']) / 60  # Convert to minutes
        
        print(f"\nStrategy {strategy}:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Wins: {stats['wins']} ({win_rate:.1f}%)")
        print(f"  Losses: {stats['losses']} ({100-win_rate:.1f}%)")
        print(f"  Total PNL: ${stats['total_pnl']:+.4f}")
        print(f"  Average PNL: ${avg_pnl:+.4f}")
        print(f"  Average Duration: {avg_duration:.1f} minutes")
        print(f"  Exit Breakdown:")
        print(f"    TP: {stats['tp_exits']} ({stats['tp_exits']/total_trades*100:.1f}%)")
        print(f"    SL: {stats['sl_exits']} ({stats['sl_exits']/total_trades*100:.1f}%)")
        print(f"    Expiry: {stats['expiry_exits']} ({stats['expiry_exits']/total_trades*100:.1f}%)")
        
        # Calculate PNL distribution
        pnls = [t['pnl'] for t in stats['trades']]
        if pnls:
            print(f"  PNL Distribution:")
            print(f"    Min: ${min(pnls):+.4f}")
            print(f"    Max: ${max(pnls):+.4f}")
            print(f"    Median: ${statistics.median(pnls):+.4f}")
            if len(pnls) > 1:
                print(f"    Std Dev: ${statistics.stdev(pnls):.4f}")
    
    # Exit Reason Analysis
    print("\n" + "="*80)
    print("EXIT REASON ANALYSIS")
    print("="*80)
    
    exit_stats = analyze_exit_patterns(trades)
    
    for exit_reason in ['TP', 'SL', 'EXPIRY']:
        if exit_reason not in exit_stats:
            continue
        
        stats = exit_stats[exit_reason]
        count = stats['count']
        
        if count == 0:
            continue
        
        win_rate = (stats['wins'] / count) * 100
        avg_pnl = stats['total_pnl'] / count
        
        print(f"\n{exit_reason}:")
        print(f"  Count: {count} ({count/len(trades)*100:.1f}% of all trades)")
        print(f"  Total PNL: ${stats['total_pnl']:+.4f}")
        print(f"  Average PNL: ${avg_pnl:+.4f}")
        print(f"  Wins: {stats['wins']} ({win_rate:.1f}%)")
        print(f"  Losses: {stats['losses']} ({100-win_rate:.1f}%)")
    
    # Choppiness Correlation Analysis
    print("\n" + "="*80)
    print("CHOPPINESS CORRELATION")
    print("="*80)
    
    chop_ranges = analyze_choppiness_correlation(trades)
    
    for range_name, pnls in chop_ranges.items():
        if not pnls:
            continue
        
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / len(pnls)) * 100 if pnls else 0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        
        print(f"\n{range_name}:")
        print(f"  Trades: {len(pnls)}")
        print(f"  Total PNL: ${sum(pnls):+.4f}")
        print(f"  Average PNL: ${avg_pnl:+.4f}")
        print(f"  Win Rate: {win_rate:.1f}%")
    
    # Time-based Performance
    print("\n" + "="*80)
    print("HOURLY PERFORMANCE")
    print("="*80)
    
    hourly_stats = analyze_time_based_performance(trades)
    
    print("\nBest Hours (by average PNL):")
    hour_performance = []
    for hour, stats in hourly_stats.items():
        if stats['count'] > 0:
            avg_pnl = sum(stats['pnl']) / stats['count']
            hour_performance.append((hour, avg_pnl, stats['count'], sum(stats['pnl'])))
    
    hour_performance.sort(key=lambda x: x[1], reverse=True)
    
    for hour, avg_pnl, count, total_pnl in hour_performance[:10]:
        print(f"  {hour:02d}:00 - Trades: {count}, Avg PNL: ${avg_pnl:+.4f}, Total: ${total_pnl:+.4f}")
    
    # Strategy C Specific Analysis (Dynamic TP/SL)
    strategy_c_trades = [t for t in trades if t['strategy'] == 'C']
    if strategy_c_trades:
        print("\n" + "="*80)
        print("STRATEGY C - DYNAMIC TP/SL ANALYSIS")
        print("="*80)
        
        tp_distances = []
        sl_distances = []
        ratios = []
        
        for trade in strategy_c_trades:
            entry = trade['entry_price']
            tp = trade['take_profit']
            sl = trade['stop_loss']
            
            tp_dist = tp - entry
            sl_dist = entry - sl
            
            tp_distances.append(tp_dist)
            sl_distances.append(sl_dist)
            
            if sl_dist > 0:
                ratios.append(tp_dist / sl_dist)
        
        print(f"\nTP/SL Distance Statistics:")
        print(f"  Average TP Distance: ${statistics.mean(tp_distances):.4f}")
        print(f"  Average SL Distance: ${statistics.mean(sl_distances):.4f}")
        print(f"  Average TP:SL Ratio: {statistics.mean(ratios):.2f}:1" if ratios else "  N/A")
        print(f"  Min TP: ${min(tp_distances):.4f}, Max TP: ${max(tp_distances):.4f}")
        print(f"  Min SL: ${min(sl_distances):.4f}, Max SL: ${max(sl_distances):.4f}")
        
        # Correlation between choppiness and TP/SL distances
        chop_tp_pairs = []
        for trade in strategy_c_trades:
            chop = trade.get('instantaneous_choppiness')
            if chop is not None:
                tp_dist = trade['take_profit'] - trade['entry_price']
                chop_tp_pairs.append((chop, tp_dist, trade['pnl']))
        
        if chop_tp_pairs:
            print(f"\nChoppiness vs TP Distance (sample):")
            chop_tp_pairs.sort(key=lambda x: x[0])
            for i in range(0, len(chop_tp_pairs), max(1, len(chop_tp_pairs)//5)):
                chop, tp_dist, pnl = chop_tp_pairs[i]
                print(f"  Chop: {chop:5.1f} -> TP Dist: ${tp_dist:.4f}, PNL: ${pnl:+.4f}")
        
        # TP/SL Adjustment Analysis
        adjustments = [t.get('tp_sl_adjustments', 0) for t in strategy_c_trades]
        if adjustments:
            print(f"\nTP/SL Adjustment Statistics:")
            print(f"  Total Adjustments: {sum(adjustments)}")
            print(f"  Average per Trade: {statistics.mean(adjustments):.1f}")
            print(f"  Max Adjustments: {max(adjustments)}")
            print(f"  Trades with Adjustments: {sum(1 for a in adjustments if a > 0)} "
                  f"({sum(1 for a in adjustments if a > 0)/len(adjustments)*100:.1f}%)")
            
            # Correlation between adjustments and PNL
            adj_pnl_pairs = [(t.get('tp_sl_adjustments', 0), t['pnl']) for t in strategy_c_trades]
            adj_groups = defaultdict(list)
            for adj, pnl in adj_pnl_pairs:
                if adj == 0:
                    adj_groups['0 adjustments'].append(pnl)
                elif adj <= 2:
                    adj_groups['1-2 adjustments'].append(pnl)
                else:
                    adj_groups['3+ adjustments'].append(pnl)
            
            print(f"\nPNL by Adjustment Count:")
            for group, pnls in sorted(adj_groups.items()):
                if pnls:
                    avg_pnl = statistics.mean(pnls)
                    wins = sum(1 for p in pnls if p > 0)
                    win_rate = wins / len(pnls) * 100
                    print(f"  {group}: {len(pnls)} trades, Avg PNL: ${avg_pnl:+.4f}, "
                          f"Win Rate: {win_rate:.1f}%")
    
    print("\n" + "="*80)

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Use today's file by default
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f'trading_results_dynamic_{date_str}.json'
    
    print(f"Loading results from: {filename}\n")
    results = load_results(filename)
    print_analysis(results)

if __name__ == "__main__":
    main()
