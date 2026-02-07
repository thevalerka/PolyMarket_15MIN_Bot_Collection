#!/usr/bin/env python3
"""
Trade Analysis Tool
Analyzes simulated trades to find optimal parameters:
- Optimal edge threshold
- Optimal sensitivity multiplier
- Best performing bins
- Trade duration analysis
- Win rate by conditions
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict
import statistics


class TradeAnalyzer:
    """Analyze trading performance and find optimal parameters"""
    
    def __init__(self, trades_file: str):
        self.trades_file = Path(trades_file)
        self.trades = []
        self.load_trades()
        
    def load_trades(self):
        """Load trades from JSON file"""
        if not self.trades_file.exists():
            print(f"❌ Error: File not found: {self.trades_file}")
            sys.exit(1)
        
        with open(self.trades_file, 'r') as f:
            data = json.load(f)
            self.trades = data.get('trades', [])
            self.date = data.get('date', 'Unknown')
            self.total_pnl = data.get('daily_pnl', 0)
        
        print(f"📂 Loaded {len(self.trades)} trades from {self.date}")
        print(f"💰 Total PNL: {self.total_pnl:+.3f}\n")
    
    def analyze_by_edge_threshold(self) -> Dict:
        """Find optimal edge threshold"""
        print("="*80)
        print("📊 EDGE THRESHOLD ANALYSIS")
        print("="*80)
        
        # Test different edge thresholds
        edge_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        results = []
        
        for threshold in edge_thresholds:
            # Filter trades by edge threshold
            filtered_trades = [t for t in self.trades if t.get('edge', 0) >= threshold]
            
            if not filtered_trades:
                continue
            
            total_pnl = sum(t['pnl'] for t in filtered_trades)
            wins = sum(1 for t in filtered_trades if t['pnl'] > 0)
            losses = sum(1 for t in filtered_trades if t['pnl'] < 0)
            win_rate = wins / len(filtered_trades) if filtered_trades else 0
            
            avg_win = statistics.mean([t['pnl'] for t in filtered_trades if t['pnl'] > 0]) if wins > 0 else 0
            avg_loss = statistics.mean([t['pnl'] for t in filtered_trades if t['pnl'] < 0]) if losses > 0 else 0
            
            profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')
            
            results.append({
                'threshold': threshold,
                'trades': len(filtered_trades),
                'pnl': total_pnl,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'pnl_per_trade': total_pnl / len(filtered_trades) if filtered_trades else 0
            })
        
        # Print results
        print(f"\n{'Threshold':<10} {'Trades':<8} {'Win Rate':<10} {'PNL':<10} {'Avg Win':<10} {'Avg Loss':<10} {'PF':<8} {'PNL/Trade':<10}")
        print("-"*90)
        
        for r in results:
            print(f"{r['threshold']:<10.2f} {r['trades']:<8} {r['win_rate']:<10.1%} "
                  f"{r['pnl']:+<10.3f} {r['avg_win']:<10.3f} {r['avg_loss']:<10.3f} "
                  f"{r['profit_factor']:<8.2f} {r['pnl_per_trade']:+<10.3f}")
        
        # Find optimal
        best_by_pnl = max(results, key=lambda x: x['pnl'])
        best_by_pnl_per_trade = max(results, key=lambda x: x['pnl_per_trade'])
        best_by_win_rate = max(results, key=lambda x: x['win_rate'])
        
        print(f"\n🏆 OPTIMAL BY TOTAL PNL: Edge >= {best_by_pnl['threshold']:.2f} "
              f"→ PNL: {best_by_pnl['pnl']:+.3f} ({best_by_pnl['trades']} trades)")
        print(f"🏆 OPTIMAL BY PNL/TRADE: Edge >= {best_by_pnl_per_trade['threshold']:.2f} "
              f"→ PNL/Trade: {best_by_pnl_per_trade['pnl_per_trade']:+.3f}")
        print(f"🏆 OPTIMAL BY WIN RATE: Edge >= {best_by_win_rate['threshold']:.2f} "
              f"→ Win Rate: {best_by_win_rate['win_rate']:.1%}")
        
        return {'best_by_pnl': best_by_pnl, 'best_by_pnl_per_trade': best_by_pnl_per_trade}
    
    def analyze_by_bin(self) -> Dict:
        """Analyze performance by bin"""
        print("\n" + "="*80)
        print("📦 BIN PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Group by open_bin
        bin_stats = defaultdict(lambda: {'trades': [], 'wins': 0, 'losses': 0})
        
        for trade in self.trades:
            bin_key = trade.get('open_bin', 'Unknown')
            bin_stats[bin_key]['trades'].append(trade)
            if trade['pnl'] > 0:
                bin_stats[bin_key]['wins'] += 1
            else:
                bin_stats[bin_key]['losses'] += 1
        
        # Calculate stats
        bin_results = []
        for bin_key, stats in bin_stats.items():
            trades = stats['trades']
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = stats['wins'] / len(trades) if trades else 0
            avg_pnl = total_pnl / len(trades) if trades else 0
            
            bin_results.append({
                'bin': bin_key,
                'trades': len(trades),
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            })
        
        # Sort by total PNL
        bin_results.sort(key=lambda x: x['total_pnl'], reverse=True)
        
        print(f"\n{'Bin':<35} {'Trades':<8} {'Win Rate':<10} {'Total PNL':<12} {'Avg PNL':<10}")
        print("-"*85)
        
        # Show top 15 and bottom 5
        for i, r in enumerate(bin_results[:15]):
            print(f"{r['bin']:<35} {r['trades']:<8} {r['win_rate']:<10.1%} "
                  f"{r['total_pnl']:+<12.3f} {r['avg_pnl']:+<10.3f}")
        
        if len(bin_results) > 20:
            print("...")
            for r in bin_results[-5:]:
                print(f"{r['bin']:<35} {r['trades']:<8} {r['win_rate']:<10.1%} "
                      f"{r['total_pnl']:+<12.3f} {r['avg_pnl']:+<10.3f}")
        
        print(f"\n🏆 BEST BIN: {bin_results[0]['bin']} "
              f"→ PNL: {bin_results[0]['total_pnl']:+.3f} ({bin_results[0]['trades']} trades)")
        print(f"❌ WORST BIN: {bin_results[-1]['bin']} "
              f"→ PNL: {bin_results[-1]['total_pnl']:+.3f} ({bin_results[-1]['trades']} trades)")
        
        return bin_results
    
    def analyze_by_type(self) -> Dict:
        """Analyze CALL vs PUT performance"""
        print("\n" + "="*80)
        print("📈 CALL vs PUT ANALYSIS")
        print("="*80)
        
        call_trades = [t for t in self.trades if t['type'] == 'CALL']
        put_trades = [t for t in self.trades if t['type'] == 'PUT']
        
        for name, trades in [('CALL', call_trades), ('PUT', put_trades)]:
            if not trades:
                continue
            
            total_pnl = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            losses = sum(1 for t in trades if t['pnl'] < 0)
            win_rate = wins / len(trades)
            avg_pnl = total_pnl / len(trades)
            avg_edge = statistics.mean([t['edge'] for t in trades])
            
            print(f"\n{name}:")
            print(f"  Trades: {len(trades)}")
            print(f"  Win Rate: {win_rate:.1%} ({wins}W/{losses}L)")
            print(f"  Total PNL: {total_pnl:+.3f}")
            print(f"  Avg PNL/Trade: {avg_pnl:+.4f}")
            print(f"  Avg Edge: {avg_edge:.4f}")
        
        return {'call': call_trades, 'put': put_trades}
    
    def analyze_trade_duration(self) -> Dict:
        """Analyze trade duration vs PNL"""
        print("\n" + "="*80)
        print("⏱️  TRADE DURATION ANALYSIS")
        print("="*80)
        
        duration_buckets = {
            '<30s': [],
            '30s-1m': [],
            '1m-2m': [],
            '2m-5m': [],
            '5m-10m': [],
            '>10m': []
        }
        
        for trade in self.trades:
            try:
                open_time = datetime.fromisoformat(trade['open_time'])
                close_time = datetime.fromisoformat(trade['close_time'])
                duration = (close_time - open_time).total_seconds()
                
                if duration < 30:
                    duration_buckets['<30s'].append(trade)
                elif duration < 60:
                    duration_buckets['30s-1m'].append(trade)
                elif duration < 120:
                    duration_buckets['1m-2m'].append(trade)
                elif duration < 300:
                    duration_buckets['2m-5m'].append(trade)
                elif duration < 600:
                    duration_buckets['5m-10m'].append(trade)
                else:
                    duration_buckets['>10m'].append(trade)
            except:
                continue
        
        print(f"\n{'Duration':<10} {'Trades':<8} {'Win Rate':<10} {'Total PNL':<12} {'Avg PNL':<10}")
        print("-"*60)
        
        for duration, trades in duration_buckets.items():
            if not trades:
                continue
            
            total_pnl = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = wins / len(trades)
            avg_pnl = total_pnl / len(trades)
            
            print(f"{duration:<10} {len(trades):<8} {win_rate:<10.1%} "
                  f"{total_pnl:+<12.3f} {avg_pnl:+<10.4f}")
        
        return duration_buckets
    
    def analyze_close_reasons(self) -> Dict:
        """Analyze performance by close reason"""
        print("\n" + "="*80)
        print("🔚 CLOSE REASON ANALYSIS")
        print("="*80)
        
        reason_stats = defaultdict(lambda: {'trades': [], 'pnl': 0})
        
        for trade in self.trades:
            reason = trade.get('close_reason', 'Unknown')
            reason_stats[reason]['trades'].append(trade)
            reason_stats[reason]['pnl'] += trade['pnl']
        
        print(f"\n{'Reason':<20} {'Trades':<8} {'Win Rate':<10} {'Total PNL':<12} {'Avg PNL':<10}")
        print("-"*70)
        
        for reason, stats in reason_stats.items():
            trades = stats['trades']
            wins = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = wins / len(trades) if trades else 0
            avg_pnl = stats['pnl'] / len(trades) if trades else 0
            
            print(f"{reason:<20} {len(trades):<8} {win_rate:<10.1%} "
                  f"{stats['pnl']:+<12.3f} {avg_pnl:+<10.4f}")
        
        return reason_stats
    
    def find_optimal_edge_by_bin(self) -> Dict:
        """Find optimal edge threshold for each bin"""
        print("\n" + "="*80)
        print("🎯 OPTIMAL EDGE BY BIN")
        print("="*80)
        
        # Group by bin
        bin_trades = defaultdict(list)
        for trade in self.trades:
            bin_trades[trade['open_bin']].append(trade)
        
        # For bins with >5 trades, find optimal edge
        optimal_edges = []
        
        for bin_key, trades in bin_trades.items():
            if len(trades) < 5:
                continue
            
            # Test edge thresholds
            best_pnl = float('-inf')
            best_threshold = 0
            
            for threshold in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                filtered = [t for t in trades if t['edge'] >= threshold]
                if not filtered:
                    continue
                
                pnl = sum(t['pnl'] for t in filtered)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_threshold = threshold
            
            optimal_edges.append({
                'bin': bin_key,
                'optimal_edge': best_threshold,
                'pnl_at_optimal': best_pnl,
                'total_trades': len(trades),
                'trades_at_optimal': len([t for t in trades if t['edge'] >= best_threshold])
            })
        
        # Sort by PNL improvement
        optimal_edges.sort(key=lambda x: x['pnl_at_optimal'], reverse=True)
        
        print(f"\n{'Bin':<35} {'Optimal Edge':<13} {'PNL':<12} {'Trades':<10}")
        print("-"*75)
        
        for oe in optimal_edges[:10]:
            print(f"{oe['bin']:<35} {oe['optimal_edge']:<13.2f} "
                  f"{oe['pnl_at_optimal']:+<12.3f} "
                  f"{oe['trades_at_optimal']}/{oe['total_trades']}")
        
        return optimal_edges
    
    def generate_summary(self):
        """Generate overall summary"""
        print("\n" + "="*80)
        print("📋 OVERALL SUMMARY")
        print("="*80)
        
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        losses = sum(1 for t in self.trades if t['pnl'] < 0)
        win_rate = wins / total_trades if total_trades else 0
        
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        avg_win = statistics.mean(winning_trades) if winning_trades else 0
        avg_loss = statistics.mean(losing_trades) if losing_trades else 0
        
        total_win = sum(winning_trades)
        total_loss = sum(losing_trades)
        
        profit_factor = abs(total_win / total_loss) if total_loss != 0 else float('inf')
        
        print(f"\nDate: {self.date}")
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins} ({win_rate:.1%})")
        print(f"Losses: {losses}")
        print(f"\nPNL Breakdown:")
        print(f"  Total PNL: {self.total_pnl:+.3f}")
        print(f"  Total Wins: {total_win:+.3f}")
        print(f"  Total Losses: {total_loss:+.3f}")
        print(f"  Avg Win: {avg_win:+.4f}")
        print(f"  Avg Loss: {avg_loss:+.4f}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  PNL per Trade: {self.total_pnl/total_trades:+.4f}")
        
        # Edge stats
        edges = [t['edge'] for t in self.trades]
        print(f"\nEdge Statistics:")
        print(f"  Avg Edge: {statistics.mean(edges):.4f}")
        print(f"  Median Edge: {statistics.median(edges):.4f}")
        print(f"  Min Edge: {min(edges):.4f}")
        print(f"  Max Edge: {max(edges):.4f}")


def main():
    """Main analysis function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_trades.py <trades_YYYYMMDD.json>")
        print("\nExample:")
        print("  python analyze_trades.py trades_20251228.json")
        sys.exit(1)
    
    trades_file = sys.argv[1]
    
    print("\n" + "="*80)
    print("🔍 TRADE PERFORMANCE ANALYZER")
    print("="*80 + "\n")
    
    analyzer = TradeAnalyzer(trades_file)
    
    # Run all analyses
    analyzer.generate_summary()
    analyzer.analyze_by_edge_threshold()
    analyzer.analyze_by_bin()
    analyzer.analyze_by_type()
    analyzer.analyze_trade_duration()
    analyzer.analyze_close_reasons()
    analyzer.find_optimal_edge_by_bin()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
