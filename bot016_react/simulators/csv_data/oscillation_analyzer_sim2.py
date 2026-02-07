import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import statistics

@dataclass
class OscillationStats:
    """Statistics for oscillations in a period"""
    period_name: str
    total_rows: int
    valid_rows: int  # After filtering
    oscillations_0_05: int
    oscillations_0_10: int
    oscillations_0_15: int
    min_price: float
    max_price: float
    price_range: float
    # Trading simulation results for multiple step sizes
    trading_results: Dict[float, Dict]  # Key is step size (0.01, 0.02, etc.)

class OscillationAnalyzer:
    """Analyze price oscillations across multiple CSV periods"""
    
    def __init__(self, csv_directory: str):
        self.csv_directory = Path(csv_directory)
        self.results: List[OscillationStats] = []
    
    def simulate_trading(self, prices_data: List[Dict], step_size: float = 0.05) -> Dict:
        """
        Simulate the MAX tracking and retracement strategy
        
        Rules:
        1. Buy CALL at ask_price at the beginning (first valid row)
        2. Sell at bid_price when profit >= step_size (this becomes new MAX)
        3. Wait for either:
           a) New MAX (price that beats previous MAX)
           b) Retracement of step_size from MAX
        4. When retracement happens, buy again at ask_price (reset MAX to this level)
        5. Repeat until end of period
        6. At last row, sell at bid_price (or 0 if null)
        """
        if not prices_data:
            return {
                'total_pnl': 0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_trade_pnl': 0
            }
        
        total_pnl = 0
        trades = []
        
        # State variables
        position = None  # {'entry_price': float, 'max_since_entry': float}
        current_max = None  # Global MAX tracker
        
        for idx, row in enumerate(prices_data):
            call_bid = row['call_bid']
            call_ask = row['call_ask']
            
            # No position - looking to enter
            if position is None:
                # First trade or re-entry after retracement
                position = {
                    'entry_price': call_ask,
                    'max_since_entry': call_ask
                }
                
                # Update global MAX
                if current_max is None or call_ask > current_max:
                    current_max = call_ask
                
            else:
                # We have a position - check for exit or MAX update
                
                # Update max_since_entry
                if call_bid > position['max_since_entry']:
                    position['max_since_entry'] = call_bid
                
                # Update global MAX
                if call_bid > current_max:
                    current_max = call_bid
                
                # Check for profit target (step_size profit)
                profit = call_bid - position['entry_price']
                
                if profit >= step_size:
                    # SELL - Take profit
                    exit_price = call_bid
                    trade_pnl = exit_price - position['entry_price']
                    total_pnl += trade_pnl
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': exit_price,
                        'pnl': trade_pnl,
                        'type': 'PROFIT_TARGET'
                    })
                    
                    # Update MAX to this exit price (new high)
                    current_max = exit_price
                    
                    # Close position, wait for retracement
                    position = None
                
                # If we're out of position, check for retracement to re-enter
                elif position is None and current_max is not None:
                    # Check if price has retraced step_size from MAX
                    retracement = current_max - call_ask
                    
                    if retracement >= step_size:
                        # Buy again on retracement
                        position = {
                            'entry_price': call_ask,
                            'max_since_entry': call_ask
                        }
        
        # Close any open position at end of period
        if position is not None:
            last_row = prices_data[-1]
            exit_price = last_row['call_bid'] if last_row['call_bid'] else 0.0
            trade_pnl = exit_price - position['entry_price']
            total_pnl += trade_pnl
            trades.append({
                'entry': position['entry_price'],
                'exit': exit_price,
                'pnl': trade_pnl,
                'type': 'END_OF_PERIOD'
            })
        
        # Calculate statistics
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] <= 0])
        avg_trade_pnl = total_pnl / len(trades) if trades else 0
        
        return {
            'total_pnl': total_pnl,
            'num_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade_pnl': avg_trade_pnl,
            'trades': trades
        }
    
    def is_valid_price(self, bid: float, ask: float) -> bool:
        """Check if prices are in valid range (0.30 to 0.70)"""
        return 0.30 <= bid <= 0.70 and 0.30 <= ask <= 0.70
    
    def count_oscillations(self, prices: List[float], threshold: float) -> int:
        """
        Count how many times price oscillates by at least threshold amount
        An oscillation is when price moves threshold in one direction, then reverses
        """
        if len(prices) < 3:
            return 0
        
        oscillations = 0
        i = 0
        
        while i < len(prices) - 1:
            # Find a significant move
            start_price = prices[i]
            j = i + 1
            
            # Look for price movement >= threshold
            max_up = 0
            max_down = 0
            
            while j < len(prices):
                move = prices[j] - start_price
                
                if move > 0:
                    max_up = max(max_up, move)
                else:
                    max_down = min(max_down, move)
                
                # Check if we hit threshold in either direction
                if max_up >= threshold:
                    # We moved up by threshold, now look for reversal
                    peak_idx = j
                    peak_price = prices[j]
                    
                    # Look for reversal down
                    k = j + 1
                    while k < len(prices):
                        if peak_price - prices[k] >= threshold:
                            # Found oscillation: up then down
                            oscillations += 1
                            i = k  # Continue from here
                            break
                        k += 1
                    
                    if k >= len(prices):
                        # No reversal found, move to peak
                        i = peak_idx
                    break
                    
                elif max_down <= -threshold:
                    # We moved down by threshold, now look for reversal
                    trough_idx = j
                    trough_price = prices[j]
                    
                    # Look for reversal up
                    k = j + 1
                    while k < len(prices):
                        if prices[k] - trough_price >= threshold:
                            # Found oscillation: down then up
                            oscillations += 1
                            i = k  # Continue from here
                            break
                        k += 1
                    
                    if k >= len(prices):
                        # No reversal found, move to trough
                        i = trough_idx
                    break
                
                j += 1
            
            # If no significant move found, advance
            if j >= len(prices):
                i += 1
            elif i == j - 1:  # Prevent infinite loop
                i += 1
        
        return oscillations
    
    def analyze_csv(self, csv_file: Path) -> OscillationStats:
        """Analyze a single CSV file"""
        call_ask_prices = []
        prices_data = []  # Store full price data for trading simulation
        total_rows = 0
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                total_rows += 1
                
                try:
                    call_bid = float(row['call_bid'])
                    call_ask = float(row['call_ask'])
                    
                    # Only check first row for validity
                    if idx == 0:
                        if self.is_valid_price(call_bid, call_ask):
                            call_ask_prices.append(call_ask)
                            prices_data.append({'call_bid': call_bid, 'call_ask': call_ask})
                        else:
                            # Skip only the first row if invalid
                            print(f"Warning: Skipping first row in {csv_file.name} (bid={call_bid}, ask={call_ask})")
                            continue
                    else:
                        # All other rows are accepted
                        call_ask_prices.append(call_ask)
                        prices_data.append({'call_bid': call_bid, 'call_ask': call_ask})
                
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row in {csv_file.name}: {e}")
                    continue
        
        if not call_ask_prices:
            print(f"Warning: No valid prices found in {csv_file.name}")
            return None
        
        # Count oscillations for different thresholds
        osc_005 = self.count_oscillations(call_ask_prices, 0.05)
        osc_010 = self.count_oscillations(call_ask_prices, 0.10)
        osc_015 = self.count_oscillations(call_ask_prices, 0.15)
        
        min_price = min(call_ask_prices)
        max_price = max(call_ask_prices)
        
        # Run trading simulation for multiple step sizes (0.01 to 0.50)
        trading_results = {}
        step_sizes = [round(i * 0.01, 2) for i in range(1, 51)]  # 0.01, 0.02, ..., 0.50
        
        for step in step_sizes:
            trading_results[step] = self.simulate_trading(prices_data, step_size=step)
        
        return OscillationStats(
            period_name=csv_file.stem,  # Filename without extension
            total_rows=total_rows,
            valid_rows=len(call_ask_prices),
            oscillations_0_05=osc_005,
            oscillations_0_10=osc_010,
            oscillations_0_15=osc_015,
            min_price=min_price,
            max_price=max_price,
            price_range=max_price - min_price,
            trading_results=trading_results
        )
    
    def analyze_all(self):
        """Analyze all CSV files in the directory"""
        csv_files = sorted(self.csv_directory.glob("prices_*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.csv_directory}")
            return
        
        print(f"Found {len(csv_files)} CSV files to analyze\n")
        
        for csv_file in csv_files:
            print(f"Analyzing {csv_file.name}...", end=" ")
            stats = self.analyze_csv(csv_file)
            
            if stats:
                self.results.append(stats)
                print(f"✓ ({stats.valid_rows}/{stats.total_rows} valid rows)")
            else:
                print("✗ No valid data")
        
        print(f"\n{'='*100}")
        self.print_results()
    
    def print_results(self):
        """Print analysis results"""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n📊 OSCILLATION ANALYSIS RESULTS")
        print(f"{'='*120}\n")
        
        # Print per-period oscillation stats
        print(f"{'Period':<25} {'Rows':<8} {'$0.05':<6} {'$0.10':<6} {'$0.15':<6} {'Range':<10}")
        print(f"{'-'*80}")
        
        for stats in self.results:
            print(f"{stats.period_name:<25} {stats.valid_rows:<8} "
                  f"{stats.oscillations_0_05:<6} {stats.oscillations_0_10:<6} {stats.oscillations_0_15:<6} "
                  f"${stats.price_range:<9.4f}")
        
        print(f"\n{'='*120}")
        
        # Calculate aggregate results for each step size across all periods
        step_sizes = sorted(self.results[0].trading_results.keys())
        
        aggregate_results = {}
        for step in step_sizes:
            total_pnl = sum([s.trading_results[step]['total_pnl'] for s in self.results])
            total_trades = sum([s.trading_results[step]['num_trades'] for s in self.results])
            total_wins = sum([s.trading_results[step]['winning_trades'] for s in self.results])
            total_losses = sum([s.trading_results[step]['losing_trades'] for s in self.results])
            avg_pnl_per_period = total_pnl / len(self.results)
            profitable_periods = len([s for s in self.results if s.trading_results[step]['total_pnl'] > 0])
            
            aggregate_results[step] = {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'avg_pnl_per_period': avg_pnl_per_period,
                'win_rate': total_wins / total_trades if total_trades > 0 else 0,
                'avg_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
                'profitable_periods': profitable_periods,
                'profitability_rate': profitable_periods / len(self.results)
            }
        
        # Print summary table for all step sizes
        print(f"\n💰 TRADING SIMULATION RESULTS - ALL STEP SIZES")
        print(f"{'='*120}")
        print(f"{'Step':<8} {'Total PNL':<12} {'Avg/Period':<12} {'Trades':<8} {'Win Rate':<10} {'Avg/Trade':<12} {'Profit %':<10}")
        print(f"{'-'*120}")
        
        for step in step_sizes:
            r = aggregate_results[step]
            print(f"${step:<7.2f} ${r['total_pnl']:>10.4f} ${r['avg_pnl_per_period']:>10.4f} "
                  f"{r['total_trades']:<8} {r['win_rate']*100:>8.1f}% ${r['avg_per_trade']:>10.5f} "
                  f"{r['profitability_rate']*100:>8.1f}%")
        
        # Find top 10 best performing step sizes
        sorted_by_pnl = sorted(aggregate_results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        
        print(f"\n{'='*120}")
        print(f"🏆 TOP 10 BEST PERFORMING STEP SIZES (by Total PNL)")
        print(f"{'='*120}")
        print(f"{'Rank':<6} {'Step':<8} {'Total PNL':<12} {'Avg/Period':<12} {'Trades':<8} {'Win Rate':<10} {'Avg/Trade':<12} {'Profit %':<10}")
        print(f"{'-'*120}")
        
        for rank, (step, r) in enumerate(sorted_by_pnl[:10], 1):
            print(f"{rank:<6} ${step:<7.2f} ${r['total_pnl']:>10.4f} ${r['avg_pnl_per_period']:>10.4f} "
                  f"{r['total_trades']:<8} {r['win_rate']*100:>8.1f}% ${r['avg_per_trade']:>10.5f} "
                  f"{r['profitability_rate']*100:>8.1f}%")
        
        # Find top 10 by win rate
        sorted_by_winrate = sorted(aggregate_results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        print(f"\n{'='*120}")
        print(f"🎯 TOP 10 HIGHEST WIN RATES")
        print(f"{'='*120}")
        print(f"{'Rank':<6} {'Step':<8} {'Win Rate':<10} {'Total PNL':<12} {'Trades':<8} {'Avg/Trade':<12}")
        print(f"{'-'*120}")
        
        for rank, (step, r) in enumerate(sorted_by_winrate[:10], 1):
            print(f"{rank:<6} ${step:<7.2f} {r['win_rate']*100:>8.1f}% ${r['total_pnl']:>10.4f} "
                  f"{r['total_trades']:<8} ${r['avg_per_trade']:>10.5f}")
        
        # Find top 10 by average per trade
        sorted_by_avg = sorted(aggregate_results.items(), key=lambda x: x[1]['avg_per_trade'], reverse=True)
        
        print(f"\n{'='*120}")
        print(f"💎 TOP 10 HIGHEST AVERAGE PER TRADE")
        print(f"{'='*120}")
        print(f"{'Rank':<6} {'Step':<8} {'Avg/Trade':<12} {'Total PNL':<12} {'Trades':<8} {'Win Rate':<10}")
        print(f"{'-'*120}")
        
        for rank, (step, r) in enumerate(sorted_by_avg[:10], 1):
            print(f"{rank:<6} ${step:<7.2f} ${r['avg_per_trade']:>10.5f} ${r['total_pnl']:>10.4f} "
                  f"{r['total_trades']:<8} {r['win_rate']*100:>8.1f}%")
        
        # Overall summary
        best_step = sorted_by_pnl[0]
        print(f"\n{'='*120}")
        print(f"📈 OVERALL SUMMARY:")
        print(f"   Total periods analyzed: {len(self.results)}")
        print(f"   Step sizes tested: {len(step_sizes)} (from $0.01 to $0.50)")
        print(f"\n   🥇 BEST OVERALL STEP SIZE: ${best_step[0]:.2f}")
        print(f"      Total PNL: ${best_step[1]['total_pnl']:+.4f}")
        print(f"      Avg PNL per period: ${best_step[1]['avg_pnl_per_period']:+.4f}")
        print(f"      Total Trades: {best_step[1]['total_trades']}")
        print(f"      Win Rate: {best_step[1]['win_rate']*100:.1f}%")
        print(f"      Avg per Trade: ${best_step[1]['avg_per_trade']:+.5f}")
        print(f"      Profitable Periods: {best_step[1]['profitable_periods']}/{len(self.results)} ({best_step[1]['profitability_rate']*100:.1f}%)")
        print(f"{'='*120}\n")
    
    def save_report(self, output_file: str = None):
        """Save detailed report to file"""
        if output_file is None:
            output_file = self.csv_directory / "oscillation_analysis_report.txt"
        
        with open(output_file, 'w') as f:
            f.write("OSCILLATION ANALYSIS REPORT\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"Analysis Date: {Path(csv_file).stat().st_mtime}\n")
            f.write(f"Total Periods: {len(self.results)}\n\n")
            
            f.write(f"{'Period':<25} {'Valid Rows':<12} {'$0.05':<8} {'$0.10':<8} {'$0.15':<8} {'Min':<8} {'Max':<8} {'Range':<8}\n")
            f.write("-"*100 + "\n")
            
            for stats in self.results:
                f.write(f"{stats.period_name:<25} {stats.valid_rows:<12} "
                       f"{stats.oscillations_0_05:<8} {stats.oscillations_0_10:<8} {stats.oscillations_0_15:<8} "
                       f"${stats.min_price:<7.3f} ${stats.max_price:<7.3f} ${stats.price_range:<7.3f}\n")
            
            # Averages
            avg_005 = statistics.mean([s.oscillations_0_05 for s in self.results])
            avg_010 = statistics.mean([s.oscillations_0_10 for s in self.results])
            avg_015 = statistics.mean([s.oscillations_0_15 for s in self.results])
            
            f.write("\n" + "="*100 + "\n")
            f.write(f"AVERAGES:\n")
            f.write(f"  $0.05 threshold: {avg_005:.2f} oscillations per period\n")
            f.write(f"  $0.10 threshold: {avg_010:.2f} oscillations per period\n")
            f.write(f"  $0.15 threshold: {avg_015:.2f} oscillations per period\n")
        
        print(f"💾 Report saved to: {output_file}")


if __name__ == "__main__":
    # Analyze all CSV files in the directory
    analyzer = OscillationAnalyzer(
        csv_directory="/home/ubuntu/013_2025_polymarket/bot016_react/simulators/csv_data"
    )
    
    analyzer.analyze_all()
    analyzer.save_report()
