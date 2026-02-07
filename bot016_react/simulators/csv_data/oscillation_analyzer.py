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

class OscillationAnalyzer:
    """Analyze price oscillations across multiple CSV periods"""
    
    def __init__(self, csv_directory: str):
        self.csv_directory = Path(csv_directory)
        self.results: List[OscillationStats] = []
    
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
                        else:
                            # Skip only the first row if invalid
                            print(f"Warning: Skipping first row in {csv_file.name} (bid={call_bid}, ask={call_ask})")
                            continue
                    else:
                        # All other rows are accepted
                        call_ask_prices.append(call_ask)
                
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
        
        return OscillationStats(
            period_name=csv_file.stem,  # Filename without extension
            total_rows=total_rows,
            valid_rows=len(call_ask_prices),
            oscillations_0_05=osc_005,
            oscillations_0_10=osc_010,
            oscillations_0_15=osc_015,
            min_price=min_price,
            max_price=max_price,
            price_range=max_price - min_price
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
        print(f"{'='*100}\n")
        
        # Print per-period results
        print(f"{'Period':<25} {'Valid Rows':<12} {'$0.05':<8} {'$0.10':<8} {'$0.15':<8} {'Min':<8} {'Max':<8} {'Range':<8}")
        print(f"{'-'*100}")
        
        for stats in self.results:
            print(f"{stats.period_name:<25} {stats.valid_rows:<12} "
                  f"{stats.oscillations_0_05:<8} {stats.oscillations_0_10:<8} {stats.oscillations_0_15:<8} "
                  f"${stats.min_price:<7.3f} ${stats.max_price:<7.3f} ${stats.price_range:<7.3f}")
        
        # Calculate averages
        print(f"{'-'*100}")
        
        avg_rows = statistics.mean([s.valid_rows for s in self.results])
        avg_005 = statistics.mean([s.oscillations_0_05 for s in self.results])
        avg_010 = statistics.mean([s.oscillations_0_10 for s in self.results])
        avg_015 = statistics.mean([s.oscillations_0_15 for s in self.results])
        avg_min = statistics.mean([s.min_price for s in self.results])
        avg_max = statistics.mean([s.max_price for s in self.results])
        avg_range = statistics.mean([s.price_range for s in self.results])
        
        print(f"{'AVERAGE':<25} {avg_rows:<12.1f} "
              f"{avg_005:<8.2f} {avg_010:<8.2f} {avg_015:<8.2f} "
              f"${avg_min:<7.3f} ${avg_max:<7.3f} ${avg_range:<7.3f}")
        
        # Calculate totals
        total_005 = sum([s.oscillations_0_05 for s in self.results])
        total_010 = sum([s.oscillations_0_10 for s in self.results])
        total_015 = sum([s.oscillations_0_15 for s in self.results])
        
        print(f"{'TOTAL':<25} {sum([s.valid_rows for s in self.results]):<12} "
              f"{total_005:<8} {total_010:<8} {total_015:<8}")
        
        print(f"\n{'='*100}")
        
        # Summary
        print(f"\n📈 SUMMARY:")
        print(f"   Total periods analyzed: {len(self.results)}")
        print(f"   Average oscillations per period:")
        print(f"      $0.05 threshold: {avg_005:.2f} oscillations")
        print(f"      $0.10 threshold: {avg_010:.2f} oscillations")
        print(f"      $0.15 threshold: {avg_015:.2f} oscillations")
        print(f"\n   Total oscillations across all periods:")
        print(f"      $0.05 threshold: {total_005} oscillations")
        print(f"      $0.10 threshold: {total_010} oscillations")
        print(f"      $0.15 threshold: {total_015} oscillations")
        print(f"\n   Average price range per period: ${avg_range:.4f}")
        print(f"{'='*100}\n")
    
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
