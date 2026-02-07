import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import glob
import os
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class BinaryOptionReversalAnalyzer:
    def __init__(self, data_folder="/home/ubuntu/013_2025_polymarket/bot004_blackScholes/daily/"):
        self.data_folder = data_folder
        self.reversal_threshold = 0.70  # 70% threshold for reversals
        self.reversal_target = 0.30     # 30% target for reversals
        self.data = []
        self.reversals = []
        
    def load_daily_files(self, start_date=None, end_date=None):
        """Load all daily JSON files from the specified date range"""
        print("📂 Loading daily files...")
        
        # Get all JSON files in the directory
        file_pattern = f"{self.data_folder}option_probabilities_M15_*.json"
        files = glob.glob(file_pattern)
        files.sort()
        
        if not files:
            print(f"❌ No files found in {self.data_folder}")
            return False
        
        print(f"📁 Found {len(files)} daily files")
        
        all_data = []
        
        for file_path in files:
            try:
                # Extract date from filename
                filename = os.path.basename(file_path)
                date_str = filename.replace("option_probabilities_M15_", "").replace(".json", "")
                
                # Skip if outside date range
                if start_date and date_str < start_date:
                    continue
                if end_date and date_str > end_date:
                    continue
                
                print(f"📖 Loading {date_str}...")
                
                # Read JSONL format (one JSON per line)
                with open(file_path, 'r') as f:
                    daily_data = []
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                # Add file info
                                record['file_date'] = date_str
                                daily_data.append(record)
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Error parsing line {line_num + 1} in {filename}: {e}")
                                continue
                    
                    all_data.extend(daily_data)
                    print(f"✅ Loaded {len(daily_data)} records from {date_str}")
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        if not all_data:
            print("❌ No data loaded")
            return False
        
        # Convert to DataFrame for easier analysis
        self.data = pd.DataFrame(all_data)
        
        # Handle mixed datetime formats (with/without microseconds)
        try:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'], format='ISO8601')
        except ValueError:
            # Fallback to mixed format parsing
            self.data['datetime'] = pd.to_datetime(self.data['datetime'], format='mixed')
        
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        
        print(f"✅ Total loaded: {len(self.data)} records from {len(files)} files")
        print(f"📅 Date range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
        
        return True
    
    def add_time_features(self):
        """Add time-based features for analysis"""
        print("🕐 Adding time features...")
        
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['minute'] = self.data['datetime'].dt.minute
        self.data['day_of_week'] = self.data['datetime'].dt.dayofweek  # 0=Monday
        
        # Quarter of hour (0-15min=0, 15-30min=1, 30-45min=2, 45-60min=3)
        self.data['quarter_of_hour'] = self.data['minute'] // 15
        
        # Time until next expiry in minutes
        def time_to_next_expiry(minute):
            if minute < 15:
                return 15 - minute
            elif minute < 30:
                return 30 - minute
            elif minute < 45:
                return 45 - minute
            else:
                return 60 - minute
        
        self.data['minutes_to_expiry'] = self.data['minute'].apply(time_to_next_expiry)
        
        # Add 4-hour price windows
        self.add_4hour_features()
    
    def add_4hour_features(self):
        """Add 4-hour rolling max/min price features"""
        print("📊 Adding 4-hour price features...")
        
        # Sort by datetime to ensure proper rolling calculations
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        
        # 4-hour window (240 minutes) = 14400 seconds
        window_seconds = 4 * 60 * 60  # 4 hours in seconds
        
        # Calculate rolling 4-hour max and min prices
        max_prices = []
        min_prices = []
        price_ranges = []
        
        for i, row in self.data.iterrows():
            current_time = row['datetime']
            start_time = current_time - pd.Timedelta(seconds=window_seconds)
            
            # Get data in the 4-hour window
            window_mask = (self.data['datetime'] >= start_time) & (self.data['datetime'] <= current_time)
            window_data = self.data[window_mask]
            
            if len(window_data) > 0:
                max_price = window_data['current_price'].max()
                min_price = window_data['current_price'].min()
                price_range = max_price - min_price
            else:
                max_price = row['current_price']
                min_price = row['current_price']
                price_range = 0
            
            max_prices.append(max_price)
            min_prices.append(min_price)
            price_ranges.append(price_range)
            
            if i % 1000 == 0:
                print(f"   Processed {i:,} records...")
        
        self.data['4h_max_price'] = max_prices
        self.data['4h_min_price'] = min_prices
        self.data['4h_price_range'] = price_ranges
        
        # Calculate current price position within 4-hour range
        self.data['price_position_in_4h'] = (
            (self.data['current_price'] - self.data['4h_min_price']) / 
            (self.data['4h_price_range'] + 1e-8)  # Add small value to avoid division by zero
        )
    
    def identify_reversals(self):
        """Identify probability reversals (70% to 30% or vice versa)"""
        print("🔄 Identifying probability reversals...")
        
        reversals = []
        
        for i in range(1, len(self.data)):
            prev_row = self.data.iloc[i-1]
            curr_row = self.data.iloc[i]
            
            # Check for call probability reversals
            call_reversal = self._check_reversal(
                prev_row['call_probability'], 
                curr_row['call_probability'],
                'call'
            )
            
            # Check for put probability reversals
            put_reversal = self._check_reversal(
                prev_row['put_probability'], 
                curr_row['put_probability'],
                'put'
            )
            
            if call_reversal or put_reversal:
                reversal_data = {
                    'index': i,
                    'datetime': curr_row['datetime'],
                    'file_date': curr_row['file_date'],
                    'hour': curr_row['hour'],
                    'minute': curr_row['minute'],
                    'quarter_of_hour': curr_row['quarter_of_hour'],
                    'minutes_to_expiry': curr_row['minutes_to_expiry'],
                    'day_of_week': curr_row['day_of_week'],
                    
                    # Before/after probabilities
                    'call_prob_before': prev_row['call_probability'],
                    'call_prob_after': curr_row['call_probability'],
                    'put_prob_before': prev_row['put_probability'],
                    'put_prob_after': curr_row['put_probability'],
                    
                    # Price data
                    'price_before': prev_row['current_price'],
                    'price_after': curr_row['current_price'],
                    'strike_price': curr_row['strike_price'],
                    'price_change_pct': curr_row['price_change_pct'],
                    
                    # Volatility
                    'volatility_before': prev_row['volatility'],
                    'volatility_after': curr_row['volatility'],
                    'volatility_change': curr_row['volatility'] - prev_row['volatility'],
                    
                    # 4-hour context
                    '4h_price_range': curr_row['4h_price_range'],
                    'price_position_in_4h': curr_row['price_position_in_4h'],
                    
                    # Reversal type
                    'reversal_type': call_reversal if call_reversal else put_reversal
                }
                
                reversals.append(reversal_data)
        
        self.reversals = pd.DataFrame(reversals)
        
        print(f"✅ Found {len(self.reversals)} probability reversals")
        
        if len(self.reversals) > 0:
            print(f"📊 Reversal types:")
            reversal_counts = self.reversals['reversal_type'].value_counts()
            for rev_type, count in reversal_counts.items():
                print(f"   • {rev_type}: {count}")
        
        return len(self.reversals)
    
    def _check_reversal(self, prob_before, prob_after, prob_type):
        """Check if a probability reversal occurred"""
        # High to low reversal (≥70% to ≤30%)
        if prob_before >= self.reversal_threshold and prob_after <= self.reversal_target:
            return f"{prob_type}_high_to_low"
        
        # Low to high reversal (≤30% to ≥70%)
        if prob_before <= self.reversal_target and prob_after >= self.reversal_threshold:
            return f"{prob_type}_low_to_high"
        
        return None
    
    def analyze_correlations(self):
        """Analyze correlations between reversals and various factors"""
        if len(self.reversals) == 0:
            print("❌ No reversals found to analyze")
            return {}
        
        print("🔍 Analyzing correlations...")
        
        correlations = {}
        
        # 1. Time of day analysis
        correlations['time_of_day'] = self._analyze_time_patterns()
        
        # 2. Quarter of hour analysis
        correlations['quarter_of_hour'] = self._analyze_quarter_patterns()
        
        # 3. Volatility correlation
        correlations['volatility'] = self._analyze_volatility_correlation()
        
        # 4. 4-hour price range correlation
        correlations['4hour_range'] = self._analyze_4hour_correlation()
        
        # 5. Minutes to expiry correlation
        correlations['expiry_timing'] = self._analyze_expiry_timing()
        
        return correlations
    
    def _analyze_time_patterns(self):
        """Analyze reversal patterns by time of day"""
        hour_counts = self.reversals['hour'].value_counts().sort_index()
        total_counts_by_hour = self.data['hour'].value_counts().sort_index()
        
        # Calculate reversal rate by hour
        reversal_rates = {}
        for hour in range(24):
            reversals_in_hour = hour_counts.get(hour, 0)
            total_in_hour = total_counts_by_hour.get(hour, 1)  # Avoid division by zero
            reversal_rates[hour] = reversals_in_hour / total_in_hour
        
        return {
            'reversal_counts_by_hour': hour_counts.to_dict(),
            'reversal_rates_by_hour': reversal_rates,
            'peak_reversal_hours': sorted(reversal_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _analyze_quarter_patterns(self):
        """Analyze reversal patterns by quarter of hour"""
        quarter_counts = self.reversals['quarter_of_hour'].value_counts().sort_index()
        total_counts_by_quarter = self.data['quarter_of_hour'].value_counts().sort_index()
        
        quarter_names = {0: "0-15min", 1: "15-30min", 2: "30-45min", 3: "45-60min"}
        
        reversal_rates = {}
        for quarter in range(4):
            reversals_in_quarter = quarter_counts.get(quarter, 0)
            total_in_quarter = total_counts_by_quarter.get(quarter, 1)
            reversal_rates[quarter_names[quarter]] = reversals_in_quarter / total_in_quarter
        
        return {
            'reversal_counts_by_quarter': {quarter_names[k]: v for k, v in quarter_counts.to_dict().items()},
            'reversal_rates_by_quarter': reversal_rates
        }
    
    def _analyze_volatility_correlation(self):
        """Analyze correlation between volatility changes and reversals"""
        vol_changes = self.reversals['volatility_change'].values
        vol_before = self.reversals['volatility_before'].values
        vol_after = self.reversals['volatility_after'].values
        
        return {
            'avg_volatility_change': np.mean(vol_changes),
            'median_volatility_change': np.median(vol_changes),
            'std_volatility_change': np.std(vol_changes),
            'avg_volatility_before': np.mean(vol_before),
            'avg_volatility_after': np.mean(vol_after),
            'volatility_increase_pct': len(vol_changes[vol_changes > 0]) / len(vol_changes) * 100
        }
    
    def _analyze_4hour_correlation(self):
        """Analyze correlation with 4-hour price movements"""
        price_ranges = self.reversals['4h_price_range'].values
        price_positions = self.reversals['price_position_in_4h'].values
        
        return {
            'avg_4h_price_range': np.mean(price_ranges),
            'median_4h_price_range': np.median(price_ranges),
            'avg_price_position_in_4h': np.mean(price_positions),
            'reversals_at_extremes_pct': len(price_positions[(price_positions < 0.2) | (price_positions > 0.8)]) / len(price_positions) * 100
        }
    
    def _analyze_expiry_timing(self):
        """Analyze correlation with time until expiry"""
        expiry_times = self.reversals['minutes_to_expiry'].values
        
        return {
            'avg_minutes_to_expiry': np.mean(expiry_times),
            'median_minutes_to_expiry': np.median(expiry_times),
            'reversals_in_last_5min_pct': len(expiry_times[expiry_times <= 5]) / len(expiry_times) * 100,
            'reversals_in_last_3min_pct': len(expiry_times[expiry_times <= 3]) / len(expiry_times) * 100
        }
    
    def generate_report(self, correlations):
        """Generate a comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("📊 BINARY OPTION PROBABILITY REVERSAL ANALYSIS REPORT")
        print("=" * 80)
        
        # Overall statistics
        print(f"\n📈 OVERALL STATISTICS")
        print(f"   • Total data points analyzed: {len(self.data):,}")
        print(f"   • Date range: {self.data['file_date'].min()} to {self.data['file_date'].max()}")
        print(f"   • Total probability reversals found: {len(self.reversals):,}")
        print(f"   • Reversal frequency: {len(self.reversals) / len(self.data) * 100:.4f}%")
        
        if len(self.reversals) == 0:
            print("\n❌ No reversals found - analysis cannot continue")
            return
        
        # Reversal types breakdown
        print(f"\n🔄 REVERSAL TYPES BREAKDOWN")
        reversal_counts = self.reversals['reversal_type'].value_counts()
        for rev_type, count in reversal_counts.items():
            percentage = count / len(self.reversals) * 100
            print(f"   • {rev_type}: {count} ({percentage:.2f}%)")
        
        # Time of day analysis
        print(f"\n🕐 TIME OF DAY ANALYSIS")
        time_data = correlations['time_of_day']
        print(f"   • Peak reversal hours (hour: rate):")
        for hour, rate in time_data['peak_reversal_hours']:
            print(f"     - Hour {hour:02d}: {rate:.6f} reversals per data point")
        
        # Quarter of hour analysis
        print(f"\n⏱️ QUARTER OF HOUR ANALYSIS")
        quarter_data = correlations['quarter_of_hour']
        print(f"   • Reversal rates by quarter:")
        for quarter, rate in quarter_data['reversal_rates_by_quarter'].items():
            print(f"     - {quarter}: {rate:.6f} reversals per data point")
        
        # Volatility analysis
        print(f"\n📊 VOLATILITY ANALYSIS")
        vol_data = correlations['volatility']
        print(f"   • Average volatility change during reversals: {vol_data['avg_volatility_change']:.4f}")
        print(f"   • Median volatility change: {vol_data['median_volatility_change']:.4f}")
        print(f"   • Volatility before reversals: {vol_data['avg_volatility_before']:.2%}")
        print(f"   • Volatility after reversals: {vol_data['avg_volatility_after']:.2%}")
        print(f"   • Reversals with volatility increase: {vol_data['volatility_increase_pct']:.1f}%")
        
        # 4-hour range analysis
        print(f"\n📈 4-HOUR PRICE RANGE ANALYSIS")
        range_data = correlations['4hour_range']
        print(f"   • Average 4h price range during reversals: ${range_data['avg_4h_price_range']:,.2f}")
        print(f"   • Median 4h price range: ${range_data['median_4h_price_range']:,.2f}")
        print(f"   • Average price position in 4h range: {range_data['avg_price_position_in_4h']:.2%}")
        print(f"   • Reversals near 4h extremes (top/bottom 20%): {range_data['reversals_at_extremes_pct']:.1f}%")
        
        # Expiry timing analysis
        print(f"\n⏰ EXPIRY TIMING ANALYSIS")
        expiry_data = correlations['expiry_timing']
        print(f"   • Average minutes to expiry: {expiry_data['avg_minutes_to_expiry']:.2f}")
        print(f"   • Median minutes to expiry: {expiry_data['median_minutes_to_expiry']:.2f}")
        print(f"   • Reversals in last 5 minutes: {expiry_data['reversals_in_last_5min_pct']:.1f}%")
        print(f"   • Reversals in last 3 minutes: {expiry_data['reversals_in_last_3min_pct']:.1f}%")
        
        print("\n" + "=" * 80)
    
    def save_detailed_results(self, filename=None):
        """Save detailed reversal data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reversal_analysis_{timestamp}.csv"
        
        if len(self.reversals) > 0:
            self.reversals.to_csv(filename, index=False)
            print(f"💾 Detailed reversal data saved to: {filename}")
        else:
            print("❌ No reversal data to save")
    
    def run_full_analysis(self, start_date=None, end_date=None, save_results=True):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Binary Option Reversal Analysis")
        print("=" * 60)
        
        # Load data
        if not self.load_daily_files(start_date, end_date):
            return None
        
        # Add time features
        self.add_time_features()
        
        # Identify reversals
        if self.identify_reversals() == 0:
            print("❌ No reversals found - analysis complete")
            return None
        
        # Analyze correlations
        correlations = self.analyze_correlations()
        
        # Generate report
        self.generate_report(correlations)
        
        # Save results
        if save_results:
            self.save_detailed_results()
        
        return correlations


def main():
    """Main execution function"""
    analyzer = BinaryOptionReversalAnalyzer()
    
    # You can specify date range here
    # start_date = "2025-01-01"  # YYYY-MM-DD format
    # end_date = "2025-01-31"    # YYYY-MM-DD format
    
    correlations = analyzer.run_full_analysis(
        # start_date=start_date,
        # end_date=end_date,
        save_results=True
    )
    
    return correlations


def analyze_specific_date(date_str):
    """Analyze reversals for a specific date"""
    analyzer = BinaryOptionReversalAnalyzer()
    correlations = analyzer.run_full_analysis(
        start_date=date_str,
        end_date=date_str,
        save_results=True
    )
    return correlations


def analyze_date_range(start_date, end_date):
    """Analyze reversals for a specific date range"""
    analyzer = BinaryOptionReversalAnalyzer()
    correlations = analyzer.run_full_analysis(
        start_date=start_date,
        end_date=end_date,
        save_results=True
    )
    return correlations


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "date" and len(sys.argv) > 2:
            # Analyze specific date: python script.py date 2025-01-15
            analyze_specific_date(sys.argv[2])
        elif sys.argv[1] == "range" and len(sys.argv) > 3:
            # Analyze date range: python script.py range 2025-01-01 2025-01-31
            analyze_date_range(sys.argv[2], sys.argv[3])
        else:
            print("Usage:")
            print("  python script.py                    # Analyze all available data")
            print("  python script.py date YYYY-MM-DD    # Analyze specific date")
            print("  python script.py range START END    # Analyze date range")
    else:
        main()