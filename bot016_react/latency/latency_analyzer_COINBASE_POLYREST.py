#!/usr/bin/env python3
"""
Latency Analyzer for BTC Spot vs BTC Call Option Prices
Monitors two JSON files to determine which is the leading indicator
"""

import json
import time
import os
from datetime import datetime
from collections import deque
import statistics

class LatencyAnalyzer:
    def __init__(self, spot_file, option_file, window_size=100):
        self.spot_file = spot_file
        self.option_file = option_file
        self.window_size = window_size
        
        # Store recent updates with timestamps
        self.spot_updates = deque(maxlen=window_size)
        self.option_updates = deque(maxlen=window_size)
        
        # Track last known values
        self.last_spot_data = None
        self.last_option_data = None
        self.last_spot_mtime = None
        self.last_option_mtime = None
        
        # Statistics
        self.spot_update_intervals = deque(maxlen=window_size)
        self.option_update_intervals = deque(maxlen=window_size)
        self.cross_correlation_data = []
        
    def read_json_file(self, filepath):
        """Read JSON file safely"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            return None
    
    def get_file_mtime(self, filepath):
        """Get file modification time"""
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return None
    
    def check_updates(self):
        """Check both files for updates and record timing"""
        current_time = time.time()
        spot_updated = False
        option_updated = False
        
        # Check spot price file
        spot_mtime = self.get_file_mtime(self.spot_file)
        if spot_mtime and spot_mtime != self.last_spot_mtime:
            spot_data = self.read_json_file(self.spot_file)
            if spot_data:
                if self.last_spot_data:
                    interval = current_time - self.spot_updates[-1]['read_time'] if self.spot_updates else 0
                    if interval > 0:
                        self.spot_update_intervals.append(interval)
                
                self.spot_updates.append({
                    'price': spot_data.get('price'),
                    'file_timestamp': spot_data.get('timestamp'),
                    'read_time': current_time,
                    'mtime': spot_mtime
                })
                self.last_spot_data = spot_data
                self.last_spot_mtime = spot_mtime
                spot_updated = True
        
        # Check option price file
        option_mtime = self.get_file_mtime(self.option_file)
        if option_mtime and option_mtime != self.last_option_mtime:
            option_data = self.read_json_file(self.option_file)
            if option_data:
                if self.last_option_data:
                    interval = current_time - self.option_updates[-1]['read_time'] if self.option_updates else 0
                    if interval > 0:
                        self.option_update_intervals.append(interval)
                
                self.option_updates.append({
                    'best_bid': option_data.get('best_bid'),
                    'best_ask': option_data.get('best_ask'),
                    'spread': option_data.get('spread'),
                    'file_timestamp': option_data.get('timestamp'),
                    'read_time': current_time,
                    'mtime': option_mtime
                })
                self.last_option_data = option_data
                self.last_option_mtime = option_mtime
                option_updated = True
        
        return spot_updated, option_updated
    
    def calculate_file_write_latency(self):
        """Calculate latency between file modifications"""
        if not self.spot_updates or not self.option_updates:
            return None
        
        # Get most recent updates
        latest_spot = self.spot_updates[-1]
        latest_option = self.option_updates[-1]
        
        # Calculate difference in file modification times
        mtime_diff = latest_spot['mtime'] - latest_option['mtime']
        
        # Calculate difference in read times
        read_diff = latest_spot['read_time'] - latest_option['read_time']
        
        return {
            'mtime_diff_ms': mtime_diff * 1000,
            'read_diff_ms': read_diff * 1000,
            'spot_mtime': latest_spot['mtime'],
            'option_mtime': latest_option['mtime']
        }
    
    def analyze_leading_indicator(self):
        """Determine which price feed leads the other"""
        if len(self.spot_updates) < 10 or len(self.option_updates) < 10:
            return None
        
        # Count which file updates first more often
        spot_leads = 0
        option_leads = 0
        total_pairs = 0
        lead_times = []
        
        # Look at recent paired updates
        for i in range(min(len(self.spot_updates), len(self.option_updates), 50)):
            spot = self.spot_updates[-(i+1)]
            option = self.option_updates[-(i+1)]
            
            mtime_diff = spot['mtime'] - option['mtime']
            lead_times.append(mtime_diff * 1000)  # Convert to ms
            
            if mtime_diff < -0.001:  # Option updated first (spot leads by being more recent)
                spot_leads += 1
            elif mtime_diff > 0.001:  # Spot updated first (option leads by being more recent)
                option_leads += 1
            
            total_pairs += 1
        
        if not lead_times:
            return None
        
        avg_lead_time = statistics.mean(lead_times)
        median_lead_time = statistics.median(lead_times)
        
        # Determine leader
        if abs(avg_lead_time) < 10:  # Within 10ms - essentially simultaneous
            leader = "Simultaneous"
        elif avg_lead_time > 0:
            leader = "Spot BTC Price"
        else:
            leader = "Call Option Price"
        
        return {
            'leader': leader,
            'avg_lead_time_ms': avg_lead_time,
            'median_lead_time_ms': median_lead_time,
            'spot_leads_count': spot_leads,
            'option_leads_count': option_leads,
            'total_comparisons': total_pairs,
            'lead_times': lead_times[-20:]  # Last 20 for display
        }
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        stats = {
            'spot_updates_count': len(self.spot_updates),
            'option_updates_count': len(self.option_updates),
        }
        
        if self.spot_update_intervals:
            stats['spot_avg_interval_s'] = statistics.mean(self.spot_update_intervals)
            stats['spot_update_rate_per_min'] = 60 / statistics.mean(self.spot_update_intervals) if statistics.mean(self.spot_update_intervals) > 0 else 0
        
        if self.option_update_intervals:
            stats['option_avg_interval_s'] = statistics.mean(self.option_update_intervals)
            stats['option_update_rate_per_min'] = 60 / statistics.mean(self.option_update_intervals) if statistics.mean(self.option_update_intervals) > 0 else 0
        
        latency = self.calculate_file_write_latency()
        if latency:
            stats['latest_latency'] = latency
        
        leading = self.analyze_leading_indicator()
        if leading:
            stats['leading_indicator'] = leading
        
        return stats
    
    def print_status(self):
        """Print current status"""
        stats = self.get_statistics()
        
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("BTC SPOT vs CALL OPTION - LATENCY ANALYZER")
        print("=" * 80)
        print(f"Monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Current prices
        if self.last_spot_data:
            print(f"📊 BTC Spot Price: ${self.last_spot_data.get('price', 'N/A'):,.2f}")
        if self.last_option_data:
            print(f"📈 Call Option Bid/Ask: {self.last_option_data.get('best_bid', 'N/A')} / {self.last_option_data.get('best_ask', 'N/A')}")
            print(f"   Spread: {self.last_option_data.get('spread', 'N/A')}")
        
        print()
        print("-" * 80)
        print("UPDATE STATISTICS")
        print("-" * 80)
        
        print(f"Spot Updates Recorded: {stats.get('spot_updates_count', 0)}")
        if 'spot_avg_interval_s' in stats:
            print(f"  Avg Interval: {stats['spot_avg_interval_s']:.3f}s")
            print(f"  Update Rate: {stats['spot_update_rate_per_min']:.1f} per minute")
        
        print()
        print(f"Option Updates Recorded: {stats.get('option_updates_count', 0)}")
        if 'option_avg_interval_s' in stats:
            print(f"  Avg Interval: {stats['option_avg_interval_s']:.3f}s")
            print(f"  Update Rate: {stats['option_update_rate_per_min']:.1f} per minute")
        
        # Latest latency
        if 'latest_latency' in stats:
            latency = stats['latest_latency']
            print()
            print("-" * 80)
            print("LATEST LATENCY")
            print("-" * 80)
            print(f"File Modification Time Difference: {latency['mtime_diff_ms']:+.2f} ms")
            print(f"Read Time Difference: {latency['read_diff_ms']:+.2f} ms")
            if latency['mtime_diff_ms'] > 0:
                print(f"  → Spot file written {latency['mtime_diff_ms']:.2f} ms AFTER option file")
            elif latency['mtime_diff_ms'] < 0:
                print(f"  → Spot file written {-latency['mtime_diff_ms']:.2f} ms BEFORE option file")
            else:
                print(f"  → Files written simultaneously")
        
        # Leading indicator analysis
        if 'leading_indicator' in stats:
            leading = stats['leading_indicator']
            print()
            print("-" * 80)
            print("LEADING INDICATOR ANALYSIS")
            print("-" * 80)
            print(f"🎯 Leader: {leading['leader']}")
            print(f"Average Lead Time: {leading['avg_lead_time_ms']:+.2f} ms")
            print(f"Median Lead Time: {leading['median_lead_time_ms']:+.2f} ms")
            print(f"Spot Leads Count: {leading['spot_leads_count']} / {leading['total_comparisons']}")
            print(f"Option Leads Count: {leading['option_leads_count']} / {leading['total_comparisons']}")
            
            if leading['avg_lead_time_ms'] > 0:
                print(f"\n💡 Spot price file updates {leading['avg_lead_time_ms']:.2f} ms after option")
                print(f"   → CALL OPTION appears to be the LEADING indicator")
            elif leading['avg_lead_time_ms'] < 0:
                print(f"\n💡 Spot price file updates {-leading['avg_lead_time_ms']:.2f} ms before option")
                print(f"   → SPOT PRICE appears to be the LEADING indicator")
            else:
                print(f"\n💡 Both feeds update nearly simultaneously")
        
        print()
        print("-" * 80)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    def run(self, interval=0.1):
        """Main monitoring loop"""
        print(f"Starting latency analyzer...")
        print(f"Spot file: {self.spot_file}")
        print(f"Option file: {self.option_file}")
        print(f"Polling interval: {interval}s")
        print()
        
        update_counter = 0
        
        try:
            while True:
                spot_updated, option_updated = self.check_updates()
                
                if spot_updated or option_updated:
                    update_counter += 1
                    
                    # Print status every update or every 10 seconds
                    if update_counter % 5 == 0 or update_counter <= 10:
                        self.print_status()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping analyzer...")
            self.print_final_summary()
    
    def print_final_summary(self):
        """Print final summary statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        if 'leading_indicator' in stats:
            leading = stats['leading_indicator']
            print(f"\n🎯 LEADING INDICATOR: {leading['leader']}")
            print(f"   Average Lead Time: {leading['avg_lead_time_ms']:+.2f} ms")
            print(f"   Median Lead Time: {leading['median_lead_time_ms']:+.2f} ms")
            print(f"   Based on {leading['total_comparisons']} comparisons")
        
        print(f"\nTotal Spot Updates: {stats.get('spot_updates_count', 0)}")
        print(f"Total Option Updates: {stats.get('option_updates_count', 0)}")
        
        if 'spot_avg_interval_s' in stats:
            print(f"\nSpot Update Rate: {stats['spot_update_rate_per_min']:.1f} per minute")
        if 'option_avg_interval_s' in stats:
            print(f"Option Update Rate: {stats['option_update_rate_per_min']:.1f} per minute")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # File paths
    SPOT_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json"
    OPTION_FILE = "/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json"
    
    # Create analyzer
    analyzer = LatencyAnalyzer(SPOT_FILE, OPTION_FILE, window_size=200)
    
    # Run monitoring (checks every 100ms)
    analyzer.run(interval=0.1)
