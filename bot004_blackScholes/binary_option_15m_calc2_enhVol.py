import json
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import time

class Enhanced15MinBinaryOptionCalculator:
    def __init__(self, price_file_path="/home/ubuntu/013_2025_polymarket/btc_price.json"):
        self.price_file_path = price_file_path
        self.risk_free_rate = 0.05
        self.results_file = "/home/ubuntu/013_2025_polymarket/option_probabilities_M15.json"
        self.current_strike = None
        self.period_start_time = None
        self.period_start_price = None
        self.binance_history = []
        self.minute_history = []
        self.volatility_method = 'combined'  # 'ewma', 'multi_timeframe', 'garch', 'combined'
        self.last_minute_data_fetch = None  # Track when minute data was last fetched

    def get_current_price(self):
        """Read current BTC price from JSON file"""
        try:
            with open(self.price_file_path, 'r') as f:
                data = json.load(f)
            return data['price'], data['timestamp']
        except Exception as e:
            print(f"Error reading price file: {e}")
            return None, None

    def calculate_time_to_expiry(self, current_timestamp=None):
        """Calculate time to next 15-minute expiry (00, 15, 30, 45) in years"""
        if current_timestamp is None:
            current_time = datetime.now()
        else:
            current_time = datetime.fromtimestamp(current_timestamp / 1000)

        # Find the next 15-minute boundary (00, 15, 30, 45)
        current_minute = current_time.minute
        
        if current_minute < 15:
            next_expiry_minute = 15
        elif current_minute < 30:
            next_expiry_minute = 30
        elif current_minute < 45:
            next_expiry_minute = 45
        else:
            next_expiry_minute = 0  # Next hour at 00

        if next_expiry_minute == 0:
            # Next expiry is at the top of next hour
            next_expiry = (current_time.replace(minute=0, second=0, microsecond=0) + 
                          timedelta(hours=1))
        else:
            # Next expiry is within current hour
            next_expiry = current_time.replace(minute=next_expiry_minute, second=0, microsecond=0)

        time_diff = (next_expiry - current_time).total_seconds() / (365.25 * 24 * 3600)
        return max(time_diff, 1e-8)

    def get_current_15min_period_start(self, current_timestamp):
        """Get the start time of the current 15-minute period"""
        current_time = datetime.fromtimestamp(current_timestamp / 1000)
        current_minute = current_time.minute
        
        if current_minute < 15:
            period_start_minute = 0
        elif current_minute < 30:
            period_start_minute = 15
        elif current_minute < 45:
            period_start_minute = 30
        else:
            period_start_minute = 45

        period_start = current_time.replace(minute=period_start_minute, second=0, microsecond=0)
        return int(period_start.timestamp() * 1000)

    def fetch_binance_klines(self, hours=168):
        """Fetch historical hourly kline data from Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit={hours}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            return data
        except Exception as e:
            print(f"Error fetching Binance hourly data: {e}")
            return None

    def fetch_binance_klines_15m(self, limit=96):
        """Fetch 15-minute kline data from Binance API for strike prices"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit={limit}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            return data
        except Exception as e:
            print(f"Error fetching 15m Binance data: {e}")
            return None

    def fetch_binance_klines_1m(self, limit=720):
        """Fetch 1-minute kline data from Binance API for enhanced volatility"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            return data
        except Exception as e:
            print(f"Error fetching 1m Binance data: {e}")
            return None

    def initialize_price_history_from_binance(self, hours=168):
        """Initialize hourly price history from Binance (for compatibility)"""
        print(f"Fetching {hours} hours of BTC price history from Binance...")

        klines = self.fetch_binance_klines(hours)
        if not klines:
            print("Failed to fetch Binance hourly data")
            return False

        self.binance_history = []
        for kline in klines:
            timestamp = int(kline[0])
            close_price = float(kline[4])

            self.binance_history.append({
                'price': close_price,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp/1000).isoformat()
            })

        print(f"Successfully loaded {len(self.binance_history)} hourly data points")
        return True

    def initialize_minute_data(self, minutes=720):
        """Initialize 1-minute price history for enhanced volatility calculation"""
        print(f"📊 Fetching {minutes} minutes of granular data for volatility...")

        klines = self.fetch_binance_klines_1m(minutes)
        if not klines:
            print("⚠️ Failed to fetch minute data, using hourly fallback")
            return False

        self.minute_history = []
        for kline in klines:
            timestamp = int(kline[0])
            close_price = float(kline[4])

            self.minute_history.append({
                'price': close_price,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp/1000).isoformat()
            })

        # Track when data was fetched
        self.last_minute_data_fetch = time.time()

        print(f"✅ Successfully loaded {len(self.minute_history)} minute-level data points")
        print(f"📅 Data range: {self.minute_history[0]['datetime']} to {self.minute_history[-1]['datetime']}")
        return True

    def calculate_ewma_volatility(self, lookback_minutes=60, recent_emphasis_minutes=30, lambda_factor=0.94):
        """Enhanced EWMA volatility with recent emphasis (adjusted for 15-min periods)"""
        if len(self.minute_history) < 2:
            print("#FB1")
            return self.fallback_volatility()

        history = sorted(self.minute_history, key=lambda x: x['timestamp'])

        if len(history) < lookback_minutes:
            recent_history = history
        else:
            recent_history = history[-lookback_minutes:]

        if len(recent_history) < 2:
            print("FB2")
            return self.fallback_volatility()

        prices = [h['price'] for h in recent_history]
        returns = []
        timestamps = []

        for i in range(1, len(prices)):
            ret = np.log(prices[i] / prices[i-1])
            returns.append(ret)
            timestamps.append(recent_history[i]['timestamp'])

        if len(returns) < 2:
            print("FB3")
            return self.fallback_volatility()

        # Calculate EWMA weights with recent emphasis
        weights = []
        current_time = timestamps[-1]

        for i, ts in enumerate(timestamps):
            age_minutes = (current_time - ts) / (1000 * 60)

            # Base EWMA weight
            ewma_weight = lambda_factor ** age_minutes

            # Boost for recent data (last 15-30 minutes)
            if age_minutes <= recent_emphasis_minutes:
                recency_boost = 2.0 * (1 - (age_minutes / recent_emphasis_minutes) ** 2)
                ewma_weight *= (1 + recency_boost)

            weights.append(ewma_weight)

        weights = np.array(weights)
        weights = weights / np.sum(weights)

        returns = np.array(returns)
        weighted_mean = np.sum(weights * returns)
        weighted_variance = np.sum(weights * (returns - weighted_mean) ** 2)

        annual_volatility = np.sqrt(weighted_variance * 525600)
        return max(0.01, min(3.0, annual_volatility))

    def calculate_multi_timeframe_volatility(self):
        """Multi-timeframe volatility with emphasis on recent periods (adjusted for 15-min)"""
        if len(self.minute_history) < 30:
            print("FB4")
            return self.fallback_volatility()

        history = sorted(self.minute_history, key=lambda x: x['timestamp'])

        timeframes = [
            {'minutes': 15, 'weight': 0.4},   # Last 15 minutes (current period)
            {'minutes': 30, 'weight': 0.3},   # Last 30 minutes (2 periods)
            {'minutes': 60, 'weight': 0.2},   # Last hour (4 periods)
            {'minutes': 120, 'weight': 0.1}   # Last 2 hours (8 periods)
        ]

        volatilities = []
        weights = []

        for tf in timeframes:
            if len(history) < tf['minutes']:
                continue

            recent_history = history[-tf['minutes']:]
            prices = [h['price'] for h in recent_history]
            returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

            if len(returns) > 1:
                vol = np.std(returns) * np.sqrt(525600 / tf['minutes'])
                volatilities.append(vol)
                weights.append(tf['weight'])

        if not volatilities:
            print("FB5")
            return self.fallback_volatility()

        weights = np.array(weights)
        weights = weights / np.sum(weights)

        combined_vol = np.sum(np.array(volatilities) * weights)
        return max(0.2, min(3.0, combined_vol))

    def calculate_garch_like_volatility(self, alpha=0.1, beta=0.85):
        """GARCH-like volatility calculation"""
        if len(self.minute_history) < 60:
            print("FB6")
            return self.fallback_volatility()

        history = sorted(self.minute_history, key=lambda x: x['timestamp'])[-60:]
        prices = [h['price'] for h in history]
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

        if len(returns) < 10:
            print("FB7")
            return self.fallback_volatility()

        variance = np.var(returns)

        for i, ret in enumerate(returns[10:], 10):
            variance = (1 - alpha - beta) * np.var(returns[:i]) + alpha * ret**2 + beta * variance

        annual_vol = np.sqrt(variance * 525600)
        return max(0.2, min(3.0, annual_vol))

    def fallback_volatility(self):
        """Fallback to hourly data volatility if minute data insufficient"""
        if len(self.binance_history) < 2:
            print("⚠️ Using default volatility of 80%")
            return 0.8

        history = sorted(self.binance_history, key=lambda x: x['timestamp'])
        recent_history = history[-24:] if len(history) > 24 else history

        prices = [h['price'] for h in recent_history]
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

        if len(returns) < 2:
            return 0.8

        returns_std = np.std(returns)
        annual_volatility = returns_std * np.sqrt(8760)
        return max(0.1, min(3.0, annual_volatility))

    def calculate_enhanced_volatility(self, method=None):
        """Enhanced volatility calculation with multiple methods"""
        if method is None:
            method = self.volatility_method

        if method == 'ewma':
            return self.calculate_ewma_volatility()
        elif method == 'multi_timeframe':
            return self.calculate_multi_timeframe_volatility()
        elif method == 'garch':
            return self.calculate_garch_like_volatility()
        elif method == 'combined':
            if len(self.minute_history) < 15:
                print("FB8a")
                return self.fallback_volatility()

            ewma_vol = self.calculate_ewma_volatility()
            multi_vol = self.calculate_multi_timeframe_volatility()
            garch_vol = self.calculate_garch_like_volatility()

            combined = 0.5 * ewma_vol + 0.3 * multi_vol + 0.2 * garch_vol
            return combined
        else:
            return self.calculate_ewma_volatility()

    def get_period_start_price_from_binance(self, period_start_timestamp):
        """Get the OPEN price of the 15-minute period from Binance data as strike"""
        try:
            # Fetch recent 15-minute klines
            klines = self.fetch_binance_klines_15m(limit=10)
            if not klines:
                return None, None

            # Find the kline that matches our period start time
            period_start_ms = period_start_timestamp
            
            for kline in reversed(klines):  # Check most recent first
                kline_start = int(kline[0])
                
                # Check if this kline starts at our period start time
                if abs(kline_start - period_start_ms) < 60000:  # Within 1 minute tolerance
                    open_price = float(kline[1])
                    timestamp = int(kline[0])
                    
                    print(f"📊 ------ NEW 15-MIN PERIOD ------:")
                    print(f"📊 Last Binance 15m Kline Data:")
                    print(f"   Timestamp: {datetime.fromtimestamp(timestamp/1000).isoformat()}")
                    print(f"   OPEN: ${open_price:,.2f}")
                    print(f"   HIGH: ${float(kline[2]):,.2f}")
                    print(f"   LOW: ${float(kline[3]):,.2f}")
                    print(f"   CLOSE: ${float(kline[4]):,.2f}")

                    return open_price, timestamp

            # If no exact match found, use the most recent kline's open
            last_kline = klines[-1]
            open_price = float(last_kline[1])
            timestamp = int(last_kline[0])
            
            print(f"📊 Using most recent 15m kline (no exact match):")
            print(f"   OPEN: ${open_price:,.2f}")
            
            return open_price, timestamp

        except Exception as e:
            print(f"Error getting period start price from Binance: {e}")
            return None, None

    def check_new_15min_period(self, current_timestamp):
        """Check if we're in a new 15-minute period and need to set a new strike"""
        current_period_start = self.get_current_15min_period_start(current_timestamp)

        if (self.period_start_time is None or
            current_period_start != self.period_start_time):

            self.period_start_time = current_period_start
            strike_price, kline_timestamp = self.get_period_start_price_from_binance(current_period_start)

            if strike_price:
                self.current_strike = strike_price
                self.period_start_price = strike_price

                current_time = datetime.fromtimestamp(current_timestamp / 1000)
                period_start_time = datetime.fromtimestamp(current_period_start / 1000)
                
                print(f"\n🕐 NEW 15-MINUTE PERIOD DETECTED:")
                print(f"Period Start: {period_start_time.strftime('%H:%M:%S')}")
                print(f"Current Time: {current_time.strftime('%H:%M:%S')}")
                print(f"Strike Price (OPEN of 15m kline): ${self.current_strike:,.2f}")

                # Refresh minute data on new period
                print("🔄 Refreshing minute-level data for enhanced volatility...")
                self.initialize_minute_data()

                return True
            else:
                print("⚠️ Failed to get strike price from Binance")

        return False

    def black_scholes_binary_call(self, S, K, T, r, sigma):
        """Calculate binary call option probability using Black-Scholes"""
        if T <= 0:
            return 1.0 if S > K else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        prob_win = norm.cdf(d2)
        return prob_win

    def black_scholes_binary_put(self, S, K, T, r, sigma):
        """Calculate binary put option probability using Black-Scholes"""
        return 1 - self.black_scholes_binary_call(S, K, T, r, sigma)

    def calculate_probabilities(self):
        """Calculate both call and put probabilities with enhanced volatility"""
        current_price, timestamp = self.get_current_price()
        if current_price is None:
            return None

        self.check_new_15min_period(timestamp)

        if self.current_strike is None:
            print("No strike price set. Waiting for new 15-minute period...")
            return None

        time_to_expiry = self.calculate_time_to_expiry(timestamp)

        # Use enhanced volatility calculation
        volatility = self.calculate_enhanced_volatility()

        call_probability = self.black_scholes_binary_call(
            current_price, self.current_strike, time_to_expiry,
            self.risk_free_rate, volatility
        )

        put_probability = self.black_scholes_binary_put(
            current_price, self.current_strike, time_to_expiry,
            self.risk_free_rate, volatility
        )

        price_change_pct = ((current_price - self.period_start_price) / self.period_start_price) * 100 if self.period_start_price else 0

        result = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp/1000).isoformat(),
            'current_price': current_price,
            'strike_price': self.current_strike,
            'period_start_price': self.period_start_price,
            'price_change_pct': price_change_pct,
            'time_to_expiry_minutes': time_to_expiry * 525600,  # Convert to minutes
            'volatility': volatility,
            'volatility_method': self.volatility_method,
            'call_probability': call_probability,
            'put_probability': put_probability,
            'moneyness': current_price / self.current_strike,
            'minute_data_age_seconds': time.time() - self.last_minute_data_fetch if self.last_minute_data_fetch else None,
            'minute_data_last_fetch': datetime.fromtimestamp(self.last_minute_data_fetch).isoformat() if self.last_minute_data_fetch else None
        }

        return result

    def save_results_to_file(self, result):
        """Save probability results to JSON file"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Error saving results: {e}")

    def print_status_with_freshness(self, result, data_age, refresh_interval):
        """Print current status with data freshness indicator"""
        vol_method = result['volatility_method'].upper()[:4]

        # Data freshness indicator
        freshness_pct = max(0, (refresh_interval - data_age) / refresh_interval * 100)
        if freshness_pct > 80:
            fresh_icon = "🟢"
        elif freshness_pct > 50:
            fresh_icon = "🟡"
        else:
            fresh_icon = "🔴"

        print(f"\r⏰ {result['datetime'][11:19]} | "
              f"Price: ${result['current_price']:>8,.2f} | "
              f"Strike: ${result['strike_price']:>8,.2f} | "
              f"Change: {result['price_change_pct']:>+6.2f}% | "
              f"Call: {result['call_probability']:>6.2%} | "
              f"Put: {result['put_probability']:>6.2%} | "
              f"Vol({vol_method}): {result['volatility']:>5.1%} | "
              f"TTExp: {result['time_to_expiry_minutes']:>5.2f}m | "
              f"Data:{fresh_icon}{freshness_pct:>3.0f}%", end="", flush=True)

    def run_continuous(self):
        """Run continuous probability calculations for 15-minute binary options"""
        print("🚀 Starting Enhanced BTC 15-Minute Binary Option Monitor")
        print("=" * 70)
        print("⏰ Expiry Schedule: Every 15 minutes at :00, :15, :30, :45")
        print(f"📈 Using volatility method: {self.volatility_method.upper()}")
        print("🔄 Minute data refresh frequency: Adaptive based on time to expiry")

        # Load both hourly and minute data
        print("📡 Loading historical data from Binance...")
        self.initialize_price_history_from_binance()
        self.initialize_minute_data()

        # Get initial volatility
        initial_vol = self.calculate_enhanced_volatility()
        print(f"📊 Initial enhanced volatility: {initial_vol:.1%}")

        # Set initial strike price
        print("\n🎯 Setting initial strike price...")
        current_price, current_timestamp = self.get_current_price()
        if current_timestamp:
            current_period_start = self.get_current_15min_period_start(current_timestamp)
            initial_strike, kline_timestamp = self.get_period_start_price_from_binance(current_period_start)
            
            if initial_strike:
                self.current_strike = initial_strike
                self.period_start_price = initial_strike
                self.period_start_time = current_period_start
                print(f"✅ Initial strike set: ${self.current_strike:,.2f}")
            else:
                print("❌ Failed to set initial strike price")
                return
        else:
            print("❌ Failed to get current price")
            return

        print(f"\n🔥 Starting enhanced continuous monitoring (Press Ctrl+C to stop)")
        print("Live updates below (single line, overwrites):")
        print("-" * 110)

        try:
            update_counter = 0
            last_minute_refresh = time.time()

            while True:
                current_time_epoch = time.time()

                # Get current time for determining refresh frequency
                now = datetime.now()
                current_minute = now.minute
                
                # Calculate minutes until next expiry
                if current_minute < 15:
                    minutes_until_expiry = 15 - current_minute
                elif current_minute < 30:
                    minutes_until_expiry = 30 - current_minute
                elif current_minute < 45:
                    minutes_until_expiry = 45 - current_minute
                else:
                    minutes_until_expiry = 60 - current_minute

                # Determine refresh frequency based on time remaining
                if minutes_until_expiry <= 3:  # Last 3 minutes - refresh every 5 seconds
                    refresh_interval = 5
                elif minutes_until_expiry <= 7:  # Last 7 minutes - refresh every 10 seconds
                    refresh_interval = 10
                elif minutes_until_expiry <= 10:  # Last 10 minutes - refresh every 15 seconds
                    refresh_interval = 15
                else:  # Normal period - refresh every 30 seconds
                    refresh_interval = 30

                # Check if it's time to refresh minute data
                if current_time_epoch - last_minute_refresh >= refresh_interval:
                    print(f"\n🔄 Refreshing minute data ({refresh_interval}s interval)... ", end="", flush=True)
                    if self.initialize_minute_data():
                        last_minute_refresh = current_time_epoch
                        refresh_time = datetime.now().strftime("%H:%M:%S")
                        print(f"✅ Updated at {refresh_time}")
                    else:
                        print("⚠️ Failed")

                result = self.calculate_probabilities()

                if result:
                    self.save_results_to_file(result)
                    # Add data freshness indicator
                    data_age = (current_time_epoch - last_minute_refresh)
                    self.print_status_with_freshness(result, data_age, refresh_interval)

                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Enhanced 15-minute monitoring stopped by user")
            print(f"Final volatility method used: {self.volatility_method}")


def main():
    """Main execution function"""
    calc = Enhanced15MinBinaryOptionCalculator()

    # You can change the volatility method here:
    # 'ewma' - Exponentially weighted with recent emphasis
    # 'multi_timeframe' - Multiple timeframe approach
    # 'garch' - GARCH-like volatility clustering
    # 'combined' - Weighted combination of all methods
    calc.volatility_method = 'ewma'  # Change this to test different methods

    calc.run_continuous()


def analyze_volatility_methods():
    """Utility function to compare volatility methods"""
    calc = Enhanced15MinBinaryOptionCalculator()
    calc.initialize_price_history_from_binance()
    calc.initialize_minute_data()

    methods = {
        'EWMA (Recent Focus)': calc.calculate_ewma_volatility(),
        'Multi-Timeframe': calc.calculate_multi_timeframe_volatility(),
        'GARCH-like': calc.calculate_garch_like_volatility(),
        'Combined': calc.calculate_enhanced_volatility('combined'),
        'Original Fallback': calc.fallback_volatility()
    }

    print("\n📊 VOLATILITY COMPARISON (15-MIN OPTIMIZED)")
    print("=" * 45)
    for method, vol in methods.items():
        print(f"{method:20}: {vol:6.2%}")


def force_refresh_data():
    """Utility function to manually refresh both hourly and minute data"""
    calc = Enhanced15MinBinaryOptionCalculator()

    print("🔄 Force refreshing all data from Binance...")
    print("-" * 50)

    # Refresh hourly data
    print("📡 Refreshing hourly data...")
    if calc.initialize_price_history_from_binance():
        print("✅ Hourly data refreshed successfully")
    else:
        print("❌ Failed to refresh hourly data")

    # Refresh minute data
    print("📊 Refreshing minute data...")
    if calc.initialize_minute_data():
        print("✅ Minute data refreshed successfully")

        # Show current volatility
        vol = calc.calculate_enhanced_volatility('combined')
        print(f"📈 Current combined volatility: {vol:.2%}")
    else:
        print("❌ Failed to refresh minute data")


def show_refresh_schedule():
    """Show the current data refresh schedule for 15-minute options"""
    print("\n⏰ DATA REFRESH SCHEDULE (15-MINUTE BINARY OPTIONS)")
    print("=" * 55)
    print("📊 Minute Data Refresh Frequency:")
    print("   • Normal period: Every 30 seconds")
    print("   • Last 10 minutes: Every 15 seconds") 
    print("   • Last 7 minutes: Every 10 seconds")
    print("   • Last 3 minutes: Every 5 seconds")
    print("\n🕐 15-Minute Period Data Refresh:")
    print("   • On new period detection (:00, :15, :30, :45)")
    print("   • When strike price changes")
    print("\n💡 Why frequent updates matter for 15-minute options:")
    print("   • Ultra-short timeframe makes timing critical")
    print("   • Last 3 minutes before expiry are extremely sensitive")
    print("   • Volatility estimation needs recent data (15-30min)")
    print("   • Price movements have amplified impact on short-term probabilities")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "refresh":
            force_refresh_data()
        elif sys.argv[1] == "compare":
            analyze_volatility_methods()
        elif sys.argv[1] == "schedule":
            show_refresh_schedule()
        else:
            print("Usage: python script.py [refresh|compare|schedule]")
    else:
        main()