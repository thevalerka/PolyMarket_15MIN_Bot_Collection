import json
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import time

class EnhancedBinaryOptionCalculator:
    def __init__(self, price_file_path="/home/ubuntu/013_2025_polymarket/btc_price.json"):
        self.price_file_path = price_file_path
        self.risk_free_rate = 0.05
        self.results_file = "/home/ubuntu/013_2025_polymarket/option_probabilities.json"
        self.current_strike = None
        self.hour_start_time = None
        self.hour_start_price = None
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
        """Calculate time to end of current hour in years"""
        if current_timestamp is None:
            current_time = datetime.now()
        else:
            current_time = datetime.fromtimestamp(current_timestamp / 1000)

        end_of_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        time_diff = (end_of_hour - current_time).total_seconds() / (365.25 * 24 * 3600)
        return max(time_diff, 1e-8)

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

    def calculate_ewma_volatility(self, lookback_minutes=120, recent_emphasis_minutes=60, lambda_factor=0.94):
        """Enhanced EWMA volatility with recent emphasis"""
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

            # Boost for recent data (last 30-60 minutes)
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
        """Multi-timeframe volatility with emphasis on recent periods"""
        if len(self.minute_history) < 60:
            print("FB4")
            return self.fallback_volatility()

        history = sorted(self.minute_history, key=lambda x: x['timestamp'])

        timeframes = [
            {'minutes': 30, 'weight': 0.4},   # Last 30 minutes
            {'minutes': 60, 'weight': 0.3},   # Last hour
            {'minutes': 180, 'weight': 0.2},  # Last 3 hours
            {'minutes': 480, 'weight': 0.1}   # Last 8 hours
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
        if len(self.minute_history) < 100:
            print("FB6")
            return self.fallback_volatility()

        history = sorted(self.minute_history, key=lambda x: x['timestamp'])[-100:]
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
            if len(self.minute_history) < 30:
                print("FB8a")
                return self.fallback_volatility()

            ewma_vol = self.calculate_ewma_volatility()
            multi_vol = self.calculate_multi_timeframe_volatility()
            garch_vol = self.calculate_garch_like_volatility()

            combined = 0.5 * ewma_vol + 0.3 * multi_vol + 0.2 * garch_vol
            return combined
        else:
            return self.calculate_ewma_volatility()

    def get_hour_start_price_from_binance(self):
        """Get the OPEN price of the last kline from Binance data as strike"""
        try:
            klines = self.fetch_binance_klines(hours=1)
            if not klines:
                return None, None

            last_kline = klines[-1]
            open_price = float(last_kline[1])
            timestamp = int(last_kline[0])
            print(f"📊 ------ NEW ------:")
            print(f"📊 Last Binance Kline Data NEW:")
            print(f"   Timestamp: {datetime.fromtimestamp(timestamp/1000).isoformat()}")
            print(f"   OPEN: ${open_price:,.2f}")
            print(f"   HIGH: ${float(last_kline[2]):,.2f}")
            print(f"   LOW: ${float(last_kline[3]):,.2f}")
            print(f"   CLOSE: ${float(last_kline[4]):,.2f}")

            return open_price, timestamp

        except Exception as e:
            print(f"Error getting hour start price from Binance: {e}")
            return None, None

    def check_new_hour(self, current_timestamp):
        """Check if we're in a new hour and need to set a new strike"""
        current_time = datetime.fromtimestamp(current_timestamp / 1000)
        current_hour_start = current_time.replace(minute=0, second=0, microsecond=0)
        current_hour_start_timestamp = int(current_hour_start.timestamp() * 1000)

        if (self.hour_start_time is None or
            current_hour_start_timestamp != self.hour_start_time):

            self.hour_start_time = current_hour_start_timestamp
            strike_price, kline_timestamp = self.get_hour_start_price_from_binance()

            if strike_price:
                self.current_strike = strike_price
                self.hour_start_price = strike_price

                print(f"\n🕐 NEW HOUR DETECTED:")
                print(f"Current Hour Start: {current_hour_start.isoformat()}")
                print(f"Strike Price (OPEN of last kline): ${self.current_strike:,.2f}")
                print(f"Kline Timestamp: {datetime.fromtimestamp(kline_timestamp/1000).isoformat()}")

                # Refresh minute data on new hour
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

        self.check_new_hour(timestamp)

        if self.current_strike is None:
            print("No strike price set. Waiting for new hour...")
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

        price_change_pct = ((current_price - self.hour_start_price) / self.hour_start_price) * 100 if self.hour_start_price else 0

        result = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp/1000).isoformat(),
            'current_price': current_price,
            'strike_price': self.current_strike,
            'hour_start_price': self.hour_start_price,
            'price_change_pct': price_change_pct,
            'time_to_expiry_hours': time_to_expiry * 8760,
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

    def print_status(self, result):
        """Print current status (overwrites previous line)"""
        vol_method = result['volatility_method'].upper()[:4]  # Abbreviate method name
        print(f"\r⏰ {result['datetime'][11:19]} | "
              f"Price: ${result['current_price']:>8,.2f} | "
              f"Strike: ${result['strike_price']:>8,.2f} | "
              f"Change: {result['price_change_pct']:>+6.2f}% | "
              f"Call: {result['call_probability']:>6.2%} | "
              f"Put: {result['put_probability']:>6.2%} | "
              f"Vol({vol_method}): {result['volatility']:>5.1%} | "
              f"TTExp: {result['time_to_expiry_hours']:>5.2f}h", end="", flush=True)

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
              f"TTExp: {result['time_to_expiry_hours']:>5.2f}h | "
              f"Data:{fresh_icon}{freshness_pct:>3.0f}%", end="", flush=True)

    def run_continuous(self):
        """Run continuous probability calculations with enhanced volatility"""
        print("🚀 Starting Enhanced BTC Binary Option Continuous Monitor")
        print("=" * 70)
        print(f"📈 Using volatility method: {self.volatility_method.upper()}")
        print("🔄 Minute data refresh frequency: Every 60 seconds")
        print("📊 Critical period data refresh: Every 30 seconds (last 15 minutes of hour)")

        # Load both hourly and minute data
        print("📡 Loading historical data from Binance...")
        self.initialize_price_history_from_binance()
        self.initialize_minute_data()

        # Get initial volatility
        initial_vol = self.calculate_enhanced_volatility()
        print(f"📊 Initial enhanced volatility: {initial_vol:.1%}")

        # Set initial strike price
        print("\n🎯 Setting initial strike price...")
        initial_strike, kline_timestamp = self.get_hour_start_price_from_binance()
        if initial_strike:
            self.current_strike = initial_strike
            self.hour_start_price = initial_strike
            current_time = datetime.now()
            self.hour_start_time = int(current_time.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
            print(f"✅ Initial strike set: ${self.current_strike:,.2f}")
        else:
            print("❌ Failed to set initial strike price")
            return

        print(f"\n🔥 Starting enhanced continuous monitoring (Press Ctrl+C to stop)")
        print("Live updates below (single line, overwrites):")
        print("-" * 100)

        try:
            update_counter = 0
            last_minute_refresh = time.time()

            while True:
                current_time = time.time()

                # Get current time for determining refresh frequency
                now = datetime.now()
                minutes_until_hour = 60 - now.minute

                # Determine refresh frequency based on time remaining in hour
                if minutes_until_hour <= 15:  # Last 15 minutes of hour - refresh every 30 seconds
                    refresh_interval = 10
                elif minutes_until_hour <= 30:  # Last 30 minutes - refresh every 45 seconds
                    refresh_interval = 15
                else:  # Normal period - refresh every 60 seconds
                    refresh_interval = 30

                # Check if it's time to refresh minute data
                if current_time - last_minute_refresh >= refresh_interval:
                    print(f"\n🔄 Refreshing minute data ({refresh_interval}s interval)... ", end="", flush=True)
                    if self.initialize_minute_data():
                        last_minute_refresh = current_time
                        refresh_time = datetime.now().strftime("%H:%M:%S")
                        print(f"✅ Updated at {refresh_time}")
                    else:
                        print("⚠️ Failed")

                result = self.calculate_probabilities()

                if result:
                    self.save_results_to_file(result)
                    # Add data freshness indicator
                    data_age = (current_time - last_minute_refresh)
                    self.print_status_with_freshness(result, data_age, refresh_interval)

                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Enhanced monitoring stopped by user")
            print(f"Final volatility method used: {self.volatility_method}")

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
              f"TTExp: {result['time_to_expiry_hours']:>5.2f}h | "
              f"Data:{fresh_icon}{freshness_pct:>3.0f}%", end="", flush=True)


def main():
    """Main execution function"""
    calc = EnhancedBinaryOptionCalculator()

    # You can change the volatility method here:
    # 'ewma' - Exponentially weighted with recent emphasis
    # 'multi_timeframe' - Multiple timeframe approach
    # 'garch' - GARCH-like volatility clustering
    # 'combined' - Weighted combination of all methods
    calc.volatility_method = 'ewma'  # Change this to test different methods

    calc.run_continuous()


def analyze_volatility_methods():
    """Utility function to compare volatility methods"""
    calc = EnhancedBinaryOptionCalculator()
    calc.initialize_price_history_from_binance()
    calc.initialize_minute_data()

    methods = {
        'EWMA (Recent Focus)': calc.calculate_ewma_volatility(),
        'Multi-Timeframe': calc.calculate_multi_timeframe_volatility(),
        'GARCH-like': calc.calculate_garch_like_volatility(),
        'Combined': calc.calculate_enhanced_volatility('combined'),
        'Original Fallback': calc.fallback_volatility()
    }

    print("\n📊 VOLATILITY COMPARISON")
    print("=" * 40)
    for method, vol in methods.items():
        print(f"{method:20}: {vol:6.2%}")


def force_refresh_data():
    """Utility function to manually refresh both hourly and minute data"""
    calc = EnhancedBinaryOptionCalculator()

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
    """Show the current data refresh schedule"""
    print("\n⏰ DATA REFRESH SCHEDULE")
    print("=" * 40)
    print("📊 Minute Data Refresh Frequency:")
    print("   • Normal period: Every 60 seconds")
    print("   • Last 30 minutes: Every 45 seconds")
    print("   • Last 15 minutes: Every 30 seconds")
    print("\n🕐 Hourly Data Refresh:")
    print("   • On new hour detection")
    print("   • When strike price changes")
    print("\n💡 Why frequent updates matter:")
    print("   • Binary options are extremely time-sensitive")
    print("   • Last 15 minutes before expiry are critical")
    print("   • Recent volatility (30-60min) heavily weighted")
    print("   • Market regime changes happen quickly in crypto")


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
