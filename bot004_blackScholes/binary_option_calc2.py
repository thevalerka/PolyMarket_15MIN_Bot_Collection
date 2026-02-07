import json
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import time

class BinaryOptionCalculator:
    def __init__(self, price_file_path="/home/ubuntu/013_2025_polymarket/btc_price.json"):
        self.price_file_path = price_file_path
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (adjust as needed)
        self.results_file = "/home/ubuntu/013_2025_polymarket/option_probabilities.json"
        self.current_strike = None
        self.hour_start_time = None
        self.hour_start_price = None
        self.binance_history = []  # Store Binance data in memory

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

        # Calculate end of current hour
        end_of_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        # Time difference in years
        time_diff = (end_of_hour - current_time).total_seconds() / (365.25 * 24 * 3600)

        return max(time_diff, 1e-8)  # Avoid division by zero

    def fetch_binance_klines(self, hours=168):
        """Fetch historical kline data from Binance API"""
        try:
            # Get data for the specified number of hours
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit={hours}"

            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())

            return data
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            return None

    def initialize_price_history_from_binance(self, hours=168):
        """Initialize price history from Binance kline data (store in memory only)"""
        print(f"Fetching {hours} hours of BTC price history from Binance...")

        klines = self.fetch_binance_klines(hours)
        if not klines:
            print("Failed to fetch Binance data")
            return False

        # Convert kline data to our format and store in memory
        self.binance_history = []

        for kline in klines:
            timestamp = int(kline[0])  # Open time
            close_price = float(kline[4])  # Close price

            self.binance_history.append({
                'price': close_price,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp/1000).isoformat()
            })

        print(f"Successfully loaded {len(self.binance_history)} data points into memory")
        print(f"Date range: {self.binance_history[0]['datetime']} to {self.binance_history[-1]['datetime']}")

        return True

    def calculate_realized_volatility(self, lookback_hours=24):
        """Calculate realized volatility from Binance data in memory"""
        if len(self.binance_history) < 2:
            print("Insufficient price history. Using default volatility of 80%.")
            return 0.8

        # Sort by timestamp (should already be sorted)
        history = sorted(self.binance_history, key=lambda x: x['timestamp'])

        # Filter to lookback period
        if len(history) > 1:
            latest_time = history[-1]['timestamp']
            cutoff_time = latest_time - (lookback_hours * 60 * 60 * 1000)
            recent_history = [h for h in history if h['timestamp'] >= cutoff_time]
        else:
            recent_history = history

        if len(recent_history) < 2:
            print(f"Insufficient recent data ({len(recent_history)} points). Using default volatility of 80%.")
            return 0.8

        # Calculate returns
        prices = [h['price'] for h in recent_history]
        returns = []

        for i in range(1, len(prices)):
            ret = np.log(prices[i] / prices[i-1])
            returns.append(ret)

        if len(returns) < 2:
            print("Insufficient returns data. Using default volatility of 80%.")
            return 0.8

        # Calculate volatility (annualized)
        returns_std = np.std(returns)

        # Determine frequency (assuming roughly hourly data)
        time_diff_hours = (recent_history[-1]['timestamp'] - recent_history[0]['timestamp']) / (1000 * 60 * 60)
        freq_per_hour = len(returns) / time_diff_hours if time_diff_hours > 0 else 1

        # Annualize volatility (8760 hours in a year)
        annual_volatility = returns_std * np.sqrt(8760 * freq_per_hour)

        # Cap volatility between reasonable bounds for BTC
        annual_volatility = max(0.2, min(3.0, annual_volatility))

        return annual_volatility

    def get_hour_start_price_from_binance(self):
        """Get the OPEN price of the last kline from Binance data as strike"""
        try:
            # Fetch the latest kline data
            klines = self.fetch_binance_klines(hours=1)  # Just get the last hour
            if not klines:
                return None, None

            # Get the last (most recent) kline
            last_kline = klines[-1]

            # Extract OPEN price (index 1) and timestamp (index 0)
            open_price = float(last_kline[1])
            timestamp = int(last_kline[0])

            print(f"📊 Last Binance Kline Data:")
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

            # New hour detected
            self.hour_start_time = current_hour_start_timestamp

            # Get OPEN price from last Binance kline as strike
            strike_price, kline_timestamp = self.get_hour_start_price_from_binance()

            if strike_price:
                self.current_strike = strike_price
                self.hour_start_price = strike_price  # Using OPEN as starting reference

                print(f"\n🕐 NEW HOUR DETECTED:")
                print(f"Current Hour Start: {current_hour_start.isoformat()}")
                print(f"Strike Price (OPEN of last kline): ${self.current_strike:,.2f}")
                print(f"Kline Timestamp: {datetime.fromtimestamp(kline_timestamp/1000).isoformat()}")
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

        # Probability of finishing above strike (call option)
        prob_win = norm.cdf(d2)
        return prob_win

    def black_scholes_binary_put(self, S, K, T, r, sigma):
        """Calculate binary put option probability using Black-Scholes"""
        return 1 - self.black_scholes_binary_call(S, K, T, r, sigma)

    def calculate_probabilities(self):
        """Calculate both call and put probabilities"""
        current_price, timestamp = self.get_current_price()
        if current_price is None:
            return None

        # Check if new hour and set new strike
        self.check_new_hour(timestamp)

        if self.current_strike is None:
            print("No strike price set. Waiting for new hour...")
            return None

        # Save current price to history
        #self.save_price_to_history(current_price, timestamp)

        # Calculate time to expiry and volatility
        time_to_expiry = self.calculate_time_to_expiry(timestamp)
        volatility = self.calculate_realized_volatility()

        # Calculate probabilities
        call_probability = self.black_scholes_binary_call(
            current_price, self.current_strike, time_to_expiry,
            self.risk_free_rate, volatility
        )

        put_probability = self.black_scholes_binary_put(
            current_price, self.current_strike, time_to_expiry,
            self.risk_free_rate, volatility
        )

        # Calculate price change from hour start
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
            'call_probability': call_probability,
            'put_probability': put_probability,
            'moneyness': current_price / self.current_strike
        }

        return result

    def save_results_to_file(self, result):
        """Save probability results to JSON file"""
        try:


            # Save back to file
            with open(self.results_file, 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"Error saving results: {e}")

    def print_status(self, result):
        """Print current status (overwrites previous line)"""
        print(f"\r⏰ {result['datetime'][11:19]} | "
              f"Price: ${result['current_price']:>8,.2f} | "
              f"Strike: ${result['strike_price']:>8,.2f} | "
              f"Change: {result['price_change_pct']:>+6.2f}% | "
              f"Call: {result['call_probability']:>6.2%} | "
              f"Put: {result['put_probability']:>6.2%} | "
              f"TTExp: {result['time_to_expiry_hours']:>5.2f}h", end="", flush=True)

    def run_continuous(self):
        """Run continuous probability calculations"""
        print("🚀 Starting BTC Binary Option Continuous Monitor")
        print("=" * 60)

        # Always load fresh Binance data
        print("📡 Loading fresh historical data from Binance...")
        self.initialize_price_history_from_binance()

        # Get initial volatility
        volatility = self.calculate_realized_volatility()
        print(f"📊 Initial 24h realized volatility: {volatility:.1%}")

        # Set initial strike price from current Binance OPEN
        print("\n📍 Setting initial strike price...")
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

        print("\n🔄 Starting continuous monitoring (Press Ctrl+C to stop)")
        print("Live updates below (single line, overwrites):")
        print("-" * 80)

        try:
            while True:
                result = self.calculate_probabilities()

                if result:
                    self.save_results_to_file(result)  # Save the latest result
                    self.print_status(result)

                time.sleep(1)  # Update every second

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")


def main():
    """Main execution function"""
    calc = BinaryOptionCalculator()
    calc.run_continuous()


# Utility functions
def refresh_binance_data():
    """Standalone function to refresh Binance data in memory"""
    calc = BinaryOptionCalculator()
    success = calc.initialize_price_history_from_binance()
    if success:
        print("Binance data refreshed successfully in memory!")
    else:
        print("Failed to refresh Binance data.")


def analyze_single_option(strike_price, option_type="call"):
    """Analyze a single option with given strike"""
    calc = BinaryOptionCalculator()
    calc.initialize_price_history_from_binance()

    current_price, timestamp = calc.get_current_price()
    if not current_price:
        return

    calc.current_strike = strike_price
    result = calc.calculate_probabilities()

    if result:
        print(f"\n📊 SINGLE OPTION ANALYSIS")
        print(f"{'='*50}")
        print(f"Current Price: ${result['current_price']:,.2f}")
        print(f"Strike Price: ${result['strike_price']:,.2f}")
        print(f"Option Type: {option_type.upper()}")
        print(f"Time to Expiry: {result['time_to_expiry_hours']:.2f} hours")
        print(f"Volatility: {result['volatility']:.2%}")

        if option_type.lower() == "call":
            print(f"WIN PROBABILITY: {result['call_probability']:.2%}")
        else:
            print(f"WIN PROBABILITY: {result['put_probability']:.2%}")


if __name__ == "__main__":
    main()
