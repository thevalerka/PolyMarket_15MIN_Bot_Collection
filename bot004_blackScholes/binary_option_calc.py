import json
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import urllib.request
import urllib.error

class BinaryOptionCalculator:
    def __init__(self, price_file_path="/home/ubuntu/013_2025_polymarket/btc_price.json"):
        self.price_file_path = price_file_path
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (adjust as needed)
        self.history_file = "/home/ubuntu/013_2025_polymarket/btc_price_history.json"

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
        """Initialize price history from Binance kline data"""
        print(f"Fetching {hours} hours of BTC price history from Binance...")

        klines = self.fetch_binance_klines(hours)
        if not klines:
            print("Failed to fetch Binance data")
            return False

        # Convert kline data to our format
        # Kline format: [timestamp, open, high, low, close, volume, close_time, ...]
        history = []

        for kline in klines:
            timestamp = int(kline[0])  # Open time
            close_price = float(kline[4])  # Close price

            history.append({
                'price': close_price,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp/1000).isoformat()
            })

        # Save to history file
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)

            print(f"Successfully initialized price history with {len(history)} data points")
            print(f"Date range: {history[0]['datetime']} to {history[-1]['datetime']}")

            return True

        except Exception as e:
            print(f"Error saving price history: {e}")
            return False

    def check_and_initialize_history(self):
        """Check if price history exists, if not initialize from Binance"""
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)

            if len(history) < 10:  # If less than 10 data points, refresh from Binance
                print("Insufficient price history found. Fetching from Binance...")
                return self.initialize_price_history_from_binance()
            else:
                print(f"Found existing price history with {len(history)} data points")
                return True

        except FileNotFoundError:
            print("No price history found. Initializing from Binance...")
            return self.initialize_price_history_from_binance()
        except Exception as e:
            print(f"Error checking price history: {e}")
            return self.initialize_price_history_from_binance()
        """Save current price to historical data"""
        try:
            # Load existing history
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = []

            # Add current price
            history.append({
                'price': price,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp/1000).isoformat()
            })

            # Keep only last 168 hours (1 week) of data
            cutoff_time = timestamp - (168 * 60 * 60 * 1000)  # 168 hours ago
            history = [h for h in history if h['timestamp'] > cutoff_time]

            # Save back to file
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save price history: {e}")

    def save_price_to_history(self, price, timestamp):
        """Save current price to historical data"""
        try:
            # Load existing history
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = []

            # Add current price
            history.append({
                'price': price,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp/1000).isoformat()
            })

            # Keep only last 168 hours (1 week) of data
            cutoff_time = timestamp - (168 * 60 * 60 * 1000)  # 168 hours ago
            history = [h for h in history if h['timestamp'] > cutoff_time]

            # Save back to file
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save price history: {e}")

    def load_price_history(self):
        """Load historical price data"""
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            return history
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Warning: Could not load price history: {e}")
            return []

    def calculate_realized_volatility(self, lookback_hours=24):
        """Calculate realized volatility from historical price data"""
        history = self.load_price_history()

        if len(history) < 2:
            print("Insufficient price history. Using default volatility of 80%.")
            return 0.8

        # Sort by timestamp
        history = sorted(history, key=lambda x: x['timestamp'])

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

        print(f"Calculated {lookback_hours}h realized volatility: {annual_volatility:.1%} ({len(returns)} price points)")

        return annual_volatility

    def black_scholes_binary_call(self, S, K, T, r, sigma):
        """
        Calculate binary call option probability using Black-Scholes

        S: Current price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility

        Returns: Probability of finishing above strike
        """
        if T <= 0:
            return 1.0 if S > K else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Probability of finishing above strike (call option)
        prob_win = norm.cdf(d2)

        return prob_win

    def black_scholes_binary_put(self, S, K, T, r, sigma):
        """
        Calculate binary put option probability using Black-Scholes

        Returns: Probability of finishing below strike
        """
        return 1 - self.black_scholes_binary_call(S, K, T, r, sigma)

    def calculate_option_probability(self, strike_price, option_type="call", starting_price=None):
        """
        Main function to calculate binary option win probability

        strike_price: The strike price for the binary option
        option_type: "call" (above) or "put" (below)
        starting_price: Starting price for comparison (optional)
        """
        # Check and initialize price history if needed
        self.check_and_initialize_history()

        # Get current price and timestamp
        current_price, timestamp = self.get_current_price()

        if current_price is None:
            print("Could not retrieve current price")
            return None

        # Save current price to history for future volatility calculations
        self.save_price_to_history(current_price, timestamp)

        # Calculate time to expiry
        time_to_expiry = self.calculate_time_to_expiry(timestamp)

        # Calculate realized volatility from historical data
        volatility = self.calculate_realized_volatility()

        # Calculate probability
        if option_type.lower() == "call":
            probability = self.black_scholes_binary_call(
                current_price, strike_price, time_to_expiry, self.risk_free_rate, volatility
            )
            direction = "ABOVE"
        else:
            probability = self.black_scholes_binary_put(
                current_price, strike_price, time_to_expiry, self.risk_free_rate, volatility
            )
            direction = "BELOW"

        # Display results
        print(f"\n{'='*50}")
        print(f"BINARY OPTION PROBABILITY ANALYSIS")
        print(f"{'='*50}")
        print(f"Current BTC Price: ${current_price:,.2f}")
        print(f"Strike Price: ${strike_price:,.2f}")
        print(f"Option Type: {option_type.upper()} ({direction} strike)")
        print(f"Time to Expiry: {time_to_expiry*8760:.2f} hours")
        print(f"Realized Volatility: {volatility:.2%}")
        print(f"Risk-Free Rate: {self.risk_free_rate:.2%}")

        if starting_price:
            price_change = ((current_price - starting_price) / starting_price) * 100
            print(f"Starting Price: ${starting_price:,.2f}")
            print(f"Price Change: {price_change:+.2f}%")

        print(f"\nWIN PROBABILITY: {probability:.2%}")
        print(f"LOSE PROBABILITY: {1-probability:.2%}")

        # Additional insights
        moneyness = current_price / strike_price
        if option_type.lower() == "call":
            if moneyness > 1.02:
                print(f"Status: Deep IN-THE-MONEY (current price {moneyness:.3f}x strike)")
            elif moneyness > 1.001:
                print(f"Status: IN-THE-MONEY (current price {moneyness:.3f}x strike)")
            elif moneyness > 0.999:
                print(f"Status: AT-THE-MONEY (current price {moneyness:.3f}x strike)")
            else:
                print(f"Status: OUT-OF-THE-MONEY (current price {moneyness:.3f}x strike)")
        else:
            if moneyness < 0.98:
                print(f"Status: Deep IN-THE-MONEY (current price {moneyness:.3f}x strike)")
            elif moneyness < 0.999:
                print(f"Status: IN-THE-MONEY (current price {moneyness:.3f}x strike)")
            elif moneyness < 1.001:
                print(f"Status: AT-THE-MONEY (current price {moneyness:.3f}x strike)")
            else:
                print(f"Status: OUT-OF-THE-MONEY (current price {moneyness:.3f}x strike)")

        return {
            'probability': probability,
            'current_price': current_price,
            'strike_price': strike_price,
            'time_to_expiry_hours': time_to_expiry * 8760,
            'volatility': volatility,
            'option_type': option_type,
            'moneyness': moneyness
        }

    def sensitivity_analysis(self, strike_price, option_type="call", base_volatility=None):
        """Perform sensitivity analysis on key parameters"""
        current_price, _ = self.get_current_price()
        if not current_price:
            return

        # Use calculated volatility if not provided
        if base_volatility is None:
            base_volatility = self.calculate_realized_volatility()

        print(f"\n{'='*50}")
        print(f"SENSITIVITY ANALYSIS")
        print(f"{'='*50}")

        # Volatility sensitivity
        vol_range = np.arange(0.2, 2.1, 0.2)
        print(f"\nVolatility Sensitivity (Current: {base_volatility:.1%}):")
        print(f"{'Volatility':<12} {'Probability':<12}")
        print("-" * 24)

        for vol in vol_range:
            time_to_expiry = self.calculate_time_to_expiry()
            if option_type.lower() == "call":
                prob = self.black_scholes_binary_call(current_price, strike_price, time_to_expiry, self.risk_free_rate, vol)
            else:
                prob = self.black_scholes_binary_put(current_price, strike_price, time_to_expiry, self.risk_free_rate, vol)
            print(f"{vol:.1%}       {prob:.2%}")

        # Price movement sensitivity
        price_changes = np.arange(-5, 6, 1)  # -5% to +5%
        print(f"\nPrice Movement Sensitivity:")
        print(f"{'Price Change':<15} {'New Price':<12} {'Probability':<12}")
        print("-" * 39)

        for change in price_changes:
            new_price = current_price * (1 + change/100)
            time_to_expiry = self.calculate_time_to_expiry()
            if option_type.lower() == "call":
                prob = self.black_scholes_binary_call(new_price, strike_price, time_to_expiry, self.risk_free_rate, base_volatility)
            else:
                prob = self.black_scholes_binary_put(new_price, strike_price, time_to_expiry, self.risk_free_rate, base_volatility)
            print(f"{change:+.1f}%           ${new_price:>8,.0f}   {prob:.2%}")


def main():
    """Example usage"""
    calc = BinaryOptionCalculator()

    print("BTC Binary Option Probability Calculator")
    print("=" * 50)

    try:
        # Ask if user wants to refresh price history
        refresh = input("Refresh price history from Binance? (y/n): ").lower()
        if refresh == 'y':
            calc.initialize_price_history_from_binance()

        # Get user inputs
        strike_price = float(input("Enter strike price: $"))
        option_type = input("Enter option type (call/put): ").lower()

        # Optional starting price for comparison
        start_price_input = input("Enter starting price for comparison (optional): $")
        starting_price = float(start_price_input) if start_price_input else None

        # Calculate probability (volatility is now calculated automatically)
        result = calc.calculate_option_probability(
            strike_price=strike_price,
            option_type=option_type,
            starting_price=starting_price
        )

        # Ask for sensitivity analysis
        if input("\nRun sensitivity analysis? (y/n): ").lower() == 'y':
            calc.sensitivity_analysis(strike_price, option_type, result['volatility'])

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")


# Additional utility function for manual history refresh
def refresh_price_history():
    """Standalone function to refresh price history"""
    calc = BinaryOptionCalculator()
    success = calc.initialize_price_history_from_binance()
    if success:
        print("Price history refreshed successfully!")
    else:
        print("Failed to refresh price history.")


if __name__ == "__main__":
    main()
