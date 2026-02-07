import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import time
import os

class Enhanced15MinBinaryOptionCalculator:
    def __init__(self, price_file_path="/home/ubuntu/013_2025_polymarket/btc_price.json"):
        self.price_file_path = price_file_path
        self.risk_free_rate = 0.05
        self.results_base_path = "/home/ubuntu/013_2025_polymarket/bot004_blackScholes/daily/"
        self.current_date = None
        self.results_file = None
        self.current_strike = None
        self.period_start_time = None
        self.period_start_price = None
        self.binance_history = []
        self.minute_history = []
        self.volatility_method = 'combined'  # 'ewma', 'multi_timeframe', 'garch', 'combined'
        self.last_minute_data_fetch = None  # Track when minute data was last fetched
        self._initialize_daily_file()

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

    def _initialize_daily_file(self):
        """Initialize daily results file path"""
        today = datetime.now().strftime("%Y-%m-%d")

        # Check if we need to start a new day
        if self.current_date != today:
            self.current_date = today

            # Create directory if it doesn't exist
            os.makedirs(self.results_base_path, exist_ok=True)

            self.results_file = f"{self.results_base_path}option_probabilities_M15_{today}.json"

            # Check if file exists to determine if it's a new day
            if os.path.exists(self.results_file):
                print(f"📂 Continuing with existing file: {self.results_file}")
            else:
                print(f"📂 Creating new daily file: {self.results_file}")

    def get_daily_filename(self):
        """Get the current daily filename"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.current_date != today:
            self._initialize_daily_file()
        return self.results_file

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

    def calculate_oscillation_metrics(self, lookback_minutes=60):
        """
        Calculate comprehensive oscillation/choppiness metrics for the last N minutes
        Returns: dict with 5 key metrics measuring path complexity and oscillation intensity
        """
        if len(self.minute_history) < lookback_minutes:
            return {
                'efficiency_ratio': None,
                'choppiness_index': None,
                'path_length_ratio': None,
                'zero_crossing_rate': None,
                'fractal_dimension': None
            }
        
        # Get the last N minutes of data
        history = sorted(self.minute_history, key=lambda x: x['timestamp'])
        recent = history[-lookback_minutes:]
        prices = np.array([h['price'] for h in recent])
        
        # 1. EFFICIENCY RATIO (0 to 1, higher = more efficient/trending)
        # Measures how directly price moved from start to end
        net_change = abs(prices[-1] - prices[0])
        total_movement = np.sum(np.abs(np.diff(prices)))
        efficiency_ratio = net_change / total_movement if total_movement > 0 else 0
        
        # 2. CHOPPINESS INDEX (0 to 100, higher = more choppy)
        # Standard measure for choppy vs trending markets
        high_low_sum = 0
        for i in range(len(prices) - 1):
            high_low_sum += abs(prices[i+1] - prices[i])
        
        price_range = np.max(prices) - np.min(prices)
        if price_range > 0 and high_low_sum > 0:
            choppiness_index = 100 * np.log10(high_low_sum / price_range) / np.log10(lookback_minutes)
        else:
            choppiness_index = 50  # Neutral value
        
        # 3. PATH LENGTH RATIO (≥1, higher = more oscillation)
        # How much longer the actual path is compared to straight line
        straight_line_distance = abs(prices[-1] - prices[0])
        if straight_line_distance > 0:
            path_length_ratio = total_movement / straight_line_distance
        else:
            path_length_ratio = 1.0 if total_movement == 0 else float('inf')
        
        # Cap at reasonable value for display
        if path_length_ratio > 1000:
            path_length_ratio = 1000
        
        # 4. ZERO-CROSSING RATE (count per minute)
        # How often price direction changes
        returns = np.diff(prices)
        direction_changes = np.sum(np.diff(np.sign(returns)) != 0)
        zero_crossing_rate = direction_changes / (lookback_minutes - 1)
        
        # 5. FRACTAL DIMENSION (1.0 to 2.0)
        # Geometric complexity: 1.0 = straight line, 2.0 = space-filling
        # Using Higuchi's method with improved implementation
        try:
            N = len(prices)
            k_max = min(20, N // 4)  # Increased k_max for better resolution
            
            if k_max < 3:
                fractal_dimension = 1.0
            else:
                Lk = []
                k_values = []
                
                for k in range(2, k_max + 1):  # Start from 2, not 1
                    Lm = []
                    for m in range(k):
                        Lmk = 0
                        max_i = int((N - m - 1) / k)
                        
                        if max_i < 1:
                            continue
                        
                        # Calculate curve length for this subsequence
                        for i in range(1, max_i + 1):
                            idx1 = m + (i - 1) * k
                            idx2 = m + i * k
                            if idx2 < N:
                                Lmk += abs(prices[idx2] - prices[idx1])
                        
                        # Normalize
                        if max_i > 0:
                            Lmk = Lmk * (N - 1) / (max_i * k * k)
                            Lm.append(Lmk)
                    
                    if len(Lm) > 0:
                        Lk.append(np.mean(Lm))
                        k_values.append(k)
                
                # Calculate fractal dimension from log-log slope
                if len(Lk) > 2 and all(lk > 0 for lk in Lk):
                    log_k = np.log(k_values)
                    log_Lk = np.log(Lk)
                    
                    # Linear regression on log-log plot
                    slope, intercept = np.polyfit(log_k, log_Lk, 1)
                    fractal_dimension = -slope  # Negative slope gives the dimension
                    
                    # Clamp to valid range and add small variation if too flat
                    fractal_dimension = max(1.0, min(2.0, fractal_dimension))
                    
                    # If result is suspiciously close to 1.0, use alternative method
                    if abs(fractal_dimension - 1.0) < 0.001:
                        # Simple variance-based estimate as fallback
                        normalized_returns = returns / np.std(returns) if np.std(returns) > 0 else returns
                        variance_ratio = np.var(normalized_returns)
                        fractal_dimension = 1.0 + min(1.0, variance_ratio * 0.5)
                else:
                    fractal_dimension = 1.0
        except Exception as e:
            # Fallback: use simple variance-based estimate
            try:
                normalized_returns = returns / np.std(returns) if np.std(returns) > 0 else returns
                variance_ratio = np.var(normalized_returns)
                fractal_dimension = 1.0 + min(1.0, variance_ratio * 0.5)
            except:
                fractal_dimension = 1.0
        
        return {
            'efficiency_ratio': round(efficiency_ratio, 4),
            'choppiness_index': round(choppiness_index, 2),
            'path_length_ratio': round(path_length_ratio, 2),
            'zero_crossing_rate': round(zero_crossing_rate, 3),
            'fractal_dimension': round(fractal_dimension, 3)
        }

    def calculate_ma_crossover_analysis(self):
        """
        Analyze 5-minute MA crossovers in the last 15 minutes
        Returns: (num_crossings, max_swing_distance, avg_last_3_swings_pct_atr)
        """
        if len(self.minute_history) < 15:
            return None, None, None
        
        # Get the last 15 minutes of data
        history = sorted(self.minute_history, key=lambda x: x['timestamp'])
        last_15_min = history[-15:]
        
        # Create pandas DataFrame for easier analysis
        df = pd.DataFrame(last_15_min)
        df['price'] = df['price'].astype(float)
        
        # Calculate 5-minute moving average (5 periods)
        df['ma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
        
        # Detect crossovers (significant ones - more than 0.01% from MA)
        threshold = 0.0001  # 0.01% threshold for "significant"
        df['above_ma'] = (df['price'] - df['ma_5']) / df['ma_5'] > threshold
        df['below_ma'] = (df['ma_5'] - df['price']) / df['ma_5'] > threshold
        
        # Count crossovers (changes in state)
        crossings = 0
        swing_distances = []
        current_swing_max = 0
        
        for i in range(1, len(df)):
            # Check for state change (crossing)
            if df['above_ma'].iloc[i] != df['above_ma'].iloc[i-1]:
                crossings += 1
                if current_swing_max > 0:
                    swing_distances.append(current_swing_max)
                current_swing_max = 0
            
            # Track maximum distance in current swing
            distance = abs(df['price'].iloc[i] - df['ma_5'].iloc[i])
            current_swing_max = max(current_swing_max, distance)
        
        # Add the last swing if it exists
        if current_swing_max > 0:
            swing_distances.append(current_swing_max)
        
        # Calculate max swing distance
        max_swing = max(swing_distances) if swing_distances else 0
        
        # Calculate ATR for the last 15 minutes using high-low from 1-min candles
        atr = self.calculate_atr_1min(periods=15)
        
        # Calculate average of last 3 swings as % of ATR
        avg_last_3_swings_pct = None
        if len(swing_distances) >= 3 and atr > 0:
            last_3_swings = swing_distances[-3:]
            avg_swing = sum(last_3_swings) / len(last_3_swings)
            avg_last_3_swings_pct = (avg_swing / atr) * 100
        elif len(swing_distances) > 0 and atr > 0:
            # If less than 3 swings, use what we have
            avg_swing = sum(swing_distances) / len(swing_distances)
            avg_last_3_swings_pct = (avg_swing / atr) * 100
        
        return crossings, max_swing, avg_last_3_swings_pct
    
    def calculate_atr_1min(self, periods=15):
        """
        Calculate Average True Range using 1-minute candles
        Uses high-low range (not open-close)
        """
        if len(self.minute_history) < periods:
            return 0
        
        # Fetch fresh 1-minute data with OHLC
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={periods}"
            with urllib.request.urlopen(url) as response:
                klines = json.loads(response.read().decode())
            
            true_ranges = []
            for i, kline in enumerate(klines):
                high = float(kline[2])
                low = float(kline[3])
                
                # For first candle, TR is just high-low
                if i == 0:
                    tr = high - low
                else:
                    prev_close = float(klines[i-1][4])
                    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
                    tr = max(high - low, 
                            abs(high - prev_close), 
                            abs(low - prev_close))
                
                true_ranges.append(tr)
            
            # Return average of true ranges
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0
            
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            # Fallback: use simple high-low from minute_history
            history = sorted(self.minute_history, key=lambda x: x['timestamp'])
            last_n = history[-periods:]
            prices = [h['price'] for h in last_n]
            if len(prices) > 1:
                return max(prices) - min(prices)
            return 0

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

        # Calculate MA crossover analysis (15 min)
        ma_crossings, max_swing, avg_swings_pct_atr = self.calculate_ma_crossover_analysis()
        
        # Calculate comprehensive oscillation metrics (60 min)
        oscillation_metrics = self.calculate_oscillation_metrics(lookback_minutes=60)

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
            'minute_data_last_fetch': datetime.fromtimestamp(self.last_minute_data_fetch).isoformat() if self.last_minute_data_fetch else None,
            'ma_5min_crossings_last_15min': ma_crossings,
            'max_swing_distance': max_swing,
            'avg_last_3_swings_pct_atr': avg_swings_pct_atr,
            # Oscillation metrics (60 min window)
            'oscillation_60min': oscillation_metrics
        }

        return result

    def save_results_to_file(self, result):
        """Save probability results immediately to daily JSON file (append mode)"""
        try:
            # Check if we need to start a new day
            self._initialize_daily_file()

            # Write result immediately as JSON line
            with open(self.results_file, 'a') as f:
                json.dump(result, f)
                f.write('\n')  # Add newline for readability


            option_prob_15M = "/home/ubuntu/013_2025_polymarket/option_probabilities_M15.json"

            # Write result immediately as JSON line
            with open(option_prob_15M, 'w') as f:
                json.dump(result, f)
                f.write('\n')  # Add newline for readability

        except Exception as e:
            print(f"❌ Error saving results: {e}")

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

        # Format MA analysis data (15 min)
        ma_crossings = result.get('ma_5min_crossings_last_15min', 'N/A')
        max_swing = result.get('max_swing_distance', 0)
        avg_swings_pct = result.get('avg_last_3_swings_pct_atr', 0)
        
        ma_info = f"MA-X:{ma_crossings} | MaxSwing:${max_swing:.2f}"
        if avg_swings_pct is not None:
            ma_info += f" | Avg3Swing:{avg_swings_pct:.1f}%ATR"
        else:
            ma_info += f" | Avg3Swing:N/A"
        
        # Format oscillation metrics (60 min)
        osc = result.get('oscillation_60min', {})
        eff_ratio = osc.get('efficiency_ratio', 0)
        chop_idx = osc.get('choppiness_index', 0)
        path_ratio = osc.get('path_length_ratio', 0)
        zcr = osc.get('zero_crossing_rate', 0)
        fractal = osc.get('fractal_dimension', 0)
        
        # Create oscillation info string
        if eff_ratio is not None:
            osc_info = f"EffRatio:{eff_ratio:.3f} | ChopIdx:{chop_idx:.1f} | PathRatio:{path_ratio:.1f}x | ZCR:{zcr:.2f}/min | Fractal:{fractal:.3f}"
        else:
            osc_info = "Oscillation metrics: Insufficient data"

        print(f"\r⏰ {result['datetime'][11:19]} | "
              f"Price: ${result['current_price']:>8,.2f} | "
              f"Strike: ${result['strike_price']:>8,.2f} | "
              f"Change: {result['price_change_pct']:>+6.2f}% | "
              f"Call: {result['call_probability']:>6.2%} | "
              f"Put: {result['put_probability']:>6.2%} | "
              f"Vol({vol_method}): {result['volatility']:>5.1%} | "
              f"TTExp: {result['time_to_expiry_minutes']:>5.2f}m\n"
              f"📊 15min: {ma_info} | Data:{fresh_icon}{freshness_pct:>3.0f}%\n"
              f"📈 60min: {osc_info}", end="", flush=True)

    def run_continuous(self):
        """Run continuous probability calculations for 15-minute binary options"""
        print("🚀 Starting Enhanced BTC 15-Minute Binary Option Monitor")
        print("=" * 70)
        print("⏰ Expiry Schedule: Every 15 minutes at :00, :15, :30, :45")
        print(f"📈 Using volatility method: {self.volatility_method.upper()}")
        print("🔄 Minute data refresh frequency: Adaptive based on time to expiry")
        print(f"💾 Daily folder: {self.results_base_path}")
        print(f"📊 File: {self.get_daily_filename()}")

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
        print("💡 All data written immediately to daily JSON file (every second)")
        print("Live updates below (overwrites):")
        print("-" * 110)

        try:
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
            print(f"💾 Data saved to: {self.results_file}")
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

    print(f"💾 Daily folder: {calc.results_base_path}")
    print(f"📊 Daily file: {calc.get_daily_filename()}")


def show_daily_stats():
    """Show basic daily file info"""
    calc = Enhanced15MinBinaryOptionCalculator()

    if os.path.exists(calc.results_file):
        file_size = os.path.getsize(calc.results_file)
        with open(calc.results_file, 'r') as f:
            line_count = sum(1 for line in f)

        print(f"\n📊 DAILY FILE INFO - {calc.current_date}")
        print("=" * 40)
        print(f"📂 File: {calc.results_file}")
        print(f"📏 Size: {file_size:,} bytes")
        print(f"📝 Lines: {line_count:,} entries")
    else:
        print(f"📊 No data file exists yet for {calc.current_date}")


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
    print("\n💾 DAILY FILE SYSTEM:")
    print("   • Folder: /home/ubuntu/013_2025_polymarket/bot004_blackScholes/daily/")
    print("   • New file created each day: option_probabilities_M15_YYYY-MM-DD.json")
    print("   • Data written immediately (every second) - NO MEMORY STORAGE")
    print("   • Each line is a complete JSON record")
    print("   • Ultra-low memory footprint")
    print("\n💡 Why frequent updates matter for 15-minute options:")
    print("   • Ultra-short timeframe makes timing critical")
    print("   • Last 3 minutes before expiry are extremely sensitive")
    print("   • Volatility estimation needs recent data (15-30min)")
    print("   • Price movements have amplified impact on short-term probabilities")


def test_oscillation_metrics():
    """Test the comprehensive oscillation metrics functionality"""
    calc = Enhanced15MinBinaryOptionCalculator()
    
    print("\n🔬 TESTING OSCILLATION METRICS (60-MINUTE WINDOW)")
    print("=" * 70)
    
    # Load minute data
    print("📊 Loading minute data from Binance...")
    if not calc.initialize_minute_data(minutes=720):
        print("❌ Failed to load minute data")
        return
    
    print("✅ Data loaded successfully\n")
    
    # Calculate oscillation metrics
    metrics = calc.calculate_oscillation_metrics(lookback_minutes=60)
    
    print("📈 OSCILLATION ANALYSIS (Last 60 Minutes):")
    print("-" * 70)
    
    # 1. Efficiency Ratio
    er = metrics['efficiency_ratio']
    if er is not None:
        print(f"\n1️⃣  EFFICIENCY RATIO: {er:.4f}")
        print(f"    Range: 0.0 (very choppy) to 1.0 (perfect trend)")
        if er > 0.7:
            print(f"    ✓ Strong trend (>0.7) - price moved efficiently")
        elif er > 0.4:
            print(f"    ✓ Moderate trend (0.4-0.7) - some oscillation")
        else:
            print(f"    ✓ Choppy market (<0.4) - lots of back-and-forth")
    else:
        print(f"\n1️⃣  EFFICIENCY RATIO: N/A (insufficient data)")
    
    # 2. Choppiness Index
    ci = metrics['choppiness_index']
    if ci is not None:
        print(f"\n2️⃣  CHOPPINESS INDEX: {ci:.2f}")
        print(f"    Range: 0 (trending) to 100 (choppy)")
        if ci > 61.8:
            print(f"    ✓ Choppy/Ranging market (>61.8)")
        elif ci < 38.2:
            print(f"    ✓ Strong trending market (<38.2)")
        else:
            print(f"    ✓ Mixed conditions (38.2-61.8)")
    else:
        print(f"\n2️⃣  CHOPPINESS INDEX: N/A (insufficient data)")
    
    # 3. Path Length Ratio
    plr = metrics['path_length_ratio']
    if plr is not None:
        print(f"\n3️⃣  PATH LENGTH RATIO: {plr:.2f}x")
        print(f"    How much longer the actual path is vs straight line")
        if plr < 2:
            print(f"    ✓ Very direct path (<2x) - strong trend")
        elif plr < 5:
            print(f"    ✓ Moderate oscillation (2-5x) - normal market")
        else:
            print(f"    ✓ Heavy oscillation (>5x) - choppy market")
    else:
        print(f"\n3️⃣  PATH LENGTH RATIO: N/A (insufficient data)")
    
    # 4. Zero-Crossing Rate
    zcr = metrics['zero_crossing_rate']
    if zcr is not None:
        print(f"\n4️⃣  ZERO-CROSSING RATE: {zcr:.3f} changes/min")
        print(f"    How often price direction changes")
        if zcr < 0.3:
            print(f"    ✓ Low frequency (<0.3) - trending")
        elif zcr < 0.6:
            print(f"    ✓ Moderate frequency (0.3-0.6) - normal")
        else:
            print(f"    ✓ High frequency (>0.6) - very choppy")
    else:
        print(f"\n4️⃣  ZERO-CROSSING RATE: N/A (insufficient data)")
    
    # 5. Fractal Dimension
    fd = metrics['fractal_dimension']
    if fd is not None:
        print(f"\n5️⃣  FRACTAL DIMENSION: {fd:.3f}")
        print(f"    Range: 1.0 (straight line) to 2.0 (space-filling)")
        if fd < 1.3:
            print(f"    ✓ Low complexity (<1.3) - smooth trend")
        elif fd < 1.6:
            print(f"    ✓ Moderate complexity (1.3-1.6) - normal oscillation")
        else:
            print(f"    ✓ High complexity (>1.6) - very jagged path")
    else:
        print(f"\n5️⃣  FRACTAL DIMENSION: N/A (insufficient data)")
    
    # Summary interpretation
    print("\n" + "=" * 70)
    print("💡 OVERALL MARKET CHARACTERIZATION:")
    print("-" * 70)
    
    if all(v is not None for v in metrics.values()):
        # Score the choppiness (0-5, higher = more choppy)
        chop_score = 0
        if er < 0.4: chop_score += 1
        if ci > 61.8: chop_score += 1
        if plr > 5: chop_score += 1
        if zcr > 0.6: chop_score += 1
        if fd > 1.6: chop_score += 1
        
        if chop_score >= 4:
            print("🔴 VERY CHOPPY MARKET - High oscillation, ranging conditions")
            print("   → Consider: fade extremes, avoid trend-following")
        elif chop_score >= 2:
            print("🟡 MIXED CONDITIONS - Moderate oscillation")
            print("   → Consider: wait for clearer signals, use wider stops")
        else:
            print("🟢 TRENDING MARKET - Low oscillation, directional movement")
            print("   → Consider: follow trend, use trend-following strategies")
    else:
        print("⚠️  Insufficient data for complete analysis")


def test_ma_crossover_analysis():
    """Test the MA crossover analysis functionality"""
    calc = Enhanced15MinBinaryOptionCalculator()
    
    print("\n🔬 TESTING MA CROSSOVER ANALYSIS")
    print("=" * 60)
    
    # Load minute data
    print("📊 Loading minute data from Binance...")
    if not calc.initialize_minute_data(minutes=20):
        print("❌ Failed to load minute data")
        return
    
    print("✅ Data loaded successfully\n")
    
    # Calculate MA crossover analysis
    crossings, max_swing, avg_swings_pct_atr = calc.calculate_ma_crossover_analysis()
    
    print("📈 ANALYSIS RESULTS (Last 15 Minutes):")
    print("-" * 60)
    print(f"1️⃣  5-Min MA Crossings: {crossings if crossings is not None else 'N/A'}")
    print(f"    (Number of times price significantly crossed the 5-min MA)")
    print()
    print(f"2️⃣  Max Swing Distance: ${max_swing if max_swing is not None else 'N/A':.2f}")
    print(f"    (Maximum absolute distance from the MA in any swing)")
    print()
    print(f"3️⃣  Avg Last 3 Swings (% of ATR): {avg_swings_pct_atr if avg_swings_pct_atr is not None else 'N/A':.1f}%")
    print(f"    (Average of last 3 swing maximums as % of 1-min ATR)")
    print()
    
    # Calculate current ATR for reference
    atr = calc.calculate_atr_1min(periods=15)
    print(f"📊 Current 1-Min ATR (15 periods): ${atr:.2f}")
    print()
    
    print("💡 INTERPRETATION:")
    print("-" * 60)
    if crossings is not None:
        if crossings == 0:
            print("✓ No MA crossings = Strong trend (bullish or bearish)")
        elif crossings <= 2:
            print("✓ Few MA crossings = Moderate trend with some oscillation")
        else:
            print("✓ Multiple MA crossings = Choppy/ranging market")
        
        if avg_swings_pct_atr is not None:
            if avg_swings_pct_atr < 30:
                print("✓ Low swing % of ATR = Tight price action")
            elif avg_swings_pct_atr < 60:
                print("✓ Medium swing % of ATR = Normal volatility")
            else:
                print("✓ High swing % of ATR = High volatility/large moves")
    else:
        print("⚠️  Not enough data for analysis")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "refresh":
            force_refresh_data()
        elif sys.argv[1] == "compare":
            analyze_volatility_methods()
        elif sys.argv[1] == "schedule":
            show_refresh_schedule()
        elif sys.argv[1] == "stats":
            show_daily_stats()
        elif sys.argv[1] == "test_ma":
            test_ma_crossover_analysis()
        elif sys.argv[1] == "test_osc":
            test_oscillation_metrics()
        else:
            print("Usage: python script.py [refresh|compare|schedule|stats|test_ma|test_osc]")
            print("\nOptions:")
            print("  refresh   - Force refresh all data from Binance")
            print("  compare   - Compare different volatility methods")
            print("  schedule  - Show data refresh schedule")
            print("  stats     - Show daily file statistics")
            print("  test_ma   - Test MA crossover analysis")
            print("  test_osc  - Test oscillation metrics (NEW)")
    else:
        main()