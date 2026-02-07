# spike_strategy.py - BTC Spike Detection and Market Analysis
"""
BTC Spike Detection System

Core Strategy:
- Monitor BTC price changes via WebSocket
- Detect spikes when price change > average absolute body of last 5min candles
- Calculate dynamic thresholds based on recent market volatility
- Trigger buy orders near ASK when spikes detected
- Implement protected sell logic with minimum quantity requirements

The system uses candlestick analysis:
- Body = abs(close - open) for each time period
- Spike threshold = average body size over 5-minute window
- Real-time detection via WebSocket integration
"""

import json
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ Numpy available - enhanced mathematical operations")
except ImportError:
    print("⚠️ Numpy not available - using basic Python math")
    print("💡 For better performance: pip3 install numpy")
    NUMPY_AVAILABLE = False
    # Create a minimal numpy substitute
    class MockNumpy:
        @staticmethod
        def mean(arr): return sum(arr) / len(arr) if arr else 0
        @staticmethod
        def std(arr):
            if not arr or len(arr) < 2: return 0
            mean_val = sum(arr) / len(arr)
            variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
            return variance ** 0.5
        @staticmethod
        def array(arr): return list(arr)
        @staticmethod
        def abs(arr): return [abs(x) for x in arr]
        @staticmethod
        def max(arr): return max(arr) if arr else 0
        @staticmethod
        def min(arr): return min(arr) if arr else 0
    np = MockNumpy()

from collar_config import state, MIN_QUANTITY_FILTER

class BTCCandlestick:
    """Represents a BTC price candlestick for spike analysis."""

    def __init__(self, timestamp: float, open_price: float, high_price: float,
                 low_price: float, close_price: float, period_seconds: int = 60):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.period_seconds = period_seconds

        # Calculate candlestick metrics
        self.body = abs(close_price - open_price)
        self.upper_wick = high_price - max(open_price, close_price)
        self.lower_wick = min(open_price, close_price) - low_price
        self.total_range = high_price - low_price
        self.body_percentage = (self.body / self.total_range) if self.total_range > 0 else 0

        # Direction
        self.is_bullish = close_price > open_price
        self.is_bearish = close_price < open_price
        self.is_doji = abs(close_price - open_price) / self.total_range < 0.1 if self.total_range > 0 else True

    def __repr__(self):
        direction = "🟢" if self.is_bullish else "🔴" if self.is_bearish else "⚪"
        return f"{direction} OHLC: {self.open:.2f}/{self.high:.2f}/{self.low:.2f}/{self.close:.2f} Body: {self.body:.2f}"

class SpikeDetector:
    """Detect BTC market spikes using candlestick body analysis."""

    def __init__(self, lookback_minutes: int = 5, min_spike_threshold: float = 0.003):
        self.lookback_minutes = lookback_minutes
        self.min_spike_threshold = min_spike_threshold  # 0.3% minimum spike

        # Price data storage
        self.price_points = deque(maxlen=1000)  # Store last 1000 price points
        self.candlesticks = deque(maxlen=100)   # Store last 100 candlesticks

        # Spike detection state
        self.last_spike_time = 0
        self.spike_cooldown = 60  # 1 minute cooldown between spikes
        self.current_candle_data = None

        # Analysis metrics
        self.avg_body_size = 0
        self.current_threshold = min_spike_threshold
        self.volatility_multiplier = 1.0

        print(f"🔍 Spike Detector initialized: {lookback_minutes}min lookback, {min_spike_threshold:.1%} min threshold")

    def add_price_point(self, price: float, timestamp_ms: int):
        """Add new price point and update candlestick data."""
        timestamp = timestamp_ms / 1000.0  # Convert to seconds

        # Add to price points
        self.price_points.append((timestamp, price))

        # Update or create current candlestick (1-minute periods)
        self._update_current_candlestick(timestamp, price)

        # Clean old data
        self._clean_old_data(timestamp)

        # Update analysis metrics
        self._update_analysis_metrics()

    def _update_current_candlestick(self, timestamp: float, price: float):
        """Update current 1-minute candlestick with new price."""
        # Determine which 1-minute period this timestamp belongs to
        period_start = math.floor(timestamp / 60) * 60

        if self.current_candle_data is None or self.current_candle_data['period_start'] != period_start:
            # Finalize previous candle if it exists
            if self.current_candle_data is not None:
                candle = BTCCandlestick(
                    timestamp=self.current_candle_data['period_start'],
                    open_price=self.current_candle_data['open'],
                    high_price=self.current_candle_data['high'],
                    low_price=self.current_candle_data['low'],
                    close_price=self.current_candle_data['close']
                )
                self.candlesticks.append(candle)

            # Start new candle
            self.current_candle_data = {
                'period_start': period_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'last_update': timestamp
            }
        else:
            # Update current candle
            self.current_candle_data['high'] = max(self.current_candle_data['high'], price)
            self.current_candle_data['low'] = min(self.current_candle_data['low'], price)
            self.current_candle_data['close'] = price
            self.current_candle_data['last_update'] = timestamp

    def _clean_old_data(self, current_timestamp: float):
        """Remove data older than the lookback period."""
        cutoff_time = current_timestamp - (self.lookback_minutes * 60)

        # Clean price points
        while self.price_points and self.price_points[0][0] < cutoff_time:
            self.price_points.popleft()

        # Clean candlesticks
        while self.candlesticks and self.candlesticks[0].timestamp < cutoff_time:
            self.candlesticks.popleft()

    def _update_analysis_metrics(self):
        """Update analysis metrics for spike detection."""
        if len(self.candlesticks) < 2:
            return

        # Calculate average body size from recent candlesticks
        
        recent_bodies = [candle.body for candle in list(self.candlesticks)[-min(5, len(self.candlesticks)):]]

        if recent_bodies:
            if NUMPY_AVAILABLE:
                self.avg_body_size = np.mean(recent_bodies)
                body_std = np.std(recent_bodies)
            else:
                self.avg_body_size = sum(recent_bodies) / len(recent_bodies)
                body_mean = self.avg_body_size
                body_std = (sum((b - body_mean) ** 2 for b in recent_bodies) / len(recent_bodies)) ** 0.5

            # Calculate dynamic threshold
            base_threshold = max(self.min_spike_threshold, self.avg_body_size * 0.5)
            volatility_adjustment = body_std * 2.0  # More volatile = higher threshold
            self.current_threshold = base_threshold + volatility_adjustment

            # Calculate volatility multiplier for context
            if self.avg_body_size > 0:
                self.volatility_multiplier = body_std / self.avg_body_size
            else:
                self.volatility_multiplier = 1.0

    def detect_spike(self) -> Tuple[bool, Dict]:
        """Detect if current price movement constitutes a spike."""
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_spike_time < self.spike_cooldown:
            return False, {'reason': 'cooldown_active', 'cooldown_remaining': self.spike_cooldown - (current_time - self.last_spike_time)}

        # Need sufficient data
        if len(self.price_points) < 10 or len(self.candlesticks) < 3:
            return False, {'reason': 'insufficient_data', 'price_points': len(self.price_points), 'candlesticks': len(self.candlesticks)}

        # Get recent price movement
        latest_prices = list(self.price_points)[-10:]  # Last 10 price points
        if len(latest_prices) < 2:
            return False, {'reason': 'insufficient_recent_data'}

        # Calculate immediate price change (last vs. first of recent points)
        start_price = latest_prices[0][1]
        end_price = latest_prices[-1][1]

        if start_price <= 0:
            return False, {'reason': 'invalid_start_price'}

        price_change = abs(end_price - start_price) / start_price
        price_change_direction = 1 if end_price > start_price else -1

        # Compare against threshold
        spike_detected = price_change > self.current_threshold

        spike_info = {
            'detected': spike_detected,
            'price_change': price_change,
            'price_change_direction': price_change_direction,
            'threshold': self.current_threshold,
            'strength': price_change / self.current_threshold if self.current_threshold > 0 else 0,
            'start_price': start_price,
            'end_price': end_price,
            'avg_body_size': self.avg_body_size,
            'volatility_multiplier': self.volatility_multiplier,
            'candlesticks_count': len(self.candlesticks),
            'timestamp': current_time
        }

        if spike_detected:
            self.last_spike_time = current_time
            direction_emoji = "📈" if price_change_direction > 0 else "📉"
            print(f"🚨 SPIKE DETECTED! {direction_emoji}")
            print(f"   Price change: {price_change:.3%} > threshold {self.current_threshold:.3%}")
            print(f"   Strength: {spike_info['strength']:.2f}x")
            print(f"   Price: ${start_price:,.2f} → ${end_price:,.2f}")

        return spike_detected, spike_info

    def get_latest_spike_status(self) -> Tuple[bool, Dict]:
        """Get latest spike detection status without triggering new detection."""
        current_time = time.time()

        # Return recent spike if within last 30 seconds
        if current_time - self.last_spike_time < 30:
            return True, {
                'detected': True,
                'age_seconds': current_time - self.last_spike_time,
                'status': 'recent_spike'
            }

        return False, {
            'detected': False,
            'last_spike_ago': current_time - self.last_spike_time if self.last_spike_time > 0 else None,
            'status': 'monitoring'
        }

    def get_analysis_summary(self) -> Dict:
        """Get comprehensive analysis summary."""
        return {
            'data_points': {
                'price_points': len(self.price_points),
                'candlesticks': len(self.candlesticks),
                'lookback_minutes': self.lookback_minutes
            },
            'metrics': {
                'avg_body_size': self.avg_body_size,
                'current_threshold': self.current_threshold,
                'min_threshold': self.min_spike_threshold,
                'volatility_multiplier': self.volatility_multiplier
            },
            'spike_status': {
                'last_spike_time': self.last_spike_time,
                'cooldown_seconds': self.spike_cooldown,
                'time_since_last_spike': time.time() - self.last_spike_time if self.last_spike_time > 0 else None
            },
            'recent_candlesticks': [
                {
                    'timestamp': candle.timestamp,
                    'body': candle.body,
                    'direction': '🟢' if candle.is_bullish else '🔴' if candle.is_bearish else '⚪',
                    'range': candle.total_range
                }
                for candle in list(self.candlesticks)[-5:]  # Last 5 candlesticks
            ]
        }

class MarketDataProcessor:
    """Process and filter market data from JSON files."""

    def __init__(self, min_quantity: float = MIN_QUANTITY_FILTER):
        self.min_quantity = min_quantity

    def load_book_data(self, file_path: str) -> Optional[Dict]:
        """Load and filter order book data."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Filter bids and asks by minimum quantity
            filtered_bids = []
            filtered_asks = []

            if 'complete_book' in data:
                bids = data['complete_book'].get('bids', [])
                asks = data['complete_book'].get('asks', [])

                for bid in bids:
                    if float(bid['size']) >= self.min_quantity:
                        filtered_bids.append({
                            'price': float(bid['price']),
                            'size': float(bid['size'])
                        })

                for ask in asks:
                    if float(ask['size']) >= self.min_quantity:
                        filtered_asks.append({
                            'price': float(ask['price']),
                            'size': float(ask['size'])
                        })

            # Sort for best prices
            filtered_bids.sort(key=lambda x: x['price'], reverse=True)  # Highest first
            filtered_asks.sort(key=lambda x: x['price'])  # Lowest first

            result = {
                'timestamp': data.get('timestamp'),
                'delay_ms': data.get('delay_ms', 0),
                'filtered_bids': filtered_bids,
                'filtered_asks': filtered_asks,
                'total_bids_filtered': len(filtered_bids),
                'total_asks_filtered': len(filtered_asks),
                'best_filtered_bid': filtered_bids[0] if filtered_bids else None,
                'best_filtered_ask': filtered_asks[0] if filtered_asks else None,
                'original_best_bid': data.get('best_bid', {}),
                'original_best_ask': data.get('best_ask', {})
            }

            # Calculate market spread and mid
            if result['best_filtered_bid'] and result['best_filtered_ask']:
                result['market_spread'] = result['best_filtered_ask']['price'] - result['best_filtered_bid']['price']
                result['market_mid'] = (result['best_filtered_bid']['price'] + result['best_filtered_ask']['price']) / 2
            else:
                result['market_spread'] = None
                result['market_mid'] = None

            # Find protected ASK levels (1000+ shares for protected sells)
            result['protected_ask_levels'] = self._find_protected_ask_levels(filtered_asks)

            return result

        except Exception as e:
            print(f"❌ Error loading book data: {e}")
            return None

    def _find_protected_ask_levels(self, asks: List[Dict]) -> List[Dict]:
        """Find ASK levels with 1000+ shares for protected sell orders."""
        protected_levels = []

        for ask in asks:
            if ask['size'] >= 1000:  # Minimum 1000 shares requirement
                protected_levels.append({
                    'price': ask['price'],
                    'size': ask['size'],
                    'protection_level': 'high' if ask['size'] >= 2000 else 'medium'
                })

        return protected_levels

    def load_btc_data(self, file_path: str) -> Optional[Dict]:
        """Load BTC price data."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            return {
                'price': float(data.get('price', 0)),
                'timestamp': data.get('timestamp'),
                'symbol': data.get('symbol', 'BTCUSDT')
            }

        except Exception as e:
            print(f"❌ Error loading BTC data: {e}")
            return None

class MarketAnalyzer:
    """Market analyzer focused on spike-driven trading opportunities."""

    def __init__(self, book_path: str, btc_path: str):
        self.book_path = book_path
        self.btc_path = btc_path
        self.data_processor = MarketDataProcessor()

    def analyze_market(self) -> Optional[Dict]:
        """Perform market analysis for spike trading."""
        # Load market data
        book_data = self.data_processor.load_book_data(self.book_path)
        btc_data = self.data_processor.load_btc_data(self.btc_path)

        if not book_data or not btc_data:
            return None

        # Calculate spike buy target (near best ASK)
        spike_buy_target = None
        if book_data.get('best_filtered_ask'):
            best_ask = book_data['best_filtered_ask']
            # Target price slightly below best ASK for better fill probability
            spike_buy_target = max(0.01, round(best_ask['price'] - 0.001, 3))

        # Calculate spread-based sell targets for non-spike periods
        spread_sell_target = None
        if book_data.get('market_mid') and book_data.get('market_spread'):
            market_mid = book_data['market_mid']
            market_spread = book_data['market_spread']
            # Sell target above mid with spread buffer
            spread_buffer = max(0.01, market_spread * 0.7)  # 70% of market spread
            spread_sell_target = min(0.99, round(market_mid + spread_buffer, 3))

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'book_data': book_data,
            'btc_data': btc_data,
            'spike_targets': {
                'buy_near_ask': spike_buy_target,
                'protected_ask_levels': book_data.get('protected_ask_levels', [])
            },
            'spread_targets': {
                'sell_above_mid': spread_sell_target,
                'spread_buffer': max(0.01, book_data.get('market_spread', 0.02) * 0.7) if book_data.get('market_spread') else 0.01
            }
        }

        # Store in global state
        state.current_analysis = analysis
        state.last_update_time = time.time()

        return analysis

def display_analysis(analysis: Dict, spike_info: Dict = None):
    """Display market analysis results with spike information."""
    if not analysis:
        print("❌ No analysis available")
        return

    book = analysis.get('book_data', {})
    btc = analysis.get('btc_data', {})
    spike_targets = analysis.get('spike_targets', {})
    spread_targets = analysis.get('spread_targets', {})

    if not book or not btc:
        print("❌ Incomplete analysis data - skipping display")
        return

    print("=" * 90)
    print("📊 BTC SPIKE TRADING ANALYSIS")
    print("=" * 90)

    # BTC and Market Data
    print(f"💰 BTC Price: ${btc.get('price', 0):,.2f}")
    print(f"⏱️ Book Delay: {book.get('delay_ms', 9999)}ms")

    # Order Book Status
    best_bid_data = book.get('best_filtered_bid')
    best_ask_data = book.get('best_filtered_ask')

    if best_bid_data and best_ask_data:
        print(f"📊 Market: ${best_bid_data['price']:.3f} x ${best_ask_data['price']:.3f}")
        market_mid = book.get('market_mid')
        market_spread = book.get('market_spread')

        if market_mid and market_mid > 0:
            print(f"📊 Market Mid: ${market_mid:.3f}")
        if market_spread:
            print(f"📊 Market Spread: ${market_spread:.3f}")
        print(f"📊 Valid Orders: {book.get('total_bids_filtered', 0)} bids, {book.get('total_asks_filtered', 0)} asks")
    else:
        print("⚠️ Empty order book detected!")

    print()

    # Spike Information
    if spike_info:
        print("⚡ BTC SPIKE STATUS")
        if spike_info.get('detected'):
            strength = spike_info.get('strength', 0)
            price_change = spike_info.get('price_change', 0)
            direction = "📈" if spike_info.get('price_change_direction', 0) > 0 else "📉"

            print(f"🚨 SPIKE ACTIVE! {direction}")
            print(f"📊 Price Change: {price_change:.3%}")
            print(f"📊 Spike Strength: {strength:.2f}x threshold")
            print(f"📊 Threshold: {spike_info.get('threshold', 0):.3%}")

            if 'age_seconds' in spike_info:
                print(f"📊 Spike Age: {spike_info['age_seconds']:.1f} seconds")
        else:
            last_spike = spike_info.get('last_spike_ago')
            if last_spike:
                print(f"⏳ Monitoring... (Last spike: {last_spike:.1f}s ago)")
            else:
                print("👁️ Monitoring for spikes...")

            if 'threshold' in spike_info:
                print(f"📊 Current Threshold: {spike_info['threshold']:.3%}")
        print()

    # Spike Trading Targets
    if spike_targets:
        print("🎯 SPIKE TRADING TARGETS")
        buy_target = spike_targets.get('buy_near_ask')
        if buy_target:
            print(f"⚡ Spike Buy Target: ${buy_target:.3f} (near ASK)")
        else:
            print("❌ No spike buy target available")

        # Protected ASK levels for sells
        protected_levels = spike_targets.get('protected_ask_levels', [])
        if protected_levels:
            print(f"🛡️ Protected ASK Levels ({len(protected_levels)} found):")
            for i, level in enumerate(protected_levels[:3]):  # Show top 3
                protection = level['protection_level']
                emoji = "🟢" if protection == 'high' else "🟡"
                print(f"   {emoji} ${level['price']:.3f} ({level['size']:.0f} shares)")
        else:
            print("❌ No protected ASK levels found (need 1000+ shares)")
        print()

    # Spread-Based Targets
    if spread_targets:
        print("📈 SPREAD-BASED TARGETS")
        sell_target = spread_targets.get('sell_above_mid')
        if sell_target:
            print(f"📊 Spread Sell Target: ${sell_target:.3f}")
            spread_buffer = spread_targets.get('spread_buffer', 0)
            print(f"📊 Spread Buffer: ${spread_buffer:.3f}")
        else:
            print("❌ No spread sell target available")
        print()

    print("=" * 90)
