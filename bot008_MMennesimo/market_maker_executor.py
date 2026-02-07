#!/usr/bin/env python3
"""
Market Maker Trading Executor

New strategy:
1. Place limit orders as market maker where book quantity > 1111
2. Cancel orders on ML prediction of sudden moves
3. Directional bias: BTC UP (CALL buy/PUT sell), BTC DOWN (PUT buy/CALL sell)
4. Position at best bid/ask on strong directional signals
"""

import time
import json
import os
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# Import ML detector for volatility prediction
from ml_arbitrage_detector import MLArbitrageDetector

@dataclass
class MarketMakerConfig:
    """Market maker configuration"""

    # API Configuration
    clob_api_url: str
    private_key: str
    api_key: str
    api_secret: str
    api_passphrase: str
    chain_id: int = 137

    # File paths
    btc_file: str = '/home/ubuntu/013_2025_polymarket/btc_price.json'
    call_file: str = '/home/ubuntu/013_2025_polymarket/CALL.json'
    put_file: str = '/home/ubuntu/013_2025_polymarket/PUT.json'

    # Market Making Parameters
    min_book_quantity: int = 1111              # Minimum quantity threshold for order placement
    max_position_size_usd: float = 5.0        # Max position size per order
    max_total_exposure_usd: float = 20.0      # Max total exposure

    # Dynamic Volatility-Based Thresholds
    volatility_lookback_minutes: int = 60       # Minutes to look back for volatility calculation
    directional_volatility_multiplier: float = 1.5  # Multiple of current volatility for directional bias
    strong_signal_volatility_multiplier: float = 2.5  # Multiple for strong signals

    # ML Volatility Prediction (dynamic)
    volatility_cancel_threshold: float = 0.70   # ML confidence to cancel orders
    spike_volatility_multiplier: float = 3.0    # Multiple of current volatility for spike detection

    # Order Management
    order_timeout_seconds: int = 120           # 2 minute order timeout
    price_improvement_cents: float = 0.001     # Try to improve price by 0.1 cent
    max_spread_percentage: float = 0.10        # Don't trade if spread > 10%

    # Risk Management
    max_orders_per_side: int = 2               # Max orders per side per option
    cancel_on_regime_change: bool = True       # Cancel orders when market regime changes

class VolatilityAnalyzer:
    """Dynamic volatility analysis using real-time Binance data"""

    def __init__(self, lookback_minutes: int = 60):
        self.lookback_minutes = lookback_minutes
        self.binance_url = "https://api.binance.com/api/v3/klines"

        # Volatility metrics cache
        self.current_volatility = 0.0
        self.volatility_percentiles = {}
        self.last_update = 0
        self.update_interval = 30  # Update every 30 seconds

        # Historical data for regime detection
        self.volatility_history = []
        self.max_history_length = 288  # 24 hours of 5-minute updates

        print("📊 Dynamic Volatility Analyzer initialized")
        print(f"📈 Lookback period: {lookback_minutes} minutes")

    def fetch_klines(self) -> Optional[pd.DataFrame]:
        """Fetch recent BTC klines from Binance"""
        try:
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'limit': self.lookback_minutes
            }

            response = requests.get(self.binance_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_timestamp', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])

            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)

            # Convert timestamp
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df

        except Exception as e:
            print(f"❌ Error fetching Binance klines: {e}")
            return None

    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive volatility metrics"""
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Remove NaN values
            returns = df['returns'].dropna()
            log_returns = df['log_returns'].dropna()

            if len(returns) < 10:
                return {}

            # Basic volatility metrics
            volatility_1m = returns.std()
            volatility_5m = returns.rolling(5).std().iloc[-1] if len(returns) >= 5 else volatility_1m
            volatility_15m = returns.rolling(15).std().iloc[-1] if len(returns) >= 15 else volatility_1m

            # Realized volatility (annualized)
            realized_vol = volatility_1m * np.sqrt(525600)  # Minutes in a year

            # High-Low volatility
            df['hl_vol'] = (df['high'] - df['low']) / df['close']
            hl_volatility = df['hl_vol'].mean()

            # Volume-weighted volatility
            df['volume_float'] = pd.to_numeric(df['volume'], errors='coerce')
            volume_weights = df['volume_float'] / df['volume_float'].sum()
            volume_weighted_vol = (returns.abs() * volume_weights[1:]).sum()  # Skip first NaN

            # Volatility percentiles for thresholds
            vol_percentiles = {
                'p10': returns.abs().quantile(0.10),
                'p25': returns.abs().quantile(0.25),
                'p50': returns.abs().quantile(0.50),
                'p75': returns.abs().quantile(0.75),
                'p90': returns.abs().quantile(0.90),
                'p95': returns.abs().quantile(0.95),
                'p99': returns.abs().quantile(0.99)
            }

            # Recent momentum
            recent_return = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0

            # Market regime detection
            current_vol = volatility_5m
            avg_vol = returns.rolling(30).std().mean() if len(returns) >= 30 else volatility_1m

            if current_vol > avg_vol * 1.5:
                regime = 'high_volatility'
            elif current_vol < avg_vol * 0.5:
                regime = 'low_volatility'
            else:
                regime = 'normal_volatility'

            metrics = {
                'current_volatility': volatility_1m,
                'volatility_5m': volatility_5m,
                'volatility_15m': volatility_15m,
                'realized_volatility': realized_vol,
                'hl_volatility': hl_volatility,
                'volume_weighted_volatility': volume_weighted_vol,
                'percentiles': vol_percentiles,
                'recent_return': recent_return,
                'regime': regime,
                'avg_volatility': avg_vol,
                'volatility_ratio': current_vol / avg_vol if avg_vol > 0 else 1.0,
                'timestamp': time.time()
            }

            return metrics

        except Exception as e:
            print(f"❌ Error calculating volatility metrics: {e}")
            return {}

    def update_volatility_metrics(self) -> bool:
        """Update volatility metrics if needed"""
        current_time = time.time()

        # Check if update is needed
        if current_time - self.last_update < self.update_interval:
            return True  # Use cached data

        print(f"\r📊 Updating volatility metrics...", end='', flush=True)

        # Fetch new data
        df = self.fetch_klines()
        if df is None or len(df) < 10:
            return False

        # Calculate new metrics
        metrics = self.calculate_volatility_metrics(df)
        if not metrics:
            return False

        # Update cached values
        self.current_volatility = metrics['current_volatility']
        self.volatility_percentiles = metrics['percentiles']
        self.last_update = current_time

        # Store in history for regime tracking
        self.volatility_history.append({
            'timestamp': current_time,
            'volatility': self.current_volatility,
            'regime': metrics['regime']
        })

        # Trim history
        if len(self.volatility_history) > self.max_history_length:
            self.volatility_history = self.volatility_history[-self.max_history_length:]

        return True

    def get_dynamic_thresholds(self, multipliers: Dict) -> Dict:
        """Get dynamic thresholds based on current volatility"""
        if not self.update_volatility_metrics():
            # Fallback to conservative fixed thresholds
            return {
                'directional_threshold': 0.0015,
                'strong_signal_threshold': 0.005,
                'spike_threshold': 0.03
            }

        base_vol = self.current_volatility

        # Calculate dynamic thresholds
        thresholds = {
            'directional_threshold': base_vol * multipliers.get('directional', 1.5),
            'strong_signal_threshold': base_vol * multipliers.get('strong_signal', 2.5),
            'spike_threshold': base_vol * multipliers.get('spike', 3.0),
            'percentile_75': self.volatility_percentiles.get('p75', 0.002),
            'percentile_90': self.volatility_percentiles.get('p90', 0.004),
            'percentile_95': self.volatility_percentiles.get('p95', 0.006),
            'current_volatility': base_vol
        }

        return thresholds

    def is_volatility_spike(self, recent_move: float) -> Tuple[bool, str]:
        """Detect if recent price move constitutes a volatility spike"""
        if not self.update_volatility_metrics():
            return False, "No volatility data"

        abs_move = abs(recent_move)

        # Check against various percentile thresholds
        if abs_move > self.volatility_percentiles.get('p99', 0.01):
            return True, "99th percentile spike"
        elif abs_move > self.volatility_percentiles.get('p95', 0.006):
            return True, "95th percentile spike"
        elif abs_move > self.volatility_percentiles.get('p90', 0.004):
            return True, "90th percentile elevated move"

        return False, "Normal volatility"

    def get_regime_info(self) -> Dict:
        """Get current market regime information"""
        if not self.volatility_history:
            return {'regime': 'unknown', 'stability': 'unknown'}

        recent_regimes = [h['regime'] for h in self.volatility_history[-10:]]
        current_regime = recent_regimes[-1] if recent_regimes else 'unknown'

        # Check regime stability
        regime_changes = sum(1 for i in range(1, len(recent_regimes))
                           if recent_regimes[i] != recent_regimes[i-1])

        if regime_changes <= 1:
            stability = 'stable'
        elif regime_changes <= 3:
            stability = 'moderate'
        else:
            stability = 'unstable'

        return {
            'regime': current_regime,
            'stability': stability,
            'regime_changes': regime_changes,
            'volatility_trend': self._get_volatility_trend()
        }

    def _get_volatility_trend(self) -> str:
        """Calculate volatility trend direction"""
        if len(self.volatility_history) < 5:
            return 'insufficient_data'

        recent_vols = [h['volatility'] for h in self.volatility_history[-5:]]
        if len(recent_vols) < 2:
            return 'insufficient_data'

        trend = np.polyfit(range(len(recent_vols)), recent_vols, 1)[0]

        if trend > 0.0001:
            return 'increasing'
        elif trend < -0.0001:
            return 'decreasing'
        else:
            return 'stable'

class MarketMakerOrder:
    """Track individual market maker orders"""

    def __init__(self, order_id: str, token_id: str, side: str, price: float,
                 quantity: float, option_type: str, strategy: str):
        self.order_id = order_id
        self.token_id = token_id
        self.side = side  # BUY or SELL
        self.price = price
        self.quantity = quantity
        self.option_type = option_type  # CALL or PUT
        self.strategy = strategy  # 'market_maker', 'directional', 'best_position'

        self.timestamp = time.time()
        self.status = 'pending'  # pending, filled, cancelled
        self.fill_price = None
        self.fill_quantity = None

    def get_age_seconds(self) -> float:
        return time.time() - self.timestamp

    def should_timeout(self, timeout_seconds: int) -> bool:
        return self.get_age_seconds() > timeout_seconds

    def get_usd_value(self) -> float:
        return self.price * self.quantity

class MarketMakerExecutor:
    """Market maker trading executor with ML volatility prediction"""

    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.client = self._setup_client()

        # ML Detector for volatility prediction
        self.ml_detector = MLArbitrageDetector(
            btc_file=config.btc_file,
            call_file=config.call_file,
            put_file=config.put_file
        )

        # Volatility Analyzer for dynamic thresholds
        self.volatility_analyzer = VolatilityAnalyzer(config.volatility_lookback_minutes)

        # Order tracking
        self.active_orders: Dict[str, MarketMakerOrder] = {}
        self.order_history: List[Dict] = []

        # Token management
        self.call_token_id = None
        self.put_token_id = None
        self._update_token_ids()

        # Market state tracking
        self.last_btc_price = None
        self.btc_momentum = 0.0
        self.current_market_regime = 'normal'
        self.last_regime_change = time.time()

        # Performance tracking
        self.total_pnl = 0.0
        self.orders_placed = 0
        self.orders_filled = 0

        print("🏪 MARKET MAKER EXECUTOR INITIALIZED")
        print(f"📊 Min book quantity: {config.min_book_quantity}")
        print(f"💰 Max position: ${config.max_position_size_usd}")
        print(f"🎯 Dynamic thresholds: {config.directional_volatility_multiplier}x volatility (directional)")
        print(f"⚡ Volatility cancel threshold: {config.volatility_cancel_threshold:.1%}")
        print(f"📈 Lookback period: {config.volatility_lookback_minutes} minutes")

    def _setup_client(self):
        """Initialize CLOB client"""
        try:
            creds = ApiCreds(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                api_passphrase=self.config.api_passphrase
            )
            client = ClobClient(
                self.config.clob_api_url,
                key=self.config.private_key,
                chain_id=self.config.chain_id,
                creds=creds
            )
            print("✅ CLOB client initialized")
            return client
        except Exception as e:
            print(f"❌ Failed to initialize CLOB client: {e}")
            raise

    def _update_token_ids(self):
        """Update token IDs from JSON files"""
        try:
            if os.path.exists(self.config.call_file):
                with open(self.config.call_file, 'r') as f:
                    call_data = json.load(f)
                    self.call_token_id = call_data.get('asset_id')

            if os.path.exists(self.config.put_file):
                with open(self.config.put_file, 'r') as f:
                    put_data = json.load(f)
                    self.put_token_id = put_data.get('asset_id')

            if self.call_token_id and self.put_token_id:
                print(f"📋 Token IDs: CALL={self.call_token_id[:8]}..., PUT={self.put_token_id[:8]}...")
            else:
                print("⚠️ Warning: Could not update token IDs")

        except Exception as e:
            print(f"❌ Error updating token IDs: {e}")

    def get_order_book_data(self, option_type: str) -> Optional[Dict]:
        """Get detailed order book data with quantities - FIXED for correct data structure"""
        try:
            if option_type == 'CALL':
                file_path = self.config.call_file
            elif option_type == 'PUT':
                file_path = self.config.put_file
            else:
                return None

            if not os.path.exists(file_path):
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # FIXED: Access complete_book structure correctly
            complete_book = data.get('complete_book', {})

            # Extract order book - convert strings to floats
            bids_raw = complete_book.get('bids', [])
            asks_raw = complete_book.get('asks', [])

            # Convert string prices/sizes to floats
            bids = []
            for bid in bids_raw:
                if isinstance(bid, dict):
                    try:
                        bids.append({
                            'price': float(bid['price']),
                            'size': float(bid['size'])
                        })
                    except (KeyError, ValueError):
                        continue

            asks = []
            for ask in asks_raw:
                if isinstance(ask, dict):
                    try:
                        asks.append({
                            'price': float(ask['price']),
                            'size': float(ask['size'])
                        })
                    except (KeyError, ValueError):
                        continue

            order_book = {
                'bids': bids,
                'asks': asks,
                'best_bid': data.get('best_bid', {}),
                'best_ask': data.get('best_ask', {}),
                'mid_price': 0,
                'spread': 0
            }

            # Calculate mid price and spread
            if order_book['best_bid'].get('price') and order_book['best_ask'].get('price'):
                bid_price = order_book['best_bid']['price']
                ask_price = order_book['best_ask']['price']
                order_book['mid_price'] = (bid_price + ask_price) / 2
                order_book['spread'] = ask_price - bid_price
                order_book['spread_pct'] = order_book['spread'] / order_book['mid_price']

            return order_book

        except Exception as e:
            print(f"❌ Error getting order book for {option_type}: {e}")
            return None

    def find_market_making_opportunities(self, order_book: Dict) -> List[Dict]:
        """Find opportunities to place market making orders where quantity > min_threshold - FIXED with debugging"""
        opportunities = []

        if not order_book or not order_book.get('bids') or not order_book.get('asks'):
            print(f"⚠️ No order book data available")
            return opportunities

        # Check spread - don't trade if too wide
        if order_book.get('spread_pct', 0) > self.config.max_spread_percentage:
            print(f"⚠️ Spread too wide: {order_book.get('spread_pct', 0):.1%} > {self.config.max_spread_percentage:.1%}")
            return opportunities

        print(f"📊 Analyzing order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")

        # Find bid opportunities (where we can sell)
        large_bids = 0
        for bid in order_book['bids']:
            if isinstance(bid, dict) and bid.get('size', 0) >= self.config.min_book_quantity:
                large_bids += 1
                opportunities.append({
                    'side': 'SELL',
                    'price': bid['price'],
                    'available_quantity': bid['size'],
                    'strategy': 'market_maker',
                    'book_position': 'bid_liquidity'
                })

        # Find ask opportunities (where we can buy)
        large_asks = 0
        for ask in order_book['asks']:
            if isinstance(ask, dict) and ask.get('size', 0) >= self.config.min_book_quantity:
                large_asks += 1
                opportunities.append({
                    'side': 'BUY',
                    'price': ask['price'],
                    'available_quantity': ask['size'],
                    'strategy': 'market_maker',
                    'book_position': 'ask_liquidity'
                })

        print(f"🎯 Found {large_bids} large bids, {large_asks} large asks (>= {self.config.min_book_quantity})")
        print(f"💡 Total market making opportunities: {len(opportunities)}")

        return opportunities

    def get_btc_momentum(self) -> Tuple[float, str, Dict]:
        """Calculate BTC momentum and direction using dynamic volatility thresholds"""
        try:
            with open(self.config.btc_file, 'r') as f:
                btc_data = json.load(f)

            current_price = btc_data['price']

            # Get dynamic thresholds based on current volatility
            multipliers = {
                'directional': self.config.directional_volatility_multiplier,
                'strong_signal': self.config.strong_signal_volatility_multiplier,
                'spike': self.config.spike_volatility_multiplier
            }

            dynamic_thresholds = self.volatility_analyzer.get_dynamic_thresholds(multipliers)

            if self.last_btc_price is not None:
                price_change = current_price - self.last_btc_price
                momentum = price_change / self.last_btc_price

                # Use dynamic thresholds for direction classification
                directional_threshold = dynamic_thresholds['directional_threshold']
                strong_threshold = dynamic_thresholds['strong_signal_threshold']

                # Determine direction using dynamic thresholds
                if momentum > strong_threshold:
                    direction = 'STRONG_UP'
                elif momentum > directional_threshold:
                    direction = 'UP'
                elif momentum < -strong_threshold:
                    direction = 'STRONG_DOWN'
                elif momentum < -directional_threshold:
                    direction = 'DOWN'
                else:
                    direction = 'NEUTRAL'

                # Check for volatility spike
                is_spike, spike_reason = self.volatility_analyzer.is_volatility_spike(momentum)
                if is_spike:
                    direction += f"_SPIKE({spike_reason})"

                self.btc_momentum = momentum
                self.last_btc_price = current_price

                return momentum, direction, dynamic_thresholds
            else:
                self.last_btc_price = current_price
                return 0.0, 'NEUTRAL', dynamic_thresholds

        except Exception as e:
            print(f"❌ Error calculating BTC momentum: {e}")
            # Return fallback thresholds
            fallback_thresholds = {
                'directional_threshold': 0.0015,
                'strong_signal_threshold': 0.005,
                'spike_threshold': 0.03,
                'current_volatility': 0.001
            }
            return 0.0, 'NEUTRAL', fallback_thresholds

    def get_directional_opportunities(self, momentum: float, direction: str,
                                    call_book: Dict, put_book: Dict,
                                    dynamic_thresholds: Dict) -> List[Dict]:
        """Generate directional trading opportunities based on BTC momentum"""
        opportunities = []

        # Strong directional signals - place at best bid/ask
        if direction == 'STRONG_UP':
            # BTC UP: CALL buy @ best bid, PUT sell @ best ask
            if call_book and call_book.get('best_bid', {}).get('price'):
                opportunities.append({
                    'option_type': 'CALL',
                    'side': 'BUY',
                    'price': call_book['best_bid']['price'],
                    'strategy': 'directional_strong',
                    'signal_strength': abs(momentum),
                    'direction': direction
                })

            if put_book and put_book.get('best_ask', {}).get('price'):
                opportunities.append({
                    'option_type': 'PUT',
                    'side': 'SELL',
                    'price': put_book['best_ask']['price'],
                    'strategy': 'directional_strong',
                    'signal_strength': abs(momentum),
                    'direction': direction
                })

        elif direction == 'STRONG_DOWN':
            # BTC DOWN: PUT buy @ best bid, CALL sell @ best ask
            if put_book and put_book.get('best_bid', {}).get('price'):
                opportunities.append({
                    'option_type': 'PUT',
                    'side': 'BUY',
                    'price': put_book['best_bid']['price'],
                    'strategy': 'directional_strong',
                    'signal_strength': abs(momentum),
                    'direction': direction
                })

            if call_book and call_book.get('best_ask', {}).get('price'):
                opportunities.append({
                    'option_type': 'CALL',
                    'side': 'SELL',
                    'price': call_book['best_ask']['price'],
                    'strategy': 'directional_strong',
                    'signal_strength': abs(momentum),
                    'direction': direction
                })

        # Moderate directional signals - improve prices slightly
        elif direction in ['UP', 'DOWN']:
            price_improvement = self.config.price_improvement_cents

            if direction == 'UP':
                # Moderate BTC up: CALL buy slightly above best bid, PUT sell slightly below best ask
                if call_book and call_book.get('best_bid', {}).get('price'):
                    opportunities.append({
                        'option_type': 'CALL',
                        'side': 'BUY',
                        'price': call_book['best_bid']['price'] + price_improvement,
                        'strategy': 'directional_moderate',
                        'signal_strength': abs(momentum),
                        'direction': direction
                    })

                if put_book and put_book.get('best_ask', {}).get('price'):
                    opportunities.append({
                        'option_type': 'PUT',
                        'side': 'SELL',
                        'price': put_book['best_ask']['price'] - price_improvement,
                        'strategy': 'directional_moderate',
                        'signal_strength': abs(momentum),
                        'direction': direction
                    })

            elif direction == 'DOWN':
                # Moderate BTC down: PUT buy slightly above best bid, CALL sell slightly below best ask
                if put_book and put_book.get('best_bid', {}).get('price'):
                    opportunities.append({
                        'option_type': 'PUT',
                        'side': 'BUY',
                        'price': put_book['best_bid']['price'] + price_improvement,
                        'strategy': 'directional_moderate',
                        'signal_strength': abs(momentum),
                        'direction': direction
                    })

                if call_book and call_book.get('best_ask', {}).get('price'):
                    opportunities.append({
                        'option_type': 'CALL',
                        'side': 'SELL',
                        'price': call_book['best_ask']['price'] - price_improvement,
                        'strategy': 'directional_moderate',
                        'signal_strength': abs(momentum),
                        'direction': direction
                    })

        return opportunities

    def check_ml_volatility_prediction(self, dynamic_thresholds: Dict) -> Tuple[bool, float, str]:
        """Check ML prediction for sudden price movements using dynamic thresholds"""
        try:
            # Get ML opportunities (volatility predictions)
            ml_opportunities = self.ml_detector.analyze_market_data()

            # Use dynamic spike threshold instead of fixed one
            spike_threshold = dynamic_thresholds.get('spike_threshold', 0.03)

            for opp in ml_opportunities:
                confidence = opp.get('confidence', 0)
                profit_potential = abs(opp.get('profit_potential', 0))

                # Check for high confidence predictions of large moves using dynamic threshold
                if (confidence > self.config.volatility_cancel_threshold and
                    profit_potential > spike_threshold):

                    return True, confidence, f"Predicted {opp['type']} {opp['action']} spike (dynamic threshold: {spike_threshold:.3f})"

            return False, 0.0, "No volatility spike predicted"

        except Exception as e:
            print(f"❌ Error checking ML volatility: {e}")
            return False, 0.0, "ML check failed"

    def should_cancel_orders(self, dynamic_thresholds: Dict) -> Tuple[bool, str]:
        """Determine if orders should be cancelled based on dynamic volatility analysis"""

        # Check ML volatility prediction with dynamic thresholds
        should_cancel, confidence, reason = self.check_ml_volatility_prediction(dynamic_thresholds)
        if should_cancel:
            return True, f"ML volatility prediction: {reason} (confidence: {confidence:.1%})"

        # Check for volatility regime changes
        regime_info = self.volatility_analyzer.get_regime_info()
        current_regime = regime_info['regime']

        if (self.config.cancel_on_regime_change and
            current_regime != self.current_market_regime and
            regime_info['stability'] == 'unstable'):

            self.current_market_regime = current_regime
            self.last_regime_change = time.time()
            return True, f"Volatile regime change to {current_regime} (unstable)"

        # Check for extreme volatility spikes using Binance data
        current_vol = dynamic_thresholds.get('current_volatility', 0)
        spike_threshold = dynamic_thresholds.get('spike_threshold', 0.03)

        if current_vol > spike_threshold:
            return True, f"Extreme volatility detected: {current_vol:.4f} > {spike_threshold:.4f}"

        return False, ""

    def calculate_position_size(self, price: float, strategy: str, signal_strength: float = 0) -> float:
        """Calculate position size based on strategy and signal strength"""

        base_usd = self.config.max_position_size_usd

        # Adjust by strategy
        if strategy == 'directional_strong':
            multiplier = 1.2 + min(signal_strength * 10, 0.5)  # Up to 1.7x for strong signals
        elif strategy == 'directional_moderate':
            multiplier = 0.8 + min(signal_strength * 5, 0.3)   # Up to 1.1x for moderate signals
        elif strategy == 'market_maker':
            multiplier = 0.6  # Conservative for market making
        else:
            multiplier = 0.5

        adjusted_usd = base_usd * multiplier
        quantity = adjusted_usd / price

        return max(1.0, round(quantity, 2))  # Minimum 1 token

    def get_balance(self, token_id: str) -> Tuple[int, float]:
        """Get token balance"""
        try:
            response = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id
                )
            )
            balance_raw = int(response.get('balance', 0))
            balance_tokens = balance_raw / 10**6
            return balance_raw, balance_tokens

        except Exception as e:
            print(f"❌ Error getting balance for {token_id[:8]}...: {e}")
            return 0, 0.0

    def check_risk_limits(self) -> Tuple[bool, str]:
        """Check risk limits before placing orders"""

        # Check total exposure
        total_exposure = sum(order.get_usd_value() for order in self.active_orders.values())

        if total_exposure >= self.config.max_total_exposure_usd:
            return False, f"Total exposure limit (${total_exposure:.2f})"

        # Check orders per side per option
        call_orders = sum(1 for o in self.active_orders.values() if o.option_type == 'CALL')
        put_orders = sum(1 for o in self.active_orders.values() if o.option_type == 'PUT')

        if call_orders >= self.config.max_orders_per_side * 2:  # Both sides
            return False, "CALL order limit reached"

        if put_orders >= self.config.max_orders_per_side * 2:
            return False, "PUT order limit reached"

        return True, ""

    def place_order(self, opportunity: Dict) -> Optional[str]:
        """Place a limit order based on opportunity"""

        try:
            option_type = opportunity.get('option_type')
            side = opportunity['side']
            price = opportunity['price']
            strategy = opportunity['strategy']

            # Get token ID
            token_id = self.call_token_id if option_type == 'CALL' else self.put_token_id
            if not token_id:
                print(f"❌ No token ID for {option_type}")
                return None

            # Check risk limits
            can_trade, limit_reason = self.check_risk_limits()
            if not can_trade:
                print(f"🚫 Risk limit: {limit_reason}")
                return None

            # Calculate position size
            signal_strength = opportunity.get('signal_strength', 0)
            quantity = self.calculate_position_size(price, strategy, signal_strength)

            # Check balance for SELL orders
            if side == 'SELL':
                balance_raw, balance_tokens = self.get_balance(token_id)
                if balance_tokens < quantity:
                    print(f"🚫 Insufficient balance: {balance_tokens:.2f} < {quantity:.2f}")
                    return None

            # Create order
            order_args = OrderArgs(
                price=price,
                size=quantity,
                side=BUY if side == 'BUY' else SELL,
                token_id=token_id
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order)

            order_id = response.get('orderId', 'unknown')

            # Track order
            order = MarketMakerOrder(
                order_id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                quantity=quantity,
                option_type=option_type,
                strategy=strategy
            )

            self.active_orders[order_id] = order
            self.orders_placed += 1

            print(f"📋 ORDER PLACED: {option_type} {side} {quantity:.2f} @ ${price:.4f} [{strategy}] - {order_id}")

            return order_id

        except Exception as e:
            print(f"❌ Error placing order: {e}")
            return None

    def cancel_order(self, order_id: str, reason: str = ""):
        """Cancel an active order"""
        try:
            if order_id in self.active_orders:
                self.client.cancel_order(order_id)

                order = self.active_orders[order_id]
                order.status = 'cancelled'

                print(f"❌ CANCELLED: {order.option_type} {order.side} @ ${order.price:.4f} - {reason}")

                # Move to history
                self.order_history.append({
                    'order_id': order_id,
                    'status': 'cancelled',
                    'reason': reason,
                    'timestamp': time.time()
                })

                del self.active_orders[order_id]

        except Exception as e:
            print(f"❌ Error cancelling order {order_id}: {e}")

    def cancel_all_orders(self, reason: str = ""):
        """Cancel all active orders"""
        orders_to_cancel = list(self.active_orders.keys())

        for order_id in orders_to_cancel:
            self.cancel_order(order_id, reason)

        print(f"🧹 CANCELLED ALL ORDERS: {reason}")

    def update_orders(self):
        """Update order status and handle timeouts"""
        orders_to_remove = []

        for order_id, order in self.active_orders.items():
            try:
                # Check for timeouts
                if order.should_timeout(self.config.order_timeout_seconds):
                    self.cancel_order(order_id, "timeout")
                    continue

                # TODO: Check order status with exchange
                # For now, simulate some fills after time
                if order.get_age_seconds() > 30 and order.status == 'pending':
                    # Simulate partial fills (in real implementation, check actual status)
                    pass

            except Exception as e:
                print(f"❌ Error updating order {order_id}: {e}")

    def print_status(self, btc_momentum: float, direction: str, dynamic_thresholds: Dict):
        """Print trading status with dynamic volatility information"""
        active_count = len(self.active_orders)
        total_exposure = sum(order.get_usd_value() for order in self.active_orders.values())

        # Get volatility regime info
        regime_info = self.volatility_analyzer.get_regime_info()
        current_vol = dynamic_thresholds.get('current_volatility', 0)
        directional_threshold = dynamic_thresholds.get('directional_threshold', 0)

        status = (f"\r{datetime.now().strftime('%H:%M:%S')} | "
                 f"Momentum: {btc_momentum:+.3%} ({direction}) | "
                 f"Vol: {current_vol:.4f} | Thresh: {directional_threshold:.4f} | "
                 f"Regime: {regime_info['regime']} ({regime_info['stability']}) | "
                 f"Orders: {active_count} | "
                 f"Exposure: ${total_exposure:.2f} | "
                 f"Placed: {self.orders_placed} | "
                 f"Filled: {self.orders_filled} | "
                 f"PnL: ${self.total_pnl:+.2f}")

        print(status, end='', flush=True)

    def run_market_making_loop(self):
        """Main market making loop"""
        print("🏪 Starting Market Maker Trading Loop")
        print("=" * 80)

        iteration = 0
        last_status_time = 0

        try:
            while True:
                iteration += 1

                # Update token IDs periodically
                if iteration % 100 == 0:
                    self._update_token_ids()

                # Get order books
                call_book = self.get_order_book_data('CALL')
                put_book = self.get_order_book_data('PUT')

                if not call_book or not put_book:
                    time.sleep(1)
                    continue

                # Get BTC momentum with dynamic thresholds
                momentum, direction, dynamic_thresholds = self.get_btc_momentum()

                # Check if we should cancel orders due to volatility
                should_cancel, cancel_reason = self.should_cancel_orders(dynamic_thresholds)
                if should_cancel:
                    self.cancel_all_orders(cancel_reason)

                # Update existing orders
                self.update_orders()

                # Find market making opportunities
                call_mm_opps = self.find_market_making_opportunities(call_book)
                put_mm_opps = self.find_market_making_opportunities(put_book)

                # Add option type to opportunities
                for opp in call_mm_opps:
                    opp['option_type'] = 'CALL'
                for opp in put_mm_opps:
                    opp['option_type'] = 'PUT'

                # Get directional opportunities with dynamic thresholds
                directional_opps = self.get_directional_opportunities(
                    momentum, direction, call_book, put_book, dynamic_thresholds
                )

                # Combine all opportunities
                all_opportunities = call_mm_opps + put_mm_opps + directional_opps

                # Execute opportunities (limit to prevent over-trading)
                executed_this_cycle = 0
                max_orders_per_cycle = 2

                print(f"\n🎯 EXECUTING OPPORTUNITIES: {len(all_opportunities)} found")

                for i, opp in enumerate(all_opportunities):
                    if executed_this_cycle >= max_orders_per_cycle:
                        print(f"⏸️ Max orders per cycle reached ({max_orders_per_cycle})")
                        break

                    print(f"📋 Opportunity {i+1}: {opp.get('option_type', 'unknown')} {opp['side']} @ ${opp['price']:.4f} [{opp['strategy']}]")

                    order_id = self.place_order(opp)
                    if order_id:
                        executed_this_cycle += 1
                        print(f"✅ Order {executed_this_cycle} placed successfully: {order_id}")
                        time.sleep(0.5)  # Brief pause between orders
                    else:
                        print(f"❌ Order {i+1} failed to place")

                if len(all_opportunities) == 0:
                    print("⚠️ No opportunities found this cycle")
                elif executed_this_cycle == 0:
                    print("❌ Opportunities found but no orders placed - check logs above")

                # Print status with dynamic volatility info
                current_time = time.time()
                if current_time - last_status_time > 5:  # Every 5 seconds
                    self.print_status(momentum, direction, dynamic_thresholds)
                    last_status_time = current_time

                time.sleep(1)  # 1 second loop

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down Market Maker...")
            self.cancel_all_orders("shutdown")
            self._print_final_report()

    def _print_final_report(self):
        """Print final performance report"""
        print("\n" + "="*80)
        print("🏪 MARKET MAKER FINAL REPORT")
        print("="*80)

        print(f"📊 Orders Placed: {self.orders_placed}")
        print(f"✅ Orders Filled: {self.orders_filled}")
        print(f"📈 Fill Rate: {self.orders_filled/max(self.orders_placed, 1):.1%}")
        print(f"💰 Total PnL: ${self.total_pnl:+.2f}")
        print(f"📋 Orders in History: {len(self.order_history)}")

        if self.active_orders:
            print(f"⚠️ {len(self.active_orders)} orders still active (cancelled)")

        print("="*80)
        print("✅ Market Maker Shutdown Complete")

def main():
    """Main function"""

    config = MarketMakerConfig(
        clob_api_url=os.getenv("CLOB_API_URL", "https://clob.polymarket.com"),
        private_key=os.getenv("PK", ""),
        api_key=os.getenv("CLOB_API_KEY", ""),
        api_secret=os.getenv("CLOB_SECRET", ""),
        api_passphrase=os.getenv("CLOB_PASS_PHRASE", ""),

        # Market making parameters
        min_book_quantity=1111,                    # Your requirement
        max_position_size_usd=25.0,                # Conservative position size
        max_total_exposure_usd=100.0,              # Total exposure limit

        # Dynamic volatility-based thresholds
        volatility_lookback_minutes=60,            # 1 hour lookback for volatility
        directional_volatility_multiplier=1.5,    # 1.5x current volatility for directional bias
        strong_signal_volatility_multiplier=2.5,  # 2.5x for strong signals

        # Volatility cancellation (dynamic)
        volatility_cancel_threshold=0.70,          # 70% ML confidence
        spike_volatility_multiplier=3.0,           # 3x volatility for spike detection

        # Order management
        order_timeout_seconds=120,           # 2 minute timeout
        price_improvement_cents=0.001,       # 0.1 cent improvement
    )

    # Validate configuration
    if not all([config.clob_api_url, config.private_key, config.api_key,
                config.api_secret, config.api_passphrase]):
        print("❌ Missing API credentials")
        return

    print("🏪 Starting Market Maker Trading System")
    print(f"📊 Strategy: Market Making + Directional Trading")
    print(f"📋 Min Book Quantity: {config.min_book_quantity}")

    executor = MarketMakerExecutor(config)
    executor.run_market_making_loop()

if __name__ == "__main__":
    main()
