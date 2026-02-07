#!/usr/bin/env python3
"""
Strategy E - Trending vs Ranging Adaptive Strategy

Market Detection: Uses ADX, Slope, and Efficiency Ratio
- TRENDING: Use Strategy D2 (EMA Bands)
- RANGING: Use Strategy C (RSI Mean Reversion)
- NEUTRAL: Use Strategy C (conservative)
"""

import requests
import time
import json
import os
from datetime import datetime, timezone
from collections import deque
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

# ============================================================================
# MARKET REGIME DETECTOR
# ============================================================================

class MarketRegimeDetector:
    """
    Detect market regime (TRENDING/RANGING) using 1 hour of 1-minute candles.
    """

    def __init__(
        self,
        lookback: int = 60,  # 60 minutes = 1 hour
        adx_period: int = 14,
        trending_threshold: float = 2.0,
        ranging_threshold: float = -2.0,
        weights: Optional[Dict[str, float]] = None
    ):
        self.lookback = lookback
        self.adx_period = adx_period
        self.trending_threshold = trending_threshold
        self.ranging_threshold = ranging_threshold

        self.weights = weights or {
            'adx': 0.4,
            'slope': 0.3,
            'efficiency': 0.3
        }

        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def fetch_bybit_data(
        self,
        symbol: str = 'BTCUSDT',
        limit: int = 200
    ) -> pd.DataFrame:
        """Fetch 1-minute candle data from Bybit API."""
        url = "https://api.bybit.com/v5/market/mark-price-kline"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': '1',
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['retCode'] != 0:
                raise ValueError(f"Bybit API error: {data['retMsg']}")

            candles = data['result']['list']
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close'])

            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)

            # Bybit returns newest first, so reverse
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch Bybit data: {e}")

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        """Calculate Average True Range."""
        if period is None:
            period = self.adx_period

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        """Calculate Average Directional Index."""
        if period is None:
            period = self.adx_period

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx

    def _calculate_normalized_slope(self, close: pd.Series, atr: pd.Series, period: Optional[int] = None) -> pd.Series:
        """Calculate linear regression slope normalized by ATR."""
        if period is None:
            period = self.lookback

        slopes = []
        for i in range(len(close)):
            if i < period:
                slopes.append(np.nan)
            else:
                y = close.iloc[i-period:i].values
                x = np.arange(period)
                slope = np.polyfit(x, y, 1)[0]
                normalized_slope = abs(slope) / atr.iloc[i] if atr.iloc[i] > 0 else 0
                slopes.append(normalized_slope)

        return pd.Series(slopes, index=close.index)

    def _calculate_efficiency_ratio(self, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        """Calculate efficiency ratio (price change / path traveled)."""
        if period is None:
            period = self.lookback

        price_change = abs(close - close.shift(period))
        path_traveled = abs(close.diff()).rolling(period).sum()
        efficiency = price_change / path_traveled
        efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        return efficiency

    def _calculate_scores(self, adx: pd.Series, norm_slope: pd.Series, efficiency: pd.Series) -> pd.Series:
        """Calculate composite trend score."""
        adx_score = (adx - 20) / 5
        slope_score = norm_slope * 10
        efficiency_score = (efficiency - 0.3) / 0.2 * 10

        trend_score = (
            self.weights['adx'] * adx_score +
            self.weights['slope'] * slope_score +
            self.weights['efficiency'] * efficiency_score
        )
        return trend_score

    def analyze(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze market regime for the given dataframe."""
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']

        atr = self._calculate_atr(high, low, close)
        adx = self._calculate_adx(high, low, close)
        norm_slope = self._calculate_normalized_slope(close, atr)
        efficiency = self._calculate_efficiency_ratio(close)
        trend_score = self._calculate_scores(adx, norm_slope, efficiency)

        regime = pd.Series('NEUTRAL', index=df.index)
        regime[trend_score > self.trending_threshold] = 'TRENDING'
        regime[trend_score < self.ranging_threshold] = 'RANGING'

        return {
            'regime': regime,
            'trend_score': trend_score,
            'adx': adx,
            'norm_slope': norm_slope,
            'efficiency': efficiency,
            'atr': atr
        }

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """Get current market regime and detailed metrics."""
        result = self.analyze(df)

        metrics = {
            'adx': result['adx'].iloc[-1],
            'norm_slope': result['norm_slope'].iloc[-1],
            'efficiency': result['efficiency'].iloc[-1],
            'atr': result['atr'].iloc[-1]
        }

        return result['regime'].iloc[-1], result['trend_score'].iloc[-1], metrics

    def analyze_from_bybit(self, symbol: str = 'BTCUSDT') -> Dict:
        """Fetch data from Bybit and analyze current regime."""
        df = self.fetch_bybit_data(symbol)
        regime, score, metrics = self.get_current_regime(df)

        return {
            'symbol': symbol,
            'regime': regime,
            'trend_score': score,
            'timestamp': df['timestamp'].iloc[-1],
            'current_price': df['close'].iloc[-1],
            'metrics': metrics
        }

# ============================================================================
# CONFIGURATION
# ============================================================================

# Market Detection (using MarketRegimeDetector)
REGIME_UPDATE_INTERVAL = 60  # Update regime every 60 seconds
REGIME_LOOKBACK = 60  # Use 60 minutes of data
REGIME_ADX_PERIOD = 14
REGIME_TRENDING_THRESHOLD = 0.5   # Lower threshold = more TRENDING signals
REGIME_RANGING_THRESHOLD = -0.5   # Higher threshold = more RANGING signals
# NEUTRAL only occurs when score is between -0.5 and 0.5 (very narrow band)

# API Configuration
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"

# Loop intervals
CHECK_INTERVAL = 0.1  # 100ms - trading loop
SAMPLE_INTERVAL = 1.0  # 1 second - indicator sampling

# Trading Parameters
MIN_BUY_PRICE = 0.03
MAX_BUY_PRICE = 0.97
CONFIRMATION_SECONDS = 1
START_DELAY = 20  # Wait 20s from period start

# RSI/EMA Parameters (Shared between D2 and C)
RSI_PERIOD_INITIAL = 60
RSI_PERIOD_MIN = 15
RSI_PERIOD_MAX = 120
RSI_PERIOD_ADJUSTMENT = 5
BREACH_WINDOW = 900  # 15 minutes

# RSI Thresholds
RSI_BREACH_UPPER = 80
RSI_BREACH_LOWER = 20

# Strategy D2 (Trending)
D2_RSI_EXIT_CALL = 90
D2_RSI_EXIT_PUT = 10
D2_MIN_TOLERANCE = 5.0

# Strategy C (Ranging)
C_RSI_BUY_CALL = 20   # Buy CALL when RSI < 20
C_RSI_SELL_CALL = 80  # Sell CALL when RSI > 80
C_RSI_BUY_PUT = 80    # Buy PUT when RSI > 80
C_RSI_SELL_PUT = 20   # Sell PUT when RSI < 20

# Persistence
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_e_state.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot017_RSI_trades"
STATE_SAVE_INTERVAL = 60

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_btc_price():
    """Get current BTC price from Bybit"""
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data['result']['list'][0]['lastPrice'])
    except Exception as e:
        return None

def get_strike_price() -> Optional[float]:
    """Get the strike price for the current 15-minute period"""
    try:
        now = datetime.now(timezone.utc)
        for start_min in [0, 15, 30, 45]:
            if now.minute >= start_min and now.minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                url = "https://api.bybit.com/v5/market/mark-price-kline"
                params = {
                    'category': 'linear',
                    'symbol': 'BTCUSDT',
                    'interval': '15',
                    'start': int(period_start.timestamp() * 1000),
                    'limit': 1
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        kline_list = data.get('result', {}).get('list', [])
                        if kline_list:
                            return float(kline_list[0][1])
        return None
    except Exception as e:
        return None

def get_option_prices():
    """Get current option prices from local JSON files"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(CALL_FILE, 'r') as f:
                content = f.read()
                if not content.strip():
                    if attempt < max_retries - 1:
                        time.sleep(0.01)
                        continue
                    return None, None, None, None
                call_data = json.loads(content)
            
            with open(PUT_FILE, 'r') as f:
                content = f.read()
                if not content.strip():
                    if attempt < max_retries - 1:
                        time.sleep(0.01)
                        continue
                    return None, None, None, None
                put_data = json.loads(content)
            
            call_best_bid = call_data.get('best_bid')
            call_best_ask = call_data.get('best_ask')
            call_bid = float(call_best_bid['price']) if call_best_bid else None
            call_ask = float(call_best_ask['price']) if call_best_ask else None
            
            put_best_bid = put_data.get('best_bid')
            put_best_ask = put_data.get('best_ask')
            put_bid = float(put_best_bid['price']) if put_best_bid else None
            put_ask = float(put_best_ask['price']) if put_best_ask else None
            
            return call_ask, call_bid, put_ask, put_bid
            
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(0.01)
                continue
            return None, None, None, None
        except Exception as e:
            return None, None, None, None
    
    return None, None, None, None

def calculate_rsi(prices: deque, period: int) -> Optional[float]:
    """Calculate RSI using proper Wilder's smoothing method"""
    if len(prices) < period + 1:
        return None
    
    # Get last period+1 prices to calculate period deltas
    prices_array = np.array(list(prices)[-(period + 1):])
    deltas = np.diff(prices_array)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # First average: Simple Moving Average of first 'period' values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Wilder's smoothing for remaining values (if any)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices: deque, period: int) -> Optional[float]:
    """Calculate EMA on 1-second samples"""
    if len(prices) < period:
        return None
    
    prices_array = np.array(list(prices))[-period:]
    k = 2.0 / (period + 1)
    ema = prices_array[0]
    
    for price in prices_array[1:]:
        ema = price * k + ema * (1 - k)
    
    return ema

def get_bin_key(timestamp):
    """Get 15-minute bin key"""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    minute_bin = (dt.minute // 15) * 15
    return dt.replace(minute=minute_bin, second=0, microsecond=0).isoformat()

def get_seconds_into_period(timestamp):
    """Get seconds elapsed in current 15-minute period"""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    minute_in_bin = dt.minute % 15
    return minute_in_bin * 60 + dt.second

def get_seconds_remaining(timestamp):
    """Get seconds remaining in current 15-minute period"""
    return 900 - get_seconds_into_period(timestamp)

# ============================================================================
# PERSISTENCE
# ============================================================================

def load_state():
    """Load bot state"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                state['price_history_1s'] = deque(state.get('price_history_1s', []), maxlen=2000)
                print(f"[STATE] Loaded: RSI period={state.get('rsi_period', RSI_PERIOD_INITIAL)}s, "
                      f"Price history={len(state['price_history_1s'])} samples")
                return state
        except Exception as e:
            print(f"[ERROR] Failed to load state: {e}")
    
    print("[STATE] Starting fresh")
    return {
        'rsi_period': RSI_PERIOD_INITIAL,
        'price_history_1s': deque(maxlen=2000),
        'total_breaches': 0,
        'last_adjustment_time': 0
    }

def save_state(state):
    """Save bot state"""
    try:
        state_copy = state.copy()
        state_copy['price_history_1s'] = list(state['price_history_1s'])
        
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state_copy, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save state: {e}")

def load_trades():
    """Load trade history"""
    os.makedirs(TRADES_DIR, exist_ok=True)
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    trades_file = os.path.join(TRADES_DIR, f"trades_E_{today}.json")
    
    if os.path.exists(trades_file):
        try:
            with open(trades_file, 'r') as f:
                trades = json.load(f)
                print(f"[TRADES] Loaded {len(trades.get('trades', []))} trades, PNL: ${trades.get('total_pnl', 0):.2f}")
                return trades
        except Exception as e:
            print(f"[ERROR] Failed to load trades: {e}")
    
    print(f"[TRADES] Starting fresh for {today}")
    return {
        'trades': [],
        'total_pnl': 0.0,
        'stats': {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'call_trades': 0,
            'put_trades': 0,
            'd2_opens': 0,
            'c_opens': 0,
            'd2_closes': 0,
            'c_closes': 0
        }
    }

def save_trade(trades, trade_record):
    """Save a completed trade"""
    try:
        trades['trades'].append(trade_record)
        trades['total_pnl'] += trade_record['pnl']
        trades['stats']['total_trades'] += 1
        
        if trade_record['pnl'] > 0:
            trades['stats']['winning_trades'] += 1
        else:
            trades['stats']['losing_trades'] += 1
        
        if trade_record['type'] == 'CALL':
            trades['stats']['call_trades'] += 1
        else:
            trades['stats']['put_trades'] += 1
        
        # Track strategy usage
        if 'D2' in trade_record['entry_strategy']:
            trades['stats']['d2_opens'] += 1
        else:
            trades['stats']['c_opens'] += 1
        
        if 'D2' in trade_record['exit_reason']:
            trades['stats']['d2_closes'] += 1
        elif 'C' in trade_record['exit_reason']:
            trades['stats']['c_closes'] += 1
        
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        trades_file = os.path.join(TRADES_DIR, f"trades_E_{today}.json")
        
        with open(trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
        
        print(f"[TRADE] {trade_record['type']} | Entry: {trade_record['entry_strategy']} | "
              f"Exit: {trade_record['exit_reason']} | PNL: ${trade_record['pnl']:+.3f} | "
              f"Total: ${trades['total_pnl']:+.2f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save trade: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("STRATEGY E - TRENDING vs RANGING ADAPTIVE STRATEGY")
    print("=" * 80)
    print(f"Market Detection: ADX + Slope + Efficiency ({REGIME_LOOKBACK}-minute lookback)")
    print(f"  Score > {REGIME_TRENDING_THRESHOLD} → TRENDING (Strategy D2)")
    print(f"  Score < {REGIME_RANGING_THRESHOLD} → RANGING (Strategy C)")
    print(f"  Between → NEUTRAL (Strategy C conservative)")
    print(f"RSI Period: {RSI_PERIOD_INITIAL}s (adaptive {RSI_PERIOD_MIN}-{RSI_PERIOD_MAX}s)")
    print("=" * 80)
    
    # Load state and trades
    state = load_state()
    trades = load_trades()
    
    # State
    price_history_1s = state['price_history_1s']
    rsi_period = state['rsi_period']
    total_breaches = state['total_breaches']
    last_adjustment_time = state.get('last_adjustment_time', 0)
    
    # Initialize Market Regime Detector
    regime_detector = MarketRegimeDetector(
        lookback=REGIME_LOOKBACK,
        adx_period=REGIME_ADX_PERIOD,
        trending_threshold=REGIME_TRENDING_THRESHOLD,
        ranging_threshold=REGIME_RANGING_THRESHOLD
    )
    
    # Regime state
    last_regime_update = 0
    market_regime = 'UNKNOWN'
    trend_score = 0.0
    regime_metrics = {}
    
    print(f"\n[INIT] Market Regime Detector initialized")
    print(f"[INIT] Will fetch {REGIME_LOOKBACK} minutes of data every {REGIME_UPDATE_INTERVAL}s\n")
    
    # Trading state
    position = None
    last_signal_type = None
    last_signal_time = 0
    current_bin = None
    strike_price = None
    
    # Indicator values (updated every 1 second)
    rsi = None
    ema = None
    volatility = 0.0
    market_regime = 'UNKNOWN'
    slope = None
    slope_ratio = 0.0
    
    # Timing
    last_sample_time = 0
    last_state_save = time.time()
    
    print(f"\n[STARTING] RSI Period: {rsi_period}s | History: {len(price_history_1s)} samples\n")
    
    while True:
        try:
            current_time = time.time()
            timestamp = current_time
            
            # Get BTC price
            btc_price = get_btc_price()
            if btc_price is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Sample price every 1 second
            if current_time - last_sample_time >= SAMPLE_INTERVAL:
                price_history_1s.append(btc_price)
                last_sample_time = current_time
                
                # Update market regime every 60 seconds
                if current_time - last_regime_update >= REGIME_UPDATE_INTERVAL:
                    try:
                        result = regime_detector.analyze_from_bybit('BTCUSDT')
                        market_regime = result['regime']
                        trend_score = result['trend_score']
                        regime_metrics = result['metrics']
                        last_regime_update = current_time
                        
                        print(f"\n[REGIME] {market_regime} | Score: {trend_score:.2f}")
                        print(f"  ADX: {regime_metrics['adx']:.1f} | Efficiency: {regime_metrics['efficiency']:.3f} | "
                              f"Norm Slope: {regime_metrics['norm_slope']:.3f} | ATR: ${regime_metrics['atr']:.2f}\n")
                    except Exception as e:
                        print(f"[ERROR] Regime detection failed: {e}")
                        market_regime = 'UNKNOWN'
                
                # Calculate RSI and EMA from 1-second data
                rsi = calculate_rsi(price_history_1s, rsi_period)
                ema = calculate_ema(price_history_1s, rsi_period)
                
                # Debug log
                if position:
                    print(f"[CHECK] RSI{rsi_period}s={rsi:.1f} | {market_regime} (score={trend_score:.2f}) | "
                          f"Pos: {position['type']}@${position['entry_price']:.2f} ({position['entry_strategy']})")
                
                # Count breaches and adjust period
                if len(price_history_1s) >= rsi_period + 1:
                    breach_window_samples = min(BREACH_WINDOW, len(price_history_1s))
                    recent_prices = list(price_history_1s)[-breach_window_samples:]
                    
                    breach_count = 0
                    in_breach = False
                    
                    num_windows = len(recent_prices) - rsi_period
                    
                    if num_windows > 0:
                        for i in range(num_windows):
                            window = recent_prices[i:i + rsi_period + 1]
                            window_rsi = calculate_rsi(deque(window), rsi_period)
                            
                            if window_rsi is not None:
                                is_outside = window_rsi > RSI_BREACH_UPPER or window_rsi < RSI_BREACH_LOWER
                                
                                if is_outside and not in_breach:
                                    breach_count += 1
                                    in_breach = True
                                elif not is_outside and in_breach:
                                    in_breach = False
                    
                    # Adjust period every 10 seconds
                    if current_time - last_adjustment_time >= 10:
                        old_period = rsi_period
                        
                        if breach_count <= 3 and rsi_period > RSI_PERIOD_MIN:
                            rsi_period = max(RSI_PERIOD_MIN, rsi_period - RSI_PERIOD_ADJUSTMENT)
                            print(f"[RSI ADJUST] Breaches: {breach_count} ≤ 3 → {old_period}s → {rsi_period}s")
                        elif breach_count >= 8 and rsi_period < RSI_PERIOD_MAX:
                            rsi_period = min(RSI_PERIOD_MAX, rsi_period + RSI_PERIOD_ADJUSTMENT)
                            print(f"[RSI ADJUST] Breaches: {breach_count} ≥ 8 → {old_period}s → {rsi_period}s")
                        
                        last_adjustment_time = current_time
                        total_breaches = breach_count
            
            # Get option prices
            call_ask, call_bid, put_ask, put_bid = get_option_prices()
            
            # Check prices available
            prices_available = True
            if position:
                if position['type'] == 'CALL' and call_bid is None:
                    prices_available = False
                elif position['type'] == 'PUT' and put_bid is None:
                    prices_available = False
            else:
                if call_ask is None or put_ask is None:
                    prices_available = False
            
            if not prices_available:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Time info
            bin_key = get_bin_key(timestamp)
            seconds_into_period = get_seconds_into_period(timestamp)
            seconds_remaining = get_seconds_remaining(timestamp)
            
            # New period handling
            if current_bin != bin_key:
                # Handle expiration
                if position:
                    next_period_strike = get_strike_price()
                    if next_period_strike is None:
                        next_period_strike = btc_price
                    
                    position_strike = position.get('strike_price')
                    final_btc = next_period_strike
                    
                    if position['type'] == 'CALL':
                        final_value = 1.0 if final_btc > position_strike else 0.0
                        result = "ITM" if final_btc > position_strike else "OTM"
                    else:
                        final_value = 1.0 if final_btc <= position_strike else 0.0
                        result = "ITM" if final_btc <= position_strike else "OTM"
                    
                    pnl = final_value - position['entry_price']
                    
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': final_value,
                        'pnl': pnl,
                        'strike_price': position_strike,
                        'final_btc': final_btc,
                        'btc_at_entry': position.get('btc_at_entry'),
                        'entry_strategy': position.get('entry_strategy', 'UNKNOWN'),
                        'exit_reason': 'EXPIRATION',
                        'result': result,
                        'max_profit': position.get('max_profit', 0.0),
                        'max_loss': position.get('max_loss', 0.0),
                        'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                        'time_to_expiry_at_exit': 0,
                        'volatility_at_entry': position.get('volatility', 0.0),
                        'volatility_at_exit': volatility,
                        'market_regime_at_entry': position.get('market_regime', 'UNKNOWN'),
                        'market_regime_at_exit': market_regime
                    }
                    save_trade(trades, trade_record)
                    print(f"\n[EXPIRATION] {position['type']} {result} | Strike: ${position_strike:.2f} | "
                          f"Final: ${final_btc:.2f} | PNL: ${pnl:+.3f}\n")
                    position = None
                
                # Reset
                current_bin = bin_key
                strike_price = get_strike_price()
                if strike_price:
                    print(f"\n{'='*80}")
                    print(f"[NEW PERIOD] {bin_key}")
                    print(f"[STRIKE] ${strike_price:.2f}")
                    print(f"{'='*80}\n")
            
            # Calculate tolerance (for D2)
            tolerance = max(D2_MIN_TOLERANCE, volatility / 8) if volatility > 0 else D2_MIN_TOLERANCE
            
            # Calculate bands (for D2)
            if ema is not None:
                upper_band = ema + tolerance
                lower_band = ema - tolerance
            else:
                upper_band = lower_band = None
            
            # Display status every 10 seconds
            if int(current_time) % 10 == 0:
                pos_str = f"{position['type']}@${position['entry_price']:.2f}" if position else "NONE"
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                
                if market_regime == 'UNKNOWN':
                    print(f"[E {datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%H:%M:%S')}] "
                          f"BTC:${btc_price:.2f} | INITIALIZING REGIME | "
                          f"RSI{rsi_period}s={rsi_str} | Pos:{pos_str} | {seconds_remaining}s")
                else:
                    print(f"[E {datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%H:%M:%S')}] "
                          f"BTC:${btc_price:.2f} | {market_regime} (score={trend_score:.2f}) | "
                          f"RSI{rsi_period}s={rsi_str} | Pos:{pos_str} | {seconds_remaining}s")
            
            # Skip trading until ready
            if rsi is None or ema is None or seconds_into_period < START_DELAY or strike_price is None or market_regime == 'UNKNOWN':
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Track max profit/loss
            if position:
                if position['type'] == 'CALL' and call_bid is not None:
                    current_pnl = call_bid - position['entry_price']
                    position['max_profit'] = max(position['max_profit'], current_pnl)
                    position['max_loss'] = min(position['max_loss'], current_pnl)
                elif position['type'] == 'PUT' and put_bid is not None:
                    current_pnl = put_bid - position['entry_price']
                    position['max_profit'] = max(position['max_profit'], current_pnl)
                    position['max_loss'] = min(position['max_loss'], current_pnl)
            
            # ========== STRATEGY E TRADING LOGIC ==========
            signal = None
            strategy = None
            
            # CLOSE CONDITIONS (Priority 1)
            if position:
                # D2 Close conditions
                if position['type'] == 'CALL' and rsi > D2_RSI_EXIT_CALL:
                    signal = 'CLOSE_CALL'
                    strategy = 'D2_RSI_EXIT'
                elif position['type'] == 'PUT' and rsi < D2_RSI_EXIT_PUT:
                    signal = 'CLOSE_PUT'
                    strategy = 'D2_RSI_EXIT'
                
                # C Close conditions (can close D2-opened positions)
                elif position['type'] == 'CALL' and rsi > C_RSI_SELL_CALL:
                    signal = 'CLOSE_CALL'
                    strategy = 'C_RSI_EXIT'
                elif position['type'] == 'PUT' and rsi < C_RSI_SELL_PUT:
                    signal = 'CLOSE_PUT'
                    strategy = 'C_RSI_EXIT'
            
            # OPEN CONDITIONS (Priority 2)
            elif not position:
                if market_regime == 'TRENDING':
                    # Use Strategy D2 for trending markets
                    if btc_price > upper_band and rsi < D2_RSI_EXIT_CALL:
                        signal = 'OPEN_CALL'
                        strategy = 'D2_TRENDING'
                    elif btc_price < lower_band and rsi > D2_RSI_EXIT_PUT:
                        signal = 'OPEN_PUT'
                        strategy = 'D2_TRENDING'
                
                elif market_regime in ['RANGING', 'NEUTRAL']:
                    # Use Strategy C for ranging/neutral markets
                    if rsi < C_RSI_BUY_CALL:
                        signal = 'OPEN_CALL'
                        strategy = 'C_RANGING'
                    elif rsi > C_RSI_BUY_PUT:
                        signal = 'OPEN_PUT'
                        strategy = 'C_RANGING'
            
            # Execute signals with confirmation
            if signal:
                if last_signal_type != signal:
                    last_signal_time = current_time
                    last_signal_type = signal
                elif current_time - last_signal_time >= CONFIRMATION_SECONDS:
                    
                    if signal == 'OPEN_CALL':
                        if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position = {
                                'type': 'CALL',
                                'entry_price': call_ask,
                                'entry_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                                'strike_price': strike_price,
                                'btc_at_entry': btc_price,
                                'rsi_at_entry': rsi,
                                'ema_at_entry': ema,
                                'tolerance': tolerance,
                                'volatility': volatility,
                                'rsi_period': rsi_period,
                                'max_profit': 0.0,
                                'max_loss': 0.0,
                                'time_to_expiry_at_entry': seconds_remaining,
                                'entry_strategy': strategy,
                                'market_regime': market_regime
                            }
                            print(f"\n✅ OPENED CALL @${call_ask:.2f} | {strategy} | TTL: {seconds_remaining}s\n")
                        last_signal_type = None
                    
                    elif signal == 'OPEN_PUT':
                        if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position = {
                                'type': 'PUT',
                                'entry_price': put_ask,
                                'entry_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                                'strike_price': strike_price,
                                'btc_at_entry': btc_price,
                                'rsi_at_entry': rsi,
                                'ema_at_entry': ema,
                                'tolerance': tolerance,
                                'volatility': volatility,
                                'rsi_period': rsi_period,
                                'max_profit': 0.0,
                                'max_loss': 0.0,
                                'time_to_expiry_at_entry': seconds_remaining,
                                'entry_strategy': strategy,
                                'market_regime': market_regime
                            }
                            print(f"\n✅ OPENED PUT @${put_ask:.2f} | {strategy} | TTL: {seconds_remaining}s\n")
                        last_signal_type = None
                    
                    elif signal == 'CLOSE_CALL':
                        pnl = call_bid - position['entry_price']
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                            'type': 'CALL',
                            'entry_price': position['entry_price'],
                            'exit_price': call_bid,
                            'pnl': pnl,
                            'strike_price': position.get('strike_price'),
                            'btc_at_entry': position.get('btc_at_entry'),
                            'entry_strategy': position.get('entry_strategy', 'UNKNOWN'),
                            'exit_reason': strategy,
                            'max_profit': position.get('max_profit', 0.0),
                            'max_loss': position.get('max_loss', 0.0),
                            'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                            'time_to_expiry_at_exit': seconds_remaining,
                            'volatility_at_entry': position.get('volatility', 0.0),
                            'volatility_at_exit': volatility,
                            'market_regime_at_entry': position.get('market_regime', 'UNKNOWN'),
                            'market_regime_at_exit': market_regime
                        }
                        save_trade(trades, trade_record)
                        print(f"\n✅ CLOSED CALL @${call_bid:.2f} | {strategy} | PNL: ${pnl:+.3f}\n")
                        position = None
                        last_signal_type = None
                        time.sleep(1)
                    
                    elif signal == 'CLOSE_PUT':
                        pnl = put_bid - position['entry_price']
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                            'type': 'PUT',
                            'entry_price': position['entry_price'],
                            'exit_price': put_bid,
                            'pnl': pnl,
                            'strike_price': position.get('strike_price'),
                            'btc_at_entry': position.get('btc_at_entry'),
                            'entry_strategy': position.get('entry_strategy', 'UNKNOWN'),
                            'exit_reason': strategy,
                            'max_profit': position.get('max_profit', 0.0),
                            'max_loss': position.get('max_loss', 0.0),
                            'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                            'time_to_expiry_at_exit': seconds_remaining,
                            'volatility_at_entry': position.get('volatility', 0.0),
                            'volatility_at_exit': volatility,
                            'market_regime_at_entry': position.get('market_regime', 'UNKNOWN'),
                            'market_regime_at_exit': market_regime
                        }
                        save_trade(trades, trade_record)
                        print(f"\n✅ CLOSED PUT @${put_bid:.2f} | {strategy} | PNL: ${pnl:+.3f}\n")
                        position = None
                        last_signal_type = None
                        time.sleep(1)
            else:
                last_signal_type = None
            
            # Save state periodically
            if current_time - last_state_save >= STATE_SAVE_INTERVAL:
                state['rsi_period'] = rsi_period
                state['price_history_1s'] = price_history_1s
                state['total_breaches'] = total_breaches
                state['last_adjustment_time'] = last_adjustment_time
                save_state(state)
                last_state_save = current_time
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Saving state...")
            state['rsi_period'] = rsi_period
            state['price_history_1s'] = price_history_1s
            state['total_breaches'] = total_breaches
            state['last_adjustment_time'] = last_adjustment_time
            save_state(state)
            print("[SHUTDOWN] Complete!")
            break
        
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    main()
