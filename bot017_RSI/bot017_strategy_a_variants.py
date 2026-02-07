#!/usr/bin/env python3
"""
RSI Trading Bot - STRATEGIES A, C, C3, C4, C5, D, D2

STRATEGY A: Adaptive Threshold (Original)
- Starting: 70/30, Adjustment: ±5, Triggers: >5/<3

STRATEGY C: Adaptive RSI Period
- Fixed thresholds: 80/20
- Adaptive RSI period: starts at 30s
- 0-2 breaches → decrease 5s, 3-5 no change, 6+ increase 5s
- Min period: 15s, Max period: 90s

STRATEGY C3: Adaptive RSI Period with ATR-based TP/SL
- Fixed thresholds: 70/30 (same as C2)
- Adaptive RSI period: starts at 30s
- 0-3 breaches → decrease 5s, 4-7 no change, 8+ increase 5s
- ATR-based dynamic TP/SL (1:3 ratio)
- ATR: 6 periods × 30 seconds from option prices
- TP/SL adapt to market volatility every loop
- Min period: 15s, Max period: 90s

STRATEGY C4: Adaptive RSI Period with Dual Thresholds
- Open thresholds: 80/15 (wider than C)
- Close thresholds: 70/30 (tighter exit)
- Adaptive RSI period: starts at 30s
- 0-3 breaches → decrease 5s, 4-7 no change, 8+ increase 5s
- Min period: 15s, NO max limit
- No TP/SL (holds until signal or expiration)

STRATEGY C5: Price-Adaptive Dynamic Thresholds
- Breach tracking: 80/20 (for period adjustment only)
- Dynamic open/close thresholds based on option price:
  * Extra-countertrend: 0.01-0.15 (PUT: 94/>60, CALL: <6/40)
  * Countertrend: 0.15-0.40 (PUT: 90/>40, CALL: <10/60)
  * Neutral: 0.40-0.60 (PUT: 80/>20, CALL: <20/80)
  * Trend: 0.60-0.80 (PUT: 70/>20, CALL: <30/80)
  * Extra-trend: 0.80-1.00 (PUT: 65/>20, CALL: <35/80)
- Adaptive RSI period: starts at 30s
- 0-3 breaches → decrease 5s, 4-7 no change, 8+ increase 5s
- Min period: 15s, NO max limit
- Trend label exported in trades

STRATEGY D: EMA Crossover with Adaptive Period
- Breach tracking: 80/20 (for period adjustment - uses RSI)
- EMA period = RSI period (adaptive, starts at 30s)
- CALL Entry: BTC crosses UP through EMA by $5+
- CALL Exit: RSI > 90 OR BTC crosses DOWN through EMA
- PUT Entry: BTC crosses DOWN through EMA by $5+
- PUT Exit: RSI < 10 OR BTC crosses UP through EMA by $5+
- Adaptive period: 0-3 breaches decrease, 4-7 hold, 8+ increase
- Min period: 15s, NO max limit

STATE PERSISTENCE:
- All thresholds, RSI periods saved to bot_state.json
- Restored on bot restart (hourly RAM management)

All strategies:
- Binary options expire at 00, 15, 30, 45 minutes
- Price limits: $0.03 - $0.97
- Time window: 20s - 880s into period


pm2 start bot_d2_only.py --cron-restart="00 * * * *" --interpreter python3
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import deque
import numpy as np


class OptionATRCalculator:
    """
    Calculate ATR based on CALL and PUT option price ranges
    - Tracks max-min range for 30-second periods
    - Averages last 6 periods (3 minutes of data)
    - Uses option prices, not BTC price
    """

    def __init__(self, period_seconds: int = 30, num_periods: int = 6):
        """
        Args:
            period_seconds: Length of each period (default 30s)
            num_periods: Number of periods to average (default 6 = 3 minutes)
        """
        self.period_seconds = period_seconds
        self.num_periods = num_periods

        # Store CALL and PUT prices with timestamps
        self.call_prices = deque(maxlen=200)  # Last ~3 min at 1Hz
        self.put_prices = deque(maxlen=200)
        self.call_timestamps = deque(maxlen=200)
        self.put_timestamps = deque(maxlen=200)

    def update(self, call_price: float, put_price: float):
        """Add new price observation"""
        now = datetime.now()
        self.call_prices.append(call_price)
        self.put_prices.append(put_price)
        self.call_timestamps.append(now)
        self.put_timestamps.append(now)

    def calculate_period_ranges(self, prices: deque, timestamps: deque) -> List[float]:
        """
        Calculate (max - min) for each 30-second period
        Returns list of ranges
        """
        if len(prices) < 10:  # Need at least 10 samples
            return []

        periods = []
        prices_list = list(prices)
        timestamps_list = list(timestamps)

        # Work backwards from most recent
        current_end = len(prices_list) - 1

        while current_end >= 0:
            # Find start of this period (30 seconds back)
            period_start = current_end
            target_time = timestamps_list[current_end] - timedelta(seconds=self.period_seconds)

            while period_start > 0 and timestamps_list[period_start] > target_time:
                period_start -= 1

            # Get prices in this period
            period_prices = prices_list[period_start:current_end + 1]

            if len(period_prices) >= 3:  # Need at least 3 samples
                price_range = max(period_prices) - min(period_prices)
                periods.append(price_range)

            # Move to next period
            current_end = period_start - 1

            # Stop after collecting enough periods
            if len(periods) >= self.num_periods:
                break

        return periods

    def calculate_atr(self) -> Optional[float]:
        """
        Calculate ATR as average of (max-min) ranges
        Averages both CALL and PUT ATRs
        """
        # Need enough data
        if len(self.call_prices) < 30 or len(self.put_prices) < 30:
            return None

        # Calculate ranges for CALL
        call_ranges = self.calculate_period_ranges(self.call_prices, self.call_timestamps)

        # Calculate ranges for PUT
        put_ranges = self.calculate_period_ranges(self.put_prices, self.put_timestamps)

        if not call_ranges or not put_ranges:
            return None

        # Average the ranges
        call_atr = sum(call_ranges) / len(call_ranges)
        put_atr = sum(put_ranges) / len(put_ranges)

        # Average CALL and PUT ATRs
        combined_atr = (call_atr + put_atr) / 2.0

        return combined_atr


# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot017_RSI_trades"
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_state.json"  # Persistent state

# Trading parameters
RSI_PERIOD = 300  # 30 seconds at 0.1s intervals
CONFIRMATION_SECONDS = 2
CHECK_INTERVAL = 0.1
MAX_BUY_PRICE = 0.97
MIN_BUY_PRICE = 0.03
MIN_SECONDS_REMAINING = 20
MAX_SECONDS_INTO_PERIOD = 20

# Strategy A parameters
RSI_UPPER_INITIAL_A = 70
RSI_LOWER_INITIAL_A = 30
THRESHOLD_ADJUSTMENT_A = 5
THRESHOLD_HIGH_TRIGGER_A = 5
THRESHOLD_LOW_TRIGGER_A = 3
MAX_RSI_UPPER = 95
MIN_RSI_LOWER = 5

# Strategy C parameters (adaptive RSI period)
RSI_UPPER_C = 80  # Fixed upper threshold
RSI_LOWER_C = 20  # Fixed lower threshold
RSI_PERIOD_INITIAL_C = 30  # Starting period in seconds
RSI_PERIOD_ADJUSTMENT_C = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_C = 5  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_C = 3  # If total breaches < this, decrease period
RSI_PERIOD_MIN_C = 15  # Minimum period in seconds
RSI_PERIOD_MAX_C = 90  # Maximum period in seconds

# Strategy C3 parameters (adaptive RSI period with ATR-based TP/SL)
RSI_UPPER_C3 = 70  # Fixed upper threshold
RSI_LOWER_C3 = 30  # Fixed lower threshold
RSI_PERIOD_INITIAL_C3 = 30  # Starting period in seconds
RSI_PERIOD_ADJUSTMENT_C3 = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_C3 = 8  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_C3 = 4  # If total breaches < this, decrease period
RSI_PERIOD_MIN_C3 = 15  # Minimum period in seconds
RSI_PERIOD_MAX_C3 = 90  # Maximum period in seconds
# ATR-based TP/SL - ratio 1:3
ATR_PERIOD_SECONDS_C3 = 15  # ATR period in seconds
ATR_NUM_PERIODS_C3 = 6  # Number of periods for ATR calculation
ATR_BASE_TP_C3 = 0.015  # Base take profit distance
ATR_BASE_SL_C3 = 0.045  # Base stop loss distance (3x TP)
ATR_TP_SL_RATIO_C3 = 3.0  # SL = TP × 3

# Strategy C4 parameters (adaptive RSI period with dual thresholds)
RSI_UPPER_OPEN_C4 = 80  # Open position threshold (same as C)
RSI_LOWER_OPEN_C4 = 15  # Open position threshold (wider than C)
RSI_UPPER_CLOSE_C4 = 70  # Close position threshold (tighter)
RSI_LOWER_CLOSE_C4 = 30  # Close position threshold (tighter)
RSI_PERIOD_INITIAL_C4 = 30  # Starting period in seconds
RSI_PERIOD_ADJUSTMENT_C4 = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_C4 = 8  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_C4 = 4  # If total breaches < this, decrease period
RSI_PERIOD_MIN_C4 = 15  # Minimum period in seconds
# NO maximum period for C4

# Strategy C5 parameters (price-adaptive dynamic thresholds)
# Uses C4 breach tracking thresholds (80/20) for period adjustment
RSI_BREACH_UPPER_C5 = 80  # Breach tracking threshold
RSI_BREACH_LOWER_C5 = 20  # Breach tracking threshold
RSI_PERIOD_INITIAL_C5 = 30  # Starting period in seconds
RSI_PERIOD_ADJUSTMENT_C5 = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_C5 = 8  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_C5 = 4  # If total breaches < this, decrease period
RSI_PERIOD_MIN_C5 = 15  # Minimum period in seconds
# NO maximum period for C5

# C5 Price-based threshold ranges (PUT)
C5_PUT_RANGES = [
    # (price_min, price_max, open_threshold, close_threshold, trend_label)
    (0.01, 0.15, 94, 60, "extra-countertrend"),
    (0.15, 0.40, 90, 40, "countertrend"),
    (0.40, 0.60, 80, 20, "neutral"),
    (0.60, 0.80, 70, 20, "trend"),
    (0.80, 1.00, 65, 20, "extra-trend"),
]

# C5 Price-based threshold ranges (CALL)
C5_CALL_RANGES = [
    # (price_min, price_max, open_threshold, close_threshold, trend_label)
    (0.01, 0.15, 6, 40, "extra-countertrend"),
    (0.15, 0.40, 10, 60, "countertrend"),
    (0.40, 0.60, 20, 80, "neutral"),
    (0.60, 0.80, 30, 80, "trend"),
    (0.80, 1.00, 35, 80, "extra-trend"),
]

# Strategy D parameters (EMA crossover with adaptive period)
# Uses C5 breach tracking thresholds (80/20) for period adjustment
RSI_BREACH_UPPER_D = 80  # Breach tracking threshold
RSI_BREACH_LOWER_D = 20  # Breach tracking threshold
RSI_PERIOD_INITIAL_D = 30  # Starting RSI period in seconds (for breach tracking)
RSI_PERIOD_ADJUSTMENT_D = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_D = 8  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_D = 4  # If total breaches < this, decrease period
RSI_PERIOD_MIN_D = 15  # Minimum RSI period in seconds
# NO maximum period for D
# EMA period = RSI period (same adaptive period for both)
EMA_CROSSOVER_THRESHOLD_D = 5  # Minimum $5 crossover to trigger signal
RSI_EXIT_CALL_D = 90  # Close CALL when RSI > 90
RSI_EXIT_PUT_D = 10  # Close PUT when RSI < 10
# Strategy D2 parameters (EMA Bands - Simplified)
# Uses same RSI/EMA period as D (shared breach tracking)
MIN_TOLERANCE_D2 = 5  # Minimum $5 tolerance
RSI_EXIT_CALL_D2 = 90  # Close CALL when RSI > 90
RSI_EXIT_PUT_D2 = 10  # Close PUT when RSI < 10
D2_START_DELAY = 20  # Wait 20s from period start
# Shares rsi_period_d and ema_d with D


# Strategy D2 parameters (EMA mean reversion)
# Uses same RSI/EMA period as D (shared breach tracking)
RSI_BREACH_UPPER_D2 = 80  # Breach tracking threshold (shared with D)
RSI_BREACH_LOWER_D2 = 20  # Breach tracking threshold (shared with D)
EMA_CROSSING_TOLERANCE_D2 = 2  # ±$2 tolerance to detect crossing
MIN_MOVE_THRESHOLD_D2 = 5  # Minimum $5 move required
RSI_EXIT_CALL_D2 = 90  # Close CALL when RSI > 90
RSI_EXIT_PUT_D2 = 10  # Close PUT when RSI < 10
# Uses same adaptive period as D (rsi_period_d, ema_period_samples_d)


def read_json(filepath: str) -> Optional[dict]:
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def write_json(filepath: str, data: dict):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_bin_key(distance: float, seconds_to_expiry: float, volatility: float) -> str:
    distance_bins = [(0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
                     (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
                     (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
                     (1280, float('inf'), "1280+")]
    time_bins = [(13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
                 (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
                 (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
                 (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
                 (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
                 (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
                 (2, 5, "5s-2s"), (0, 2, "last-2s")]
    vol_bins = [(0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
                (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"), (120, 240, "120-240"),
                (240, float('inf'), "240+")]

    def get_bin_label(value, bins):
        for min_val, max_val, label in bins:
            if min_val <= value < max_val:
                return label
        return bins[-1][2]

    return f"{get_bin_label(distance, distance_bins)}|{get_bin_label(seconds_to_expiry, time_bins)}|{get_bin_label(volatility, vol_bins)}"


def get_strike_price() -> Optional[float]:
    try:
        import requests
        now = datetime.now(timezone.utc)
        for start_min in [0, 15, 30, 45]:
            if now.minute >= start_min and now.minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                url = "https://api.bybit.com/v5/market/mark-price-kline"
                params = {'category': 'linear', 'symbol': 'BTCUSDT', 'interval': '15',
                         'start': int(period_start.timestamp() * 1000), 'limit': 1}
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        kline_list = data.get('result', {}).get('list', [])
                        if kline_list:
                            return float(kline_list[0][1])
        return None
    except:
        return None


def get_seconds_to_expiry() -> float:
    now = datetime.now()
    for start_min in [0, 15, 30, 45]:
        if now.minute >= start_min and now.minute < start_min + 15:
            return 900 - ((now.minute - start_min) * 60 + now.second)
    return 0


def calculate_btc_volatility(price_history: deque) -> float:
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)
    return max(prices) - min(prices)


def calculate_rsi(prices: deque, period: int = RSI_PERIOD) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    price_array = np.array(list(prices))[-(period + 1):]
    deltas = np.diff(price_array)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain, avg_loss = np.mean(gains), np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema(prices: deque, period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average
    EMA = Price(t) * k + EMA(y) * (1 - k)
    where k = 2 / (N + 1)
    """
    if len(prices) < period:
        return None

    prices_array = np.array(list(prices))

    # Use SMA for first EMA value
    sma = np.mean(prices_array[:period])

    # Calculate multiplier
    k = 2.0 / (period + 1)

    # Calculate EMA
    ema = sma
    for price in prices_array[period:]:
        ema = price * k + ema * (1 - k)

    return ema


def calculate_rsi_slope(rsi_history: deque, period_seconds: int = 30) -> Optional[float]:
    """
    Calculate slope of RSI over specified period
    Args:
        rsi_history: deque of RSI values (sampled at 0.1s intervals)
        period_seconds: time period in seconds to calculate slope over
    """
    samples_needed = period_seconds * 10  # Convert seconds to samples (0.1s intervals)

    # If we don't have enough samples yet, use whatever we have (minimum 50 samples = 5s)
    available_samples = len(rsi_history)
    if available_samples < 50:  # Need at least 5s of data
        return None

    # Use requested period if available, otherwise use all available data
    samples_to_use = min(samples_needed, available_samples)
    recent_rsi = list(rsi_history)[-samples_to_use:]

    x = np.arange(len(recent_rsi))
    y = np.array(recent_rsi)
    slope = np.polyfit(x, y, 1)[0]  # Linear regression slope
    return slope


def save_bot_state(state: Dict):
    """Save bot state to persistent file"""
    write_json(STATE_FILE, state)


def load_bot_state() -> Optional[Dict]:
    """Load bot state from persistent file"""
    state = read_json(STATE_FILE)
    if state:
        print(f"\n📥 Loaded bot state from previous session:")
        print(f"   A:  Thresholds [{state['a_lower']:.0f}/{state['a_upper']:.0f}]")
        print(f"   C:  RSI Period {state['c_period']}s")
        print(f"   C3: RSI Period {state.get('c3_period', RSI_PERIOD_INITIAL_C3)}s")
        print(f"   C4: RSI Period {state.get('c4_period', RSI_PERIOD_INITIAL_C4)}s")
        print(f"   C5: RSI Period {state.get('c5_period', RSI_PERIOD_INITIAL_C5)}s")
        print(f"   D:  EMA/RSI Period {state.get('d_period', RSI_PERIOD_INITIAL_D)}s")
    return state


def adjust_thresholds(breach_history: deque, current_upper: float, current_lower: float,
                      adjustment: int, high_trigger: int, low_trigger: int) -> tuple:
    """Adjust RSI thresholds based on breach frequency"""
    if len(breach_history) == 0:
        return current_upper, current_lower, 0, 0

    # Handle both tuple format (breach_type, timestamp) and string format
    upper_breaches = 0
    lower_breaches = 0
    for item in breach_history:
        if isinstance(item, tuple):
            breach_type, _ = item
        else:
            breach_type = item

        if breach_type == 'UPPER':
            upper_breaches += 1
        elif breach_type == 'LOWER':
            lower_breaches += 1

    new_upper = current_upper
    new_lower = current_lower

    # Adjust upper threshold
    if upper_breaches > high_trigger:
        new_upper = min(MAX_RSI_UPPER, current_upper + adjustment)
    elif upper_breaches < low_trigger:
        new_lower_bound = current_lower + 10
        new_upper = max(new_lower_bound, current_upper - adjustment)

    # Adjust lower threshold
    if lower_breaches > high_trigger:
        new_lower = max(MIN_RSI_LOWER, current_lower - adjustment)
    elif lower_breaches < low_trigger:
        new_upper_bound = current_upper - 10
        new_lower = min(new_upper_bound, current_lower + adjustment)

    return new_upper, new_lower, upper_breaches, lower_breaches


def adjust_rsi_period(breach_history: deque, current_period: int, adjustment: int,
                      high_trigger: int, low_trigger: int, min_period: int, max_period: int = None) -> tuple:
    """
    Adjust RSI period based on total breach frequency
    More breaches → Increase period (slower, smoother RSI)
    Fewer breaches → Decrease period (faster, more responsive RSI)
    """
    # Count total breaches (upper + lower)
    total_breaches = 0
    for item in breach_history:
        if isinstance(item, tuple):
            breach_type, _ = item
        else:
            breach_type = item

        if breach_type in ['UPPER', 'LOWER']:
            total_breaches += 1

    new_period = current_period

    # Adjust period based on total breaches
    if total_breaches > high_trigger:
        # Too many breaches → increase period (slower RSI, smoother)
        new_period = current_period + adjustment
        if max_period is not None:
            new_period = min(max_period, new_period)
    elif total_breaches < low_trigger:
        # Too few breaches (including 0) → decrease period (faster RSI, more sensitive)
        new_period = max(min_period, current_period - adjustment)

    return new_period, total_breaches


def load_trades(strategy: str) -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    if strategy == "A":
        filename = f"trades_{today}.json"
    else:
        filename = f"strategy{strategy}_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    if Path(filepath).exists():
        data = read_json(filepath)
        if data:
            print(f"📂 Strategy {strategy} - Loaded: {len(data.get('trades', []))} trades, PNL: ${data.get('total_pnl', 0):.2f}")
            return data
    return {"strategy": f"{strategy}_Adaptive", "date": today, "total_pnl": 0.0, "trades": [],
            "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "call_trades": 0, "put_trades": 0}}


def save_trades(trades_data: Dict, strategy: str):
    today = datetime.now().strftime("%Y-%m-%d")
    if strategy == "A":
        filename = f"trades_{today}.json"
    else:
        filename = f"strategy{strategy}_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    write_json(filepath, trades_data)


def check_tp_sl(position, current_bid: float, take_profit: float, stop_loss: float) -> Optional[str]:
    """
    Check if position hits take profit or stop loss
    Returns: 'TP', 'SL', or None
    """
    if not position:
        return None

    current_pnl = current_bid - position['entry_price']

    if current_pnl >= take_profit:
        return 'TP'
    elif current_pnl <= -stop_loss:
        return 'SL'
    return None


def calculate_dynamic_tpsl_c3(atr: float, entry_price: float) -> Tuple[float, float]:
    """
    Calculate dynamic TP and SL for Strategy C3 using option ATR

    Rules:
    - TP = entry_price + ATR (minimum 0.035, maximum 0.20)
    - SL = ATR * 3 (minimum 0.10, maximum 0.30)
    - Maintains approximately 1:3 ratio

    Args:
        atr: Average True Range from option prices
        entry_price: Position entry price

    Returns:
        (take_profit_price, stop_loss_distance)
    """
    # TP distance: ATR with limits
    tp_distance = max(0.035, min(0.20, atr))

    # SL distance: 3× ATR with limits
    sl_distance = max(0.10, min(0.30, atr * 3.0))

    # Calculate TP target price
    take_profit_price = entry_price + tp_distance

    return take_profit_price, sl_distance




def get_c5_thresholds(option_type: str, option_price: float) -> Tuple[float, float, str]:
    """
    Get dynamic RSI thresholds for C5 based on option price
    Returns: (open_threshold, close_threshold, trend_label)
    """
    ranges = C5_PUT_RANGES if option_type == 'PUT' else C5_CALL_RANGES

    for price_min, price_max, open_thresh, close_thresh, label in ranges:
        if price_min < option_price <= price_max:
            return (open_thresh, close_thresh, label)

    # Default to neutral if price out of range
    if option_type == 'PUT':
        return (80, 20, "neutral")
    else:
        return (20, 80, "neutral")


def close_position_expiration(position, strike_price, final_btc, trades_data, strategy, timestamp):
    if position['type'] == 'CALL':
        exit_price = 1.0 if final_btc > strike_price else 0.0
    else:
        exit_price = 1.0 if final_btc < strike_price else 0.0

    pnl = exit_price - position['entry_price']
    trades_data['total_pnl'] += pnl

    trade_record = {**position, 'exit_price': exit_price, 'exit_time': timestamp,
                   'exit_btc': final_btc, 'pnl': pnl, 'exit_reason': 'EXPIRATION',
                   'expired_itm': exit_price == 1.0}
    trades_data['trades'].append(trade_record)
    trades_data['stats']['total_trades'] += 1
    if pnl > 0:
        trades_data['stats']['winning_trades'] += 1
    else:
        trades_data['stats']['losing_trades'] += 1

    save_trades(trades_data, strategy)
    result_emoji = "✅" if exit_price == 1.0 else "❌"
    print(f"[{strategy}] 💰 {result_emoji} Expired ${exit_price:.2f} | PNL:${pnl:+.3f} | Total:${trades_data['total_pnl']:+.2f}")
    return None


def close_position_signal(position, exit_price, timestamp, trades_data, strategy, exit_reason='RSI_SIGNAL'):
    pnl = exit_price - position['entry_price']
    trades_data['total_pnl'] += pnl

    trade_record = {**position, 'exit_price': exit_price, 'exit_time': timestamp,
                   'pnl': pnl, 'exit_reason': exit_reason}
    trades_data['trades'].append(trade_record)
    trades_data['stats']['total_trades'] += 1
    if pnl > 0:
        trades_data['stats']['winning_trades'] += 1
    else:
        trades_data['stats']['losing_trades'] += 1

    save_trades(trades_data, strategy)
    reason_emoji = "💰" if exit_reason == 'TP' else "🛑" if exit_reason == 'SL' else "🔴"
    print(f"[{strategy}] {reason_emoji} SOLD {position['type']} @${exit_price:.2f} | PNL:${pnl:+.3f} | {exit_reason}")
    return None


def open_position(opt_type, ask_price, timestamp, strike_price, btc_price, rsi, bin_key,
                 seconds_remaining, trades_data, strategy, rsi_upper=None, rsi_lower=None,
                 rsi_upper_enhanced=None, rsi_lower_enhanced=None, rsi_period=None, btc_volatility=None, rsi_slope=None):
    position = {'type': opt_type, 'entry_price': ask_price, 'entry_time': timestamp,
               'entry_strike': strike_price, 'entry_btc': btc_price, 'entry_rsi': rsi,
               'entry_bin': bin_key, 'entry_seconds_remaining': seconds_remaining}
    if rsi_upper is not None and rsi_lower is not None:
        position['entry_rsi_upper'] = rsi_upper
        position['entry_rsi_lower'] = rsi_lower
    if rsi_upper_enhanced is not None and rsi_lower_enhanced is not None:
        position['entry_rsi_upper_enhanced'] = rsi_upper_enhanced
        position['entry_rsi_lower_enhanced'] = rsi_lower_enhanced
    if rsi_period is not None:
        position['entry_rsi_period'] = rsi_period
    if btc_volatility is not None:
        position['entry_btc_volatility'] = btc_volatility
    if rsi_slope is not None:
        position['entry_rsi_slope'] = rsi_slope

    trades_data['stats'][f"{opt_type.lower()}_trades"] += 1
    save_trades(trades_data, strategy)

    thresh_str = f" [{rsi_lower:.0f}/{rsi_upper:.0f}]" if rsi_upper is not None else ""
    enh_str = f" E[{rsi_lower_enhanced:.0f}/{rsi_upper_enhanced:.0f}]" if rsi_upper_enhanced is not None else ""
    period_str = f" RSI{rsi_period}s" if rsi_period is not None else ""
    vol_str = f" Vol:{btc_volatility:.1f}" if btc_volatility is not None else ""
    slope_str = f" Slope:{rsi_slope:+.4f}" if rsi_slope is not None else ""
    print(f"[{strategy}] 🟢 BOUGHT {opt_type} @${ask_price:.2f} | RSI:{rsi:.1f}{thresh_str}{enh_str}{period_str}{vol_str}{slope_str}")
    return position


def main():
    print("\n" + "="*150)
    print("RSI TRADING BOT - STRATEGIES A, C, C3, C4, C5, D, D2")
    print(f"A: Original Adaptive (±{THRESHOLD_ADJUSTMENT_A}, >{THRESHOLD_HIGH_TRIGGER_A}/<{THRESHOLD_LOW_TRIGGER_A})")
    print(f"C: Adaptive RSI Period ({RSI_LOWER_C}/{RSI_UPPER_C}, 0-2/3-5/6+, Max:{RSI_PERIOD_MAX_C}s)")
    print(f"C3: Adaptive RSI Period + ATR TP/SL ({RSI_LOWER_C3}/{RSI_UPPER_C3}, 0-3/4-7/8+, 1:3 ratio, Max:{RSI_PERIOD_MAX_C3}s)")
    print(f"C4: Dual Threshold (Open:{RSI_LOWER_OPEN_C4}/{RSI_UPPER_OPEN_C4}, Close:{RSI_LOWER_CLOSE_C4}/{RSI_UPPER_CLOSE_C4}, 0-3/4-7/8+, NO MAX)")
    print(f"C5: Price-Adaptive Thresholds (Breach:80/20, Dynamic open/close, 0-3/4-7/8+, NO MAX)")
    print(f"D: EMA Crossover (Breach:80/20, EMA±$5, RSI exits 90/10, 0-3/4-7/8+, NO MAX)")
    print(f"D2: EMA Bands (BTC>EMA+tol→CALL, BTC<EMA-tol→PUT, RSI exits 90/10)")
    print(f"Limits: ${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f} | Window: {MAX_SECONDS_INTO_PERIOD}s-{900-MIN_SECONDS_REMAINING}s")
    print("="*150)

    # Load persistent state
    saved_state = load_bot_state()

    trades_a = load_trades("A")
    trades_c = load_trades("C")
    trades_c3 = load_trades("C3")
    trades_c4 = load_trades("C4")
    trades_c5 = load_trades("C5")
    trades_d = load_trades("D")
    trades_d2 = load_trades("D2")

    position_a, position_c, position_c3, position_c4, position_c5, position_d, position_d2 = None, None, None, None, None, None, None
    btc_price_history = deque(maxlen=RSI_PERIOD + 1)
    btc_vol_history = deque(maxlen=600)

    # Strategy A state - Load from saved state or use defaults
    rsi_upper_a = saved_state['a_upper'] if saved_state else RSI_UPPER_INITIAL_A
    rsi_lower_a = saved_state['a_lower'] if saved_state else RSI_LOWER_INITIAL_A
    breach_history_a = deque(maxlen=100)  # Max ~100 breaches in 15 min
    currently_breaching_upper_a = False
    currently_breaching_lower_a = False
    last_signal_time_a, last_signal_type_a = 0, None

    # Strategy C state (adaptive RSI period)
    rsi_period_c = saved_state['c_period'] if saved_state else RSI_PERIOD_INITIAL_C
    rsi_period_samples_c = rsi_period_c * 10  # Convert seconds to samples (0.1s intervals)
    breach_history_c = deque(maxlen=100)  # Max ~100 breaches in 15 min
    currently_breaching_upper_c = False
    currently_breaching_lower_c = False
    last_signal_time_c, last_signal_type_c = 0, None
    btc_price_history_c = deque(maxlen=2000)  # Large enough for any RSI period

    # Strategy C3 state (adaptive RSI period with ATR-based TP/SL)
    rsi_period_c3 = saved_state.get('c3_period', RSI_PERIOD_INITIAL_C3) if saved_state else RSI_PERIOD_INITIAL_C3
    rsi_period_samples_c3 = rsi_period_c3 * 10
    breach_history_c3 = deque(maxlen=100)  # Max ~100 breaches in 15 min
    currently_breaching_upper_c3 = False
    currently_breaching_lower_c3 = False
    last_signal_time_c3, last_signal_type_c3 = 0, None
    btc_price_history_c3 = deque(maxlen=2000)
    # ATR tracking for C3
    atr_calculator_c3 = OptionATRCalculator(period_seconds=30, num_periods=6)

    # Strategy C4 state (adaptive RSI period with dual thresholds)
    rsi_period_c4 = saved_state.get('c4_period', RSI_PERIOD_INITIAL_C4) if saved_state else RSI_PERIOD_INITIAL_C4
    rsi_period_samples_c4 = rsi_period_c4 * 10
    breach_history_c4 = deque(maxlen=100)  # Max ~100 breaches in 15 min
    currently_breaching_upper_c4 = False
    currently_breaching_lower_c4 = False
    last_signal_time_c4, last_signal_type_c4 = 0, None
    btc_price_history_c4 = deque(maxlen=2000)


    # Strategy C5 state (price-adaptive dynamic thresholds)
    rsi_period_c5 = saved_state.get('c5_period', RSI_PERIOD_INITIAL_C5) if saved_state else RSI_PERIOD_INITIAL_C5
    rsi_period_samples_c5 = rsi_period_c5 * 10
    breach_history_c5 = deque(maxlen=100)  # Max ~100 breaches in 15 min
    currently_breaching_upper_c5 = False
    currently_breaching_lower_c5 = False
    last_signal_type_c5, last_signal_time_c5 = None, 0
    btc_price_history_c5 = deque(maxlen=2000)


    # Strategy D state (EMA crossover with adaptive period)
    rsi_period_d = saved_state.get('d_period', RSI_PERIOD_INITIAL_D) if saved_state else RSI_PERIOD_INITIAL_D
    rsi_period_samples_d = rsi_period_d * 10  # For RSI breach tracking
    ema_period_samples_d = rsi_period_d * 10  # EMA uses same period as RSI
    breach_history_d = deque(maxlen=100)  # Max ~100 breaches in 15 min
    currently_breaching_upper_d = False
    currently_breaching_lower_d = False
    last_signal_type_d, last_signal_time_d = None, 0
    btc_price_history_d = deque(maxlen=2000)  # For both RSI and EMA
    last_ema_d = None
    last_btc_price_d = None
    ema_cross_direction_d = None  # 'UP' or 'DOWN'


    # Strategy D2 state (EMA bands - simplified)
    last_d2_type_opened = None  # Track last type to enforce alternation
    last_signal_type_d2, last_signal_time_d2 = None, 0

    last_threshold_adjustment_time = time.time()
    threshold_check_interval = 60

    strike_price, last_strike_update_minute = None, None
    waiting_for_prices = False

    try:
        while True:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            is_period_start = now.minute in [0, 15, 30, 45]

            # Handle period end expirations
            if is_period_start and now.second == 0:
                btc_data = read_json(BTC_FILE)
                final_btc = btc_data.get('price', 0) if btc_data else 0

                if position_a and strike_price:
                    position_a = close_position_expiration(position_a, strike_price, final_btc, trades_a, "A", timestamp)
                if position_c and strike_price:
                    position_c = close_position_expiration(position_c, strike_price, final_btc, trades_c, "C", timestamp)
                if position_c3 and strike_price:
                    position_c3 = close_position_expiration(position_c3, strike_price, final_btc, trades_c3, "C3", timestamp)
                if position_c4 and strike_price:
                    position_c4 = close_position_expiration(position_c4, strike_price, final_btc, trades_c4, "C4", timestamp)
                if position_c5 and strike_price:
                    position_c5 = close_position_expiration(position_c5, strike_price, final_btc, trades_c5, "C5", timestamp)
                if position_d and strike_price:
                    position_d = close_position_expiration(position_d, strike_price, final_btc, trades_d, "D", timestamp)
                if position_d2 and strike_price:
                    position_d2 = close_position_expiration(position_d2, strike_price, final_btc, trades_d2, "D2", timestamp)
                # Reset D2 state at period end
                last_d2_type_opened = None

                if position_a or position_c or position_c3 or position_c4 or position_c5 or position_d or position_d2:
                    time.sleep(2)

            # Update strike price
            if is_period_start and now.second >= 5 and last_strike_update_minute != now.minute:
                new_strike = get_strike_price()
                if new_strike:
                    strike_price = new_strike
                    last_strike_update_minute = now.minute
                    print(f"\n🔄 New strike: ${strike_price:.2f}")

            if strike_price is None:
                strike_price = get_strike_price()
                if strike_price:
                    print(f"Initial Strike: ${strike_price:.2f}\n")

            # Read market data
            btc_data = read_json(BTC_FILE)
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)

            if not all([btc_data, call_data, put_data]):
                time.sleep(CHECK_INTERVAL)
                continue

            btc_price = btc_data.get('price', 0)
            btc_price_history.append(btc_price)
            btc_vol_history.append(btc_price)

            call_bid_data = call_data.get('best_bid')
            call_ask_data = call_data.get('best_ask')
            put_bid_data = put_data.get('best_bid')
            put_ask_data = put_data.get('best_ask')

            call_bid = call_bid_data.get('price', 0) if call_bid_data else 0
            call_ask = call_ask_data.get('price', 0) if call_ask_data else 0
            put_bid = put_bid_data.get('price', 0) if put_bid_data else 0
            put_ask = put_ask_data.get('price', 0) if put_ask_data else 0

            has_valid_prices = call_bid > 0 and call_ask > 0 and put_bid > 0 and put_ask > 0
            if not has_valid_prices:
                if not waiting_for_prices:
                    print(f"\n⚠️  Waiting for valid prices...")
                    waiting_for_prices = True
                time.sleep(CHECK_INTERVAL)
                continue
            else:
                if waiting_for_prices:
                    waiting_for_prices = False

            # Calculate RSI
            rsi = calculate_rsi(btc_price_history)
            if rsi is None:
                time.sleep(CHECK_INTERVAL)
                continue

            # Calculate RSI for Strategy C (with adaptive period)
            btc_price_history_c.append(btc_price)
            rsi_c = calculate_rsi(btc_price_history_c, period=rsi_period_samples_c)

            # Calculate RSI for Strategy C3 (with adaptive period)
            btc_price_history_c3.append(btc_price)
            rsi_c3 = calculate_rsi(btc_price_history_c3, period=rsi_period_samples_c3)

            # Update ATR calculator for C3 with CALL and PUT option prices
            atr_calculator_c3.update(call_ask, put_ask)

            # Calculate RSI for Strategy C4 (with adaptive period)
            btc_price_history_c4.append(btc_price)
            rsi_c4 = calculate_rsi(btc_price_history_c4, period=rsi_period_samples_c4)

            # Calculate RSI for Strategy C5 (with adaptive period)
            btc_price_history_c5.append(btc_price)
            rsi_c5 = calculate_rsi(btc_price_history_c5, period=rsi_period_samples_c5)

            # Calculate RSI and EMA for Strategy D (EMA crossover)
            btc_price_history_d.append(btc_price)
            rsi_d = calculate_rsi(btc_price_history_d, period=rsi_period_samples_d)
            ema_d = calculate_ema(btc_price_history_d, period=ema_period_samples_d)

            # Track breaches for all strategies
            # Strategy A
            if rsi > rsi_upper_a:
                if not currently_breaching_upper_a:
                    breach_history_a.append(('UPPER', time.time()))
                    currently_breaching_upper_a = True
                    currently_breaching_lower_a = False
            elif rsi < rsi_lower_a:
                if not currently_breaching_lower_a:
                    breach_history_a.append(('LOWER', time.time()))
                    currently_breaching_lower_a = True
                    currently_breaching_upper_a = False
            else:
                currently_breaching_upper_a = False
                currently_breaching_lower_a = False

            # Strategy C (fixed thresholds, adaptive period)
            if rsi_c is not None:
                if rsi_c > RSI_UPPER_C:
                    if not currently_breaching_upper_c:
                        breach_history_c.append(('UPPER', time.time()))
                        currently_breaching_upper_c = True
                        currently_breaching_lower_c = False
                elif rsi_c < RSI_LOWER_C:
                    if not currently_breaching_lower_c:
                        breach_history_c.append(('LOWER', time.time()))
                        currently_breaching_lower_c = True
                        currently_breaching_upper_c = False
                else:
                    currently_breaching_upper_c = False
                    currently_breaching_lower_c = False

            # Strategy C3 (fixed thresholds, adaptive period, ATR TP/SL)
            if rsi_c3 is not None:
                if rsi_c3 > RSI_UPPER_C3:
                    if not currently_breaching_upper_c3:
                        breach_history_c3.append(('UPPER', time.time()))
                        currently_breaching_upper_c3 = True
                        currently_breaching_lower_c3 = False
                elif rsi_c3 < RSI_LOWER_C3:
                    if not currently_breaching_lower_c3:
                        breach_history_c3.append(('LOWER', time.time()))
                        currently_breaching_lower_c3 = True
                        currently_breaching_upper_c3 = False
                else:
                    currently_breaching_upper_c3 = False
                    currently_breaching_lower_c3 = False

            # Strategy C4 (dual thresholds, adaptive period)
            # Track breaches using OPEN thresholds for period adjustment
            if rsi_c4 is not None:
                if rsi_c4 > RSI_UPPER_OPEN_C4:
                    if not currently_breaching_upper_c4:
                        breach_history_c4.append(('UPPER', time.time()))
                        currently_breaching_upper_c4 = True
                        currently_breaching_lower_c4 = False
                elif rsi_c4 < RSI_LOWER_OPEN_C4:
                    if not currently_breaching_lower_c4:
                        breach_history_c4.append(('LOWER', time.time()))
                        currently_breaching_lower_c4 = True
                        currently_breaching_upper_c4 = False
                else:
                    currently_breaching_upper_c4 = False
                    currently_breaching_lower_c4 = False


            # Strategy C5 (price-adaptive, breach tracking on 80/20 for period adjustment)
            if rsi_c5 is not None:
                if rsi_c5 > RSI_BREACH_UPPER_C5:
                    if not currently_breaching_upper_c5:
                        breach_history_c5.append(('UPPER', time.time()))
                        currently_breaching_upper_c5 = True
                        currently_breaching_lower_c5 = False
                elif rsi_c5 < RSI_BREACH_LOWER_C5:
                    if not currently_breaching_lower_c5:
                        breach_history_c5.append(('LOWER', time.time()))
                        currently_breaching_lower_c5 = True
                        currently_breaching_upper_c5 = False
                else:
                    currently_breaching_upper_c5 = False
                    currently_breaching_lower_c5 = False


            # Strategy D (EMA crossover, RSI breach tracking on 80/20 for period adjustment)
            if rsi_d is not None:
                if rsi_d > RSI_BREACH_UPPER_D:
                    if not currently_breaching_upper_d:
                        breach_history_d.append(('UPPER', time.time()))
                        currently_breaching_upper_d = True
                        currently_breaching_lower_d = False
                elif rsi_d < RSI_BREACH_LOWER_D:
                    if not currently_breaching_lower_d:
                        breach_history_d.append(('LOWER', time.time()))
                        currently_breaching_lower_d = True
                        currently_breaching_upper_d = False
                else:
                    currently_breaching_upper_d = False
                    currently_breaching_lower_d = False

            # Adjust thresholds/periods every 60 seconds
            current_time_threshold = time.time()
            if current_time_threshold - last_threshold_adjustment_time >= threshold_check_interval:
                print(f"\n⏱️  Running threshold/period checks at {datetime.now().strftime('%H:%M:%S')}...")

                # Strategy A
                recent_breaches_a = [b for b in breach_history_a if current_time_threshold - b[1] <= 900]
                recent_breach_types_a = deque([b[0] for b in recent_breaches_a])
                old_upper_a, old_lower_a = rsi_upper_a, rsi_lower_a
                rsi_upper_a, rsi_lower_a, upper_count_a, lower_count_a = adjust_thresholds(
                    recent_breach_types_a, rsi_upper_a, rsi_lower_a,
                    THRESHOLD_ADJUSTMENT_A, THRESHOLD_HIGH_TRIGGER_A, THRESHOLD_LOW_TRIGGER_A)
                if rsi_upper_a != old_upper_a or rsi_lower_a != old_lower_a:
                    print(f"\n📊 [A] Thresholds: {old_lower_a:.0f}/{old_upper_a:.0f} → {rsi_lower_a:.0f}/{rsi_upper_a:.0f} (Breaches: U={upper_count_a}, L={lower_count_a})")

                # Strategy C - Adjust RSI period
                recent_breaches_c = [b for b in breach_history_c if current_time_threshold - b[1] <= 900]
                recent_breach_types_c = deque([b[0] for b in recent_breaches_c])
                old_period_c = rsi_period_c
                rsi_period_c, total_breaches_c = adjust_rsi_period(
                    recent_breach_types_c, rsi_period_c, RSI_PERIOD_ADJUSTMENT_C,
                    RSI_PERIOD_HIGH_TRIGGER_C, RSI_PERIOD_LOW_TRIGGER_C, RSI_PERIOD_MIN_C, RSI_PERIOD_MAX_C)
                rsi_period_samples_c = rsi_period_c * 10  # Convert to samples

                # Always show C adjustment check (even if no change)
                if rsi_period_c != old_period_c:
                    print(f"\n📊 [C] RSI Period: {old_period_c}s → {rsi_period_c}s (Total Breaches: {total_breaches_c})")
                else:
                    # Debug: show why no change
                    if total_breaches_c > RSI_PERIOD_HIGH_TRIGGER_C:
                        print(f"\n📊 [C] Period check: {total_breaches_c} breaches (>{RSI_PERIOD_HIGH_TRIGGER_C}) → at max {rsi_period_c}s")
                    elif total_breaches_c < RSI_PERIOD_LOW_TRIGGER_C:
                        print(f"\n📊 [C] Period check: {total_breaches_c} breaches (<{RSI_PERIOD_LOW_TRIGGER_C}) → at min {rsi_period_c}s")
                    else:
                        print(f"\n📊 [C] Period check: {total_breaches_c} breaches (staying at {rsi_period_c}s)")

                # Strategy C3 - Adjust RSI period
                recent_breaches_c3 = [b for b in breach_history_c3 if current_time_threshold - b[1] <= 900]
                recent_breach_types_c3 = deque([b[0] for b in recent_breaches_c3])
                old_period_c3 = rsi_period_c3
                rsi_period_c3, total_breaches_c3 = adjust_rsi_period(
                    recent_breach_types_c3, rsi_period_c3, RSI_PERIOD_ADJUSTMENT_C3,
                    RSI_PERIOD_HIGH_TRIGGER_C3, RSI_PERIOD_LOW_TRIGGER_C3, RSI_PERIOD_MIN_C3, RSI_PERIOD_MAX_C3)
                rsi_period_samples_c3 = rsi_period_c3 * 10

                if rsi_period_c3 != old_period_c3:
                    print(f"\n📊 [C3] RSI Period: {old_period_c3}s → {rsi_period_c3}s (Total Breaches: {total_breaches_c3})")

                # Strategy C4 - Adjust RSI period (NO MAX LIMIT)
                recent_breaches_c4 = [b for b in breach_history_c4 if current_time_threshold - b[1] <= 900]
                recent_breach_types_c4 = deque([b[0] for b in recent_breaches_c4])
                old_period_c4 = rsi_period_c4
                rsi_period_c4, total_breaches_c4 = adjust_rsi_period(
                    recent_breach_types_c4, rsi_period_c4, RSI_PERIOD_ADJUSTMENT_C4,
                    RSI_PERIOD_HIGH_TRIGGER_C4, RSI_PERIOD_LOW_TRIGGER_C4, RSI_PERIOD_MIN_C4, max_period=None)
                rsi_period_samples_c4 = rsi_period_c4 * 10

                if rsi_period_c4 != old_period_c4:
                    print(f"\n📊 [C4] RSI Period: {old_period_c4}s → {rsi_period_c4}s (Total Breaches: {total_breaches_c4}) [NO MAX]")


                # Strategy C5 - Adjust RSI period (NO MAX LIMIT)
                recent_breaches_c5 = [b for b in breach_history_c5 if current_time_threshold - b[1] <= 900]
                recent_breach_types_c5 = deque([b[0] for b in recent_breaches_c5])
                old_period_c5 = rsi_period_c5
                rsi_period_c5, total_breaches_c5 = adjust_rsi_period(
                    recent_breach_types_c5, rsi_period_c5, RSI_PERIOD_ADJUSTMENT_C5,
                    RSI_PERIOD_HIGH_TRIGGER_C5, RSI_PERIOD_LOW_TRIGGER_C5, RSI_PERIOD_MIN_C5, max_period=None)
                rsi_period_samples_c5 = rsi_period_c5 * 10

                if rsi_period_c5 != old_period_c5:
                    print(f"\n📊 [C5] RSI Period: {old_period_c5}s → {rsi_period_c5}s (Total Breaches: {total_breaches_c5}) [NO MAX]")


                # Strategy D - Adjust EMA/RSI period (NO MAX LIMIT)
                recent_breaches_d = [b for b in breach_history_d if current_time_threshold - b[1] <= 900]
                recent_breach_types_d = deque([b[0] for b in recent_breaches_d])
                old_period_d = rsi_period_d
                rsi_period_d, total_breaches_d = adjust_rsi_period(
                    recent_breach_types_d, rsi_period_d, RSI_PERIOD_ADJUSTMENT_D,
                    RSI_PERIOD_HIGH_TRIGGER_D, RSI_PERIOD_LOW_TRIGGER_D, RSI_PERIOD_MIN_D, max_period=None)
                rsi_period_samples_d = rsi_period_d * 10
                ema_period_samples_d = rsi_period_d * 10  # EMA uses same period

                if rsi_period_d != old_period_d:
                    print(f"\n📊 [D] EMA/RSI Period: {old_period_d}s → {rsi_period_d}s (Total Breaches: {total_breaches_d}) [NO MAX]")

                last_threshold_adjustment_time = current_time_threshold

                # Save bot state for persistence
                bot_state = {
                    'a_upper': rsi_upper_a,
                    'a_lower': rsi_lower_a,
                    'c_period': rsi_period_c,
                    'c3_period': rsi_period_c3,
                    'c4_period': rsi_period_c4,
                    'c5_period': rsi_period_c5,
                    'd_period': rsi_period_d,
                    'timestamp': datetime.now().isoformat()
                }
                save_bot_state(bot_state)

            # Market metrics
            seconds_remaining = get_seconds_to_expiry()
            seconds_into_period = 900 - seconds_remaining
            distance = abs(btc_price - strike_price) if strike_price else 0
            volatility = calculate_btc_volatility(btc_vol_history)
            bin_key = get_bin_key(distance, seconds_remaining, volatility)
            in_time_window = (seconds_into_period >= MAX_SECONDS_INTO_PERIOD and
                            seconds_remaining >= MIN_SECONDS_REMAINING)

            # Display
            pos_a_str = f"{position_a['type']}@${position_a['entry_price']:.2f}" if position_a else "NONE"
            pos_c_str = f"{position_c['type']}@${position_c['entry_price']:.2f}" if position_c else "NONE"
            pos_c3_str = f"{position_c3['type']}@${position_c3['entry_price']:.2f}" if position_c3 else "NONE"
            pos_c4_str = f"{position_c4['type']}@${position_c4['entry_price']:.2f}" if position_c4 else "NONE"
            pos_c5_str = f"{position_c5['type']}@${position_c5['entry_price']:.2f}" if position_c5 else "NONE"
            pos_d_str = f"{position_d['type']}@${position_d['entry_price']:.2f}" if position_d else "NONE"
            pos_d2_str = f"{position_d2['type']}@${position_d2['entry_price']:.2f}" if position_d2 else "NONE"
            rsi_c_str = f"{rsi_c:.1f}" if rsi_c is not None else "N/A"
            rsi_c3_str = f"{rsi_c3:.1f}" if rsi_c3 is not None else "N/A"
            rsi_c4_str = f"{rsi_c4:.1f}" if rsi_c4 is not None else "N/A"
            rsi_c5_str = f"{rsi_c5:.1f}" if rsi_c5 is not None else "N/A"
            rsi_d_str = f"{rsi_d:.1f}" if rsi_d is not None else "N/A"
            ema_d_str = f"${ema_d:.2f}" if ema_d is not None else "N/A"

            # Calculate ATR for C3 display
            atr_c3 = atr_calculator_c3.calculate_atr()
            atr_c3_str = f"{atr_c3:.4f}" if atr_c3 is not None else "N/A"

            # Split display across 2 lines for readability
            print(f"[{timestamp}] BTC:${btc_price:.2f} | A:[{rsi_lower_a:.0f}/{rsi_upper_a:.0f}] {pos_a_str:<12} ${trades_a['total_pnl']:+.2f} | C:RSI{rsi_period_c}s={rsi_c_str} {pos_c_str:<12} ${trades_c['total_pnl']:+.2f}")
            print(f"         C3:RSI{rsi_period_c3}s={rsi_c3_str} ATR:{atr_c3_str} {pos_c3_str:<12} ${trades_c3['total_pnl']:+.2f} | C4:RSI{rsi_period_c4}s={rsi_c4_str} {pos_c4_str:<12} ${trades_c4['total_pnl']:+.2f}")
            last_type_str = f"Last:{last_d2_type_opened}" if last_d2_type_opened else "Last:None"
            print(f"         C5:RSI{rsi_period_c5}s={rsi_c5_str} {pos_c5_str:<12} ${trades_c5['total_pnl']:+.2f} | D:EMA{rsi_period_d}s={ema_d_str} {pos_d_str:<12} ${trades_d['total_pnl']:+.2f} | D2:{last_type_str} {pos_d2_str:<12} ${trades_d2['total_pnl']:+.2f}", end='\r')

            if not in_time_window:
                time.sleep(CHECK_INTERVAL)
                continue

            current_time = time.time()

            # ===== CHECK TP/SL FOR C2 AND C3 =====
            # Check TP/SL for C3 (ATR-based adaptive TP/SL)
            if position_c3:
                # Calculate current ATR-based TP/SL
                atr_c3_current = atr_calculator_c3.calculate_atr()

                if atr_c3_current:
                    tp_target, sl_distance = calculate_dynamic_tpsl_c3(atr_c3_current, position_c3['entry_price'])
                    current_bid_c3 = put_bid if position_c3['type'] == 'PUT' else call_bid
                    current_pnl = current_bid_c3 - position_c3['entry_price']

                    # Debug output every 10 seconds
                    if int(time.time()) % 10 == 0:
                        tp_distance = tp_target - position_c3['entry_price']
                        print(f"\n[C3 DEBUG] Entry:${position_c3['entry_price']:.3f} Bid:${current_bid_c3:.3f} PNL:${current_pnl:+.3f} | ATR:{atr_c3_current:.4f} TP:${tp_target:.3f}(+${tp_distance:.3f}) SL:${sl_distance:.3f}(-${sl_distance:.3f})")

                    tp_sl_result = check_tp_sl(position_c3, current_bid_c3, tp_target - position_c3['entry_price'], sl_distance)

                    if tp_sl_result:
                        position_c3 = close_position_signal(position_c3, current_bid_c3, timestamp, trades_c3, "C3", tp_sl_result)
                        time.sleep(2)

            # ===== STRATEGY A =====
            signal_a = None
            if rsi < rsi_lower_a:
                signal_a = 'BUY'
            elif rsi > rsi_upper_a:
                signal_a = 'SELL'

            if signal_a:
                if last_signal_type_a != signal_a:
                    last_signal_time_a = current_time
                    last_signal_type_a = signal_a
                elif current_time - last_signal_time_a >= CONFIRMATION_SECONDS:
                    if signal_a == 'BUY' and (not position_a or position_a['type'] == 'PUT'):
                        if position_a:
                            position_a = close_position_signal(position_a, put_bid, timestamp, trades_a, "A")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position_a = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi,
                                                      bin_key, seconds_remaining, trades_a, "A", rsi_upper_a, rsi_lower_a)
                        last_signal_type_a = None
                    elif signal_a == 'SELL' and (not position_a or position_a['type'] == 'CALL'):
                        if position_a:
                            position_a = close_position_signal(position_a, call_bid, timestamp, trades_a, "A")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position_a = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi,
                                                      bin_key, seconds_remaining, trades_a, "A", rsi_upper_a, rsi_lower_a)
                        last_signal_type_a = None
            else:
                last_signal_type_a = None


            # ===== STRATEGY C (Adaptive RSI Period) =====
            if rsi_c is not None:
                signal_c = None
                if rsi_c < RSI_LOWER_C:
                    signal_c = 'BUY'
                elif rsi_c > RSI_UPPER_C:
                    signal_c = 'SELL'

                if signal_c:
                    if last_signal_type_c != signal_c:
                        last_signal_time_c = current_time
                        last_signal_type_c = signal_c
                    elif current_time - last_signal_time_c >= CONFIRMATION_SECONDS:
                        if signal_c == 'BUY' and (not position_c or position_c['type'] == 'PUT'):
                            if position_c:
                                position_c = close_position_signal(position_c, put_bid, timestamp, trades_c, "C")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_c = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi_c,
                                                          bin_key, seconds_remaining, trades_c, "C",
                                                          rsi_period=rsi_period_c, btc_volatility=volatility)
                            last_signal_type_c = None
                        elif signal_c == 'SELL' and (not position_c or position_c['type'] == 'CALL'):
                            if position_c:
                                position_c = close_position_signal(position_c, call_bid, timestamp, trades_c, "C")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_c = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi_c,
                                                          bin_key, seconds_remaining, trades_c, "C",
                                                          rsi_period=rsi_period_c, btc_volatility=volatility)
                            last_signal_type_c = None
                else:
                    last_signal_type_c = None

            # ===== STRATEGY C3 (Adaptive RSI Period with ATR-based TP/SL) =====
            if rsi_c3 is not None:
                signal_c3 = None
                if rsi_c3 < RSI_LOWER_C3:
                    signal_c3 = 'BUY'
                elif rsi_c3 > RSI_UPPER_C3:
                    signal_c3 = 'SELL'

                if signal_c3:
                    if last_signal_type_c3 != signal_c3:
                        last_signal_time_c3 = current_time
                        last_signal_type_c3 = signal_c3
                    elif current_time - last_signal_time_c3 >= CONFIRMATION_SECONDS:
                        if signal_c3 == 'BUY' and (not position_c3 or position_c3['type'] == 'PUT'):
                            if position_c3:
                                position_c3 = close_position_signal(position_c3, put_bid, timestamp, trades_c3, "C3")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_c3 = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi_c3,
                                                            bin_key, seconds_remaining, trades_c3, "C3",
                                                            rsi_period=rsi_period_c3, btc_volatility=volatility)
                            last_signal_type_c3 = None
                        elif signal_c3 == 'SELL' and (not position_c3 or position_c3['type'] == 'CALL'):
                            if position_c3:
                                position_c3 = close_position_signal(position_c3, call_bid, timestamp, trades_c3, "C3")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_c3 = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi_c3,
                                                            bin_key, seconds_remaining, trades_c3, "C3",
                                                            rsi_period=rsi_period_c3, btc_volatility=volatility)
                            last_signal_type_c3 = None
                else:
                    last_signal_type_c3 = None

            # ===== STRATEGY C4 (Dual Thresholds with Adaptive Period) =====
            if rsi_c4 is not None:
                signal_c4 = None

                # OPEN signals use wider thresholds (80/15)
                if not position_c4:
                    if rsi_c4 < RSI_LOWER_OPEN_C4:  # 15
                        signal_c4 = 'BUY'
                    elif rsi_c4 > RSI_UPPER_OPEN_C4:  # 80
                        signal_c4 = 'SELL'
                # CLOSE signals use tighter thresholds (70/30)
                else:
                    if position_c4['type'] == 'CALL' and rsi_c4 > RSI_UPPER_CLOSE_C4:  # 70
                        signal_c4 = 'CLOSE_CALL'
                    elif position_c4['type'] == 'PUT' and rsi_c4 < RSI_LOWER_CLOSE_C4:  # 30
                        signal_c4 = 'CLOSE_PUT'

                if signal_c4:
                    if last_signal_type_c4 != signal_c4:
                        last_signal_time_c4 = current_time
                        last_signal_type_c4 = signal_c4
                    elif current_time - last_signal_time_c4 >= CONFIRMATION_SECONDS:
                        if signal_c4 == 'BUY':
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_c4 = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi_c4,
                                                            bin_key, seconds_remaining, trades_c4, "C4",
                                                            rsi_period=rsi_period_c4, btc_volatility=volatility)
                            last_signal_type_c4 = None
                        elif signal_c4 == 'SELL':
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_c4 = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi_c4,
                                                            bin_key, seconds_remaining, trades_c4, "C4",
                                                            rsi_period=rsi_period_c4, btc_volatility=volatility)
                            last_signal_type_c4 = None
                        elif signal_c4 == 'CLOSE_CALL':
                            position_c4 = close_position_signal(position_c4, call_bid, timestamp, trades_c4, "C4")
                            last_signal_type_c4 = None
                            time.sleep(2)
                        elif signal_c4 == 'CLOSE_PUT':
                            position_c4 = close_position_signal(position_c4, put_bid, timestamp, trades_c4, "C4")
                            last_signal_type_c4 = None
                            time.sleep(2)
                else:
                    last_signal_type_c4 = None

            # ===== STRATEGY C5 (Price-Adaptive Dynamic Thresholds) =====
            if rsi_c5 is not None:
                signal_c5 = None

                # Determine which option to trade and get dynamic thresholds
                # OPEN signals - check both CALL and PUT prices
                if not position_c5:
                    # Check PUT thresholds
                    put_open_thresh, put_close_thresh, put_trend = get_c5_thresholds('PUT', put_ask)
                    # Check CALL thresholds
                    call_open_thresh, call_close_thresh, call_trend = get_c5_thresholds('CALL', call_ask)

                    # PUT: Open when RSI > threshold
                    if rsi_c5 > put_open_thresh:
                        signal_c5 = 'BUY_PUT'
                        c5_trend_label = put_trend
                        c5_open_thresh = put_open_thresh
                        c5_close_thresh = put_close_thresh
                    # CALL: Open when RSI < threshold
                    elif rsi_c5 < call_open_thresh:
                        signal_c5 = 'BUY_CALL'
                        c5_trend_label = call_trend
                        c5_open_thresh = call_open_thresh
                        c5_close_thresh = call_close_thresh

                # CLOSE signals - use stored thresholds from position
                else:
                    current_price = put_ask if position_c5['type'] == 'PUT' else call_ask
                    stored_close_thresh = position_c5.get('close_threshold', 20 if position_c5['type'] == 'PUT' else 80)

                    if position_c5['type'] == 'PUT' and rsi_c5 < stored_close_thresh:
                        signal_c5 = 'CLOSE_PUT'
                    elif position_c5['type'] == 'CALL' and rsi_c5 > stored_close_thresh:
                        signal_c5 = 'CLOSE_CALL'

                if signal_c5:
                    if last_signal_type_c5 != signal_c5:
                        last_signal_time_c5 = current_time
                        last_signal_type_c5 = signal_c5
                    elif current_time - last_signal_time_c5 >= CONFIRMATION_SECONDS:
                        if signal_c5 == 'BUY_PUT':
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_c5 = {
                                    'type': 'PUT',
                                    'entry_price': put_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_c5,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'rsi_period': rsi_period_c5,
                                    'btc_volatility': volatility,
                                    'open_threshold': c5_open_thresh,
                                    'close_threshold': c5_close_thresh,
                                    'trend': c5_trend_label
                                }
                                print(f"\n[C5] 🟢 BUY PUT @${put_ask:.2f} | RSI={rsi_c5:.1f} | Trend:{c5_trend_label} | Open>{c5_open_thresh}/Close<{c5_close_thresh} | {seconds_remaining}s left | Period={rsi_period_c5}s")
                            last_signal_type_c5 = None
                        elif signal_c5 == 'BUY_CALL':
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_c5 = {
                                    'type': 'CALL',
                                    'entry_price': call_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_c5,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'rsi_period': rsi_period_c5,
                                    'btc_volatility': volatility,
                                    'open_threshold': c5_open_thresh,
                                    'close_threshold': c5_close_thresh,
                                    'trend': c5_trend_label
                                }
                                print(f"\n[C5] 🟢 BUY CALL @${call_ask:.2f} | RSI={rsi_c5:.1f} | Trend:{c5_trend_label} | Open<{c5_open_thresh}/Close>{c5_close_thresh} | {seconds_remaining}s left | Period={rsi_period_c5}s")
                            last_signal_type_c5 = None
                        elif signal_c5 == 'CLOSE_PUT':
                            pnl = put_bid - position_c5['entry_price']
                            trades_c5['total_pnl'] += pnl

                            won = pnl > 0
                            trades_c5['stats']['total_trades'] += 1
                            if won:
                                trades_c5['stats']['winning_trades'] += 1
                            else:
                                trades_c5['stats']['losing_trades'] += 1
                            trades_c5['stats']['put_trades'] += 1

                            trade_record = {
                                'entry_time': position_c5['entry_time'],
                                'exit_time': timestamp,
                                'type': 'PUT',
                                'entry_price': position_c5['entry_price'],
                                'exit_price': put_bid,
                                'pnl': pnl,
                                'strike_price': position_c5.get('strike_price'),
                                'btc_at_entry': position_c5.get('btc_at_entry'),
                                'rsi_at_entry': position_c5.get('rsi_at_entry'),
                                'rsi_period': position_c5.get('rsi_period'),
                                'btc_volatility': position_c5.get('btc_volatility'),
                                'bin': position_c5.get('bin'),
                                'seconds_remaining_at_entry': position_c5.get('seconds_remaining'),
                                'exit_reason': 'RSI_SIGNAL',
                                'open_threshold': position_c5.get('open_threshold'),
                                'close_threshold': position_c5.get('close_threshold'),
                                'trend': position_c5.get('trend')
                            }

                            trades_c5['trades'].append(trade_record)
                            save_trades(trades_c5, "C5")

                            print(f"[C5] 🔴 SOLD PUT @${put_bid:.2f} | PNL:{pnl:+.3f} | Trend:{position_c5.get('trend')} | RSI_SIGNAL")
                            position_c5 = None
                            last_signal_type_c5 = None
                            time.sleep(2)
                        elif signal_c5 == 'CLOSE_CALL':
                            pnl = call_bid - position_c5['entry_price']
                            trades_c5['total_pnl'] += pnl

                            won = pnl > 0
                            trades_c5['stats']['total_trades'] += 1
                            if won:
                                trades_c5['stats']['winning_trades'] += 1
                            else:
                                trades_c5['stats']['losing_trades'] += 1
                            trades_c5['stats']['call_trades'] += 1

                            trade_record = {
                                'entry_time': position_c5['entry_time'],
                                'exit_time': timestamp,
                                'type': 'CALL',
                                'entry_price': position_c5['entry_price'],
                                'exit_price': call_bid,
                                'pnl': pnl,
                                'strike_price': position_c5.get('strike_price'),
                                'btc_at_entry': position_c5.get('btc_at_entry'),
                                'rsi_at_entry': position_c5.get('rsi_at_entry'),
                                'rsi_period': position_c5.get('rsi_period'),
                                'btc_volatility': position_c5.get('btc_volatility'),
                                'bin': position_c5.get('bin'),
                                'seconds_remaining_at_entry': position_c5.get('seconds_remaining'),
                                'exit_reason': 'RSI_SIGNAL',
                                'open_threshold': position_c5.get('open_threshold'),
                                'close_threshold': position_c5.get('close_threshold'),
                                'trend': position_c5.get('trend')
                            }

                            trades_c5['trades'].append(trade_record)
                            save_trades(trades_c5, "C5")

                            print(f"[C5] 🔴 SOLD CALL @${call_bid:.2f} | PNL:{pnl:+.3f} | Trend:{position_c5.get('trend')} | RSI_SIGNAL")
                            position_c5 = None
                            last_signal_type_c5 = None
                            time.sleep(2)
                else:
                    last_signal_type_c5 = None

            # ===== STRATEGY D (EMA Crossover with Adaptive Period) =====
            if ema_d is not None and rsi_d is not None:
                signal_d = None

                # Debug: Show EMA calculation status every 10 seconds
                if int(time.time()) % 10 == 0:
                    ema_str = f"{ema_d:.2f}" if ema_d is not None else "N/A"
                    rsi_str = f"{rsi_d:.1f}" if rsi_d is not None else "N/A"
                    last_ema_str = f"{last_ema_d:.2f}" if last_ema_d is not None else "None"
                    last_btc_str = f"{last_btc_price_d:.2f}" if last_btc_price_d is not None else "None"
                    if last_ema_d is not None and last_btc_price_d is not None:
                        curr_dist = btc_price - ema_d
                        last_dist = last_btc_price_d - last_ema_d
                        cross_dir_str = f" | CrossDir:{ema_cross_direction_d}" if ema_cross_direction_d else ""
                        print(f"\n[D STATUS] BTC:{btc_price:.2f} EMA:{ema_str} | Dist:{curr_dist:+.2f} LastDist:{last_dist:+.2f} | RSI:{rsi_str}{cross_dir_str}")
                    else:
                        print(f"\n[D STATUS] EMA:{ema_str} | RSI:{rsi_str} | BTC:{btc_price:.2f} | History:{len(btc_price_history_d)} samples")

                # Detect EMA crossover - TWO STEP LOGIC
                if last_ema_d is not None and last_btc_price_d is not None:
                    # Calculate distance from EMA (positive = above, negative = below)
                    current_distance = btc_price - ema_d
                    last_distance = last_btc_price_d - last_ema_d

                    # Step 1: Detect crossover (sets direction flag)
                    if last_distance < 0 and current_distance >= 0:
                        # Just crossed UP
                        if ema_cross_direction_d != 'UP':
                            ema_cross_direction_d = 'UP'
                            print(f"\n[D CROSS] ⬆️ Crossed UP! Now tracking for ${EMA_CROSSOVER_THRESHOLD_D:.0f}+ move...")
                    elif last_distance > 0 and current_distance <= 0:
                        # Just crossed DOWN
                        if ema_cross_direction_d != 'DOWN':
                            ema_cross_direction_d = 'DOWN'
                            print(f"\n[D CROSS] ⬇️ Crossed DOWN! Now tracking for ${EMA_CROSSOVER_THRESHOLD_D:.0f}+ move...")

                    # Step 2: Check if we have enough distance in the cross direction
                    if not position_d and ema_cross_direction_d:
                        # CALL: We crossed up, now check if $5+ above
                        if ema_cross_direction_d == 'UP' and current_distance >= EMA_CROSSOVER_THRESHOLD_D:
                            signal_d = 'BUY_CALL'
                            print(f"\n[D SIGNAL] 🎯 CALL entry! Distance:${current_distance:.2f} >= ${EMA_CROSSOVER_THRESHOLD_D:.0f}")
                            # Don't reset direction here - wait for trade execution

                        # PUT: We crossed down, now check if $5+ below
                        elif ema_cross_direction_d == 'DOWN' and current_distance <= -EMA_CROSSOVER_THRESHOLD_D:
                            signal_d = 'BUY_PUT'
                            print(f"\n[D SIGNAL] 🎯 PUT entry! Distance:${abs(current_distance):.2f} >= ${EMA_CROSSOVER_THRESHOLD_D:.0f}")
                            # Don't reset direction here - wait for trade execution

                        # Reset cross direction if we cross back before reaching threshold
                        elif ema_cross_direction_d == 'UP' and current_distance < 0:
                            print(f"\n[D CROSS] ❌ Crossed back down before reaching ${EMA_CROSSOVER_THRESHOLD_D:.0f} threshold")
                            ema_cross_direction_d = None
                        elif ema_cross_direction_d == 'DOWN' and current_distance > 0:
                            print(f"\n[D CROSS] ❌ Crossed back up before reaching ${EMA_CROSSOVER_THRESHOLD_D:.0f} threshold")
                            ema_cross_direction_d = None

                    # Exit signals
                    if position_d:
                        if position_d['type'] == 'CALL':
                            # Exit CALL: RSI > 90 OR crossed back down through EMA
                            if rsi_d > RSI_EXIT_CALL_D:
                                signal_d = 'CLOSE_CALL_RSI'
                            elif last_distance > 0 and current_distance < 0:
                                signal_d = 'CLOSE_CALL_EMA'

                        elif position_d['type'] == 'PUT':
                            # Exit PUT: RSI < 10 OR crossed back up through EMA by $5+
                            if rsi_d < RSI_EXIT_PUT_D:
                                signal_d = 'CLOSE_PUT_RSI'
                            elif last_distance < 0 and current_distance >= EMA_CROSSOVER_THRESHOLD_D:
                                signal_d = 'CLOSE_PUT_EMA'

                # Update last values for next iteration
                last_ema_d = ema_d
                last_btc_price_d = btc_price

                if signal_d:
                    if last_signal_type_d != signal_d:
                        last_signal_time_d = current_time
                        last_signal_type_d = signal_d
                        print(f"\n[D] Signal '{signal_d}' detected, waiting {CONFIRMATION_SECONDS}s confirmation...")
                    elif current_time - last_signal_time_d >= CONFIRMATION_SECONDS:
                        if signal_d == 'BUY_CALL':
                            print(f"\n[D] Attempting CALL: ask=${call_ask:.2f} | Limits:[${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f}] | InWindow:{in_time_window}")
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_d = {
                                    'type': 'CALL',
                                    'entry_price': call_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_d,
                                    'ema_at_entry': ema_d,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'ema_period': rsi_period_d,
                                    'btc_volatility': volatility,
                                    'entry_reason': 'EMA_CROSS_UP'
                                }
                                print(f"\n[D] 🟢 BUY CALL @${call_ask:.2f} | BTC:${btc_price:.2f} crossed UP EMA:${ema_d:.2f} | RSI={rsi_d:.1f} | {seconds_remaining}s left | Period={rsi_period_d}s")
                                # Reset cross direction after successful trade
                                ema_cross_direction_d = None
                            else:
                                print(f"\n[D] ❌ CALL blocked: ask=${call_ask:.2f} outside limits [${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f}]")
                                # Reset cross direction after failed trade
                                ema_cross_direction_d = None
                            last_signal_type_d = None
                        elif signal_d == 'BUY_PUT':
                            print(f"\n[D] Attempting PUT: ask=${put_ask:.2f} | Limits:[${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f}] | InWindow:{in_time_window}")
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_d = {
                                    'type': 'PUT',
                                    'entry_price': put_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_d,
                                    'ema_at_entry': ema_d,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'ema_period': rsi_period_d,
                                    'btc_volatility': volatility,
                                    'entry_reason': 'EMA_CROSS_DOWN'
                                }
                                print(f"\n[D] 🟢 BUY PUT @${put_ask:.2f} | BTC:${btc_price:.2f} crossed DOWN EMA:${ema_d:.2f} | RSI={rsi_d:.1f} | {seconds_remaining}s left | Period={rsi_period_d}s")
                                # Reset cross direction after successful trade
                                ema_cross_direction_d = None
                            else:
                                print(f"\n[D] ❌ PUT blocked: ask=${put_ask:.2f} outside limits [${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f}]")
                                # Reset cross direction after failed trade
                                ema_cross_direction_d = None
                            last_signal_type_d = None
                        elif signal_d in ['CLOSE_CALL_RSI', 'CLOSE_CALL_EMA']:
                            exit_reason = 'RSI_EXIT' if 'RSI' in signal_d else 'EMA_CROSS_DOWN'
                            pnl = call_bid - position_d['entry_price']
                            trades_d['total_pnl'] += pnl

                            won = pnl > 0
                            trades_d['stats']['total_trades'] += 1
                            if won:
                                trades_d['stats']['winning_trades'] += 1
                            else:
                                trades_d['stats']['losing_trades'] += 1
                            trades_d['stats']['call_trades'] += 1

                            trade_record = {
                                'entry_time': position_d['entry_time'],
                                'exit_time': timestamp,
                                'type': 'CALL',
                                'entry_price': position_d['entry_price'],
                                'exit_price': call_bid,
                                'pnl': pnl,
                                'strike_price': position_d.get('strike_price'),
                                'btc_at_entry': position_d.get('btc_at_entry'),
                                'rsi_at_entry': position_d.get('rsi_at_entry'),
                                'ema_at_entry': position_d.get('ema_at_entry'),
                                'ema_period': position_d.get('ema_period'),
                                'btc_volatility': position_d.get('btc_volatility'),
                                'bin': position_d.get('bin'),
                                'seconds_remaining_at_entry': position_d.get('seconds_remaining'),
                                'entry_reason': position_d.get('entry_reason'),
                                'exit_reason': exit_reason
                            }

                            trades_d['trades'].append(trade_record)
                            save_trades(trades_d, "D")

                            print(f"[D] 🔴 SOLD CALL @${call_bid:.2f} | PNL:{pnl:+.3f} | {exit_reason} | BTC:${btc_price:.2f} EMA:${ema_d:.2f} RSI:{rsi_d:.1f}")
                            position_d = None
                            last_signal_type_d = None
                            time.sleep(2)
                        elif signal_d in ['CLOSE_PUT_RSI', 'CLOSE_PUT_EMA']:
                            exit_reason = 'RSI_EXIT' if 'RSI' in signal_d else 'EMA_CROSS_UP'
                            pnl = put_bid - position_d['entry_price']
                            trades_d['total_pnl'] += pnl

                            won = pnl > 0
                            trades_d['stats']['total_trades'] += 1
                            if won:
                                trades_d['stats']['winning_trades'] += 1
                            else:
                                trades_d['stats']['losing_trades'] += 1
                            trades_d['stats']['put_trades'] += 1

                            trade_record = {
                                'entry_time': position_d['entry_time'],
                                'exit_time': timestamp,
                                'type': 'PUT',
                                'entry_price': position_d['entry_price'],
                                'exit_price': put_bid,
                                'pnl': pnl,
                                'strike_price': position_d.get('strike_price'),
                                'btc_at_entry': position_d.get('btc_at_entry'),
                                'rsi_at_entry': position_d.get('rsi_at_entry'),
                                'ema_at_entry': position_d.get('ema_at_entry'),
                                'ema_period': position_d.get('ema_period'),
                                'btc_volatility': position_d.get('btc_volatility'),
                                'bin': position_d.get('bin'),
                                'seconds_remaining_at_entry': position_d.get('seconds_remaining'),
                                'entry_reason': position_d.get('entry_reason'),
                                'exit_reason': exit_reason
                            }

                            trades_d['trades'].append(trade_record)
                            save_trades(trades_d, "D")

                            print(f"[D] 🔴 SOLD PUT @${put_bid:.2f} | PNL:{pnl:+.3f} | {exit_reason} | BTC:${btc_price:.2f} EMA:${ema_d:.2f} RSI:{rsi_d:.1f}")
                            position_d = None
                            last_signal_type_d = None
                            time.sleep(2)
                else:
                    last_signal_type_d = None

            # ===== STRATEGY D2 (EMA Bands - Simplified) =====
            # D2 shares EMA and RSI with D
            if ema_d is not None and rsi_d is not None and seconds_into_period >= D2_START_DELAY:
                signal_d2 = None

                # Calculate tolerance
                vol_tolerance = volatility / 8 if volatility > 0 else MIN_TOLERANCE_D2
                tolerance = max(MIN_TOLERANCE_D2, vol_tolerance)

                # Calculate bands
                upper_band = ema_d + tolerance
                lower_band = ema_d - tolerance

                # Debug every 10 seconds
                if int(time.time()) % 10 == 0:
                    pos_str = f"{position_d2['type']}" if position_d2 else "NONE"
                    last_str = last_d2_type_opened if last_d2_type_opened else "None"
                    btc_vs_bands = "ABOVE" if btc_price > upper_band else "BELOW" if btc_price < lower_band else "INSIDE"
                    print(f"\n[D2] BTC:{btc_price:.2f} {btc_vs_bands} [{lower_band:.2f} EMA:{ema_d:.2f} {upper_band:.2f}] Tol:{tolerance:.2f} Pos:{pos_str} Last:{last_str} RSI:{rsi_d:.1f}")

                # PRIORITY 1: Close on RSI extremes
                if position_d2 and position_d2['type'] == 'CALL' and rsi_d > RSI_EXIT_CALL_D2:
                    signal_d2 = 'CLOSE_CALL'
                    print(f"\n[D2] 🔴 Close CALL - RSI:{rsi_d:.1f} > {RSI_EXIT_CALL_D2}")

                elif position_d2 and position_d2['type'] == 'PUT' and rsi_d < RSI_EXIT_PUT_D2:
                    signal_d2 = 'CLOSE_PUT'
                    print(f"\n[D2] 🔴 Close PUT - RSI:{rsi_d:.1f} < {RSI_EXIT_PUT_D2}")

                # PRIORITY 2: Flip wrong position to correct one
                elif position_d2 and position_d2['type'] == 'CALL' and btc_price < lower_band:
                    # Holding CALL but should be PUT
                    signal_d2 = 'FLIP_CALL_TO_PUT'
                    print(f"\n[D2] 🔄 FLIP CALL→PUT - BTC:{btc_price:.2f} < LB:{lower_band:.2f}")

                elif position_d2 and position_d2['type'] == 'PUT' and btc_price > upper_band:
                    # Holding PUT but should be CALL
                    signal_d2 = 'FLIP_PUT_TO_CALL'
                    print(f"\n[D2] 🔄 FLIP PUT→CALL - BTC:{btc_price:.2f} > UB:{upper_band:.2f}")

                # PRIORITY 3: Open new position if none exists
                elif not position_d2 and btc_price > upper_band:
                    if last_d2_type_opened != 'CALL':
                        signal_d2 = 'OPEN_CALL'
                        print(f"\n[D2] 🟢 Open CALL - BTC:{btc_price:.2f} > {upper_band:.2f}")
                    else:
                        if int(time.time()) % 30 == 0:
                            print(f"\n[D2] ⏸️ CALL condition met but must open PUT first (last was CALL)")

                elif not position_d2 and btc_price < lower_band:
                    if last_d2_type_opened != 'PUT':
                        signal_d2 = 'OPEN_PUT'
                        print(f"\n[D2] 🟢 Open PUT - BTC:{btc_price:.2f} < {lower_band:.2f}")
                    else:
                        if int(time.time()) % 30 == 0:
                            print(f"\n[D2] ⏸️ PUT condition met but must open CALL first (last was PUT)")

                # Execute signals
                if signal_d2:
                    if last_signal_type_d2 != signal_d2:
                        last_signal_time_d2 = current_time
                        last_signal_type_d2 = signal_d2
                    elif current_time - last_signal_time_d2 >= CONFIRMATION_SECONDS:

                        if signal_d2 == 'OPEN_CALL':
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_d2 = {
                                    'type': 'CALL',
                                    'entry_price': call_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_d,
                                    'ema_at_entry': ema_d,
                                    'tolerance': tolerance,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'ema_period': rsi_period_d,
                                    'btc_volatility': volatility
                                }
                                last_d2_type_opened = 'CALL'
                                print(f"\n[D2] 🟢 OPENED CALL @${call_ask:.2f} | BTC:{btc_price:.2f} > UB:{upper_band:.2f}")
                            last_signal_type_d2 = None

                        elif signal_d2 == 'OPEN_PUT':
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_d2 = {
                                    'type': 'PUT',
                                    'entry_price': put_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_d,
                                    'ema_at_entry': ema_d,
                                    'tolerance': tolerance,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'ema_period': rsi_period_d,
                                    'btc_volatility': volatility
                                }
                                last_d2_type_opened = 'PUT'
                                print(f"\n[D2] 🟢 OPENED PUT @${put_ask:.2f} | BTC:{btc_price:.2f} < LB:{lower_band:.2f}")
                            last_signal_type_d2 = None

                        elif signal_d2 == 'CLOSE_CALL':
                            pnl = call_bid - position_d2['entry_price']
                            trades_d2['total_pnl'] += pnl
                            trades_d2['stats']['total_trades'] += 1
                            if pnl > 0:
                                trades_d2['stats']['winning_trades'] += 1
                            else:
                                trades_d2['stats']['losing_trades'] += 1
                            trades_d2['stats']['call_trades'] += 1

                            trade_record = {
                                'entry_time': position_d2['entry_time'],
                                'exit_time': timestamp,
                                'type': 'CALL',
                                'entry_price': position_d2['entry_price'],
                                'exit_price': call_bid,
                                'pnl': pnl,
                                'strike_price': position_d2.get('strike_price'),
                                'btc_at_entry': position_d2.get('btc_at_entry'),
                                'ema_at_entry': position_d2.get('ema_at_entry'),
                                'tolerance': position_d2.get('tolerance'),
                                'ema_period': position_d2.get('ema_period'),
                                'exit_reason': 'RSI_EXIT'
                            }
                            trades_d2['trades'].append(trade_record)
                            save_trades(trades_d2, "D2")

                            print(f"[D2] 🔴 CLOSED CALL @${call_bid:.2f} | PNL:{pnl:+.3f} | RSI:{rsi_d:.1f}")
                            position_d2 = None
                            last_signal_type_d2 = None
                            time.sleep(2)

                        elif signal_d2 == 'CLOSE_PUT':
                            pnl = put_bid - position_d2['entry_price']
                            trades_d2['total_pnl'] += pnl
                            trades_d2['stats']['total_trades'] += 1
                            if pnl > 0:
                                trades_d2['stats']['winning_trades'] += 1
                            else:
                                trades_d2['stats']['losing_trades'] += 1
                            trades_d2['stats']['put_trades'] += 1

                            trade_record = {
                                'entry_time': position_d2['entry_time'],
                                'exit_time': timestamp,
                                'type': 'PUT',
                                'entry_price': position_d2['entry_price'],
                                'exit_price': put_bid,
                                'pnl': pnl,
                                'strike_price': position_d2.get('strike_price'),
                                'btc_at_entry': position_d2.get('btc_at_entry'),
                                'ema_at_entry': position_d2.get('ema_at_entry'),
                                'tolerance': position_d2.get('tolerance'),
                                'ema_period': position_d2.get('ema_period'),
                                'exit_reason': 'RSI_EXIT'
                            }
                            trades_d2['trades'].append(trade_record)
                            save_trades(trades_d2, "D2")

                            print(f"[D2] 🔴 CLOSED PUT @${put_bid:.2f} | PNL:{pnl:+.3f} | RSI:{rsi_d:.1f}")
                            position_d2 = None
                            last_signal_type_d2 = None
                            time.sleep(2)

                        # FLIP CALL TO PUT
                        elif signal_d2 == 'FLIP_CALL_TO_PUT':
                            # Close CALL
                            pnl = call_bid - position_d2['entry_price']
                            trades_d2['total_pnl'] += pnl
                            trades_d2['stats']['total_trades'] += 1
                            if pnl > 0:
                                trades_d2['stats']['winning_trades'] += 1
                            else:
                                trades_d2['stats']['losing_trades'] += 1
                            trades_d2['stats']['call_trades'] += 1

                            trade_record = {
                                'entry_time': position_d2['entry_time'],
                                'exit_time': timestamp,
                                'type': 'CALL',
                                'entry_price': position_d2['entry_price'],
                                'exit_price': call_bid,
                                'pnl': pnl,
                                'strike_price': position_d2.get('strike_price'),
                                'btc_at_entry': position_d2.get('btc_at_entry'),
                                'ema_at_entry': position_d2.get('ema_at_entry'),
                                'tolerance': position_d2.get('tolerance'),
                                'ema_period': position_d2.get('ema_period'),
                                'exit_reason': 'FLIP_TO_PUT'
                            }
                            trades_d2['trades'].append(trade_record)
                            save_trades(trades_d2, "D2")

                            print(f"[D2] 🔄 Closed CALL @${call_bid:.2f} PNL:{pnl:+.3f}")
                            position_d2 = None
                            time.sleep(2)

                            # Open PUT
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_d2 = {
                                    'type': 'PUT',
                                    'entry_price': put_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_d,
                                    'ema_at_entry': ema_d,
                                    'tolerance': tolerance,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'ema_period': rsi_period_d,
                                    'btc_volatility': volatility
                                }
                                last_d2_type_opened = 'PUT'
                                print(f"[D2] 🟢 Opened PUT @${put_ask:.2f}")
                            last_signal_type_d2 = None

                        # FLIP PUT TO CALL
                        elif signal_d2 == 'FLIP_PUT_TO_CALL':
                            # Close PUT
                            pnl = put_bid - position_d2['entry_price']
                            trades_d2['total_pnl'] += pnl
                            trades_d2['stats']['total_trades'] += 1
                            if pnl > 0:
                                trades_d2['stats']['winning_trades'] += 1
                            else:
                                trades_d2['stats']['losing_trades'] += 1
                            trades_d2['stats']['put_trades'] += 1

                            trade_record = {
                                'entry_time': position_d2['entry_time'],
                                'exit_time': timestamp,
                                'type': 'PUT',
                                'entry_price': position_d2['entry_price'],
                                'exit_price': put_bid,
                                'pnl': pnl,
                                'strike_price': position_d2.get('strike_price'),
                                'btc_at_entry': position_d2.get('btc_at_entry'),
                                'ema_at_entry': position_d2.get('ema_at_entry'),
                                'tolerance': position_d2.get('tolerance'),
                                'ema_period': position_d2.get('ema_period'),
                                'exit_reason': 'FLIP_TO_CALL'
                            }
                            trades_d2['trades'].append(trade_record)
                            save_trades(trades_d2, "D2")

                            print(f"[D2] 🔄 Closed PUT @${put_bid:.2f} PNL:{pnl:+.3f}")
                            position_d2 = None
                            time.sleep(2)

                            # Open CALL
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_d2 = {
                                    'type': 'CALL',
                                    'entry_price': call_ask,
                                    'entry_time': timestamp,
                                    'strike_price': strike_price,
                                    'btc_at_entry': btc_price,
                                    'rsi_at_entry': rsi_d,
                                    'ema_at_entry': ema_d,
                                    'tolerance': tolerance,
                                    'bin': bin_key,
                                    'seconds_remaining': seconds_remaining,
                                    'ema_period': rsi_period_d,
                                    'btc_volatility': volatility
                                }
                                last_d2_type_opened = 'CALL'
                                print(f"[D2] 🟢 Opened CALL @${call_ask:.2f}")
                            last_signal_type_d2 = None
                else:
                    last_signal_type_d2 = None


            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")

        # Close any open positions
        for pos, data, strat, cb, pb in [(position_a, trades_a, "A", call_bid, put_bid),
                                          (position_c, trades_c, "C", call_bid, put_bid),
                                          (position_c3, trades_c3, "C3", call_bid, put_bid),
                                          (position_c4, trades_c4, "C4", call_bid, put_bid),
                                          (position_c5, trades_c5, "C5", call_bid, put_bid),
                                          (position_d, trades_d, "D", call_bid, put_bid),
                                          (position_d2, trades_d2, "D2", call_bid, put_bid)]:
            if pos:
                exit_price = cb if pos['type'] == 'CALL' else pb
                if exit_price > 0:
                    pos = close_position_signal(pos, exit_price, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data, strat)

        # Print summaries
        for strat, data in [("A", trades_a), ("C", trades_c), ("C3", trades_c3), ("C4", trades_c4), ("C5", trades_c5), ("D", trades_d), ("D2", trades_d2)]:
            print(f"\n{'='*80}\nStrategy {strat} Summary\n{'='*80}")
            print(f"Total PNL: ${data['total_pnl']:+.2f}")
            print(f"Trades: {data['stats']['total_trades']} | Wins: {data['stats']['winning_trades']} | Losses: {data['stats']['losing_trades']}")
            if data['stats']['total_trades'] > 0:
                print(f"Win Rate: {data['stats']['winning_trades']/data['stats']['total_trades']*100:.1f}%")
            print("="*80)


if __name__ == "__main__":
    main()
