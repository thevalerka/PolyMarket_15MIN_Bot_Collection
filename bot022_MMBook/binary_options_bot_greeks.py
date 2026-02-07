#!/usr/bin/env python3
"""
15-Minute Binary Options Trading Bot v3
========================================
Trades CALL/PUT options on Polymarket based on BTC volume spikes and momentum.
Considers option Greeks (delta/gamma) for trade sizing and entry decisions.

Key Features:
- Volume spike detection (1s vs 30s average)
- Buy/Sell pressure momentum analysis
- Delta/Gamma awareness for option sensitivity
- Max 12 trades per 15-minute period with COOLDOWNS
- Dynamic trade sizing based on probability edge
- Full PNL tracking and simulation
- Proper strike price from Bybit API
- Early exit in last 10 seconds at market price
"""

import json
import time
import math
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class Trade:
    """Represents a single trade"""
    trade_id: int
    timestamp: datetime
    direction: TradeDirection
    entry_price: float  # Option price at entry (0-1)
    size: float  # Position size in dollars
    btc_price_at_entry: float  # BTC price at entry
    strike_price: float  # BTC strike price for this period
    time_to_expiry_seconds: int
    delta: float
    gamma: float
    volume_ratio: float  # Volume spike ratio that triggered the trade
    buy_sell_ratio: float  # Buy/sell pressure ratio
    edge: float  # Expected edge at entry
    exit_price: Optional[float] = None  # Market exit price or 1.0/0.0 at expiry
    btc_price_at_exit: Optional[float] = None
    pnl: Optional[float] = None
    settled: bool = False
    expiry_time: Optional[datetime] = None
    exit_type: str = "expiry"  # "expiry" or "early_exit"


@dataclass
class TradingPeriod:
    """Represents a 15-minute trading period"""
    start_time: datetime
    end_time: datetime
    strike_price: Optional[float]  # BTC price at period start (the strike)
    trades: List[Trade] = field(default_factory=list)
    max_trades: int = 12
    settled: bool = False
    early_exit_done: bool = False

    @property
    def trades_remaining(self) -> int:
        return self.max_trades - len(self.trades)

    @property
    def open_trades(self) -> List[Trade]:
        return [t for t in self.trades if not t.settled]

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl is not None)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl is not None and t.pnl > 0)


class BinaryOptionsBot:
    def __init__(
        self,
        call_file: str = "/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        put_file: str = "/home/ubuntu/013_2025_polymarket/15M_PUT.json",
        btc_file: str = "binance_BTC_details.json",
        base_trade_size: float = 10.0,  # Base trade size in dollars
        max_trades_per_period: int = 12,
        min_volume_spike_ratio: float = 2.0,  # Minimum 2x volume spike to consider
        min_buy_sell_imbalance: float = 1.5,  # Minimum buy/sell ratio imbalance
        min_edge_threshold: float = 0.05,  # Minimum expected edge to trade
        trade_cooldown_seconds: int = 45,  # Minimum seconds between trades
        signal_confirmation_readings: int = 3,  # Need N consecutive readings confirming signal
        early_exit_seconds: int = 10,  # Exit positions this many seconds before expiry
        simulation_mode: bool = True
    ):
        self.call_file = Path(call_file)
        self.put_file = Path(put_file)
        self.btc_file = Path(btc_file)

        self.base_trade_size = base_trade_size
        self.max_trades_per_period = max_trades_per_period
        self.min_volume_spike_ratio = min_volume_spike_ratio
        self.min_buy_sell_imbalance = min_buy_sell_imbalance
        self.min_edge_threshold = min_edge_threshold
        self.trade_cooldown_seconds = trade_cooldown_seconds
        self.signal_confirmation_readings = signal_confirmation_readings
        self.early_exit_seconds = early_exit_seconds
        self.simulation_mode = simulation_mode

        # State
        self.current_period: Optional[TradingPeriod] = None
        self.all_periods: List[TradingPeriod] = []
        self.last_btc_data: Optional[Dict] = None
        self.last_call_data: Optional[Dict] = None
        self.last_put_data: Optional[Dict] = None
        self.last_trade_time: Optional[datetime] = None

        # Signal confirmation buffer
        self.signal_buffer: List[Dict] = []
        self.max_signal_buffer = 10

        # Trade counter
        self.trade_id_counter = 0

        # Overall stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # PNL tracking file
        self.pnl_file = Path("bot_pnl_tracking.json")
        self.trades_log_file = Path("bot_trades_log.json")
        self.load_pnl_history()

    def load_pnl_history(self):
        """Load PNL history from file"""
        try:
            if self.pnl_file.exists():
                with open(self.pnl_file, 'r') as f:
                    data = json.load(f)
                    self.total_trades = data.get('total_trades', 0)
                    self.winning_trades = data.get('winning_trades', 0)
                    self.total_pnl = data.get('total_pnl', 0.0)
                    self.trade_id_counter = data.get('last_trade_id', 0)
                    logger.info(f"📂 Loaded PNL history: {self.total_trades} trades, ${self.total_pnl:+.2f}")
        except Exception as e:
            logger.warning(f"Could not load PNL history: {e}")

    def save_pnl_history(self):
        """Save PNL history to file"""
        try:
            data = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'total_pnl': round(self.total_pnl, 2),
                'last_trade_id': self.trade_id_counter,
                'win_rate': round(self.winning_trades / self.total_trades * 100, 2) if self.total_trades > 0 else 0,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.pnl_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save PNL history: {e}")

    def log_trade(self, trade: Trade):
        """Append trade to trades log file"""
        try:
            trades = []
            if self.trades_log_file.exists():
                with open(self.trades_log_file, 'r') as f:
                    trades = json.load(f)

            # Find existing trade record or create new
            existing_idx = None
            for idx, t in enumerate(trades):
                if t.get('trade_id') == trade.trade_id:
                    existing_idx = idx
                    break

            trade_record = {
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp.isoformat(),
                'direction': trade.direction.value,
                'entry_price': trade.entry_price,
                'size': trade.size,
                'btc_price_at_entry': trade.btc_price_at_entry,
                'strike_price': trade.strike_price,
                'time_to_expiry_seconds': trade.time_to_expiry_seconds,
                'delta': round(trade.delta, 4),
                'gamma': round(trade.gamma, 4),
                'volume_ratio': round(trade.volume_ratio, 2),
                'buy_sell_ratio': round(trade.buy_sell_ratio, 2),
                'edge': round(trade.edge, 4),
                'expiry_time': trade.expiry_time.isoformat() if trade.expiry_time else None,
                'exit_price': trade.exit_price,
                'exit_type': trade.exit_type,
                'btc_price_at_exit': trade.btc_price_at_exit,
                'pnl': round(trade.pnl, 2) if trade.pnl is not None else None,
                'settled': trade.settled
            }

            if existing_idx is not None:
                trades[existing_idx] = trade_record
            else:
                trades.append(trade_record)

            with open(self.trades_log_file, 'w') as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not log trade: {e}")

    def get_strike_price_from_bybit(self) -> Optional[float]:
        """Get strike price from Bybit API - the open price of the current 15-min candle"""
        try:
            now = datetime.now(timezone.utc)
            current_minute = now.minute

            # Find which 15-min period we're in
            for start_min in [0, 15, 30, 45]:
                if current_minute >= start_min and current_minute < start_min + 15:
                    period_start = now.replace(minute=start_min, second=0, microsecond=0)
                    start_timestamp = int(period_start.timestamp() * 1000)

                    url = "https://api.bybit.com/v5/market/mark-price-kline"
                    params = {
                        'category': 'linear',
                        'symbol': 'BTCUSDT',
                        'interval': '15',
                        'start': start_timestamp,
                        'limit': 1
                    }

                    response = requests.get(url, params=params, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('retCode') == 0:
                            kline_list = data.get('result', {}).get('list', [])
                            if kline_list:
                                # kline format: [startTime, open, high, low, close]
                                strike = float(kline_list[0][1])
                                logger.debug(f"Got strike price from Bybit: ${strike:,.2f}")
                                return strike
            return None
        except Exception as e:
            logger.warning(f"Could not get strike price from Bybit: {e}")
            return None

    def get_next_expiry(self, now: datetime) -> datetime:
        """Get the next 15-minute expiry time (00, 15, 30, 45)"""
        minute = now.minute
        if minute < 15:
            next_minute = 15
        elif minute < 30:
            next_minute = 30
        elif minute < 45:
            next_minute = 45
        else:
            next_minute = 0
            now = now + timedelta(hours=1)

        return now.replace(minute=next_minute, second=0, microsecond=0)

    def get_current_period_start(self, now: datetime) -> datetime:
        """Get the start time of the current 15-minute period"""
        minute = now.minute
        if minute < 15:
            period_minute = 0
        elif minute < 30:
            period_minute = 15
        elif minute < 45:
            period_minute = 30
        else:
            period_minute = 45

        return now.replace(minute=period_minute, second=0, microsecond=0)

    def calculate_time_to_expiry(self, now: datetime) -> int:
        """Calculate seconds until next expiry"""
        expiry = self.get_next_expiry(now)
        return int((expiry - now).total_seconds())

    def calculate_delta(self, option_price: float, time_to_expiry_seconds: int) -> float:
        """
        Estimate delta for a binary option.
        Delta peaks at ATM (0.5), lower at ITM/OTM.
        """
        distance_from_atm = abs(option_price - 0.5)
        base_delta = math.exp(-8 * distance_from_atm ** 2)

        minutes_to_expiry = time_to_expiry_seconds / 60
        if minutes_to_expiry > 0:
            time_factor = min(2.0, 15 / max(minutes_to_expiry, 1))
        else:
            time_factor = 2.0

        return base_delta * time_factor

    def calculate_gamma(self, option_price: float, time_to_expiry_seconds: int) -> float:
        """
        Estimate gamma for a binary option.
        Gamma is highest at ATM and near expiry.
        """
        distance_from_atm = abs(option_price - 0.5)
        base_gamma = math.exp(-12 * distance_from_atm ** 2)

        minutes_to_expiry = time_to_expiry_seconds / 60
        if minutes_to_expiry > 0:
            time_factor = min(5.0, (15 / max(minutes_to_expiry, 0.5)) ** 1.5)
        else:
            time_factor = 5.0

        return base_gamma * time_factor

    def read_json_file(self, filepath: Path) -> Optional[Dict]:
        """Safely read a JSON file"""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    last_brace = content.rfind('}')
                    if last_brace != -1:
                        content = content[:last_brace + 1]
                    return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Error reading {filepath}: {e}")
        return None

    def get_option_price(self, option_data: Optional[Dict], side: str = 'ask') -> Optional[float]:
        """
        Safely get option price from order book data.
        Returns None if price is not available.

        side: 'ask' for buying, 'bid' for selling
        """
        if option_data is None:
            return None

        try:
            if side == 'ask':
                best = option_data.get('best_ask')
                if best and best.get('price') is not None:
                    return float(best['price'])
            else:  # bid
                best = option_data.get('best_bid')
                if best and best.get('price') is not None:
                    return float(best['price'])
        except (KeyError, TypeError, ValueError):
            pass

        return None

    def analyze_momentum(self, btc_data: Dict) -> Dict:
        """Analyze BTC momentum from volume and buy/sell data."""
        stats_1s = btc_data.get('stats_1s', {})
        stats_5s = btc_data.get('stats_5s', {})
        stats_30s = btc_data.get('stats_30s', {})
        volume_analysis = btc_data.get('volume_analysis', {})

        volume_ratio_15s = volume_analysis.get('ratio_1s_vs_15s_avg', 1.0)
        volume_ratio_30s = volume_analysis.get('ratio_1s_vs_30s_avg', 1.0)
        is_volume_spike = volume_analysis.get('is_volume_spike', False)

        buy_vol_1s = stats_1s.get('buy_volume', 0)
        sell_vol_1s = stats_1s.get('sell_volume', 0)
        buy_vol_5s = stats_5s.get('buy_volume', 0)
        sell_vol_5s = stats_5s.get('sell_volume', 0)
        buy_vol_30s = stats_30s.get('buy_volume', 0)
        sell_vol_30s = stats_30s.get('sell_volume', 0)

        def safe_ratio(a, b):
            if b == 0:
                return 2.0 if a > 0 else 1.0
            return a / b

        buy_sell_vol_ratio_1s = safe_ratio(buy_vol_1s, sell_vol_1s)
        buy_sell_vol_ratio_5s = safe_ratio(buy_vol_5s, sell_vol_5s)
        buy_sell_vol_ratio_30s = safe_ratio(buy_vol_30s, sell_vol_30s)

        total_vol_1s = buy_vol_1s + sell_vol_1s
        net_pressure_1s = (buy_vol_1s - sell_vol_1s) / total_vol_1s if total_vol_1s > 0 else 0

        total_vol_5s = buy_vol_5s + sell_vol_5s
        net_pressure_5s = (buy_vol_5s - sell_vol_5s) / total_vol_5s if total_vol_5s > 0 else 0

        total_vol_30s = buy_vol_30s + sell_vol_30s
        net_pressure_30s = (buy_vol_30s - sell_vol_30s) / total_vol_30s if total_vol_30s > 0 else 0

        # STRICTER signals - need multiple timeframe confirmation
        bullish_signal = (
            buy_sell_vol_ratio_1s > self.min_buy_sell_imbalance and
            buy_sell_vol_ratio_5s > 1.2 and
            buy_sell_vol_ratio_30s > 1.1 and
            net_pressure_1s > 0.15 and
            net_pressure_5s > 0.1
        )

        bearish_signal = (
            buy_sell_vol_ratio_1s < (1 / self.min_buy_sell_imbalance) and
            buy_sell_vol_ratio_5s < 0.83 and
            buy_sell_vol_ratio_30s < 0.9 and
            net_pressure_1s < -0.15 and
            net_pressure_5s < -0.1
        )

        return {
            'volume_ratio_15s': volume_ratio_15s,
            'volume_ratio_30s': volume_ratio_30s,
            'is_volume_spike': is_volume_spike,
            'buy_sell_vol_ratio_1s': buy_sell_vol_ratio_1s,
            'buy_sell_vol_ratio_5s': buy_sell_vol_ratio_5s,
            'buy_sell_vol_ratio_30s': buy_sell_vol_ratio_30s,
            'net_pressure_1s': net_pressure_1s,
            'net_pressure_5s': net_pressure_5s,
            'net_pressure_30s': net_pressure_30s,
            'bullish_signal': bullish_signal,
            'bearish_signal': bearish_signal,
            'btc_price': btc_data.get('price', 0),
            'timestamp': datetime.now()
        }

    def check_signal_confirmation(self, current_signal: Dict) -> Optional[str]:
        """
        Check if we have N consecutive readings confirming the same signal.
        Returns 'CALL', 'PUT', or None.
        """
        self.signal_buffer.append(current_signal)
        if len(self.signal_buffer) > self.max_signal_buffer:
            self.signal_buffer.pop(0)

        if len(self.signal_buffer) < self.signal_confirmation_readings:
            return None

        # Check last N readings
        recent = self.signal_buffer[-self.signal_confirmation_readings:]

        # All must be bullish
        if all(s['bullish_signal'] for s in recent):
            if any(s['volume_ratio_30s'] >= self.min_volume_spike_ratio for s in recent):
                return 'CALL'

        # All must be bearish
        if all(s['bearish_signal'] for s in recent):
            if any(s['volume_ratio_30s'] >= self.min_volume_spike_ratio for s in recent):
                return 'PUT'

        return None

    def is_cooldown_active(self) -> bool:
        """Check if we're still in cooldown from last trade"""
        if self.last_trade_time is None:
            return False

        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed < self.trade_cooldown_seconds

    def get_cooldown_remaining(self) -> int:
        """Get seconds remaining in cooldown"""
        if self.last_trade_time is None:
            return 0

        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        remaining = self.trade_cooldown_seconds - elapsed
        return max(0, int(remaining))

    def calculate_expected_edge(
        self,
        direction: TradeDirection,
        option_price: float,
        momentum: Dict,
        delta: float,
        time_to_expiry_seconds: int
    ) -> float:
        """Calculate expected edge for a trade."""
        if direction == TradeDirection.CALL:
            momentum_strength = momentum['net_pressure_5s']
            vol_ratio = momentum['buy_sell_vol_ratio_1s']
        else:
            momentum_strength = -momentum['net_pressure_5s']
            vol_ratio = 1 / momentum['buy_sell_vol_ratio_1s'] if momentum['buy_sell_vol_ratio_1s'] > 0 else 1

        volume_spike_factor = min(momentum['volume_ratio_30s'], 3.0) / 3.0

        minutes_to_expiry = time_to_expiry_seconds / 60
        if minutes_to_expiry < 2:
            time_reliability = 0.5
        elif minutes_to_expiry < 5:
            time_reliability = 1.2
        elif minutes_to_expiry < 10:
            time_reliability = 1.0
        else:
            time_reliability = 0.8

        prob_adjustment = (
            momentum_strength * 0.15 +
            (vol_ratio - 1) * 0.05 * volume_spike_factor +
            volume_spike_factor * 0.05
        ) * time_reliability * delta

        estimated_prob = option_price + prob_adjustment
        estimated_prob = max(0.01, min(0.99, estimated_prob))

        edge = estimated_prob - option_price
        return edge

    def calculate_trade_size(
        self,
        edge: float,
        delta: float,
        gamma: float,
        trades_remaining: int,
        time_to_expiry_seconds: int
    ) -> float:
        """Calculate optimal trade size."""
        if edge <= 0 or trades_remaining <= 0:
            return 0

        kelly_fraction = edge * 0.5
        gamma_adjustment = max(0.3, 1 - gamma * 0.2)

        minutes_to_expiry = time_to_expiry_seconds / 60
        if minutes_to_expiry < 2:
            time_adjustment = 0.5
        elif minutes_to_expiry < 5:
            time_adjustment = 0.8
        else:
            time_adjustment = 1.0

        capacity_adjustment = min(1.0, trades_remaining / 6)

        size = (
            self.base_trade_size *
            (1 + kelly_fraction * 2) *
            gamma_adjustment *
            time_adjustment *
            capacity_adjustment
        )

        min_size = 5.0
        max_size = self.base_trade_size * 3

        return max(min_size, min(max_size, size))

    def should_trade(
        self,
        confirmed_signal: str,
        momentum: Dict,
        call_data: Dict,
        put_data: Dict,
        time_to_expiry_seconds: int
    ) -> Optional[Dict]:
        """Determine if we should trade based on confirmed signal."""

        # Don't trade in last 60 seconds (too risky, plus we're exiting)
        if time_to_expiry_seconds < 60:
            logger.debug("Too close to expiry, skipping")
            return None

        # Don't trade in first 2 minutes (let market settle)
        minutes_in_period = (15 * 60 - time_to_expiry_seconds) / 60
        if minutes_in_period < 2:
            logger.debug("Period just started, waiting for market to settle")
            return None

        # Check cooldown
        if self.is_cooldown_active():
            logger.debug(f"Cooldown active: {self.get_cooldown_remaining()}s remaining")
            return None

        # Determine direction from confirmed signal
        if confirmed_signal == 'CALL':
            direction = TradeDirection.CALL
            option_price = self.get_option_price(call_data, 'ask')
        elif confirmed_signal == 'PUT':
            direction = TradeDirection.PUT
            option_price = self.get_option_price(put_data, 'ask')
        else:
            return None

        # Check if we have a valid price
        if option_price is None:
            logger.debug(f"No ask price available for {confirmed_signal}")
            return None

        # Calculate Greeks
        delta = self.calculate_delta(option_price, time_to_expiry_seconds)
        gamma = self.calculate_gamma(option_price, time_to_expiry_seconds)

        # Skip if delta is too low
        if delta < 0.25:
            logger.debug(f"Delta too low: {delta:.3f}")
            return None

        # Skip if option is too far ITM or OTM
        if option_price < 0.15 or option_price > 0.85:
            logger.debug(f"Option too far from ATM: {option_price:.2f}")
            return None

        # Calculate edge
        edge = self.calculate_expected_edge(
            direction, option_price, momentum, delta, time_to_expiry_seconds
        )

        if edge < self.min_edge_threshold:
            logger.debug(f"Edge too low: {edge:.4f} < {self.min_edge_threshold}")
            return None

        # Calculate trade size
        trades_remaining = self.current_period.trades_remaining if self.current_period else self.max_trades_per_period
        size = self.calculate_trade_size(edge, delta, gamma, trades_remaining, time_to_expiry_seconds)

        return {
            'direction': direction,
            'option_price': option_price,
            'size': size,
            'delta': delta,
            'gamma': gamma,
            'edge': edge,
            'volume_ratio': momentum['volume_ratio_30s'],
            'buy_sell_ratio': momentum['buy_sell_vol_ratio_1s'],
            'btc_price': momentum['btc_price']
        }

    def execute_trade(self, trade_params: Dict, time_to_expiry_seconds: int) -> Trade:
        """Execute a trade (simulated)"""
        now = datetime.now()
        self.trade_id_counter += 1

        trade = Trade(
            trade_id=self.trade_id_counter,
            timestamp=now,
            direction=trade_params['direction'],
            entry_price=trade_params['option_price'],
            size=trade_params['size'],
            btc_price_at_entry=trade_params['btc_price'],
            strike_price=self.current_period.strike_price,
            time_to_expiry_seconds=time_to_expiry_seconds,
            delta=trade_params['delta'],
            gamma=trade_params['gamma'],
            volume_ratio=trade_params['volume_ratio'],
            buy_sell_ratio=trade_params['buy_sell_ratio'],
            edge=trade_params['edge'],
            expiry_time=self.current_period.end_time
        )

        # Update cooldown
        self.last_trade_time = now

        # Clear signal buffer after trade
        self.signal_buffer.clear()

        # Log trade
        self.log_trade(trade)

        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 TRADE #{trade.trade_id} EXECUTED: {trade.direction.value}")
        logger.info(f"   💵 Entry Price: ${trade.entry_price:.2f} | Size: ${trade.size:.2f}")
        logger.info(f"   📊 BTC: ${trade.btc_price_at_entry:,.2f} | Strike: ${trade.strike_price:,.2f}")
        logger.info(f"   📈 Delta: {trade.delta:.3f} | Gamma: {trade.gamma:.3f} | Edge: {trade.edge:.4f}")
        logger.info(f"   🔥 Volume Spike: {trade.volume_ratio:.2f}x | B/S Ratio: {trade.buy_sell_ratio:.2f}")
        logger.info(f"   ⏱️  Time to Expiry: {time_to_expiry_seconds}s | Expiry: {trade.expiry_time.strftime('%H:%M')}")
        logger.info(f"   ⏳ Next trade cooldown: {self.trade_cooldown_seconds}s")
        logger.info(f"{'='*70}\n")

        return trade

    def early_exit_positions(self, call_data: Optional[Dict], put_data: Optional[Dict], btc_price: float):
        """
        Exit all open positions at market price in the last N seconds before expiry.
        This locks in profits/losses before the binary settlement.
        """
        if not self.current_period or self.current_period.early_exit_done:
            return

        open_trades = self.current_period.open_trades
        if not open_trades:
            self.current_period.early_exit_done = True
            return

        logger.info(f"\n{'!'*70}")
        logger.info(f"⚡ EARLY EXIT - {len(open_trades)} positions | {self.early_exit_seconds}s before expiry")
        logger.info(f"{'!'*70}")

        for trade in open_trades:
            # Get the bid price (we're selling)
            if trade.direction == TradeDirection.CALL:
                exit_price = self.get_option_price(call_data, 'bid')
            else:
                exit_price = self.get_option_price(put_data, 'bid')

            # If no bid available, estimate based on BTC vs strike
            if exit_price is None:
                if self.current_period.strike_price:
                    # Estimate: if deep ITM, assume ~0.95, if deep OTM assume ~0.05
                    if trade.direction == TradeDirection.CALL:
                        if btc_price > self.current_period.strike_price + 50:
                            exit_price = 0.95
                        elif btc_price < self.current_period.strike_price - 50:
                            exit_price = 0.05
                        else:
                            exit_price = 0.50  # ATM estimate
                    else:  # PUT
                        if btc_price < self.current_period.strike_price - 50:
                            exit_price = 0.95
                        elif btc_price > self.current_period.strike_price + 50:
                            exit_price = 0.05
                        else:
                            exit_price = 0.50
                else:
                    exit_price = 0.50  # Fallback

                logger.warning(f"   ⚠️  No bid for {trade.direction.value}, estimated exit: ${exit_price:.2f}")

            # Calculate PNL
            trade.exit_price = exit_price
            trade.btc_price_at_exit = btc_price
            trade.pnl = (exit_price - trade.entry_price) * trade.size
            trade.settled = True
            trade.exit_type = "early_exit"

            # Update stats
            self.total_trades += 1
            if trade.pnl > 0:
                self.winning_trades += 1
            self.total_pnl += trade.pnl

            # Log updated trade
            self.log_trade(trade)

            result = "✅ WIN" if trade.pnl > 0 else "❌ LOSS"
            logger.info(f"   {result} | Trade #{trade.trade_id} {trade.direction.value}")
            logger.info(f"       Entry: ${trade.entry_price:.2f} → Exit: ${exit_price:.2f} | PnL: ${trade.pnl:+.2f}")

        self.current_period.early_exit_done = True

        # Period summary
        period_pnl = self.current_period.total_pnl
        period_wins = self.current_period.winning_trades
        logger.info(f"\n   📈 Period Summary: {len(self.current_period.trades)} trades | {period_wins} wins | PnL: ${period_pnl:+.2f}")
        if self.total_trades > 0:
            logger.info(f"   📊 Overall: {self.total_trades} trades | {self.winning_trades} wins ({self.winning_trades/self.total_trades*100:.1f}%) | Total PnL: ${self.total_pnl:+.2f}")
        logger.info(f"{'!'*70}\n")

        # Save PNL history
        self.save_pnl_history()

    def settle_period(self, final_btc_price: float):
        """Settle all trades in the current period at expiry (binary 0 or 1)."""
        if not self.current_period or self.current_period.settled:
            return

        # Check if there are unsettled trades (shouldn't be if early exit worked)
        unsettled_trades = [t for t in self.current_period.trades if not t.settled]

        if not unsettled_trades:
            self.current_period.settled = True
            return

        strike = self.current_period.strike_price

        logger.info(f"\n{'#'*70}")
        logger.info(f"📊 SETTLING PERIOD AT EXPIRY: {self.current_period.start_time.strftime('%H:%M')} - {self.current_period.end_time.strftime('%H:%M')}")
        logger.info(f"   Strike: ${strike:,.2f} | Final BTC: ${final_btc_price:,.2f}")
        logger.info(f"   Result: BTC {'>' if final_btc_price > strike else '<'} Strike → {'CALL wins' if final_btc_price > strike else 'PUT wins'}")
        logger.info(f"{'#'*70}")

        for trade in unsettled_trades:
            # Determine if ITM or OTM
            if trade.direction == TradeDirection.CALL:
                is_itm = final_btc_price > strike
            else:
                is_itm = final_btc_price < strike

            trade.exit_price = 1.0 if is_itm else 0.0
            trade.btc_price_at_exit = final_btc_price
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
            trade.settled = True
            trade.exit_type = "expiry"

            self.total_trades += 1
            if trade.pnl > 0:
                self.winning_trades += 1
            self.total_pnl += trade.pnl

            # Update trade log
            self.log_trade(trade)

            result = "✅ WIN" if is_itm else "❌ LOSS"
            logger.info(f"   Trade #{trade.trade_id}: {result} | {trade.direction.value} @ ${trade.entry_price:.2f} | PnL: ${trade.pnl:+.2f}")

        self.current_period.settled = True

        if self.current_period.trades:
            period_pnl = self.current_period.total_pnl
            period_wins = self.current_period.winning_trades
            logger.info(f"\n   📈 Period Summary: {len(self.current_period.trades)} trades | {period_wins} wins | PnL: ${period_pnl:+.2f}")
            if self.total_trades > 0:
                logger.info(f"   📊 Overall: {self.total_trades} trades | {self.winning_trades} wins ({self.winning_trades/self.total_trades*100:.1f}%) | Total PnL: ${self.total_pnl:+.2f}")
        logger.info(f"{'#'*70}\n")

        # Save PNL history
        self.save_pnl_history()

    def check_and_update_period(self, now: datetime, btc_price: float):
        """Check if we need to start a new trading period"""
        period_start = self.get_current_period_start(now)
        period_end = self.get_next_expiry(now)

        # Check if we're in a new period
        if self.current_period is None or period_start != self.current_period.start_time:
            # Settle previous period if exists
            if self.current_period and not self.current_period.settled:
                self.settle_period(btc_price)
                self.all_periods.append(self.current_period)

            # Get strike price from Bybit
            strike_price = self.get_strike_price_from_bybit()
            if strike_price is None:
                # Fallback to current BTC price
                strike_price = btc_price
                logger.warning(f"Could not get strike from Bybit, using current BTC price: ${strike_price:,.2f}")

            # Start new period
            self.current_period = TradingPeriod(
                start_time=period_start,
                end_time=period_end,
                strike_price=strike_price,
                max_trades=self.max_trades_per_period
            )

            # Clear signal buffer for new period
            self.signal_buffer.clear()

            logger.info(f"\n{'='*70}")
            logger.info(f"📅 NEW PERIOD STARTED")
            logger.info(f"   Time: {period_start.strftime('%H:%M')} - {period_end.strftime('%H:%M')}")
            logger.info(f"   Strike Price: ${strike_price:,.2f} (from Bybit)")
            logger.info(f"   Current BTC: ${btc_price:,.2f}")
            logger.info(f"   Max Trades: {self.max_trades_per_period}")
            logger.info(f"{'='*70}\n")

    def run_iteration(self) -> bool:
        """Run one iteration of the trading bot."""
        now = datetime.now()

        # Read data files
        btc_data = self.read_json_file(self.btc_file)
        call_data = self.read_json_file(self.call_file)
        put_data = self.read_json_file(self.put_file)

        if not btc_data:
            return False

        btc_price = btc_data.get('price', 0)
        if btc_price <= 0:
            return False

        # Update period
        self.check_and_update_period(now, btc_price)

        # Store last data
        self.last_btc_data = btc_data
        self.last_call_data = call_data
        self.last_put_data = put_data

        # Calculate time to expiry
        time_to_expiry = self.calculate_time_to_expiry(now)

        # Check for early exit (last N seconds)
        if time_to_expiry <= self.early_exit_seconds and self.current_period and not self.current_period.early_exit_done:
            self.early_exit_positions(call_data, put_data, btc_price)
            return False

        # Check if we can still trade
        if self.current_period.trades_remaining <= 0:
            return False

        # Analyze momentum
        momentum = self.analyze_momentum(btc_data)

        # Check for confirmed signal
        confirmed_signal = self.check_signal_confirmation(momentum)

        if not confirmed_signal:
            return False

        # Need valid option data for trading
        if not call_data or not put_data:
            logger.debug("Missing option data")
            return False

        # Check if we should trade
        trade_params = self.should_trade(confirmed_signal, momentum, call_data, put_data, time_to_expiry)

        if trade_params:
            trade = self.execute_trade(trade_params, time_to_expiry)
            self.current_period.trades.append(trade)
            return True

        return False

    def print_status(self):
        """Print current bot status"""
        now = datetime.now()
        time_to_expiry = self.calculate_time_to_expiry(now)

        print(f"\n{'─'*70}")
        print(f"⏰ {now.strftime('%H:%M:%S')} | Expiry in: {time_to_expiry//60}m {time_to_expiry%60}s", end="")

        if self.is_cooldown_active():
            print(f" | 🔒 Cooldown: {self.get_cooldown_remaining()}s", end="")
        elif time_to_expiry <= self.early_exit_seconds:
            print(f" | 🚪 EXIT WINDOW", end="")
        else:
            print(f" | 🟢 Ready to trade", end="")
        print()

        if self.current_period:
            strike_str = f"${self.current_period.strike_price:,.2f}" if self.current_period.strike_price else "N/A"
            print(f"📅 Period: {self.current_period.start_time.strftime('%H:%M')}-{self.current_period.end_time.strftime('%H:%M')} | Strike: {strike_str}")
            print(f"📈 Trades: {len(self.current_period.trades)}/{self.current_period.max_trades} | Open: {len(self.current_period.open_trades)} | Period PnL: ${self.current_period.total_pnl:+.2f}")

        if self.last_btc_data:
            momentum = self.analyze_momentum(self.last_btc_data)
            print(f"💰 BTC: ${momentum['btc_price']:,.2f}", end="")
            if self.current_period and self.current_period.strike_price:
                diff = momentum['btc_price'] - self.current_period.strike_price
                direction = "CALL↑" if diff > 0 else "PUT↓"
                print(f" ({'+' if diff >= 0 else ''}{diff:,.2f} → {direction})", end="")
            print()

            print(f"📊 Vol: {momentum['volume_ratio_30s']:.2f}x | B/S: {momentum['buy_sell_vol_ratio_1s']:.2f} (1s) {momentum['buy_sell_vol_ratio_5s']:.2f} (5s) {momentum['buy_sell_vol_ratio_30s']:.2f} (30s)")
            print(f"📈 Pressure: {momentum['net_pressure_1s']:+.3f} (1s) | {momentum['net_pressure_5s']:+.3f} (5s) | {momentum['net_pressure_30s']:+.3f} (30s)")

            signal = "🟢 BULLISH" if momentum['bullish_signal'] else "🔴 BEARISH" if momentum['bearish_signal'] else "⚪ NEUTRAL"
            confirmations = sum(1 for s in self.signal_buffer[-self.signal_confirmation_readings:]
                              if s.get('bullish_signal') or s.get('bearish_signal'))
            print(f"🎯 Signal: {signal} | Confirmations: {confirmations}/{self.signal_confirmation_readings}")

        # Safely get option prices
        call_ask = self.get_option_price(self.last_call_data, 'ask')
        call_bid = self.get_option_price(self.last_call_data, 'bid')
        put_ask = self.get_option_price(self.last_put_data, 'ask')
        put_bid = self.get_option_price(self.last_put_data, 'bid')

        call_str = f"${call_ask:.2f}" if call_ask else "N/A"
        put_str = f"${put_ask:.2f}" if put_ask else "N/A"

        if call_ask:
            call_delta = self.calculate_delta(call_ask, time_to_expiry)
            call_str += f" (Δ={call_delta:.2f})"
        if put_ask:
            put_delta = self.calculate_delta(put_ask, time_to_expiry)
            put_str += f" (Δ={put_delta:.2f})"

        print(f"📞 CALL: {call_str} | 📉 PUT: {put_str}")

        if call_bid is not None and put_bid is not None:
            print(f"   Bids: CALL ${call_bid:.2f} | PUT ${put_bid:.2f}")

        # Overall stats
        print(f"{'─'*70}")
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades * 100
            print(f"📊 OVERALL: {self.total_trades} trades | {win_rate:.1f}% wins | PnL: ${self.total_pnl:+.2f}")
        else:
            print(f"📊 OVERALL: No trades yet")
        print(f"{'─'*70}")

    def run(self, interval_seconds: float = 0.5):
        """Main bot loop"""
        logger.info(f"\n{'='*70}")
        logger.info("🚀 BINARY OPTIONS TRADING BOT v3")
        logger.info(f"{'='*70}")
        logger.info(f"   Base Trade Size: ${self.base_trade_size}")
        logger.info(f"   Max Trades/Period: {self.max_trades_per_period}")
        logger.info(f"   Trade Cooldown: {self.trade_cooldown_seconds}s")
        logger.info(f"   Signal Confirmations: {self.signal_confirmation_readings}")
        logger.info(f"   Min Volume Spike: {self.min_volume_spike_ratio}x")
        logger.info(f"   Min B/S Imbalance: {self.min_buy_sell_imbalance}x")
        logger.info(f"   Min Edge: {self.min_edge_threshold}")
        logger.info(f"   Early Exit: {self.early_exit_seconds}s before expiry")
        logger.info(f"   Mode: {'SIMULATION' if self.simulation_mode else 'LIVE'}")
        logger.info(f"{'='*70}\n")

        last_status_time = time.time()
        status_interval = 10  # Print status every 10 seconds

        try:
            while True:
                # Run trading iteration
                self.run_iteration()

                # Print status periodically
                if time.time() - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = time.time()

                # Wait for next iteration
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")

            # Settle current period if has trades
            if self.current_period and not self.current_period.settled:
                if self.last_btc_data:
                    # First try early exit, then settle any remaining
                    if not self.current_period.early_exit_done:
                        self.early_exit_positions(self.last_call_data, self.last_put_data, self.last_btc_data.get('price', 0))
                    self.settle_period(self.last_btc_data.get('price', 0))

            self.print_status()
            self.save_pnl_history()


if __name__ == "__main__":
    bot = BinaryOptionsBot(
        call_file="/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_PUT.json",
        btc_file="binance_BTC_details.json",
        base_trade_size=5.0,
        max_trades_per_period=8,
        min_volume_spike_ratio=2.0,      # Need 2x volume spike
        min_buy_sell_imbalance=1.5,      # Need 1.5x buy/sell imbalance
        min_edge_threshold=0.05,          # Need 5% edge
        trade_cooldown_seconds=45,        # 45s minimum between trades
        signal_confirmation_readings=3,   # Need 3 consecutive confirmations
        early_exit_seconds=10,            # Exit 10s before expiry
        simulation_mode=True
    )

    bot.run(interval_seconds=0.5)
