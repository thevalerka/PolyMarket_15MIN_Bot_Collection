#!/usr/bin/env python3
"""
Choppiness-Based Pullback Trading Simulator for Polymarket Binary Options
- Trade only when choppiness > threshold
- Max positions per side with balance enforcement
- Entry trigger: when ASK price drops below (max_ask - pullback)
- Execution: Enter at BID price for better fills
- Real-time PNL calculation using Binance strike and final prices
- SIMULATION ONLY - No real orders placed
pm2 start bot_choppiness_pullback_bidask.py --cron-restart="00 */6 * * *" --interpreter python3
"""


import sys
import time
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHECK_INTERVAL = 1  # Check every second
CHOPPINESS_THRESHOLD = 20
PULLBACK_AMOUNT = 0.03
MAX_POSITIONS_PER_SIDE = 5

# Market configuration
MARKETS = {
    "BTC": "71674365894582536756414132424821018076679382255498799165083572074620560449536",
}

class Position:
    """Track an open position"""
    def __init__(self, side: str, entry_price: float, size: float, market: str,
                 entry_time: Optional[datetime] = None, choppiness: float = 0.0,
                 time_to_expiry_seconds: int = 0):
        self.side = side  # 'CALL' or 'PUT'
        self.entry_price = entry_price
        self.size = size
        self.market = market
        self.entry_time = entry_time if entry_time else datetime.now(timezone.utc)
        self.choppiness = choppiness
        self.time_to_expiry_seconds = time_to_expiry_seconds
        self.pnl = 0.0
        self.final_value = 0.0

    def __repr__(self):
        return f"Position({self.side}, entry=${self.entry_price:.3f}, size={self.size}, PNL=${self.pnl:.2f})"

    def to_dict(self):
        """Convert position to dictionary for JSON serialization"""
        return {
            'side': self.side,
            'entry_price': self.entry_price,
            'size': self.size,
            'market': self.market,
            'entry_time': self.entry_time.isoformat(),
            'choppiness': self.choppiness,
            'time_to_expiry_seconds': self.time_to_expiry_seconds,
            'pnl': self.pnl,
            'final_value': self.final_value
        }

    @staticmethod
    def from_dict(data: dict):
        """Create position from dictionary"""
        pos = Position(
            side=data['side'],
            entry_price=data['entry_price'],
            size=data['size'],
            market=data['market'],
            entry_time=datetime.fromisoformat(data['entry_time']),
            choppiness=data.get('choppiness', 0.0),
            time_to_expiry_seconds=data.get('time_to_expiry_seconds', 0)
        )
        pos.pnl = data.get('pnl', 0.0)
        pos.final_value = data.get('final_value', 0.0)
        return pos


class RetracementTracker:
    """Track price retracements (swings) for bid and ask prices"""
    def __init__(self, min_retracement: float = 0.04):
        self.min_retracement = min_retracement
        self.reset()

    def reset(self):
        """Reset for new period"""
        self.retracements = []  # List of detected retracements
        self.last_swing_price = None
        self.last_swing_type = None  # 'high' or 'low'
        self.current_high = None
        self.current_low = None

    def add_price(self, price: float, price_type: str):
        """Add new price point and detect retracements"""
        timestamp = datetime.now(timezone.utc)

        # Initialize
        if self.last_swing_price is None:
            self.last_swing_price = price
            self.current_high = price
            self.current_low = price
            return

        # Update current high/low
        if price > self.current_high:
            self.current_high = price
        if price < self.current_low:
            self.current_low = price

        # Check for swing high (price was going up, now going down)
        if self.last_swing_type != 'high' and price < self.current_high:
            # We have a potential swing high at current_high
            if self.last_swing_type == 'low':
                # Calculate retracement from last low to this high
                retracement_width = self.current_high - self.last_swing_price
                if retracement_width >= self.min_retracement:
                    self.retracements.append({
                        'direction': 'up',
                        'from': self.last_swing_price,
                        'to': self.current_high,
                        'width': retracement_width,
                        'price_type': price_type,
                        'timestamp': timestamp.isoformat()
                    })

            self.last_swing_price = self.current_high
            self.last_swing_type = 'high'
            self.current_low = price

        # Check for swing low (price was going down, now going up)
        elif self.last_swing_type != 'low' and price > self.current_low:
            # We have a potential swing low at current_low
            if self.last_swing_type == 'high':
                # Calculate retracement from last high to this low
                retracement_width = self.last_swing_price - self.current_low
                if retracement_width >= self.min_retracement:
                    self.retracements.append({
                        'direction': 'down',
                        'from': self.last_swing_price,
                        'to': self.current_low,
                        'width': retracement_width,
                        'price_type': price_type,
                        'timestamp': timestamp.isoformat()
                    })

            self.last_swing_price = self.current_low
            self.last_swing_type = 'low'
            self.current_high = price

    def get_stats(self):
        """Get retracement statistics"""
        if not self.retracements:
            return {
                'count': 0,
                'total_width': 0.0,
                'avg_width': 0.0,
                'max_width': 0.0,
                'min_width': 0.0,
                'up_count': 0,
                'down_count': 0,
                'retracements': []
            }

        widths = [r['width'] for r in self.retracements]
        up_retracements = [r for r in self.retracements if r['direction'] == 'up']
        down_retracements = [r for r in self.retracements if r['direction'] == 'down']

        return {
            'count': len(self.retracements),
            'total_width': sum(widths),
            'avg_width': sum(widths) / len(widths),
            'max_width': max(widths),
            'min_width': min(widths),
            'up_count': len(up_retracements),
            'down_count': len(down_retracements),
            'retracements': self.retracements
        }


class ChoppinessPullbackBot:
    def __init__(self):
        self.positions: List[Position] = []

        # Track max prices for pullback detection
        self.max_call_ask = 0.0
        self.max_put_ask = 0.0

        # Position counters
        self.call_positions = 0
        self.put_positions = 0

        # Strike price for current period
        self.strike_price: Optional[float] = None
        self.final_price: Optional[float] = None
        self.period_start: Optional[datetime] = None

        # Retracement tracking (minimum 0.04$ retracement)
        self.call_bid_retracements = RetracementTracker(min_retracement=0.04)
        self.call_ask_retracements = RetracementTracker(min_retracement=0.04)
        self.put_bid_retracements = RetracementTracker(min_retracement=0.04)
        self.put_ask_retracements = RetracementTracker(min_retracement=0.04)

        # Data persistence
        self.data_dir = "/home/ubuntu/013_2025_polymarket/bot019_Randomwalk"
        self.trades_dir = f"{self.data_dir}/tradesCBID"
        self.state_file = f"{self.data_dir}/stateCBID.json"
        self.trades_history = []  # All completed trades
        self.state_history = []  # Hourly state snapshots
        self.last_save_time = datetime.now(timezone.utc)

        # Create directories if they don't exist
        import os
        os.makedirs(self.trades_dir, exist_ok=True)

        # Load previous state
        self.load_state()

        logger.info("✓ Bot initialized (SIMULATION MODE - Real Prices)")
        logger.info(f"✓ Reading REAL prices from:")
        logger.info(f"  - CALL: /home/ubuntu/013_2025_polymarket/15M_CALL.json")
        logger.info(f"  - PUT: /home/ubuntu/013_2025_polymarket/15M_PUT.json")
        logger.info(f"  - BTC: /home/ubuntu/013_2025_polymarket/bybit_btc_price.json")
        logger.info(f"✓ State file: {self.state_file}")
        logger.info(f"✓ Trades directory: {self.trades_dir}")
        logger.info(f"✓ Loaded {len(self.trades_history)} historical trades")
        logger.info(f"✓ Choppiness threshold: > {CHOPPINESS_THRESHOLD}")
        logger.info(f"✓ Pullback entry: ${PULLBACK_AMOUNT} below max")
        logger.info(f"✓ Max positions: {MAX_POSITIONS_PER_SIDE} CALL + {MAX_POSITIONS_PER_SIDE} PUT")

    def load_state(self):
        """Load previous state from JSON file"""
        try:
            import json
            import os

            if not os.path.exists(self.state_file):
                logger.info("📂 No previous state file found - starting fresh")
                return

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Restore trades history
            self.trades_history = state.get('trades_history', [])

            # Restore state history
            self.state_history = state.get('state_history', [])

            # Restore current positions (if any)
            if 'current_positions' in state:
                for pos_data in state['current_positions']:
                    pos = Position.from_dict(pos_data)
                    self.positions.append(pos)
                    if pos.side == 'CALL':
                        self.call_positions += 1
                    else:
                        self.put_positions += 1

            # Restore max prices
            self.max_call_ask = state.get('max_call_ask', 0.0)
            self.max_put_ask = state.get('max_put_ask', 0.0)

            logger.info(f"✅ Loaded state from {self.state_file}")

        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def save_state(self, market_data: Optional[Dict] = None):
        """Save current state to JSON file"""
        try:
            import json

            # Prepare state data
            state = {
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'trades_history': self.trades_history,
                'state_history': self.state_history,
                'current_positions': [pos.to_dict() for pos in self.positions],
                'max_call_ask': self.max_call_ask,
                'max_put_ask': self.max_put_ask,
                'call_positions': self.call_positions,
                'put_positions': self.put_positions,
                'strike_price': self.strike_price,
                'final_price': self.final_price,
                'period_start': self.period_start.isoformat() if self.period_start else None
            }

            # Add current market data snapshot if provided
            if market_data:
                state['last_market_snapshot'] = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'btc_price': market_data.get('btc_price'),
                    'call_ask': market_data.get('call_ask'),
                    'put_ask': market_data.get('put_ask'),
                    'choppiness': market_data.get('choppiness')
                }

            # Write to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            self.last_save_time = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def save_period_trades(self, period_data: Dict):
        """Save trades from completed period to separate JSON file"""
        try:
            import json
            from datetime import datetime

            # Create filename based on period start time
            period_start_str = period_data['period_start'].replace(':', '-').replace('.', '-')
            filename = f"{self.trades_dir}/trades_{period_start_str}.json"

            with open(filename, 'w') as f:
                json.dump(period_data, f, indent=2)

            logger.info(f"💾 Period trades saved to: {filename}")

        except Exception as e:
            logger.error(f"Error saving period trades: {e}")

    def save_hourly_snapshot(self, market_data: Dict):
        """Save hourly state snapshot"""
        try:
            snapshot = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'btc_price': market_data.get('btc_price'),
                'call_ask': market_data.get('call_ask'),
                'put_ask': market_data.get('put_ask'),
                'choppiness': market_data.get('choppiness'),
                'strike_price': self.strike_price,
                'call_positions': self.call_positions,
                'put_positions': self.put_positions,
                'total_pnl': sum(p.pnl for p in self.positions),
                'retracements_summary': {
                    'call_bid': self.call_bid_retracements.get_stats()['count'],
                    'call_ask': self.call_ask_retracements.get_stats()['count'],
                    'put_bid': self.put_bid_retracements.get_stats()['count'],
                    'put_ask': self.put_ask_retracements.get_stats()['count']
                }
            }

            self.state_history.append(snapshot)

            # Keep only last 30 days (24 * 30 = 720 hourly snapshots)
            if len(self.state_history) > 720:
                self.state_history = self.state_history[-720:]

        except Exception as e:
            logger.error(f"Error saving hourly snapshot: {e}")

    def get_time_to_expiry(self) -> int:
        """Calculate time to expiry in seconds from now until period end"""
        if not self.period_start:
            return 0

        from datetime import timedelta
        # Period ends 15 minutes after it starts
        period_end = self.period_start + timedelta(minutes=15)
        now = datetime.now(timezone.utc)

        time_remaining = (period_end - now).total_seconds()
        return max(0, int(time_remaining))

    def force_save_period(self, final_price: float):
        """Force save previous period when new period starts"""
        if not self.strike_price or len(self.positions) == 0:
            return

        print()
        logger.info("=" * 80)
        logger.info("📊 FORCING PERIOD SAVE (new period detected)")
        logger.info(f"Strike (OPEN): ${self.strike_price:,.2f}")
        logger.info(f"Final (CLOSE = Next OPEN): ${final_price:,.2f}")
        logger.info(f"Price movement: ${final_price - self.strike_price:+,.2f}")

        # Calculate PNL
        for position in self.positions:
            if position.side == 'CALL':
                position.final_value = 1.0 if final_price > self.strike_price else 0.0
            else:  # PUT
                position.final_value = 1.0 if final_price < self.strike_price else 0.0
            position.pnl = (position.final_value - position.entry_price) * position.size

        # Save period data
        period_data = {
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': datetime.now(timezone.utc).isoformat(),
            'strike_price': float(self.strike_price),
            'final_price': float(final_price),
            'price_movement': float(final_price - self.strike_price),
            'calls_won': final_price > self.strike_price,
            'puts_won': final_price < self.strike_price,
            'retracements': {
                'call_bid': self.call_bid_retracements.get_stats(),
                'call_ask': self.call_ask_retracements.get_stats(),
                'put_bid': self.put_bid_retracements.get_stats(),
                'put_ask': self.put_ask_retracements.get_stats()
            },
            'trades': [],
            'summary': {'total_trades': len(self.positions), 'call_trades': 0, 'put_trades': 0, 'total_pnl': 0.0, 'call_pnl': 0.0, 'put_pnl': 0.0}
        }

        # Add trades
        total_pnl = call_pnl = put_pnl = 0.0
        for pos in self.positions:
            trade_record = {
                'entry_time': pos.entry_time.isoformat(),
                'side': pos.side,
                'entry_price': pos.entry_price,
                'size': pos.size,
                'choppiness': pos.choppiness,
                'time_to_expiry_seconds': pos.time_to_expiry_seconds,
                'strike_price': self.strike_price,
                'final_price': final_price,
                'final_value': pos.final_value,
                'pnl': pos.pnl,
                'won': pos.pnl > 0
            }
            period_data['trades'].append(trade_record)
            self.trades_history.append(trade_record)
            total_pnl += pos.pnl
            if pos.side == 'CALL':
                call_pnl += pos.pnl
                period_data['summary']['call_trades'] += 1
            else:
                put_pnl += pos.pnl
                period_data['summary']['put_trades'] += 1

        period_data['summary']['total_pnl'] = total_pnl
        period_data['summary']['call_pnl'] = call_pnl
        period_data['summary']['put_pnl'] = put_pnl

        logger.info(f"💰 Total PNL: ${total_pnl:+.2f}")
        logger.info("=" * 80)
        print()

        # Save files
        self.save_period_trades(period_data)
        self.save_state()

        # Reset
        self.positions = []
        self.call_positions = 0
        self.put_positions = 0
        self.max_call_ask = 0.0
        self.max_put_ask = 0.0
        self.call_bid_retracements.reset()
        self.call_ask_retracements.reset()
        self.put_bid_retracements.reset()
        self.put_ask_retracements.reset()

    def get_strike_price(self) -> Optional[float]:
        """Get strike price from Binance API for current 15-min period"""
        try:
            now = datetime.now(timezone.utc)
            current_minute = now.minute

            for start_min in [0, 15, 30, 45]:
                if current_minute >= start_min and current_minute < start_min + 15:
                    period_start = now.replace(minute=start_min, second=0, microsecond=0)
                    start_timestamp = int(period_start.timestamp() * 1000)

                    time.sleep(5)

                    url = "https://api.binance.com/api/v3/klines"
                    params = {
                        'symbol': 'BTCUSDT',
                        'interval': '15m',
                        'startTime': start_timestamp,
                        'limit': 1,
                        'timeZone': '0'  # UTC
                    }

                    response = requests.get(url, params=params, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            # Binance kline format: [openTime, open, high, low, close, ...]
                            strike = float(data[0][1])  # Index 1 is open price


                            # Store period start if this is new period
                            if self.period_start != period_start:
                                self.period_start = period_start
                                logger.info(f"📍 New period started at {period_start.strftime('%H:%M:%S')} UTC")
                                logger.info(f"📍 Strike price: ${strike:,.2f}")

                            return strike

            return None
        except Exception as e:
            logger.error(f"Error getting strike price: {e}")
            return None

    def get_final_price(self) -> Optional[float]:
        """Get final (close) price from Binance for current period
        The close of current period = open of next period = final settlement price"""
        try:
            if not self.period_start:
                return None

            start_timestamp = int(self.period_start.timestamp() * 1000)

            time.sleep(5)

            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '15m',
                'startTime': start_timestamp,
                'limit': 1,
                'timeZone': '0'
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Check if candle is closed
                    close_time = data[0][6]  # Close time in ms
                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                    if now_ms >= close_time:
                        # Candle is closed, get the close price
                        # Close price of current period = Open price of next period
                        final = float(data[0][4])  # Index 4 is close price
                        return final

            return None
        except Exception as e:
            logger.error(f"Error getting final price: {e}")
            return None

    def verify_final_price_from_next_candle(self) -> Optional[float]:
        """Verify final price by getting OPEN of next 15min candle
        This is more accurate: close of current = open of next"""
        try:
            if not self.period_start:
                return None

            # Get next period's timestamp (current + 15 min)
            from datetime import timedelta
            next_period_start = self.period_start + timedelta(minutes=15)
            next_start_timestamp = int(next_period_start.timestamp() * 1000)

            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '15m',
                'startTime': next_start_timestamp,
                'limit': 1,
                'timeZone': '0'
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Get OPEN of next candle (this is the close of previous candle)
                    next_open = float(data[0][1])  # Index 1 is open price

                    # Verify this candle exists and has started
                    open_time = data[0][0]  # Open time in ms
                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                    if now_ms >= open_time:
                        return next_open

            return None
        except Exception as e:
            logger.error(f"Error verifying final price: {e}")
            return None

    def calculate_choppiness(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Choppiness Index"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return 0.0

        import math

        highs = highs[-period:]
        lows = lows[-period:]
        closes = closes[-period:]

        high_max = max(highs)
        low_min = min(lows)

        atr_sum = 0.0
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            atr_sum += true_range

        if high_max - low_min == 0:
            return 0.0

        chop = 100 * math.log10(atr_sum / (high_max - low_min)) / math.log10(period)
        return chop

    def get_market_data(self, market: str) -> Optional[Dict]:
        """Get real prices from local JSON files - both bid and ask"""
        try:
            import json

            # Read CALL prices (both bid and ask)
            try:
                with open('/home/ubuntu/013_2025_polymarket/15M_CALL.json', 'r') as f:
                    call_data = json.load(f)
                    call_ask = call_data.get('best_ask', {}).get('price') if call_data.get('best_ask') else None
                    call_bid = call_data.get('best_bid', {}).get('price') if call_data.get('best_bid') else None
            except:
                call_ask = None
                call_bid = None

            # Read PUT prices (both bid and ask)
            try:
                with open('/home/ubuntu/013_2025_polymarket/15M_PUT.json', 'r') as f:
                    put_data = json.load(f)
                    put_ask = put_data.get('best_ask', {}).get('price') if put_data.get('best_ask') else None
                    put_bid = put_data.get('best_bid', {}).get('price') if put_data.get('best_bid') else None
            except:
                put_ask = None
                put_bid = None

            # Read BTC price
            try:
                with open('/home/ubuntu/013_2025_polymarket/bybit_btc_price.json', 'r') as f:
                    btc_data = json.load(f)
                    btc_price = btc_data.get('price')
            except:
                btc_price = None

            if call_ask is None or put_ask is None or btc_price is None:
                return None

            # Get historical data for choppiness from Binance
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'limit': 30  # Need at least 14 for choppiness
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return None

            klines = response.json()
            if len(klines) < 14:
                return None

            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]

            choppiness = self.calculate_choppiness(highs, lows, closes)

            return {
                'call_ask': float(call_ask) if call_ask else None,
                'call_bid': float(call_bid) if call_bid else None,
                'put_ask': float(put_ask) if put_ask else None,
                'put_bid': float(put_bid) if put_bid else None,
                'choppiness': choppiness,
                'spread': abs(float(call_ask) - float(put_ask)) if (call_ask and put_ask) else 0,
                'btc_price': float(btc_price)
            }

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    def open_position(self, market: str, side: str, price: float, size: float = 5.0, choppiness: float = 0.0) -> bool:
        """Simulate opening a position with balance enforcement"""
        try:
            position_side = 'CALL' if side == 'BUY' else 'PUT'

            # Check balance constraint: abs(num_calls - num_puts) <= 1
            if position_side == 'CALL':
                new_call_count = self.call_positions + 1
                if abs(new_call_count - self.put_positions) > 1:
                    logger.info(f"⚠️  CALL rejected: would create imbalance ({new_call_count}C vs {self.put_positions}P)")
                    return False
            else:  # PUT
                new_put_count = self.put_positions + 1
                if abs(self.call_positions - new_put_count) > 1:
                    logger.info(f"⚠️  PUT rejected: would create imbalance ({self.call_positions}C vs {new_put_count}P)")
                    return False

            # Calculate time to expiry
            time_to_expiry = self.get_time_to_expiry()

            logger.info(f"✅ SIMULATED ORDER: {side} @ ${price:.3f}, size={size}, chop={choppiness:.1f}, TTL={time_to_expiry}s")

            # Create position with choppiness and time to expiry
            position = Position(position_side, price, size, market,
                              choppiness=choppiness,
                              time_to_expiry_seconds=time_to_expiry)
            self.positions.append(position)

            # Update counters
            if position_side == 'CALL':
                self.call_positions += 1
                self.max_call_ask = 0.0  # Reset max after opening position
                logger.info(f"🔵 CALL position opened. Balance: {self.call_positions}C/{self.put_positions}P")
            else:
                self.put_positions += 1
                self.max_put_ask = 0.0  # Reset max after opening position
                logger.info(f"🔴 PUT position opened. Balance: {self.call_positions}C/{self.put_positions}P")

            return True

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False

    def update_pnl(self):
        """Update PNL for all positions based on strike and final prices
        CALL wins (ITM) if final > strike → final_value = $1, else $0
        PUT wins (ITM) if final < strike → final_value = $1, else $0
        PNL = (final_value - entry_price) * size"""
        if not self.strike_price:
            return

        # Try to get final price (verified from next candle's open)
        final = self.verify_final_price_from_next_candle()

        # Fallback to current candle's close if next candle not available yet
        if final is None:
            final = self.get_final_price()

        for position in self.positions:
            if final is not None:
                # Period is closed, calculate final PNL
                if position.side == 'CALL':
                    # CALL is ITM (wins) if final > strike
                    if final > self.strike_price:
                        position.final_value = 1.0  # Worth $1
                    else:
                        position.final_value = 0.0  # Worth $0
                else:  # PUT
                    # PUT is ITM (wins) if final < strike
                    if final < self.strike_price:
                        position.final_value = 1.0  # Worth $1
                    else:
                        position.final_value = 0.0  # Worth $0

                # PNL = (final_value - entry_price) * size
                position.pnl = (position.final_value - position.entry_price) * position.size
            else:
                # Period still open, PNL is pending
                position.pnl = 0.0
                position.final_value = 0.0

    def display_status(self, market_data: Dict):
        """Display real-time status"""
        # Calculate total PNL
        total_pnl = sum(p.pnl for p in self.positions)

        # Status line
        status = f"\r⏱️  "
        status += f"BTC: ${market_data['btc_price']:,.2f} "
        status += f"| Chop: {market_data['choppiness']:.1f} "

        # Show CALL bid/ask
        call_bid = market_data.get('call_bid')
        call_ask = market_data.get('call_ask')
        if call_bid and call_ask:
            status += f"| CALL: ${call_bid:.3f}/${call_ask:.3f} (max: ${self.max_call_ask:.3f}) "
        elif call_ask:
            status += f"| CALL ask: ${call_ask:.3f} (max: ${self.max_call_ask:.3f}) "

        # Show PUT bid/ask
        put_bid = market_data.get('put_bid')
        put_ask = market_data.get('put_ask')
        if put_bid and put_ask:
            status += f"| PUT: ${put_bid:.3f}/${put_ask:.3f} (max: ${self.max_put_ask:.3f}) "
        elif put_ask:
            status += f"| PUT ask: ${put_ask:.3f} (max: ${self.max_put_ask:.3f}) "

        status += f"| Balance: {self.call_positions}C/{self.put_positions}P "
        status += f"| PNL: ${total_pnl:.2f}"

        if self.strike_price:
            status += f" | Strike: ${self.strike_price:,.2f}"
        if self.final_price:
            status += f" | Final: ${self.final_price:,.2f}"

        # Show time to expiry
        ttl = self.get_time_to_expiry()
        if ttl > 0:
            status += f" | TTL: {ttl}s"

        print(status, end='', flush=True)

    def check_period_end(self):
        """Check if current period has ended and log results"""
        # First try verified final price (next candle's open)
        final = self.verify_final_price_from_next_candle()

        # Fallback to current candle's close
        if final is None:
            final = self.get_final_price()

        # Only proceed if we have a final price AND it's different from last time
        # OR if we have positions but haven't saved yet (period just ended)
        if final is not None and final != self.final_price:
            self.final_price = final

            # Check if we have a strike price (should always have one, but safety check)
            if self.strike_price is None:
                logger.warning("⚠️  Period ended but no strike price recorded. Skipping.")
                return

            print()  # New line after status updates
            logger.info("=" * 80)
            logger.info("📊 PERIOD ENDED")
            logger.info(f"Period start: {self.period_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info(f"Strike (OPEN): ${self.strike_price:,.2f}")
            logger.info(f"Final (CLOSE = Next OPEN): ${final:,.2f}")
            logger.info(f"Price movement: ${final - self.strike_price:+,.2f}")

            if final > self.strike_price:
                logger.info("✅ CALLs are ITM (Final > Strike) → CALL worth $1, PUT worth $0")
            elif final < self.strike_price:
                logger.info("✅ PUTs are ITM (Final < Strike) → PUT worth $1, CALL worth $0")
            else:
                logger.info("⚖️  At Strike (Final = Strike) → Both worth $0.50")

            # CRITICAL: Calculate PNL for all positions now that we have final price
            for position in self.positions:
                if position.side == 'CALL':
                    # CALL is ITM if final > strike → worth $1, else $0
                    if final > self.strike_price:
                        position.final_value = 1.0
                    else:
                        position.final_value = 0.0
                else:  # PUT
                    # PUT is ITM if final < strike → worth $1, else $0
                    if final < self.strike_price:
                        position.final_value = 1.0
                    else:
                        position.final_value = 0.0

                # PNL = (final_value - entry_price) * size
                position.pnl = (position.final_value - position.entry_price) * position.size

            # Prepare period data for saving
            period_data = {
                'period_start': self.period_start.isoformat() if self.period_start else None,
                'period_end': datetime.now(timezone.utc).isoformat(),
                'strike_price': float(self.strike_price) if self.strike_price is not None else None,
                'final_price': float(final) if final is not None else None,
                'price_movement': float(final - self.strike_price) if (final and self.strike_price) else 0.0,
                'calls_won': final > self.strike_price if (final and self.strike_price) else False,
                'puts_won': final < self.strike_price if (final and self.strike_price) else False,
                'retracements': {
                    'call_bid': self.call_bid_retracements.get_stats(),
                    'call_ask': self.call_ask_retracements.get_stats(),
                    'put_bid': self.put_bid_retracements.get_stats(),
                    'put_ask': self.put_ask_retracements.get_stats()
                },
                'trades': [],
                'summary': {
                    'total_trades': len(self.positions),
                    'call_trades': 0,
                    'put_trades': 0,
                    'total_pnl': 0.0,
                    'call_pnl': 0.0,
                    'put_pnl': 0.0
                }
            }

            # Show all positions and their PNL
            logger.info("-" * 80)

            if len(self.positions) == 0:
                logger.info("No positions were opened this period")
                logger.info("-" * 80)
                logger.info("💰 Period PNL: $0.00")
                logger.info("=" * 80)
                print()

                # Still save the period data (empty trades)
                self.save_period_trades(period_data)
                self.save_state()

                # Reset for next period
                self.positions = []
                self.call_positions = 0
                self.put_positions = 0
                self.max_call_ask = 0.0
                self.max_put_ask = 0.0
                self.strike_price = None
                self.final_price = None
                return

            total_pnl = 0.0
            call_pnl = 0.0
            put_pnl = 0.0

            for i, pos in enumerate(self.positions, 1):
                logger.info(f"Position {i}: {pos.side} entry=${pos.entry_price:.3f} size={pos.size} "
                          f"chop={pos.choppiness:.1f} TTL={pos.time_to_expiry_seconds}s → "
                          f"final=${pos.final_value:.2f} → PNL=${pos.pnl:+.2f}")

                total_pnl += pos.pnl

                if pos.side == 'CALL':
                    call_pnl += pos.pnl
                    period_data['summary']['call_trades'] += 1
                else:
                    put_pnl += pos.pnl
                    period_data['summary']['put_trades'] += 1

                # Save trade to history and period data
                trade_record = {
                    'entry_time': pos.entry_time.isoformat(),
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'size': pos.size,
                    'choppiness': pos.choppiness,
                    'time_to_expiry_seconds': pos.time_to_expiry_seconds,
                    'strike_price': self.strike_price,
                    'final_price': final,
                    'final_value': pos.final_value,
                    'pnl': pos.pnl,
                    'won': pos.pnl > 0
                }
                self.trades_history.append(trade_record)
                period_data['trades'].append(trade_record)

            period_data['summary']['total_pnl'] = total_pnl
            period_data['summary']['call_pnl'] = call_pnl
            period_data['summary']['put_pnl'] = put_pnl

            logger.info("-" * 80)
            logger.info(f"💰 Period PNL Breakdown:")
            logger.info(f"   CALL PNL: ${call_pnl:+.2f} ({period_data['summary']['call_trades']} positions)")
            logger.info(f"   PUT PNL:  ${put_pnl:+.2f} ({period_data['summary']['put_trades']} positions)")
            logger.info(f"   TOTAL:    ${total_pnl:+.2f}")
            logger.info("-" * 80)
            logger.info(f"📊 Retracement Statistics:")

            # Log retracement stats
            for price_type, stats in period_data['retracements'].items():
                if stats['count'] > 0:
                    logger.info(f"   {price_type.upper()}: {stats['count']} retracements, "
                              f"avg width: ${stats['avg_width']:.3f}, max: ${stats['max_width']:.3f}")
                else:
                    logger.info(f"   {price_type.upper()}: No retracements detected")

            logger.info("-" * 80)
            logger.info(f"📝 Saved {len(self.positions)} trades to history")
            logger.info("=" * 80)
            print()

            # Save period trades to separate file
            self.save_period_trades(period_data)

            # Save state after period ends
            self.save_state()

            # Reset for next period
            self.positions = []
            self.call_positions = 0
            self.put_positions = 0
            self.max_call_ask = 0.0
            self.max_put_ask = 0.0
            self.strike_price = None
            self.final_price = None

            # Reset retracement trackers for new period
            self.call_bid_retracements.reset()
            self.call_ask_retracements.reset()
            self.put_bid_retracements.reset()
            self.put_ask_retracements.reset()

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting trading bot...")
        logger.info("=" * 80)

        market = MARKETS["BTC"]

        while True:
            try:
                # Update strike price
                self.strike_price = self.get_strike_price()

                # Check if period ended (this will use the retracement data we just collected)
                self.check_period_end()



                # Get market data
                market_data = self.get_market_data(market)

                if not market_data:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Track retracements for bid/ask prices (BEFORE checking period end)
                if market_data.get('call_bid') is not None:
                    self.call_bid_retracements.add_price(market_data['call_bid'], 'call_bid')
                if market_data.get('call_ask') is not None:
                    self.call_ask_retracements.add_price(market_data['call_ask'], 'call_ask')
                if market_data.get('put_bid') is not None:
                    self.put_bid_retracements.add_price(market_data['put_bid'], 'put_bid')
                if market_data.get('put_ask') is not None:
                    self.put_ask_retracements.add_price(market_data['put_ask'], 'put_ask')

                # Update PNL
                self.update_pnl()

                # Display status every second
                self.display_status(market_data)

                # Save state hourly
                now = datetime.now(timezone.utc)
                if (now - self.last_save_time).total_seconds() >= 3600:  # 1 hour
                    self.save_hourly_snapshot(market_data)
                    self.save_state(market_data)
                    logger.info(f"\n💾 Hourly state saved at {now.strftime('%H:%M:%S')} UTC")

                choppiness = market_data['choppiness']

                # Only trade if choppiness > 30
                if choppiness <= CHOPPINESS_THRESHOLD:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Get ASK prices for tracking max and detecting pullback
                call_ask = market_data['call_ask']
                put_ask = market_data['put_ask']

                # Get BID prices for actual entry
                call_bid = market_data.get('call_bid')
                put_bid = market_data.get('put_bid')

                # Update max prices using ASK prices
                if call_ask and call_ask > self.max_call_ask:
                    self.max_call_ask = call_ask

                if put_ask and put_ask > self.max_put_ask:
                    self.max_put_ask = put_ask

                # Check for CALL entry (ASK dropped from max, but enter at BID)
                if self.call_positions < MAX_POSITIONS_PER_SIDE:
                    pullback_weighted = PULLBACK_AMOUNT*choppiness/10
                    if self.max_call_ask > 0 and call_ask and call_ask <= (self.max_call_ask - pullback_weighted):
                        # Entry condition met based on ASK price pullback
                        # But enter at BID price for better execution
                        entry_price = call_bid if call_bid is not None else call_ask
                        print()  # New line before order log
                        logger.info(f"🔵 CALL entry signal: ASK ${call_ask:.3f} is ${pullback_weighted:.3f} below max ${self.max_call_ask:.3f}")
                        logger.info(f"🔵 Entering at BID: ${entry_price:.3f}")
                        if self.open_position(market, 'BUY', entry_price, choppiness=choppiness):
                            logger.info(f"🔵 CALL opened. Resetting max_call_ask. Next entry on new high -${pullback_weighted:.3f}")

                # Check for PUT entry (ASK dropped from max, but enter at BID)
                if self.put_positions < MAX_POSITIONS_PER_SIDE:
                    pullback_weighted = PULLBACK_AMOUNT*choppiness/10
                    if self.max_put_ask > 0 and put_ask and put_ask <= (self.max_put_ask - pullback_weighted):
                        # Entry condition met based on ASK price pullback
                        # But enter at BID price for better execution
                        entry_price = put_bid if put_bid is not None else put_ask
                        print()  # New line before order log
                        logger.info(f"🔴 PUT entry signal: ASK ${put_ask:.3f} is ${pullback_weighted:.3f} below max ${self.max_put_ask:.3f}")
                        logger.info(f"🔴 Entering at BID: ${entry_price:.3f}")
                        if self.open_position(market, 'SELL', entry_price, choppiness=choppiness):
                            logger.info(f"🔴 PUT opened. Resetting max_put_ask. Next entry on new high -${pullback_weighted:.3f}")

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user")
                logger.info("💾 Saving final state before shutdown...")
                self.save_state(market_data if 'market_data' in locals() else None)
                logger.info("✅ State saved successfully")
                break
            except Exception as e:
                print()  # New line after status line
                logger.error(f"Error in main loop: {e}")
                time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    bot = ChoppinessPullbackBot()
    bot.run()
