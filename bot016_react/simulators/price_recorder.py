import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import copy

@dataclass
class PriceSnapshot:
    """Record of prices at a specific moment"""
    timestamp: str
    timestamp_ms: int
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    call_mid: float
    put_mid: float
    spread_total: float
    seconds_to_expiry: float

    # Derived metrics
    price_sum: float  # call_mid + put_mid (should be ~1.0)
    arbitrage_opportunity: float  # deviation from 1.0

@dataclass
class OscillationMetrics:
    """Metrics describing price oscillation"""
    period_start: str
    period_end: str

    # Price statistics
    call_min: float
    call_max: float
    call_range: float
    call_volatility: float

    put_min: float
    put_max: float
    put_range: float
    put_volatility: float

    # Oscillation patterns
    num_reversals: int  # How many times direction changed
    avg_swing_size: float  # Average size of price swings
    max_swing_size: float  # Largest single swing

    # Mean reversion
    mean_reversion_speed: float  # How fast prices return to 0.5
    time_above_60: float  # % of time CALL was above 0.6
    time_below_40: float  # % of time CALL was below 0.4

    # Opportunity analysis
    best_buy_and_hold_call: float  # Best result from buying and holding CALL
    best_buy_and_hold_put: float  # Best result from buying and holding PUT
    theoretical_max_swing_trading: float  # If you caught every swing perfectly

class PriceRecorder:
    """Records all price data for retroactive strategy analysis"""

    def __init__(self, call_file: str, put_file: str, output_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/simulators/price_analysis"):
        self.call_file = Path(call_file)
        self.put_file = Path(put_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track last modified times
        self.last_call_mtime = 0
        self.last_put_mtime = 0

        # Current period tracking
        self.current_period_start = None
        self.current_period_snapshots: List[PriceSnapshot] = []

        # Cache for current prices
        self.current_call_bid = None
        self.current_call_ask = None
        self.current_put_bid = None
        self.current_put_ask = None

    def get_current_period_start(self) -> datetime:
        """Get the start time of the current 15-minute period"""
        now = datetime.now()
        minute = now.minute

        if minute < 15:
            period_minute = 0
        elif minute < 30:
            period_minute = 15
        elif minute < 45:
            period_minute = 30
        else:
            period_minute = 45

        period_start = now.replace(minute=period_minute, second=0, microsecond=0)
        return period_start

    def get_next_expiry(self, from_time: datetime = None) -> datetime:
        """Calculate the next 15-minute expiry time"""
        if from_time is None:
            from_time = datetime.now()

        minute = from_time.minute

        if minute < 15:
            next_minute = 15
        elif minute < 30:
            next_minute = 30
        elif minute < 45:
            next_minute = 45
        else:
            next_minute = 0

        expiry = from_time.replace(second=0, microsecond=0)

        if next_minute == 0:
            expiry = expiry + timedelta(hours=1)
            expiry = expiry.replace(minute=0)
        else:
            expiry = expiry.replace(minute=next_minute)

        return expiry

    def read_price_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Read price data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            ask_price = data.get('best_ask')
            bid_price = data.get('best_bid')

            if ask_price is not None:
                ask_price = float(ask_price)
            if bid_price is not None:
                bid_price = float(bid_price)

            return {
                'ask': ask_price,
                'bid': bid_price,
                'timestamp': data.get('timestamp'),
                'timestamp_readable': data.get('timestamp_readable')
            }
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def record_snapshot(self):
        """Record current price snapshot"""
        if None in [self.current_call_bid, self.current_call_ask,
                    self.current_put_bid, self.current_put_ask]:
            return

        now = datetime.now()
        expiry = self.get_next_expiry(now)
        seconds_to_expiry = (expiry - now).total_seconds()

        call_mid = (self.current_call_bid + self.current_call_ask) / 2
        put_mid = (self.current_put_bid + self.current_put_ask) / 2

        snapshot = PriceSnapshot(
            timestamp=now.isoformat(),
            timestamp_ms=int(now.timestamp() * 1000),
            call_bid=self.current_call_bid,
            call_ask=self.current_call_ask,
            put_bid=self.current_put_bid,
            put_ask=self.current_put_ask,
            call_mid=call_mid,
            put_mid=put_mid,
            spread_total=(self.current_call_ask - self.current_call_bid) +
                        (self.current_put_ask - self.current_put_bid),
            seconds_to_expiry=seconds_to_expiry,
            price_sum=call_mid + put_mid,
            arbitrage_opportunity=abs((call_mid + put_mid) - 1.0)
        )

        self.current_period_snapshots.append(snapshot)

    def calculate_oscillation_metrics(self, snapshots: List[PriceSnapshot]) -> OscillationMetrics:
        """Calculate comprehensive oscillation metrics from price snapshots"""
        if len(snapshots) < 2:
            return None

        # Extract price series
        call_mids = [s.call_mid for s in snapshots]
        put_mids = [s.put_mid for s in snapshots]

        # Basic statistics
        call_min = min(call_mids)
        call_max = max(call_mids)
        put_min = min(put_mids)
        put_max = max(put_mids)

        # Volatility (standard deviation)
        call_volatility = statistics.stdev(call_mids) if len(call_mids) > 1 else 0
        put_volatility = statistics.stdev(put_mids) if len(put_mids) > 1 else 0

        # Count reversals (direction changes)
        reversals = 0
        swings = []

        for i in range(2, len(call_mids)):
            prev_change = call_mids[i-1] - call_mids[i-2]
            curr_change = call_mids[i] - call_mids[i-1]

            if prev_change * curr_change < 0:  # Sign change = reversal
                reversals += 1
                swings.append(abs(curr_change))

        avg_swing = statistics.mean(swings) if swings else 0
        max_swing = max(swings) if swings else 0

        # Mean reversion analysis
        deviations = [abs(c - 0.5) for c in call_mids]
        mean_reversion_speed = statistics.mean(deviations) if deviations else 0

        time_above_60 = sum(1 for c in call_mids if c > 0.6) / len(call_mids)
        time_below_40 = sum(1 for c in call_mids if c < 0.4) / len(call_mids)

        # Theoretical maximum profits
        best_call_buy_hold = call_max - call_min
        best_put_buy_hold = put_max - put_min

        # Theoretical max from swing trading (catch every peak and trough)
        theoretical_max = 0
        for i in range(1, len(call_mids)):
            theoretical_max += abs(call_mids[i] - call_mids[i-1])

        return OscillationMetrics(
            period_start=snapshots[0].timestamp,
            period_end=snapshots[-1].timestamp,
            call_min=call_min,
            call_max=call_max,
            call_range=call_max - call_min,
            call_volatility=call_volatility,
            put_min=put_min,
            put_max=put_max,
            put_range=put_max - put_min,
            put_volatility=put_volatility,
            num_reversals=reversals,
            avg_swing_size=avg_swing,
            max_swing_size=max_swing,
            mean_reversion_speed=mean_reversion_speed,
            time_above_60=time_above_60,
            time_below_40=time_below_40,
            best_buy_and_hold_call=best_call_buy_hold,
            best_buy_and_hold_put=best_put_buy_hold,
            theoretical_max_swing_trading=theoretical_max
        )

    def backtest_strategies(self, snapshots: List[PriceSnapshot]) -> Dict[str, Any]:
        """Retroactively test multiple strategies on recorded data"""

        results = {}

        # MOMENTUM STRATEGIES - Multiple variations
        results['momentum_basic'] = self._backtest_momentum_basic(snapshots)
        results['momentum_confirmation'] = self._backtest_momentum_confirmation(snapshots)
        results['momentum_acceleration'] = self._backtest_momentum_acceleration(snapshots)
        results['momentum_strength'] = self._backtest_momentum_strength(snapshots)
        results['momentum_adaptive'] = self._backtest_momentum_adaptive(snapshots)

        # FIXED PROFIT STRATEGIES - Multiple variations
        results['fixed_profit_3pct'] = self._backtest_fixed_profit(snapshots, profit_target=0.03)
        results['fixed_profit_5pct'] = self._backtest_fixed_profit(snapshots, profit_target=0.05)
        results['fixed_profit_7pct'] = self._backtest_fixed_profit(snapshots, profit_target=0.07)
        results['fixed_profit_10pct'] = self._backtest_fixed_profit(snapshots, profit_target=0.10)

        # PAIR TRADING
        results['pair_trading'] = self._backtest_pair_trading(snapshots)

        return results

    def _backtest_momentum_basic(self, snapshots: List[PriceSnapshot]) -> Dict:
        """Basic momentum: Enter on 2 consecutive moves, exit on reversal"""
        if len(snapshots) < 3:
            return {'total_pnl': 0, 'num_trades': 0, 'trades': []}

        position = None
        trades = []
        pnl = 0

        for i in range(2, len(snapshots)):
            prev_change = snapshots[i-1].call_mid - snapshots[i-2].call_mid
            curr_change = snapshots[i].call_mid - snapshots[i-1].call_mid

            if position is None:
                # Enter on momentum (2 consecutive moves in same direction)
                if curr_change > 0 and prev_change > 0:
                    position = {'entry': snapshots[i].call_ask, 'timestamp': snapshots[i].timestamp, 'type': 'momentum_up'}
                elif curr_change < 0 and prev_change < 0:
                    position = {'entry': snapshots[i].put_ask, 'timestamp': snapshots[i].timestamp, 'type': 'momentum_down'}
            else:
                # Exit on reversal
                if position['type'] == 'momentum_up' and curr_change < 0:
                    exit_price = snapshots[i].call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl': trade_pnl,
                        'entry_time': position['timestamp'],
                        'exit_time': snapshots[i].timestamp
                    })
                    position = None
                elif position['type'] == 'momentum_down' and curr_change > 0:
                    exit_price = snapshots[i].put_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl': trade_pnl,
                        'entry_time': position['timestamp'],
                        'exit_time': snapshots[i].timestamp
                    })
                    position = None

        # Close at expiry
        if position is not None:
            final_snapshot = snapshots[-1]
            if position['type'] == 'momentum_up':
                exit_price = 1.0 if final_snapshot.call_mid >= 0.5 else 0.0
            else:
                exit_price = 1.0 if final_snapshot.put_mid >= 0.5 else 0.0
            trade_pnl = exit_price - position['entry']
            pnl += trade_pnl
            trades.append({
                'entry': position['entry'],
                'exit': exit_price,
                'pnl': trade_pnl,
                'entry_time': position['timestamp'],
                'exit_time': final_snapshot.timestamp,
                'closed_at_expiry': True
            })

        return {
            'total_pnl': pnl,
            'num_trades': len(trades),
            'trades': trades,
            'avg_pnl_per_trade': pnl / len(trades) if trades else 0
        }

    def _backtest_momentum_confirmation(self, snapshots: List[PriceSnapshot]) -> Dict:
        """Momentum with 3-tick confirmation: Need 3 consecutive moves"""
        if len(snapshots) < 4:
            return {'total_pnl': 0, 'num_trades': 0, 'trades': []}

        position = None
        trades = []
        pnl = 0

        for i in range(3, len(snapshots)):
            change1 = snapshots[i-2].call_mid - snapshots[i-3].call_mid
            change2 = snapshots[i-1].call_mid - snapshots[i-2].call_mid
            change3 = snapshots[i].call_mid - snapshots[i-1].call_mid

            if position is None:
                # Enter only with 3 consecutive confirmations
                if change1 > 0 and change2 > 0 and change3 > 0:
                    position = {'entry': snapshots[i].call_ask, 'timestamp': snapshots[i].timestamp, 'type': 'up'}
            else:
                # Exit on reversal
                if change3 < 0:
                    exit_price = snapshots[i].call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl})
                    position = None

        # Close at expiry
        if position is not None:
            final_snapshot = snapshots[-1]
            exit_price = 1.0 if final_snapshot.call_mid >= 0.5 else 0.0
            trade_pnl = exit_price - position['entry']
            pnl += trade_pnl
            trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl, 'closed_at_expiry': True})

        return {'total_pnl': pnl, 'num_trades': len(trades), 'trades': trades, 'avg_pnl_per_trade': pnl / len(trades) if trades else 0}

    def _backtest_momentum_acceleration(self, snapshots: List[PriceSnapshot]) -> Dict:
        """Momentum with acceleration: Enter when moves are getting larger"""
        if len(snapshots) < 4:
            return {'total_pnl': 0, 'num_trades': 0, 'trades': []}

        position = None
        trades = []
        pnl = 0

        for i in range(3, len(snapshots)):
            change1 = abs(snapshots[i-2].call_mid - snapshots[i-3].call_mid)
            change2 = abs(snapshots[i-1].call_mid - snapshots[i-2].call_mid)
            change3 = abs(snapshots[i].call_mid - snapshots[i-1].call_mid)

            direction = snapshots[i].call_mid - snapshots[i-1].call_mid

            if position is None:
                # Enter when acceleration detected (moves getting larger)
                if change3 > change2 > change1 and change3 > 0.01:  # Minimum threshold
                    if direction > 0:
                        position = {'entry': snapshots[i].call_ask, 'timestamp': snapshots[i].timestamp, 'type': 'up'}
            else:
                # Exit when deceleration or reversal
                if change3 < change2 or direction < 0:
                    exit_price = snapshots[i].call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl})
                    position = None

        # Close at expiry
        if position is not None:
            final_snapshot = snapshots[-1]
            exit_price = 1.0 if final_snapshot.call_mid >= 0.5 else 0.0
            trade_pnl = exit_price - position['entry']
            pnl += trade_pnl
            trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl, 'closed_at_expiry': True})

        return {'total_pnl': pnl, 'num_trades': len(trades), 'trades': trades, 'avg_pnl_per_trade': pnl / len(trades) if trades else 0}

    def _backtest_momentum_strength(self, snapshots: List[PriceSnapshot]) -> Dict:
        """Momentum with strength filter: Only enter on strong moves (>2 cents)"""
        if len(snapshots) < 3:
            return {'total_pnl': 0, 'num_trades': 0, 'trades': []}

        position = None
        trades = []
        pnl = 0

        for i in range(2, len(snapshots)):
            prev_change = snapshots[i-1].call_mid - snapshots[i-2].call_mid
            curr_change = snapshots[i].call_mid - snapshots[i-1].call_mid

            if position is None:
                # Enter only on strong momentum (both moves > 0.02)
                if curr_change > 0.02 and prev_change > 0.02:
                    position = {'entry': snapshots[i].call_ask, 'timestamp': snapshots[i].timestamp}
            else:
                # Exit on reversal
                if curr_change < 0:
                    exit_price = snapshots[i].call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl})
                    position = None

        # Close at expiry
        if position is not None:
            final_snapshot = snapshots[-1]
            exit_price = 1.0 if final_snapshot.call_mid >= 0.5 else 0.0
            trade_pnl = exit_price - position['entry']
            pnl += trade_pnl
            trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl, 'closed_at_expiry': True})

        return {'total_pnl': pnl, 'num_trades': len(trades), 'trades': trades, 'avg_pnl_per_trade': pnl / len(trades) if trades else 0}

    def _backtest_momentum_adaptive(self, snapshots: List[PriceSnapshot]) -> Dict:
        """Adaptive momentum: Adjust sensitivity based on volatility"""
        if len(snapshots) < 20:
            return {'total_pnl': 0, 'num_trades': 0, 'trades': []}

        position = None
        trades = []
        pnl = 0

        for i in range(20, len(snapshots)):
            # Calculate recent volatility
            recent_prices = [s.call_mid for s in snapshots[i-20:i]]
            volatility = statistics.stdev(recent_prices)

            # Adaptive threshold based on volatility
            threshold = max(0.01, volatility * 0.5)

            prev_change = snapshots[i-1].call_mid - snapshots[i-2].call_mid
            curr_change = snapshots[i].call_mid - snapshots[i-1].call_mid

            if position is None:
                # Enter with adaptive threshold
                if curr_change > threshold and prev_change > threshold:
                    position = {'entry': snapshots[i].call_ask, 'timestamp': snapshots[i].timestamp}
            else:
                # Exit on reversal
                if curr_change < 0:
                    exit_price = snapshots[i].call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl})
                    position = None

        # Close at expiry
        if position is not None:
            final_snapshot = snapshots[-1]
            exit_price = 1.0 if final_snapshot.call_mid >= 0.5 else 0.0
            trade_pnl = exit_price - position['entry']
            pnl += trade_pnl
            trades.append({'entry': position['entry'], 'exit': exit_price, 'pnl': trade_pnl, 'closed_at_expiry': True})

        return {'total_pnl': pnl, 'num_trades': len(trades), 'trades': trades, 'avg_pnl_per_trade': pnl / len(trades) if trades else 0}

    def _backtest_fixed_profit(self, snapshots: List[PriceSnapshot], profit_target: float) -> Dict:
        """Test fixed profit target strategy"""
        position = None
        trades = []
        pnl = 0

        for snapshot in snapshots:
            if position is None:
                # Always have a position - buy at ask
                position = {'entry': snapshot.call_ask, 'timestamp': snapshot.timestamp}
            else:
                # Check if profit target hit
                potential_profit = snapshot.call_bid - position['entry']
                if potential_profit >= profit_target:
                    exit_price = snapshot.call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl': trade_pnl,
                        'entry_time': position['timestamp'],
                        'exit_time': snapshot.timestamp
                    })
                    # Immediately reopen
                    position = {'entry': snapshot.call_ask, 'timestamp': snapshot.timestamp}

        # Close any open position at end
        if position is not None:
            final_snapshot = snapshots[-1]
            exit_price = 1.0 if final_snapshot.call_mid >= 0.5 else 0.0
            trade_pnl = exit_price - position['entry']
            pnl += trade_pnl
            trades.append({
                'entry': position['entry'],
                'exit': exit_price,
                'pnl': trade_pnl,
                'entry_time': position['timestamp'],
                'exit_time': final_snapshot.timestamp,
                'closed_at_expiry': True
            })

        return {
            'total_pnl': pnl,
            'num_trades': len(trades),
            'trades': trades,
            'avg_pnl_per_trade': pnl / len(trades) if trades else 0
        }

    def _backtest_pair_trading(self, snapshots: List[PriceSnapshot]) -> Dict:
        """Test pair trading strategy - exploit when CALL+PUT != 1.0"""
        trades = []
        pnl = 0

        for snapshot in snapshots:
            price_sum = snapshot.call_mid + snapshot.put_mid

            # If sum > 1.0, sell both (they're overpriced)
            # If sum < 1.0, buy both (they're underpriced)

            if abs(price_sum - 1.0) > 0.02:  # Significant deviation
                arbitrage_profit = abs(price_sum - 1.0) - snapshot.spread_total
                if arbitrage_profit > 0:
                    pnl += arbitrage_profit
                    trades.append({
                        'timestamp': snapshot.timestamp,
                        'price_sum': price_sum,
                        'arbitrage_profit': arbitrage_profit
                    })

        return {
            'total_pnl': pnl,
            'num_trades': len(trades),
            'trades': trades,
            'avg_pnl_per_trade': pnl / len(trades) if trades else 0
        }

    def find_optimal_tp_sl(self, snapshots: List[PriceSnapshot]) -> Dict:
        """
        HINDSIGHT ANALYSIS: Find the best TP/SL combination for the completed period
        Tests a grid of TP/SL values and returns the best performing combination
        """
        if len(snapshots) < 2:
            return None

        # Grid of TP/SL values to test (in dollars)
        tp_values = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
        sl_values = [0.05, 0.07, 0.10, 0.12, 0.15, 0.20, None]  # None = no SL

        best_result = None
        best_pnl = float('-inf')
        all_results = []

        for tp in tp_values:
            for sl in sl_values:
                result = self._test_tp_sl_combination(snapshots, tp, sl)
                all_results.append(result)

                if result['total_pnl'] > best_pnl:
                    best_pnl = result['total_pnl']
                    best_result = result

        return {
            'best_combination': best_result,
            'all_combinations': all_results,
            'total_tested': len(all_results)
        }

    def _test_tp_sl_combination(self, snapshots: List[PriceSnapshot], tp: float, sl: Optional[float]) -> Dict:
        """Test a specific TP/SL combination on historical data"""
        position = None
        trades = []
        pnl = 0

        for snapshot in snapshots:
            if position is None:
                # Open position
                position = {
                    'entry': snapshot.call_ask,
                    'timestamp': snapshot.timestamp,
                    'tp': snapshot.call_ask + tp,
                    'sl': (snapshot.call_ask - sl) if sl is not None else -1.0
                }
            else:
                # Check TP
                if snapshot.call_bid >= position['tp']:
                    exit_price = snapshot.call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl': trade_pnl,
                        'reason': 'TP'
                    })
                    # Reopen
                    position = {
                        'entry': snapshot.call_ask,
                        'timestamp': snapshot.timestamp,
                        'tp': snapshot.call_ask + tp,
                        'sl': (snapshot.call_ask - sl) if sl is not None else -1.0
                    }
                # Check SL
                elif sl is not None and snapshot.call_bid <= position['sl'] and position['sl'] >= 0:
                    exit_price = snapshot.call_bid
                    trade_pnl = exit_price - position['entry']
                    pnl += trade_pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl': trade_pnl,
                        'reason': 'SL'
                    })
                    # Reopen
                    position = {
                        'entry': snapshot.call_ask,
                        'timestamp': snapshot.timestamp,
                        'tp': snapshot.call_ask + tp,
                        'sl': (snapshot.call_ask - sl) if sl is not None else -1.0
                    }

        # Close at expiry
        if position is not None:
            final_snapshot = snapshots[-1]
            exit_price = 1.0 if final_snapshot.call_mid >= 0.5 else 0.0
            trade_pnl = exit_price - position['entry']
            pnl += trade_pnl
            trades.append({
                'entry': position['entry'],
                'exit': exit_price,
                'pnl': trade_pnl,
                'reason': 'EXPIRY'
            })

        tp_trades = len([t for t in trades if t.get('reason') == 'TP'])
        sl_trades = len([t for t in trades if t.get('reason') == 'SL'])

        return {
            'tp': tp,
            'sl': sl,
            'total_pnl': pnl,
            'num_trades': len(trades),
            'tp_hits': tp_trades,
            'sl_hits': sl_trades,
            'avg_pnl_per_trade': pnl / len(trades) if trades else 0,
            'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
        }

    def save_period_analysis(self):
        """Save complete period analysis"""
        if len(self.current_period_snapshots) < 2:
            return

        period_end = self.current_period_start + timedelta(minutes=15)
        filename = f"analysis_{self.current_period_start.strftime('%Y%m%d_%H%M')}.json"
        output_file = self.output_dir / filename

        # Calculate metrics
        metrics = self.calculate_oscillation_metrics(self.current_period_snapshots)

        # Backtest strategies
        strategy_results = self.backtest_strategies(self.current_period_snapshots)

        # Find optimal TP/SL (hindsight)
        optimal_tp_sl = self.find_optimal_tp_sl(self.current_period_snapshots)

        # Compile results
        result = {
            'period_start': self.current_period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'num_snapshots': len(self.current_period_snapshots),
            'oscillation_metrics': asdict(metrics) if metrics else None,
            'strategy_backtests': strategy_results,
            'optimal_tp_sl_hindsight': optimal_tp_sl,
            'price_snapshots': [asdict(s) for s in self.current_period_snapshots]
        }

        # Save
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*100}")
        print(f"📊 PERIOD ANALYSIS: {self.current_period_start.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*100}")
        if metrics:
            print(f"📈 CALL Range: ${metrics.call_min:.4f} - ${metrics.call_max:.4f} (Range: ${metrics.call_range:.4f})")
            print(f"📉 PUT Range: ${metrics.put_min:.4f} - ${metrics.put_max:.4f} (Range: ${metrics.put_range:.4f})")
            print(f"🔄 Reversals: {metrics.num_reversals} | Avg Swing: ${metrics.avg_swing_size:.4f} | Max Swing: ${metrics.max_swing_size:.4f}")
            print(f"📊 Volatility: CALL=${metrics.call_volatility:.4f} | PUT=${metrics.put_volatility:.4f}")
            print(f"🎯 Best Buy & Hold: CALL=${metrics.best_buy_and_hold_call:+.4f} | PUT=${metrics.best_buy_and_hold_put:+.4f}")
            print(f"💰 Theoretical Max (swing trading): ${metrics.theoretical_max_swing_trading:.4f}")

        print(f"\n🧪 MOMENTUM STRATEGY VARIATIONS:")
        for strategy_name in ['momentum_basic', 'momentum_confirmation', 'momentum_acceleration', 'momentum_strength', 'momentum_adaptive']:
            if strategy_name in strategy_results:
                r = strategy_results[strategy_name]
                print(f"   {strategy_name}: PNL=${r['total_pnl']:+.4f} | Trades={r['num_trades']} | Avg=${r['avg_pnl_per_trade']:+.4f}")

        print(f"\n💵 FIXED PROFIT STRATEGIES:")
        for strategy_name in ['fixed_profit_3pct', 'fixed_profit_5pct', 'fixed_profit_7pct', 'fixed_profit_10pct']:
            if strategy_name in strategy_results:
                r = strategy_results[strategy_name]
                print(f"   {strategy_name}: PNL=${r['total_pnl']:+.4f} | Trades={r['num_trades']} | Avg=${r['avg_pnl_per_trade']:+.4f}")

        print(f"\n🔄 PAIR TRADING:")
        if 'pair_trading' in strategy_results:
            r = strategy_results['pair_trading']
            print(f"   pair_trading: PNL=${r['total_pnl']:+.4f} | Opportunities={r['num_trades']}")

        if optimal_tp_sl and optimal_tp_sl['best_combination']:
            best = optimal_tp_sl['best_combination']
            sl_str = f"${best['sl']:.2f}" if best['sl'] is not None else "None"
            print(f"\n🎯 OPTIMAL TP/SL (HINDSIGHT):")
            print(f"   Best: TP=${best['tp']:.2f}, SL={sl_str}")
            print(f"   PNL=${best['total_pnl']:+.4f} | Trades={best['num_trades']} | Win Rate={best['win_rate']:.1%}")
            print(f"   TP Hits={best['tp_hits']} | SL Hits={best['sl_hits']}")

        print(f"💾 Saved to: {output_file}")
        print(f"{'='*100}\n")

    def check_period_change(self):
        """Check if we've moved to a new period"""
        current_period = self.get_current_period_start()

        if self.current_period_start is None:
            self.current_period_start = current_period
            return False

        if current_period > self.current_period_start:
            # Save analysis for completed period
            self.save_period_analysis()

            # Reset for new period
            self.current_period_start = current_period
            self.current_period_snapshots = []

            return True

        return False

    def update_prices(self):
        """Update prices from files"""
        updated = False

        # Check CALL file
        if self.call_file.exists():
            mtime = self.call_file.stat().st_mtime
            if mtime > self.last_call_mtime:
                prices = self.read_price_file(self.call_file)
                if prices and prices['ask'] is not None and prices['bid'] is not None:
                    self.current_call_ask = prices['ask']
                    self.current_call_bid = prices['bid']
                    self.last_call_mtime = mtime
                    updated = True

        # Check PUT file
        if self.put_file.exists():
            mtime = self.put_file.stat().st_mtime
            if mtime > self.last_put_mtime:
                prices = self.read_price_file(self.put_file)
                if prices and prices['ask'] is not None and prices['bid'] is not None:
                    self.current_put_ask = prices['ask']
                    self.current_put_bid = prices['bid']
                    self.last_put_mtime = mtime
                    updated = True

        return updated

    def run(self, check_interval: float = 1.0):
        """Main recording loop"""
        print("Starting Price Recorder & Strategy Analyzer...")
        print(f"Monitoring: {self.call_file}")
        print(f"Monitoring: {self.put_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"\n📊 Recording all price movements second-by-second")
        print(f"🧪 Testing 10+ strategy variations per period:")
        print(f"   - 5 Momentum variations (basic, confirmation, acceleration, strength, adaptive)")
        print(f"   - 4 Fixed profit targets (3%, 5%, 7%, 10%)")
        print(f"   - Pair trading (CALL+PUT arbitrage)")
        print(f"🎯 HINDSIGHT: Finding optimal TP/SL for each completed period\n")

        try:
            while True:
                self.check_period_change()

                if self.update_prices():
                    self.record_snapshot()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nShutting down recorder...")
            self.save_period_analysis()
            print("Analysis saved. Goodbye!")


if __name__ == "__main__":
    recorder = PriceRecorder(
        call_file="/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json",
        output_dir="/home/ubuntu/013_2025_polymarket/bot016_react/simulators/price_analysis"
    )

    recorder.run(check_interval=1.0)
