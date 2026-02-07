import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import deque

# pm2 start simulator_dynamic_tpsl.py --cron-restart="00 * * * *" --interpreter python3

class Strategy(Enum):
    A = "A"      # BTC_CALL only, buy at ask, TP +0.06, SL -0.09
    A2 = "A2"    # Same as A but only when choppiness > 30
    B = "B"      # Both BTC_CALL and BTC_PUT with same TP/SL as A
    C = "C"      # BTC_CALL with dynamic TP/SL based on instantaneous choppiness

@dataclass
class Position:
    """Represents an open position"""
    asset_type: str  # "BTC_CALL" or "BTC_PUT"
    entry_price: float
    entry_time: datetime
    period_end: datetime
    strategy: str
    position_id: str  # Unique identifier
    take_profit: float  # Absolute price level for TP
    stop_loss: float    # Absolute price level for SL
    size: float = 1.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "TP", "SL", "EXPIRY"
    pnl: Optional[float] = None
    market_conditions: Optional[Dict] = None
    atr_at_entry: Optional[float] = None  # ATR value when position opened
    tp_sl_adjustments: int = 0  # Track how many times TP/SL was adjusted (Strategy C)

@dataclass
class Trade:
    """Represents a completed trade"""
    timestamp: datetime
    period_end: datetime
    strategy: str
    asset_type: str
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    take_profit: float
    stop_loss: float
    market_conditions: Optional[Dict] = None
    atr_at_entry: Optional[float] = None  # ATR value when trade opened
    duration_seconds: float = 0.0
    tp_sl_adjustments: int = 0  # Number of times TP/SL was adjusted during the trade

class PriceBuffer:
    """Stores recent price history for calculating instantaneous choppiness"""
    def __init__(self, max_seconds: int = 15):
        self.max_seconds = max_seconds
        self.prices = deque(maxlen=max_seconds * 10)  # Assuming ~10 updates per second max
        self.timestamps = deque(maxlen=max_seconds * 10)
    
    def add(self, price: float, timestamp: datetime):
        """Add a new price point"""
        self.prices.append(price)
        self.timestamps.append(timestamp)
        
        # Clean old data
        cutoff_time = timestamp - timedelta(seconds=self.max_seconds)
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.prices.popleft()
            self.timestamps.popleft()
    
    def get_recent_prices(self, seconds: int = 15) -> List[float]:
        """Get prices from the last N seconds"""
        if not self.timestamps:
            return []
        
        cutoff_time = self.timestamps[-1] - timedelta(seconds=seconds)
        recent_prices = []
        
        for price, ts in zip(self.prices, self.timestamps):
            if ts >= cutoff_time:
                recent_prices.append(price)
        
        return recent_prices

class ATRCalculator:
    """Calculate Average True Range with exponential weighting"""
    
    @staticmethod
    def calculate_period_ranges(prices: List[float], timestamps: List[datetime], 
                                period_seconds: int = 15) -> List[Tuple[float, float, datetime]]:
        """
        Calculate max-min range for each period
        Returns list of (max_price, min_price, period_end_time)
        """
        if not prices or not timestamps:
            return []
        
        periods = []
        current_period_prices = []
        current_period_start = timestamps[0]
        
        for price, ts in zip(prices, timestamps):
            # Check if we've moved to a new period
            time_diff = (ts - current_period_start).total_seconds()
            
            if time_diff >= period_seconds:
                # Save current period
                if current_period_prices:
                    max_price = max(current_period_prices)
                    min_price = min(current_period_prices)
                    periods.append((max_price, min_price, ts))
                
                # Start new period
                current_period_prices = [price]
                current_period_start = ts
            else:
                current_period_prices.append(price)
        
        # Add final period if it has data
        if current_period_prices:
            max_price = max(current_period_prices)
            min_price = min(current_period_prices)
            periods.append((max_price, min_price, timestamps[-1]))
        
        return periods
    
    @staticmethod
    def calculate_atr(prices: List[float], timestamps: List[datetime], 
                     period_seconds: int = 15, num_periods: int = 6) -> Optional[float]:
        """
        Calculate ATR using exponential weighting
        More weight on recent periods
        
        Args:
            prices: List of bid prices
            timestamps: Corresponding timestamps
            period_seconds: Length of each period in seconds (default 15)
            num_periods: Number of periods to use (default 6)
        
        Returns:
            ATR value or None if insufficient data
        """
        # Get period ranges
        periods = ATRCalculator.calculate_period_ranges(prices, timestamps, period_seconds)
        
        if len(periods) < 2:
            return None
        
        # Take last num_periods
        recent_periods = periods[-num_periods:] if len(periods) >= num_periods else periods
        
        if len(recent_periods) < 2:
            return None
        
        # Calculate True Range for each period
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        true_ranges = []
        
        for i in range(len(recent_periods)):
            max_price, min_price, _ = recent_periods[i]
            
            if i == 0:
                # First period: just use high-low
                tr = max_price - min_price
            else:
                # Use previous period's close (approximated as average)
                prev_max, prev_min, _ = recent_periods[i-1]
                prev_close = (prev_max + prev_min) / 2
                
                high_low = max_price - min_price
                high_prev_close = abs(max_price - prev_close)
                low_prev_close = abs(min_price - prev_close)
                
                tr = max(high_low, high_prev_close, low_prev_close)
            
            true_ranges.append(tr)
        
        if not true_ranges:
            return None
        
        # Apply exponential weighting - more weight on recent periods
        # Using exponential decay factor alpha = 2 / (N + 1)
        n = len(true_ranges)
        alpha = 2.0 / (n + 1)
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, tr in enumerate(true_ranges):
            # Weight increases exponentially for more recent periods
            weight = (1 - alpha) ** (n - 1 - i)
            weighted_sum += tr * weight
            weight_sum += weight
        
        atr = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        return atr
    
    @staticmethod
    def estimate_movement_probability(atr: float, current_price: float) -> Tuple[float, float]:
        """
        Estimate probability and expected magnitude of price movement based on ATR
        
        Args:
            atr: Average True Range value
            current_price: Current asset price
        
        Returns: (probability_of_movement, expected_move_magnitude)
        
        Higher ATR relative to price = More volatile, larger expected moves
        Lower ATR relative to price = Less volatile, smaller expected moves
        """
        if current_price == 0:
            return 0.5, 0.05
        
        # Calculate ATR as percentage of price
        atr_pct = (atr / current_price) * 100
        
        # ATR% interpretation:
        # 0-5%: Low volatility
        # 5-15%: Medium volatility  
        # 15%+: High volatility
        
        if atr_pct > 15:
            # High volatility - larger expected moves
            prob_movement = min(0.95, 0.65 + (atr_pct - 15) / 50)
            expected_magnitude = min(0.15, atr * 1.5)
        elif atr_pct > 5:
            # Medium volatility
            prob_movement = 0.45 + (atr_pct - 5) / 50
            expected_magnitude = atr * 1.2
        else:
            # Low volatility - smaller moves
            prob_movement = 0.35 + atr_pct / 25
            expected_magnitude = max(0.02, atr * 0.8)
        
        return prob_movement, expected_magnitude

class BinaryOptionsTrader:
    def __init__(self):
        self.data_files = {
            'BTC_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
            'BTC_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
        }
        
        self.oscillation_file = '/home/ubuntu/013_2025_polymarket/bot004_blackScholes/data/latest_oscillation.json'
        
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.total_pnl = {strategy.value: 0.0 for strategy in Strategy}
        self.total_pnl['ALL'] = 0.0
        
        # Price buffers for ATR calculation
        self.price_buffers = {
            'BTC_CALL': PriceBuffer(max_seconds=100),  # Store up to 100 seconds for 6 periods of 15s
            'BTC_PUT': PriceBuffer(max_seconds=100)
        }
        
        self.atr_calc = ATRCalculator()
        
        # Load existing data from today's file on startup
        self.load_existing_data()
    
    def get_next_period_end(self, current_time: datetime) -> datetime:
        """Calculate the next period end time (00, 15, 30, 45 minutes)"""
        minute = current_time.minute
        
        if minute < 15:
            target_minute = 15
        elif minute < 30:
            target_minute = 30
        elif minute < 45:
            target_minute = 45
        else:
            target_minute = 0
            current_time += timedelta(hours=1)
        
        period_end = current_time.replace(minute=target_minute, second=0, microsecond=0)
        return period_end
    
    def read_market_data(self) -> Dict[str, Dict]:
        """Read all market data files and normalize the structure"""
        data = {}
        for asset_name, filepath in self.data_files.items():
            try:
                with open(filepath, 'r') as f:
                    raw_data = json.load(f)
                    
                    # Normalize the data structure
                    # Extract bid/ask from best_bid/best_ask if they exist
                    normalized_data = {}
                    
                    if 'best_bid' in raw_data and raw_data['best_bid']:
                        normalized_data['bid'] = raw_data['best_bid'].get('price')
                    else:
                        normalized_data['bid'] = raw_data.get('bid')
                    
                    if 'best_ask' in raw_data and raw_data['best_ask']:
                        normalized_data['ask'] = raw_data['best_ask'].get('price')
                    else:
                        normalized_data['ask'] = raw_data.get('ask')
                    
                    # Keep the original data too
                    normalized_data['raw'] = raw_data
                    
                    data[asset_name] = normalized_data
                    
            except FileNotFoundError:
                print(f"Warning: {filepath} not found")
                data[asset_name] = None
            except json.JSONDecodeError:
                print(f"Warning: Error decoding {filepath}")
                data[asset_name] = None
        return data
    
    def read_oscillation_data(self) -> Optional[Dict]:
        """Read market oscillation data"""
        try:
            with open(self.oscillation_file, 'r') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding oscillation file: {e}")
            return None
        except Exception as e:
            print(f"Warning: Error reading oscillation data: {e}")
            return None
    
    def update_price_buffers(self, data: Dict[str, Dict], current_time: datetime):
        """Update price buffers with current bid prices"""
        for asset_type in ['BTC_CALL', 'BTC_PUT']:
            if data.get(asset_type) and 'bid' in data[asset_type]:
                bid_price = data[asset_type]['bid']
                if bid_price is not None:
                    self.price_buffers[asset_type].add(bid_price, current_time)
    
    def calculate_atr(self, asset_type: str, period_seconds: int = 15, num_periods: int = 6) -> Optional[float]:
        """Calculate ATR for the asset using last N periods"""
        recent_prices = self.price_buffers[asset_type].prices
        recent_timestamps = self.price_buffers[asset_type].timestamps
        
        if len(recent_prices) < 10:  # Need some minimum data
            return None
        
        return self.atr_calc.calculate_atr(
            list(recent_prices), 
            list(recent_timestamps), 
            period_seconds, 
            num_periods
        )
    
    def get_dynamic_tpsl(self, atr: float, entry_price: float) -> Tuple[float, float]:
        """
        Calculate dynamic TP and SL based on ATR
        Maintains 2:3 ratio (TP:SL)
        
        Returns: (take_profit_price, stop_loss_price)
        """
        prob_movement, expected_magnitude = self.atr_calc.estimate_movement_probability(
            atr, entry_price
        )
        
        # Scale the TP/SL based on ATR and expected magnitude
        # Base TP/SL with 2:3 ratio
        base_tp = 0.03  # Minimum TP
        base_sl = 0.045  # Minimum SL (maintains 2:3 ratio)
        
        # Scale based on ATR-derived expected magnitude and probability
        scaling_factor = expected_magnitude * prob_movement
        
        tp_distance = base_tp + (scaling_factor * 1.0)
        sl_distance = base_sl + (scaling_factor * 1.5)
        
        # Ensure minimum values and maintain ratio
        tp_distance = max(0.02, min(0.20, tp_distance))  # Clamp TP
        sl_distance = tp_distance * 1.5  # Maintain 2:3 ratio
        
        take_profit = min(0.99, entry_price + tp_distance)  # Cap at 0.99
        stop_loss = entry_price - sl_distance
        
        return take_profit, stop_loss
    
    def load_existing_data(self):
        """Load existing trades and PNL data from today's file on startup"""
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f'trading_results_dynamic_{date_str}.json'
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
                # Load total PNL
                if 'total_pnl' in data:
                    self.total_pnl.update(data['total_pnl'])
                
                # Load completed trades
                if 'trades' in data:
                    for trade_data in data['trades']:
                        trade = Trade(
                            timestamp=datetime.fromisoformat(trade_data['timestamp']),
                            period_end=datetime.fromisoformat(trade_data['period_end']),
                            strategy=trade_data['strategy'],
                            asset_type=trade_data['asset_type'],
                            entry_price=trade_data['entry_price'],
                            exit_price=trade_data['exit_price'],
                            pnl=trade_data['pnl'],
                            exit_reason=trade_data['exit_reason'],
                            take_profit=trade_data['take_profit'],
                            stop_loss=trade_data['stop_loss'],
                            market_conditions=trade_data.get('market_conditions'),
                            atr_at_entry=trade_data.get('atr_at_entry') or trade_data.get('instantaneous_choppiness'),
                            duration_seconds=trade_data.get('duration_seconds', 0.0),
                            tp_sl_adjustments=trade_data.get('tp_sl_adjustments', 0)
                        )
                        self.trades.append(trade)
                
                # Load open positions
                if 'open_positions_detail' in data:
                    for pos_data in data['open_positions_detail']:
                        pos = Position(
                            asset_type=pos_data['asset_type'],
                            entry_price=pos_data['entry_price'],
                            entry_time=datetime.fromisoformat(pos_data['entry_time']),
                            period_end=datetime.fromisoformat(pos_data['period_end']),
                            strategy=pos_data['strategy'],
                            position_id=pos_data['position_id'],
                            take_profit=pos_data['take_profit'],
                            stop_loss=pos_data['stop_loss'],
                            size=pos_data.get('size', 1.0),
                            market_conditions=pos_data.get('market_conditions'),
                            atr_at_entry=pos_data.get('atr_at_entry') or pos_data.get('instantaneous_choppiness'),
                            tp_sl_adjustments=pos_data.get('tp_sl_adjustments', 0)
                        )
                        self.positions.append(pos)
                
                print(f"Loaded existing data: {len(self.trades)} trades, {len(self.positions)} open positions")
                print(f"Current PNL: ${self.total_pnl['ALL']:+.2f}")
                
        except FileNotFoundError:
            print(f"No existing data file found for today. Starting fresh.")
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    def check_trade_conditions(self, data: Dict[str, Dict], current_time: datetime, 
                               oscillation_data: Optional[Dict]) -> List[Tuple]:
        """
        Check if conditions are met to open new positions
        Strategies run continuously - check every second
        Returns list of (asset_type, entry_price, period_end, strategy, tp, sl, instantaneous_chop)
        """
        signals = []
        period_end = self.get_next_period_end(current_time)
        
        # Get BTC_CALL data
        btc_call_data = data.get('BTC_CALL')
        btc_put_data = data.get('BTC_PUT')
        
        if not btc_call_data or 'ask' not in btc_call_data or btc_call_data['ask'] is None:
            return signals
        
        btc_call_ask = btc_call_data['ask']
        
        # Get oscillation data for choppiness
        choppiness_60min = None
        if oscillation_data and 'oscillation_60min' in oscillation_data:
            choppiness_60min = oscillation_data['oscillation_60min'].get('choppiness_index')
        
        # Calculate ATR for Strategy C
        atr_value = self.calculate_atr('BTC_CALL', period_seconds=15, num_periods=6)
        
        # Check if each strategy already has an open position
        has_strategy_a = any(p.strategy == 'A' and p.asset_type == 'BTC_CALL' for p in self.positions)
        has_strategy_a2 = any(p.strategy == 'A2' and p.asset_type == 'BTC_CALL' for p in self.positions)
        has_strategy_b_call = any(p.strategy == 'B' and p.asset_type == 'BTC_CALL' for p in self.positions)
        has_strategy_b_put = any(p.strategy == 'B' and p.asset_type == 'BTC_PUT' for p in self.positions)
        has_strategy_c = any(p.strategy == 'C' and p.asset_type == 'BTC_CALL' for p in self.positions)
        
        # Strategy A: Buy BTC_CALL at ask, TP +0.06, SL -0.09
        if not has_strategy_a:
            tp = min(0.99, btc_call_ask + 0.06)  # Cap at 0.99
            sl = btc_call_ask - 0.09
            signals.append(('BTC_CALL', btc_call_ask, period_end, 'A', tp, sl, None))
        
        # Strategy A2: Same as A but only when choppiness > 30
        if not has_strategy_a2:
            if choppiness_60min is not None and choppiness_60min > 30:
                tp = min(0.99, btc_call_ask + 0.06)  # Cap at 0.99
                sl = btc_call_ask - 0.09
                signals.append(('BTC_CALL', btc_call_ask, period_end, 'A2', tp, sl, None))
        
        # Strategy B: Buy both BTC_CALL and BTC_PUT independently
        # Each position can re-enter even if the other is still open
        
        # Strategy B - CALL position (independent)
        if not has_strategy_b_call:
            tp_call = min(0.99, btc_call_ask + 0.06)  # Cap at 0.99
            sl_call = btc_call_ask - 0.09
            signals.append(('BTC_CALL', btc_call_ask, period_end, 'B', tp_call, sl_call, None))
        
        # Strategy B - PUT position (independent)
        if not has_strategy_b_put:
            if btc_put_data and 'ask' in btc_put_data and btc_put_data['ask'] is not None:
                btc_put_ask = btc_put_data['ask']
                tp_put = min(0.99, btc_put_ask + 0.06)  # Cap at 0.99
                sl_put = btc_put_ask - 0.09
                signals.append(('BTC_PUT', btc_put_ask, period_end, 'B', tp_put, sl_put, None))
        
        # Strategy C: Dynamic TP/SL based on ATR
        if not has_strategy_c:
            if atr_value is not None:
                tp, sl = self.get_dynamic_tpsl(atr_value, btc_call_ask)
                tp = min(0.99, tp)  # Cap at 0.99
                signals.append(('BTC_CALL', btc_call_ask, period_end, 'C', tp, sl, atr_value))
        
        return signals
    
    def open_position(self, asset_type: str, entry_price: float, entry_time: datetime,
                     period_end: datetime, strategy: str, take_profit: float, stop_loss: float,
                     market_conditions: Optional[Dict], atr_value: Optional[float]):
        """Open a new position"""
        position_id = f"{strategy}_{asset_type}_{period_end.strftime('%Y%m%d%H%M')}_{len(self.positions)}"
        
        position = Position(
            asset_type=asset_type,
            entry_price=entry_price,
            entry_time=entry_time,
            period_end=period_end,
            strategy=strategy,
            position_id=position_id,
            take_profit=take_profit,
            stop_loss=stop_loss,
            market_conditions=market_conditions,
            atr_at_entry=atr_value
        )
        
        self.positions.append(position)
        
        print(f"\n{'='*80}")
        print(f"POSITION OPENED - {strategy}")
        print(f"{'='*80}")
        print(f"Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Asset: {asset_type}")
        print(f"Entry Price: ${entry_price:.4f}")
        print(f"Take Profit: ${take_profit:.4f} (+${take_profit - entry_price:.4f})")
        if take_profit == 0.99:
            print(f"  ⚠️  TP capped at $0.99 (maximum allowed)")
        print(f"Stop Loss: ${stop_loss:.4f} (-${entry_price - stop_loss:.4f})")
        
        # For Strategy C, display ATR metrics
        if strategy == 'C' and atr_value is not None:
            prob_movement, expected_magnitude = self.atr_calc.estimate_movement_probability(
                atr_value, entry_price
            )
            atr_pct = (atr_value / entry_price) * 100
            
            print(f"\n🎯 Strategy C Market Analysis:")
            print(f"   ATR (6 periods, 15s each): ${atr_value:.4f} ({atr_pct:.2f}% of price)")
            print(f"   Probability of Movement: {prob_movement:.1%}")
            print(f"   Expected Magnitude: ${expected_magnitude:.4f}")
            print(f"   TP/SL Ratio: {(take_profit - entry_price)/(entry_price - stop_loss):.2f}:1")
            print(f"   Note: TP/SL will adapt to changing ATR")
        
        if atr_value is None and strategy == 'C':
            print(f"\n⚠️  Strategy C: Waiting for price history (need 90+ seconds)...")
        
        if market_conditions and 'oscillation_60min' in market_conditions:
            chop_60 = market_conditions['oscillation_60min'].get('choppiness_index')
            if chop_60 is not None:
                print(f"60min Choppiness: {chop_60:.2f}")
        
        print(f"Period Ends: {period_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    def check_tpsl_exit(self, data: Dict[str, Dict], current_time: datetime):
        """Check if any positions hit TP or SL"""
        positions_to_close = []
        
        for pos in self.positions:
            asset_data = data.get(pos.asset_type)
            if not asset_data or 'bid' not in asset_data or asset_data['bid'] is None:
                continue
            
            current_bid = asset_data['bid']
            
            # CRITICAL: Force close ALL positions when bid reaches 0.99
            if current_bid >= 0.99:
                pos.exit_price = current_bid
                pos.exit_time = current_time
                pos.exit_reason = "TP_MAX"  # Special reason for 0.99 exit
                pos.pnl = (pos.exit_price - pos.entry_price) * pos.size
                positions_to_close.append(pos)
                continue
            
            # Strategy C: Dynamically adjust TP/SL based on current ATR
            if pos.strategy == 'C':
                current_atr = self.calculate_atr(pos.asset_type, period_seconds=15, num_periods=6)
                
                if current_atr is not None:
                    new_tp, new_sl = self.get_dynamic_tpsl(current_atr, pos.entry_price)
                    new_tp = min(0.99, new_tp)  # Always cap at 0.99
                    
                    # Get probability metrics for display
                    prob_movement, expected_magnitude = self.atr_calc.estimate_movement_probability(
                        current_atr, pos.entry_price
                    )
                    
                    # Only update if significantly different (to avoid constant tiny adjustments)
                    tp_change = abs(new_tp - pos.take_profit)
                    sl_change = abs(new_sl - pos.stop_loss)
                    
                    if tp_change > 0.01 or sl_change > 0.01:
                        old_tp = pos.take_profit
                        old_sl = pos.stop_loss
                        old_atr = pos.atr_at_entry or 0
                        pos.take_profit = new_tp
                        pos.stop_loss = new_sl
                        pos.atr_at_entry = current_atr  # Update current ATR
                        pos.tp_sl_adjustments += 1  # Increment adjustment counter
                        
                        atr_pct = (current_atr / pos.entry_price) * 100
                        atr_change_pct = ((current_atr - old_atr) / old_atr * 100) if old_atr > 0 else 0
                        
                        print(f"\n{'~'*80}")
                        print(f"STRATEGY C - TP/SL ADJUSTED (#{pos.tp_sl_adjustments})")
                        print(f"{'~'*80}")
                        print(f"Time: {current_time.strftime('%H:%M:%S')}")
                        print(f"Asset: {pos.asset_type}")
                        print(f"Entry Price: ${pos.entry_price:.4f}")
                        print(f"Current Bid: ${current_bid:.4f}")
                        print(f"\nMarket Analysis:")
                        print(f"  ATR: ${current_atr:.4f} ({atr_pct:.2f}% of price)")
                        if old_atr > 0:
                            print(f"  ATR Change: {atr_change_pct:+.1f}% from previous")
                        print(f"  Probability of Movement: {prob_movement:.1%}")
                        print(f"  Expected Magnitude: ${expected_magnitude:.4f}")
                        print(f"\nTP/SL Adjustment:")
                        print(f"  Old TP: ${old_tp:.4f} → New TP: ${new_tp:.4f} (Δ ${new_tp - old_tp:+.4f})")
                        if new_tp == 0.99:
                            print(f"  ⚠️  TP capped at $0.99 (max allowed)")
                        print(f"  Old SL: ${old_sl:.4f} → New SL: ${new_sl:.4f} (Δ ${new_sl - old_sl:+.4f})")
                        print(f"  TP Distance: ${new_tp - pos.entry_price:.4f}")
                        print(f"  SL Distance: ${pos.entry_price - new_sl:.4f}")
                        print(f"  Ratio: {(new_tp - pos.entry_price)/(pos.entry_price - new_sl):.2f}:1")
                        print(f"{'~'*80}\n")
            
            # Check Take Profit
            if current_bid >= pos.take_profit:
                pos.exit_price = current_bid
                pos.exit_time = current_time
                pos.exit_reason = "TP"
                pos.pnl = (pos.exit_price - pos.entry_price) * pos.size
                positions_to_close.append(pos)
                continue
            
            # Check Stop Loss
            if current_bid <= pos.stop_loss:
                pos.exit_price = current_bid
                pos.exit_time = current_time
                pos.exit_reason = "SL"
                pos.pnl = (pos.exit_price - pos.entry_price) * pos.size
                positions_to_close.append(pos)
                continue
        
        # Close positions that hit TP/SL
        for pos in positions_to_close:
            self.close_position(pos, data)
    
    def close_positions_at_expiry(self, data: Dict[str, Dict], current_time: datetime):
        """Close positions that have reached their period end"""
        positions_to_close = []
        
        for pos in self.positions:
            if current_time >= pos.period_end:
                # Try to get exit price from bid
                asset_data = data.get(pos.asset_type)
                if asset_data and 'bid' in asset_data and asset_data['bid'] is not None:
                    pos.exit_price = asset_data['bid']
                else:
                    pos.exit_price = 0.0  # Total loss if no price available
                
                pos.exit_time = current_time
                pos.exit_reason = "EXPIRY"
                pos.pnl = (pos.exit_price - pos.entry_price) * pos.size
                positions_to_close.append(pos)
        
        for pos in positions_to_close:
            self.close_position(pos, data)
    
    def close_position(self, pos: Position, data: Dict[str, Dict]):
        """Close a position and record the trade"""
        duration_seconds = (pos.exit_time - pos.entry_time).total_seconds()
        
        trade = Trade(
            timestamp=pos.entry_time,
            period_end=pos.period_end,
            strategy=pos.strategy,
            asset_type=pos.asset_type,
            entry_price=pos.entry_price,
            exit_price=pos.exit_price,
            pnl=pos.pnl,
            exit_reason=pos.exit_reason,
            take_profit=pos.take_profit,
            stop_loss=pos.stop_loss,
            market_conditions=pos.market_conditions,
            atr_at_entry=pos.atr_at_entry,
            duration_seconds=duration_seconds,
            tp_sl_adjustments=pos.tp_sl_adjustments
        )
        
        self.trades.append(trade)
        self.total_pnl[pos.strategy] += pos.pnl
        self.total_pnl['ALL'] += pos.pnl
        
        print(f"\n{'*'*80}")
        print(f"POSITION CLOSED - {pos.strategy} - {pos.exit_reason}")
        print(f"{'*'*80}")
        print(f"Asset: {pos.asset_type}")
        print(f"Entry: ${pos.entry_price:.4f} at {pos.entry_time.strftime('%H:%M:%S')}")
        print(f"Exit:  ${pos.exit_price:.4f} at {pos.exit_time.strftime('%H:%M:%S')}")
        print(f"Duration: {duration_seconds:.0f}s ({duration_seconds/60:.1f}min)")
        print(f"TP: ${pos.take_profit:.4f}, SL: ${pos.stop_loss:.4f}")
        
        # For Strategy C, show adjustment info and ATR
        if pos.strategy == 'C':
            print(f"TP/SL Adjustments: {pos.tp_sl_adjustments} times")
            if pos.atr_at_entry is not None:
                atr_pct = (pos.atr_at_entry / pos.entry_price) * 100
                print(f"Final ATR: ${pos.atr_at_entry:.4f} ({atr_pct:.2f}% of price)")
        
        print(f"PNL: ${pos.pnl:+.4f}")
        print(f"Strategy {pos.strategy} PNL: ${self.total_pnl[pos.strategy]:+.4f}")
        print(f"Total PNL: ${self.total_pnl['ALL']:+.4f}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"{'*'*80}\n")
        
        # Remove from positions list
        self.positions.remove(pos)
        
        # Save results
        self.save_results()
    
    def print_status(self):
        """Print current status"""
        print(f"\n--- Status Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Completed Trades: {len(self.trades)}")
        print(f"Total PNL: ${self.total_pnl['ALL']:+.4f}")
        
        print("\nPNL by Strategy:")
        for strategy in Strategy:
            trades_count = sum(1 for t in self.trades if t.strategy == strategy.value)
            wins = sum(1 for t in self.trades if t.strategy == strategy.value and t.pnl > 0)
            win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
            print(f"  {strategy.value}: ${self.total_pnl[strategy.value]:+.4f} "
                  f"({trades_count} trades, {win_rate:.1f}% win rate)")
        
        if self.positions:
            print("\nOpen Positions:")
            for pos in self.positions:
                time_remaining = (pos.period_end - datetime.now()).total_seconds() / 60
                
                # Basic position info
                print(f"  [{pos.strategy}] {pos.asset_type}: Entry ${pos.entry_price:.4f}, "
                      f"TP ${pos.take_profit:.4f}, SL ${pos.stop_loss:.4f}, "
                      f"Closes in {time_remaining:.1f} min")
                
                # For Strategy C, show current ATR metrics
                if pos.strategy == 'C':
                    current_atr = self.calculate_atr(pos.asset_type, period_seconds=15, num_periods=6)
                    if current_atr is not None:
                        prob_movement, expected_magnitude = self.atr_calc.estimate_movement_probability(
                            current_atr, pos.entry_price
                        )
                        atr_pct = (current_atr / pos.entry_price) * 100
                        print(f"       🎯 ATR: ${current_atr:.4f} ({atr_pct:.2f}%), "
                              f"P(move): {prob_movement:.1%}, "
                              f"Exp.Mag: ${expected_magnitude:.4f}")
        print()
    
    def save_results(self):
        """Save trading results to file"""
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f'trading_results_dynamic_{date_str}.json'
        
        results = {
            'last_updated': datetime.now().isoformat(),
            'date': date_str,
            'total_pnl': self.total_pnl,
            'total_trades': len(self.trades),
            'open_positions': len(self.positions),
            'trades': [],
            'open_positions_detail': []
        }
        
        # Convert trades to dict
        for trade in self.trades:
            trade_dict = asdict(trade)
            trade_dict['timestamp'] = trade.timestamp.isoformat()
            trade_dict['period_end'] = trade.period_end.isoformat()
            results['trades'].append(trade_dict)
        
        # Convert open positions to dict
        for pos in self.positions:
            pos_dict = asdict(pos)
            pos_dict['entry_time'] = pos.entry_time.isoformat()
            pos_dict['period_end'] = pos.period_end.isoformat()
            if pos.exit_time:
                pos_dict['exit_time'] = pos.exit_time.isoformat()
            results['open_positions_detail'].append(pos_dict)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def run(self):
        """Main trading loop"""
        print("="*80)
        print("Binary Options Trading Simulator - Dynamic TP/SL - Continuous")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nStrategies (all run continuously):")
        print("  A:  BTC_CALL only, TP +$0.06, SL -$0.09")
        print("  A2: BTC_CALL only when choppiness > 30, TP +$0.06, SL -$0.09")
        print("  B:  Both BTC_CALL and BTC_PUT (INDEPENDENT)")
        print("      - Each can re-enter immediately when closed")
        print("      - CALL can re-open even if PUT is still active (and vice versa)")
        print("      - TP +$0.06, SL -$0.09 for both")
        print("  C:  BTC_CALL with dynamic TP/SL (2:3 ratio)")
        print("      Based on ATR (6 periods of 15 seconds, exponentially weighted)")
        print("      Higher ATR = Larger TP/SL, Lower ATR = Smaller TP/SL")
        print("\n⚠️  CRITICAL LIMITS:")
        print("  - ALL Take Profits capped at $0.99 (max allowed)")
        print("  - ALL positions force-close when bid reaches $0.99")
        print("\nAll strategies enter immediately when no position exists")
        print("Price checks every 1 second for TP/SL exits")
        print(f"Results saved to: trading_results_dynamic_YYYYMMDD.json")
        print("="*80)
        
        last_status_time = time.time()
        
        try:
            while True:
                current_time = datetime.now()
                
                # Read market data
                data = self.read_market_data()
                
                # Update price buffers for choppiness calculation
                self.update_price_buffers(data, current_time)
                
                # Check for TP/SL exits first (most frequent)
                self.check_tpsl_exit(data, current_time)
                
                # Check for positions at expiry
                self.close_positions_at_expiry(data, current_time)
                
                # Read oscillation data
                oscillation_data = self.read_oscillation_data()
                
                # Check for new trade opportunities
                trade_signals = self.check_trade_conditions(data, current_time, oscillation_data)
                
                for signal in trade_signals:
                    asset_type, entry_price, period_end, strategy, tp, sl, atr_value = signal
                    
                    self.open_position(
                        asset_type, entry_price, current_time, period_end, 
                        strategy, tp, sl, oscillation_data, atr_value
                    )
                
                # Print status every 30 seconds
                if time.time() - last_status_time > 30:
                    self.print_status()
                    last_status_time = time.time()
                
                # Sleep for 1 second
                time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\n\nTrading stopped by user")
            self.print_status()
            self.save_results()
            
            print("\nFinal Summary:")
            print(f"  Total Trades: {len(self.trades)}")
            print(f"  Total PNL: ${self.total_pnl['ALL']:+.4f}")
            print("\nBy Strategy:")
            for strategy in Strategy:
                strategy_trades = [t for t in self.trades if t.strategy == strategy.value]
                if strategy_trades:
                    winning = sum(1 for t in strategy_trades if t.pnl > 0)
                    win_rate = 100 * winning / len(strategy_trades)
                    avg_pnl = sum(t.pnl for t in strategy_trades) / len(strategy_trades)
                    print(f"  {strategy.value}: {len(strategy_trades)} trades, "
                          f"${self.total_pnl[strategy.value]:+.4f}, "
                          f"Win Rate: {win_rate:.1f}%, Avg PNL: ${avg_pnl:+.4f}")
            
            print("\nBy Exit Reason:")
            exit_reasons = {}
            for trade in self.trades:
                if trade.exit_reason not in exit_reasons:
                    exit_reasons[trade.exit_reason] = {'count': 0, 'pnl': 0.0}
                exit_reasons[trade.exit_reason]['count'] += 1
                exit_reasons[trade.exit_reason]['pnl'] += trade.pnl
            
            for reason, stats in sorted(exit_reasons.items()):
                avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  {reason}: {stats['count']} trades, "
                      f"${stats['pnl']:+.4f}, Avg: ${avg_pnl:+.4f}")

if __name__ == "__main__":
    trader = BinaryOptionsTrader()
    trader.run()
