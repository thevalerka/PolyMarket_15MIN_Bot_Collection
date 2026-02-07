import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# pm2 start simulator_5.py --cron-restart="00 * * * *" --interpreter python3

class Strategy(Enum):
    A = "A"          # 0.30-0.42, >10 min, combined max 0.80, sell both when combined bid < 0.24 or > 0.95
    A2 = "A2"        # 0.30-0.42, >7 min, combined max 0.80, sell both when combined bid < 0.24 or > 0.95
    B = "B"          # 0.24-0.47, >10 min, combined max 0.77, sell both when combined bid < 0.24 or > 0.95
    B2 = "B2"        # 0.24-0.47, >10 min, combined max 0.77, sell both when combined bid < 0.24 or > 0.93
    C = "C"          # Combined <0.8, >10 min, sell both when combined bid < 0.24 or > 0.95
    D = "D_trend"    # Buy when COIN1 is >0.75, and sum COIN1+COIN2 is <0.90, sell @ combined 0.99

@dataclass
class Position:
    """Represents an open position"""
    asset_type: str  # "BTC_PUT", "ETH_CALL", "SOL_PUT", "XRP_CALL", etc.
    entry_price: float
    entry_time: datetime
    period_end: datetime
    strategy: str
    pair_id: str  # Unique identifier for the trading pair
    market_conditions: Optional[Dict] = None  # Market oscillation data at entry
    correlation: Optional[float] = None  # Correlation between the pair assets
    size: float = 1.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None

@dataclass
class Trade:
    """Represents a completed trade pair"""
    timestamp: datetime
    period_end: datetime
    strategy: str
    pair_type: str  # "BTC-ETH", "BTC-SOL", "ETH-SOL", "BTC-XRP", etc.
    asset1: str
    asset1_entry: float
    asset1_exit: float
    asset2: str
    asset2_entry: float
    asset2_exit: float
    asset1_pnl: float
    asset2_pnl: float
    total_pnl: float
    market_conditions: Optional[Dict] = None  # Market conditions at entry
    correlation: Optional[float] = None  # Correlation between the pair assets

class BinaryOptionsTrader:
    def __init__(self):
        self.data_files = {
            'BTC_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
            'BTC_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
            'ETH_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_ETH.json',
            'ETH_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_ETH.json',
            'SOL_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_SOL.json',
            'SOL_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_SOL.json',
            'XRP_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_XRP.json',
            'XRP_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_XRP.json'
        }

        self.oscillation_file = '/home/ubuntu/013_2025_polymarket/bot004_blackScholes/data/latest_oscillation.json'
        self.correlation_file = '/home/ubuntu/013_2025_polymarket/bybit_correlations.json'

        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.total_pnl = {strategy.value: 0.0 for strategy in Strategy}
        self.total_pnl['ALL'] = 0.0

        # Track which strategy-pair combinations have been used for each period
        self.period_strategy_pairs = {}

        # Load existing data from today's file on startup
        self.load_existing_data()

    def get_next_period_end(self, current_time: datetime) -> datetime:
        """Calculate the next period end time (00, 15, 30, 45 minutes)"""
        minute = current_time.minute

        # Find next period marker
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
        """Read all market data files"""
        data = {}
        for asset_name, filepath in self.data_files.items():
            try:
                with open(filepath, 'r') as f:
                    data[asset_name] = json.load(f)
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
            print(f"Warning: Oscillation file not found: {self.oscillation_file}")
            return None
        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding {self.oscillation_file}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Unexpected error reading oscillation data: {e}")
            return None

    def read_correlation_data(self) -> Optional[Dict]:
        """Read correlation data between cryptocurrency pairs"""
        try:
            with open(self.correlation_file, 'r') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            print(f"Warning: Correlation file not found: {self.correlation_file}")
            return None
        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding {self.correlation_file}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Unexpected error reading correlation data: {e}")
            return None

    def get_correlation(self, coin1: str, coin2: str, correlation_data: Optional[Dict]) -> Optional[float]:
        """
        Get correlation value for a pair of coins.
        Handles both directions (BTC_SOL and SOL_BTC are the same).
        Expects data structure: {"correlations": {"BTC_SOL": {"correlation": 0.89, ...}, ...}}
        """
        if not correlation_data:
            return None
        
        # Get the correlations dictionary
        correlations = correlation_data.get('correlations', {})
        if not correlations:
            print(f"Warning: No 'correlations' key found in correlation data")
            return None
        
        # Try both combinations
        pair_key1 = f"{coin1}_{coin2}"
        pair_key2 = f"{coin2}_{coin1}"
        
        # Look for the pair in either direction
        pair_data = correlations.get(pair_key1) or correlations.get(pair_key2)
        
        if pair_data is None:
            print(f"Warning: Correlation not found for {coin1}-{coin2}")
            return None
        
        # Extract the actual correlation value from the nested structure
        correlation = pair_data.get('correlation')
        
        if correlation is None:
            print(f"Warning: Correlation value missing for {coin1}-{coin2}")
            return None
        
        return correlation

    def load_existing_data(self):
        """Load existing trades and PNL data from today's file on startup"""
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f'trading_results5_{date_str}.json'

        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)

                # Load existing trades
                for trade_dict in existing_data.get('trades', []):
                    trade = Trade(
                        timestamp=datetime.fromisoformat(trade_dict['timestamp']),
                        period_end=datetime.fromisoformat(trade_dict['period_end']),
                        strategy=trade_dict['strategy'],
                        pair_type=trade_dict['pair_type'],
                        asset1=trade_dict['asset1'],
                        asset1_entry=trade_dict['asset1_entry'],
                        asset1_exit=trade_dict['asset1_exit'],
                        asset2=trade_dict['asset2'],
                        asset2_entry=trade_dict['asset2_entry'],
                        asset2_exit=trade_dict['asset2_exit'],
                        asset1_pnl=trade_dict['asset1_pnl'],
                        asset2_pnl=trade_dict['asset2_pnl'],
                        total_pnl=trade_dict['total_pnl'],
                        market_conditions=trade_dict.get('market_conditions'),
                        correlation=trade_dict.get('correlation')
                    )
                    self.trades.append(trade)

                # Restore PNL totals
                if 'total_pnl' in existing_data:
                    self.total_pnl = existing_data['total_pnl']

                print(f"\n{'='*80}")
                print(f"LOADED EXISTING DATA FROM {filename}")
                print(f"  Existing Trades: {len(self.trades)}")
                print(f"  Cumulative PNL: ${self.total_pnl.get('ALL', 0.0):+.2f}")
                print(f"{'='*80}\n")

        except FileNotFoundError:
            print(f"\nNo existing file found for today ({filename}). Starting fresh.")
        except Exception as e:
            print(f"\nWarning: Could not load existing data: {e}. Starting fresh.")

    def get_pair_id(self, strategy: str, period_end: datetime, coin1: str, coin2: str) -> str:
        """Generate unique pair identifier"""
        coins = tuple(sorted([coin1, coin2]))
        return f"{strategy}_{period_end.isoformat()}_{coins[0]}_{coins[1]}"

    def is_pair_used(self, strategy: str, period_end: datetime, coin1: str, coin2: str) -> bool:
        """Check if a strategy-pair combination has already been used for this period"""
        pair_id = self.get_pair_id(strategy, period_end, coin1, coin2)
        period_key = period_end.isoformat()

        if period_key not in self.period_strategy_pairs:
            self.period_strategy_pairs[period_key] = set()

        return pair_id in self.period_strategy_pairs[period_key]

    def mark_pair_used(self, strategy: str, period_end: datetime, coin1: str, coin2: str):
        """Mark a strategy-pair combination as used for this period"""
        pair_id = self.get_pair_id(strategy, period_end, coin1, coin2)
        period_key = period_end.isoformat()

        if period_key not in self.period_strategy_pairs:
            self.period_strategy_pairs[period_key] = set()

        self.period_strategy_pairs[period_key].add(pair_id)

    def check_strategy_a(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy A: 0.30-0.42, >10 minutes, sum <= 0.80 - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []
        coins = ['BTC', 'ETH', 'SOL', 'XRP']

        # Check all possible coin pair combinations
        for i in range(len(coins)):
            for j in range(i + 1, len(coins)):
                coin1, coin2 = coins[i], coins[j]

                if not self.is_pair_used(Strategy.A.value, period_end, coin1, coin2):
                    # Check PUT-CALL combination
                    put_key = f'{coin1}_PUT'
                    call_key = f'{coin2}_CALL'
                    if (prices[put_key] and prices[call_key] and
                        0.30 <= prices[put_key] <= 0.42 and
                        0.30 <= prices[call_key] <= 0.42 and
                        (prices[put_key] + prices[call_key]) <= 0.80):
                        signals.append((put_key, prices[put_key], call_key, prices[call_key],
                                      period_end, Strategy.A.value, f'{coin1}-{coin2}'))

                    # Check CALL-PUT combination
                    call_key = f'{coin1}_CALL'
                    put_key = f'{coin2}_PUT'
                    if (prices[call_key] and prices[put_key] and
                        0.30 <= prices[call_key] <= 0.42 and
                        0.30 <= prices[put_key] <= 0.42 and
                        (prices[call_key] + prices[put_key]) <= 0.80):
                        signals.append((call_key, prices[call_key], put_key, prices[put_key],
                                      period_end, Strategy.A.value, f'{coin1}-{coin2}'))

        return signals

    def check_strategy_a2(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy A2: 0.30-0.42, >7 minutes, sum <= 0.80 - check all coin pairs"""
        if time_to_end <= 7:
            return []

        signals = []
        coins = ['BTC', 'ETH', 'SOL', 'XRP']

        # Check all possible coin pair combinations
        for i in range(len(coins)):
            for j in range(i + 1, len(coins)):
                coin1, coin2 = coins[i], coins[j]

                if not self.is_pair_used(Strategy.A2.value, period_end, coin1, coin2):
                    # Check PUT-CALL combination
                    put_key = f'{coin1}_PUT'
                    call_key = f'{coin2}_CALL'
                    if (prices[put_key] and prices[call_key] and
                        0.30 <= prices[put_key] <= 0.42 and
                        0.30 <= prices[call_key] <= 0.42 and
                        (prices[put_key] + prices[call_key]) <= 0.80):
                        signals.append((put_key, prices[put_key], call_key, prices[call_key],
                                      period_end, Strategy.A2.value, f'{coin1}-{coin2}'))

                    # Check CALL-PUT combination
                    call_key = f'{coin1}_CALL'
                    put_key = f'{coin2}_PUT'
                    if (prices[call_key] and prices[put_key] and
                        0.30 <= prices[call_key] <= 0.42 and
                        0.30 <= prices[put_key] <= 0.42 and
                        (prices[call_key] + prices[put_key]) <= 0.80):
                        signals.append((call_key, prices[call_key], put_key, prices[put_key],
                                      period_end, Strategy.A2.value, f'{coin1}-{coin2}'))

        return signals

    def check_strategy_b(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy B: 0.24-0.47, >10 minutes, sum <= 0.77 - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []
        coins = ['BTC', 'ETH', 'SOL', 'XRP']

        # Check all possible coin pair combinations
        for i in range(len(coins)):
            for j in range(i + 1, len(coins)):
                coin1, coin2 = coins[i], coins[j]

                if not self.is_pair_used(Strategy.B.value, period_end, coin1, coin2):
                    # Check PUT-CALL combination
                    put_key = f'{coin1}_PUT'
                    call_key = f'{coin2}_CALL'
                    if (prices[put_key] and prices[call_key] and
                        0.24 <= prices[put_key] <= 0.47 and
                        0.24 <= prices[call_key] <= 0.47 and
                        (prices[put_key] + prices[call_key]) <= 0.77):
                        signals.append((put_key, prices[put_key], call_key, prices[call_key],
                                      period_end, Strategy.B.value, f'{coin1}-{coin2}'))

                    # Check CALL-PUT combination
                    call_key = f'{coin1}_CALL'
                    put_key = f'{coin2}_PUT'
                    if (prices[call_key] and prices[put_key] and
                        0.24 <= prices[call_key] <= 0.47 and
                        0.24 <= prices[put_key] <= 0.47 and
                        (prices[call_key] + prices[put_key]) <= 0.77):
                        signals.append((call_key, prices[call_key], put_key, prices[put_key],
                                      period_end, Strategy.B.value, f'{coin1}-{coin2}'))

        return signals

    def check_strategy_b2(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy B2: 0.24-0.47, >10 minutes, sum <= 0.77 - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []
        coins = ['BTC', 'ETH', 'SOL', 'XRP']

        # Check all possible coin pair combinations
        for i in range(len(coins)):
            for j in range(i + 1, len(coins)):
                coin1, coin2 = coins[i], coins[j]

                if not self.is_pair_used(Strategy.B2.value, period_end, coin1, coin2):
                    # Check PUT-CALL combination
                    put_key = f'{coin1}_PUT'
                    call_key = f'{coin2}_CALL'
                    if (prices[put_key] and prices[call_key] and
                        0.24 <= prices[put_key] <= 0.47 and
                        0.24 <= prices[call_key] <= 0.47 and
                        (prices[put_key] + prices[call_key]) <= 0.77):
                        signals.append((put_key, prices[put_key], call_key, prices[call_key],
                                      period_end, Strategy.B2.value, f'{coin1}-{coin2}'))

                    # Check CALL-PUT combination
                    call_key = f'{coin1}_CALL'
                    put_key = f'{coin2}_PUT'
                    if (prices[call_key] and prices[put_key] and
                        0.24 <= prices[call_key] <= 0.47 and
                        0.24 <= prices[put_key] <= 0.47 and
                        (prices[call_key] + prices[put_key]) <= 0.77):
                        signals.append((call_key, prices[call_key], put_key, prices[put_key],
                                      period_end, Strategy.B2.value, f'{coin1}-{coin2}'))

        return signals

    def check_strategy_c(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy C: Combined price <0.8, >10 minutes - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []
        coins = ['BTC', 'ETH', 'SOL', 'XRP']

        # Check all possible coin pair combinations
        for i in range(len(coins)):
            for j in range(i + 1, len(coins)):
                coin1, coin2 = coins[i], coins[j]

                if not self.is_pair_used(Strategy.C.value, period_end, coin1, coin2):
                    # Check PUT-CALL combination
                    put_key = f'{coin1}_PUT'
                    call_key = f'{coin2}_CALL'
                    if (prices[put_key] and prices[call_key] and
                        (prices[put_key] + prices[call_key]) < 0.8):
                        signals.append((put_key, prices[put_key], call_key, prices[call_key],
                                      period_end, Strategy.C.value, f'{coin1}-{coin2}'))

                    # Check CALL-PUT combination
                    call_key = f'{coin1}_CALL'
                    put_key = f'{coin2}_PUT'
                    if (prices[call_key] and prices[put_key] and
                        (prices[call_key] + prices[put_key]) < 0.8):
                        signals.append((call_key, prices[call_key], put_key, prices[put_key],
                                      period_end, Strategy.C.value, f'{coin1}-{coin2}'))

        return signals

    def check_strategy_d(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy D: Buy when COIN1 is >0.75, and sum COIN1+COIN2 is <0.90 - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []
        coins = ['BTC', 'ETH', 'SOL', 'XRP']

        # Check all possible coin pair combinations
        for i in range(len(coins)):
            for j in range(i + 1, len(coins)):
                coin1, coin2 = coins[i], coins[j]

                if not self.is_pair_used(Strategy.D.value, period_end, coin1, coin2):
                    # Check PUT-CALL combination (COIN1 = PUT asset)
                    put_key = f'{coin1}_PUT'
                    call_key = f'{coin2}_CALL'
                    if (prices[put_key] and prices[call_key] and
                        prices[put_key] > 0.75 and
                        (prices[put_key] + prices[call_key]) < 0.90):
                        signals.append((put_key, prices[put_key], call_key, prices[call_key],
                                      period_end, Strategy.D.value, f'{coin1}-{coin2}'))

                    # Check CALL-PUT combination (COIN1 = CALL asset)
                    call_key = f'{coin1}_CALL'
                    put_key = f'{coin2}_PUT'
                    if (prices[call_key] and prices[put_key] and
                        prices[call_key] > 0.75 and
                        (prices[call_key] + prices[put_key]) < 0.90):
                        signals.append((call_key, prices[call_key], put_key, prices[put_key],
                                      period_end, Strategy.D.value, f'{coin1}-{coin2}'))

        return signals

    def check_trade_conditions(self, data: Dict[str, Dict], current_time: datetime) -> List[tuple]:
        """Check all strategies for trade conditions across all coin pairs"""
        period_end = self.get_next_period_end(current_time)
        time_to_end = (period_end - current_time).total_seconds() / 60

        # Extract all prices
        try:
            prices = {
                'BTC_PUT': data['BTC_PUT']['best_ask']['price'] if data['BTC_PUT'] else None,
                'BTC_CALL': data['BTC_CALL']['best_ask']['price'] if data['BTC_CALL'] else None,
                'ETH_PUT': data['ETH_PUT']['best_ask']['price'] if data['ETH_PUT'] else None,
                'ETH_CALL': data['ETH_CALL']['best_ask']['price'] if data['ETH_CALL'] else None,
                'SOL_PUT': data['SOL_PUT']['best_ask']['price'] if data['SOL_PUT'] else None,
                'SOL_CALL': data['SOL_CALL']['best_ask']['price'] if data['SOL_CALL'] else None,
                'XRP_PUT': data['XRP_PUT']['best_ask']['price'] if data['XRP_PUT'] else None,
                'XRP_CALL': data['XRP_CALL']['best_ask']['price'] if data['XRP_CALL'] else None,
            }
        except (KeyError, TypeError):
            return []

        # Collect all signals from all strategies
        all_signals = []

        all_signals.extend(self.check_strategy_a(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_a2(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_b(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_b2(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_c(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_d(prices, time_to_end, period_end))

        return all_signals

    def open_position(self, asset_type: str, entry_price: float, entry_time: datetime,
                     period_end: datetime, strategy: str, pair_id: str, 
                     market_conditions: Optional[Dict] = None, correlation: Optional[float] = None):
        """Open a new position"""

        print(f"\n[DEBUG] open_position called")
        print(f"[DEBUG] market_conditions type: {type(market_conditions)}")
        print(f"[DEBUG] market_conditions is None: {market_conditions is None}")
        if market_conditions:
            print(f"[DEBUG] market_conditions keys: {list(market_conditions.keys())}")

        position = Position(
            asset_type=asset_type,
            entry_price=entry_price,
            entry_time=entry_time,
            period_end=period_end,
            strategy=strategy,
            pair_id=pair_id,
            market_conditions=market_conditions,
            correlation=correlation
        )

        print(f"[DEBUG] Position created, market_conditions in position: {position.market_conditions is not None}")
        print(f"[DEBUG] Correlation: {correlation}")

        self.positions.append(position)

        coin = asset_type.split('_')[0]
        option = asset_type.split('_')[1]

        print(f"\n{'='*80}")
        print(f"[OPEN] {entry_time.strftime('%Y-%m-%d %H:%M:%S')} - Strategy {strategy}")
        print(f"  Asset: {coin} {option}")
        print(f"  Entry Price: ${entry_price:.2f}")
        print(f"  Period End: {period_end.strftime('%H:%M')}")
        print(f"  Time to End: {((period_end - entry_time).total_seconds() / 60):.1f} minutes")

        if market_conditions:
            print(f"\n  Market Conditions:")
            print(f"    Volatility: {market_conditions.get('volatility', 'N/A'):.4f}" if market_conditions.get('volatility') else "    Volatility: N/A")
            print(f"    Time to Expiry: {market_conditions.get('time_to_expiry_minutes', 'N/A'):.2f} min" if market_conditions.get('time_to_expiry_minutes') else "    Time to Expiry: N/A")
            print(f"    MA Crossings (15m): {market_conditions.get('ma_5min_crossings_last_15min', 'N/A')}")
            print(f"    Max Swing Distance: {market_conditions.get('max_swing_distance', 'N/A'):.2f}" if market_conditions.get('max_swing_distance') else "    Max Swing Distance: N/A")
            if 'oscillation_60min' in market_conditions:
                osc = market_conditions['oscillation_60min']
                print(f"    Efficiency Ratio: {osc.get('efficiency_ratio', 'N/A'):.4f}" if osc.get('efficiency_ratio') else "    Efficiency Ratio: N/A")
                print(f"    Choppiness Index: {osc.get('choppiness_index', 'N/A'):.2f}" if osc.get('choppiness_index') else "    Choppiness Index: N/A")
        else:
            print(f"\n  [WARNING] Market Conditions: NOT AVAILABLE")

        if correlation is not None:
            print(f"  Correlation: {correlation:.4f}")
        else:
            print(f"  Correlation: NOT AVAILABLE")

        print(f"{'='*80}\n")

        # Save to JSON immediately
        self.save_results()

    def check_early_exit(self, data: Dict[str, Dict], current_time: datetime):
        """Check if any positions should be closed early based on strategy rules"""
        positions_to_close = []
        
        # Group positions by pair_id to check combined prices
        pair_positions = {}
        for pos in self.positions:
            if pos.pair_id not in pair_positions:
                pair_positions[pos.pair_id] = []
            pair_positions[pos.pair_id].append(pos)
        
        # Check each pair for early exit conditions
        for pair_id, positions in pair_positions.items():
            if len(positions) != 2:
                continue
            
            pos1, pos2 = positions
            strategy = pos1.strategy
            
            # Get current bid prices for both positions
            try:
                bid1 = data[pos1.asset_type]['best_bid']['price']
                bid2 = data[pos2.asset_type]['best_bid']['price']
            except (KeyError, TypeError):
                continue
            
            combined_bid = bid1 + bid2
            should_close = False
            exit_reason = ""
            
            # Strategy A: sell both when combined price bid < 0.24 or > 0.95
            if strategy == Strategy.A.value:
                if combined_bid < 0.24:
                    should_close = True
                    exit_reason = "Combined bid < 0.24"
                elif combined_bid > 0.95:
                    should_close = True
                    exit_reason = "Combined bid > 0.95"
            
            # Strategy A2: sell both when combined price bid < 0.24 or > 0.95
            elif strategy == Strategy.A2.value:
                if combined_bid < 0.24:
                    should_close = True
                    exit_reason = "Combined bid < 0.24"
                elif combined_bid > 0.95:
                    should_close = True
                    exit_reason = "Combined bid > 0.95"
            
            # Strategy B: sell both when combined price bid < 0.24 or > 0.95
            elif strategy == Strategy.B.value:
                if combined_bid < 0.24:
                    should_close = True
                    exit_reason = "Combined bid < 0.24"
                elif combined_bid > 0.95:
                    should_close = True
                    exit_reason = "Combined bid > 0.95"
            
            # Strategy B2: sell both when combined price bid < 0.24 or > 0.93
            elif strategy == Strategy.B2.value:
                if combined_bid < 0.24:
                    should_close = True
                    exit_reason = "Combined bid < 0.24"
                elif combined_bid > 0.93:
                    should_close = True
                    exit_reason = "Combined bid > 0.93"
            
            # Strategy C: sell both when combined price bid < 0.24 or > 0.95
            elif strategy == Strategy.C.value:
                if combined_bid < 0.24:
                    should_close = True
                    exit_reason = "Combined bid < 0.24"
                elif combined_bid > 0.95:
                    should_close = True
                    exit_reason = "Combined bid > 0.95"
            
            # Strategy D: sell @ combined 0.99
            elif strategy == Strategy.D.value:
                if combined_bid >= 0.99:
                    should_close = True
                    exit_reason = "Combined bid >= 0.99"
            
            if should_close:
                print(f"\n[EARLY EXIT] Strategy {strategy} - {exit_reason}")
                print(f"  Combined Bid: ${combined_bid:.2f}")
                positions_to_close.extend(positions)
        
        # Close the positions
        if positions_to_close:
            self._execute_close(positions_to_close, data, current_time, early_exit=True)

    def close_positions(self, data: Dict[str, Dict], current_time: datetime):
        """Close positions that have reached their period end"""
        period_end = self.get_next_period_end(current_time)

        # Find positions to close (within 5 seconds of period end)
        positions_to_close = [pos for pos in self.positions
                             if pos.period_end == period_end and
                             (period_end - current_time).total_seconds() <= 5]

        if not positions_to_close:
            return

        self._execute_close(positions_to_close, data, current_time, early_exit=False)

    def _execute_close(self, positions_to_close: List[Position], data: Dict[str, Dict], 
                      current_time: datetime, early_exit: bool = False):
        """Execute the closing of positions"""
        # Group by pair_id for trade recording
        trade_pairs = {}

        for pos in positions_to_close:
            # Get exit price from best bid, or 0.0 if unavailable
            try:
                exit_price = data[pos.asset_type]['best_bid']['price']
            except (KeyError, TypeError):
                print(f"Warning: Exit price unavailable for {pos.asset_type}, assuming total loss (0.0)")
                exit_price = 0.0

            pos.exit_price = exit_price
            pos.exit_time = current_time
            pos.pnl = (exit_price - pos.entry_price) * pos.size

            # Update total PNL
            self.total_pnl[pos.strategy] += pos.pnl
            self.total_pnl['ALL'] += pos.pnl

            coin = pos.asset_type.split('_')[0]
            option = pos.asset_type.split('_')[1]

            exit_type = "[EARLY EXIT]" if early_exit else "[CLOSE]"
            print(f"\n{'='*80}")
            print(f"{exit_type} {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Strategy {pos.strategy}")
            print(f"  Asset: {coin} {option}")
            print(f"  Entry: ${pos.entry_price:.2f} @ {pos.entry_time.strftime('%H:%M:%S')}")
            print(f"  Exit:  ${pos.exit_price:.2f} @ {pos.exit_time.strftime('%H:%M:%S')}")
            print(f"  PNL: ${pos.pnl:+.2f}")
            print(f"{'='*80}\n")

            # Group positions by pair_id for trade pair creation
            if pos.pair_id not in trade_pairs:
                trade_pairs[pos.pair_id] = []
            trade_pairs[pos.pair_id].append(pos)

        # Record trade pairs
        for pair_id, pair_positions in trade_pairs.items():
            if len(pair_positions) == 2:
                pos1, pos2 = pair_positions

                # Determine pair type from the positions
                coin1 = pos1.asset_type.split('_')[0]
                coin2 = pos2.asset_type.split('_')[0]
                pair_type = f"{coin1}-{coin2}" if coin1 < coin2 else f"{coin2}-{coin1}"

                trade = Trade(
                    timestamp=pos1.entry_time,
                    period_end=pos1.period_end,
                    strategy=pos1.strategy,
                    pair_type=pair_type,
                    asset1=pos1.asset_type,
                    asset1_entry=pos1.entry_price,
                    asset1_exit=pos1.exit_price,
                    asset2=pos2.asset_type,
                    asset2_entry=pos2.entry_price,
                    asset2_exit=pos2.exit_price,
                    asset1_pnl=pos1.pnl,
                    asset2_pnl=pos2.pnl,
                    total_pnl=pos1.pnl + pos2.pnl,
                    market_conditions=pos1.market_conditions,  # Use market conditions from first position
                    correlation=pos1.correlation  # Use correlation from first position (both should be the same)
                )
                self.trades.append(trade)

                exit_type = "EARLY EXIT" if early_exit else "TRADE PAIR COMPLETED"
                print(f"\n{'*'*80}")
                print(f"{exit_type} - Strategy {trade.strategy} - {pair_type}")
                print(f"  {pos1.asset_type}: ${pos1.pnl:+.2f}")
                print(f"  {pos2.asset_type}: ${pos2.pnl:+.2f}")
                print(f"  Total PNL: ${trade.total_pnl:+.2f}")
                print(f"  Strategy {trade.strategy} PNL: ${self.total_pnl[trade.strategy]:+.2f}")
                print(f"  Cumulative PNL: ${self.total_pnl['ALL']:+.2f}")
                print(f"  Total Trades: {len(self.trades)}")
                print(f"{'*'*80}\n")

        # Remove closed positions
        self.positions = [pos for pos in self.positions if pos not in positions_to_close]

        # Save to JSON immediately after closing
        self.save_results()

    def print_status(self):
        """Print current status"""
        print(f"\n--- Status Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Completed Trades: {len(self.trades)}")
        print(f"Total PNL: ${self.total_pnl['ALL']:+.2f}")

        # PNL by strategy
        print("\nPNL by Strategy:")
        for strategy in Strategy:
            trades_count = sum(1 for t in self.trades if t.strategy == strategy.value)
            print(f"  {strategy.value}: ${self.total_pnl[strategy.value]:+.2f} ({trades_count} trades)")

        # PNL by pair type
        pair_types = {}
        for trade in self.trades:
            if trade.pair_type not in pair_types:
                pair_types[trade.pair_type] = {'pnl': 0.0, 'count': 0}
            pair_types[trade.pair_type]['pnl'] += trade.total_pnl
            pair_types[trade.pair_type]['count'] += 1

        if pair_types:
            print("\nPNL by Pair Type:")
            for pair, stats in sorted(pair_types.items()):
                print(f"  {pair}: ${stats['pnl']:+.2f} ({stats['count']} trades)")

        if self.positions:
            print("\nOpen Positions:")
            for pos in self.positions:
                time_remaining = (pos.period_end - datetime.now()).total_seconds() / 60
                coin = pos.asset_type.split('_')[0]
                option = pos.asset_type.split('_')[1]
                print(f"  [{pos.strategy}] {coin} {option}: Entry ${pos.entry_price:.2f}, Closes in {time_remaining:.1f} min")
        print()

    def save_results(self):
        """Save trading results to file"""
        # Create filename with date
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f'trading_results5_{date_str}.json'

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
        print("Binary Options Trading Simulator - DRY RUN")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nCoin Pairs: BTC-ETH, BTC-SOL, ETH-SOL, BTC-XRP, ETH-XRP, SOL-XRP")
        print("\nStrategies:")
        print("  A:  $0.30-$0.42, sum ≤ $0.80, >10 min, exit: combined bid < $0.24 or > $0.95")
        print("  A2: $0.30-$0.42, sum ≤ $0.80, >7 min, exit: combined bid < $0.24 or > $0.95")
        print("  B:  $0.24-$0.47, sum ≤ $0.77, >10 min, exit: combined bid < $0.24 or > $0.95")
        print("  B2: $0.24-$0.47, sum ≤ $0.77, >10 min, exit: combined bid < $0.24 or > $0.93")
        print("  C:  Combined < $0.80, >10 min, exit: combined bid < $0.24 or > $0.95")
        print("  D:  COIN1 > $0.75 and sum < $0.90, exit: combined bid >= $0.99")
        print("\nAll strategies can run simultaneously per period for each pair")
        print("Exit price unavailable = Total loss ($0.00)")
        print(f"Results saved daily to: trading_results5_YYYYMMDD.json")
        print("="*80)

        last_status_time = time.time()

        try:
            while True:
                current_time = datetime.now()

                # Read market data
                data = self.read_market_data()

                # Check for early exit conditions first
                self.check_early_exit(data, current_time)

                # Check for position closes at period end
                self.close_positions(data, current_time)

                # Check for new trade opportunities (returns list of signals)
                trade_signals = self.check_trade_conditions(data, current_time)

                # Read oscillation and correlation data once per iteration if there are signals
                if trade_signals:
                    oscillation_data = self.read_oscillation_data()
                    correlation_data = self.read_correlation_data()

                    for signal in trade_signals:
                        asset1, price1, asset2, price2, period_end, strategy, pair_type = signal

                        # Generate unique pair_id for this trade pair
                        coin1 = asset1.split('_')[0]
                        coin2 = asset2.split('_')[0]
                        pair_id = self.get_pair_id(strategy, period_end, coin1, coin2)

                        # Get correlation for this pair
                        correlation = self.get_correlation(coin1, coin2, correlation_data)

                        # Open both positions with market conditions and correlation
                        self.open_position(asset1, price1, current_time, period_end, strategy, pair_id, 
                                         oscillation_data, correlation)
                        self.open_position(asset2, price2, current_time, period_end, strategy, pair_id, 
                                         oscillation_data, correlation)

                        # Mark this strategy-pair combination as used
                        self.mark_pair_used(strategy, period_end, coin1, coin2)

                # Print status every 30 seconds
                if time.time() - last_status_time > 30:
                    self.print_status()
                    last_status_time = time.time()

                # Sleep for 1 second
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nTrading stopped by user")
            self.print_status()
            self.save_results()

            print("\nFinal Summary:")
            print(f"  Total Trades: {len(self.trades)}")
            print(f"  Total PNL: ${self.total_pnl['ALL']:+.2f}")
            print("\nBy Strategy:")
            for strategy in Strategy:
                strategy_trades = [t for t in self.trades if t.strategy == strategy.value]
                if strategy_trades:
                    winning = sum(1 for t in strategy_trades if t.total_pnl > 0)
                    print(f"  {strategy.value}: {len(strategy_trades)} trades, "
                          f"${self.total_pnl[strategy.value]:+.2f}, "
                          f"Win Rate: {100*winning/len(strategy_trades):.1f}%")

            print("\nBy Pair Type:")
            pair_types = {}
            for trade in self.trades:
                if trade.pair_type not in pair_types:
                    pair_types[trade.pair_type] = {'pnl': 0.0, 'count': 0, 'wins': 0}
                pair_types[trade.pair_type]['pnl'] += trade.total_pnl
                pair_types[trade.pair_type]['count'] += 1
                if trade.total_pnl > 0:
                    pair_types[trade.pair_type]['wins'] += 1

            for pair, stats in sorted(pair_types.items()):
                win_rate = 100 * stats['wins'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  {pair}: {stats['count']} trades, ${stats['pnl']:+.2f}, "
                      f"Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    trader = BinaryOptionsTrader()
    trader.run()
