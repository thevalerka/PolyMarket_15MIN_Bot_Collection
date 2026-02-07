import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# pm2 start simulator_3.py  --interpreter python3

class Strategy(Enum):
    A = "A"      # 0.30-0.44, >10 min
    A2 = "A2"    # 0.30-0.44, >7 min
    B = "B"      # 0.24-0.47, >10 min
    C = "C"      # Combined <0.8, >10 min

@dataclass
class Position:
    """Represents an open position"""
    asset_type: str  # "BTC_PUT", "ETH_CALL", "SOL_PUT", etc.
    entry_price: float
    entry_time: datetime
    period_end: datetime
    strategy: str
    pair_id: str  # Unique identifier for the trading pair
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
    pair_type: str  # "BTC-ETH", "BTC-SOL", "ETH-SOL"
    asset1: str
    asset1_entry: float
    asset1_exit: float
    asset2: str
    asset2_entry: float
    asset2_exit: float
    asset1_pnl: float
    asset2_pnl: float
    total_pnl: float

class BinaryOptionsTrader:
    def __init__(self):
        self.data_files = {
            'BTC_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
            'BTC_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
            'ETH_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_ETH.json',
            'ETH_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_ETH.json',
            'SOL_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_SOL.json',
            'SOL_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_SOL.json'
        }

        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.total_pnl = {strategy.value: 0.0 for strategy in Strategy}
        self.total_pnl['ALL'] = 0.0

        # Track which strategy-pair combinations have been used for each period
        self.period_strategy_pairs = {}

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
        """Strategy A: 0.30-0.44, >10 minutes - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []

        # BTC-ETH combinations
        if not self.is_pair_used(Strategy.A.value, period_end, 'BTC', 'ETH'):
            if (prices['BTC_PUT'] and prices['ETH_CALL'] and
                0.30 <= prices['BTC_PUT'] <= 0.44 and
                0.30 <= prices['ETH_CALL'] <= 0.44):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'ETH_CALL', prices['ETH_CALL'],
                              period_end, Strategy.A.value, 'BTC-ETH'))

            elif (prices['BTC_CALL'] and prices['ETH_PUT'] and
                  0.30 <= prices['BTC_CALL'] <= 0.44 and
                  0.30 <= prices['ETH_PUT'] <= 0.44):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'ETH_PUT', prices['ETH_PUT'],
                              period_end, Strategy.A.value, 'BTC-ETH'))

        # BTC-SOL combinations
        if not self.is_pair_used(Strategy.A.value, period_end, 'BTC', 'SOL'):
            if (prices['BTC_PUT'] and prices['SOL_CALL'] and
                0.30 <= prices['BTC_PUT'] <= 0.44 and
                0.30 <= prices['SOL_CALL'] <= 0.44):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.A.value, 'BTC-SOL'))

            elif (prices['BTC_CALL'] and prices['SOL_PUT'] and
                  0.30 <= prices['BTC_CALL'] <= 0.44 and
                  0.30 <= prices['SOL_PUT'] <= 0.44):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.A.value, 'BTC-SOL'))

        # ETH-SOL combinations
        if not self.is_pair_used(Strategy.A.value, period_end, 'ETH', 'SOL'):
            if (prices['ETH_PUT'] and prices['SOL_CALL'] and
                0.30 <= prices['ETH_PUT'] <= 0.44 and
                0.30 <= prices['SOL_CALL'] <= 0.44):
                signals.append(('ETH_PUT', prices['ETH_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.A.value, 'ETH-SOL'))

            elif (prices['ETH_CALL'] and prices['SOL_PUT'] and
                  0.30 <= prices['ETH_CALL'] <= 0.44 and
                  0.30 <= prices['SOL_PUT'] <= 0.44):
                signals.append(('ETH_CALL', prices['ETH_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.A.value, 'ETH-SOL'))

        return signals

    def check_strategy_a2(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy A2: 0.30-0.44, >7 minutes - check all coin pairs"""
        if time_to_end <= 7:
            return []

        signals = []

        # BTC-ETH combinations
        if not self.is_pair_used(Strategy.A2.value, period_end, 'BTC', 'ETH'):
            if (prices['BTC_PUT'] and prices['ETH_CALL'] and
                0.30 <= prices['BTC_PUT'] <= 0.44 and
                0.30 <= prices['ETH_CALL'] <= 0.44):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'ETH_CALL', prices['ETH_CALL'],
                              period_end, Strategy.A2.value, 'BTC-ETH'))

            elif (prices['BTC_CALL'] and prices['ETH_PUT'] and
                  0.30 <= prices['BTC_CALL'] <= 0.44 and
                  0.30 <= prices['ETH_PUT'] <= 0.44):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'ETH_PUT', prices['ETH_PUT'],
                              period_end, Strategy.A2.value, 'BTC-ETH'))

        # BTC-SOL combinations
        if not self.is_pair_used(Strategy.A2.value, period_end, 'BTC', 'SOL'):
            if (prices['BTC_PUT'] and prices['SOL_CALL'] and
                0.30 <= prices['BTC_PUT'] <= 0.44 and
                0.30 <= prices['SOL_CALL'] <= 0.44):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.A2.value, 'BTC-SOL'))

            elif (prices['BTC_CALL'] and prices['SOL_PUT'] and
                  0.30 <= prices['BTC_CALL'] <= 0.44 and
                  0.30 <= prices['SOL_PUT'] <= 0.44):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.A2.value, 'BTC-SOL'))

        # ETH-SOL combinations
        if not self.is_pair_used(Strategy.A2.value, period_end, 'ETH', 'SOL'):
            if (prices['ETH_PUT'] and prices['SOL_CALL'] and
                0.30 <= prices['ETH_PUT'] <= 0.44 and
                0.30 <= prices['SOL_CALL'] <= 0.44):
                signals.append(('ETH_PUT', prices['ETH_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.A2.value, 'ETH-SOL'))

            elif (prices['ETH_CALL'] and prices['SOL_PUT'] and
                  0.30 <= prices['ETH_CALL'] <= 0.44 and
                  0.30 <= prices['SOL_PUT'] <= 0.44):
                signals.append(('ETH_CALL', prices['ETH_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.A2.value, 'ETH-SOL'))

        return signals

    def check_strategy_b(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy B: 0.24-0.47, >10 minutes - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []

        # BTC-ETH combinations
        if not self.is_pair_used(Strategy.B.value, period_end, 'BTC', 'ETH'):
            if (prices['BTC_PUT'] and prices['ETH_CALL'] and
                0.24 <= prices['BTC_PUT'] <= 0.47 and
                0.24 <= prices['ETH_CALL'] <= 0.47):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'ETH_CALL', prices['ETH_CALL'],
                              period_end, Strategy.B.value, 'BTC-ETH'))

            elif (prices['BTC_CALL'] and prices['ETH_PUT'] and
                  0.24 <= prices['BTC_CALL'] <= 0.47 and
                  0.24 <= prices['ETH_PUT'] <= 0.47):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'ETH_PUT', prices['ETH_PUT'],
                              period_end, Strategy.B.value, 'BTC-ETH'))

        # BTC-SOL combinations
        if not self.is_pair_used(Strategy.B.value, period_end, 'BTC', 'SOL'):
            if (prices['BTC_PUT'] and prices['SOL_CALL'] and
                0.24 <= prices['BTC_PUT'] <= 0.47 and
                0.24 <= prices['SOL_CALL'] <= 0.47):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.B.value, 'BTC-SOL'))

            elif (prices['BTC_CALL'] and prices['SOL_PUT'] and
                  0.24 <= prices['BTC_CALL'] <= 0.47 and
                  0.24 <= prices['SOL_PUT'] <= 0.47):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.B.value, 'BTC-SOL'))

        # ETH-SOL combinations
        if not self.is_pair_used(Strategy.B.value, period_end, 'ETH', 'SOL'):
            if (prices['ETH_PUT'] and prices['SOL_CALL'] and
                0.24 <= prices['ETH_PUT'] <= 0.47 and
                0.24 <= prices['SOL_CALL'] <= 0.47):
                signals.append(('ETH_PUT', prices['ETH_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.B.value, 'ETH-SOL'))

            elif (prices['ETH_CALL'] and prices['SOL_PUT'] and
                  0.24 <= prices['ETH_CALL'] <= 0.47 and
                  0.24 <= prices['SOL_PUT'] <= 0.47):
                signals.append(('ETH_CALL', prices['ETH_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.B.value, 'ETH-SOL'))

        return signals

    def check_strategy_c(self, prices: Dict, time_to_end: float, period_end: datetime) -> List[tuple]:
        """Strategy C: Combined price <0.8, >10 minutes - check all coin pairs"""
        if time_to_end <= 10:
            return []

        signals = []

        # BTC-ETH combinations
        if not self.is_pair_used(Strategy.C.value, period_end, 'BTC', 'ETH'):
            if (prices['BTC_PUT'] and prices['ETH_CALL'] and
                (prices['BTC_PUT'] + prices['ETH_CALL']) < 0.8):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'ETH_CALL', prices['ETH_CALL'],
                              period_end, Strategy.C.value, 'BTC-ETH'))

            elif (prices['BTC_CALL'] and prices['ETH_PUT'] and
                  (prices['BTC_CALL'] + prices['ETH_PUT']) < 0.8):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'ETH_PUT', prices['ETH_PUT'],
                              period_end, Strategy.C.value, 'BTC-ETH'))

        # BTC-SOL combinations
        if not self.is_pair_used(Strategy.C.value, period_end, 'BTC', 'SOL'):
            if (prices['BTC_PUT'] and prices['SOL_CALL'] and
                (prices['BTC_PUT'] + prices['SOL_CALL']) < 0.8):
                signals.append(('BTC_PUT', prices['BTC_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.C.value, 'BTC-SOL'))

            elif (prices['BTC_CALL'] and prices['SOL_PUT'] and
                  (prices['BTC_CALL'] + prices['SOL_PUT']) < 0.8):
                signals.append(('BTC_CALL', prices['BTC_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.C.value, 'BTC-SOL'))

        # ETH-SOL combinations
        if not self.is_pair_used(Strategy.C.value, period_end, 'ETH', 'SOL'):
            if (prices['ETH_PUT'] and prices['SOL_CALL'] and
                (prices['ETH_PUT'] + prices['SOL_CALL']) < 0.8):
                signals.append(('ETH_PUT', prices['ETH_PUT'], 'SOL_CALL', prices['SOL_CALL'],
                              period_end, Strategy.C.value, 'ETH-SOL'))

            elif (prices['ETH_CALL'] and prices['SOL_PUT'] and
                  (prices['ETH_CALL'] + prices['SOL_PUT']) < 0.8):
                signals.append(('ETH_CALL', prices['ETH_CALL'], 'SOL_PUT', prices['SOL_PUT'],
                              period_end, Strategy.C.value, 'ETH-SOL'))

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
            }
        except (KeyError, TypeError):
            return []

        # Collect all signals from all strategies
        all_signals = []

        all_signals.extend(self.check_strategy_a(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_a2(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_b(prices, time_to_end, period_end))
        all_signals.extend(self.check_strategy_c(prices, time_to_end, period_end))

        return all_signals

    def open_position(self, asset_type: str, entry_price: float, entry_time: datetime,
                     period_end: datetime, strategy: str, pair_id: str):
        """Open a new position"""
        position = Position(
            asset_type=asset_type,
            entry_price=entry_price,
            entry_time=entry_time,
            period_end=period_end,
            strategy=strategy,
            pair_id=pair_id
        )
        self.positions.append(position)

        coin = asset_type.split('_')[0]
        option = asset_type.split('_')[1]

        print(f"\n{'='*80}")
        print(f"[OPEN] {entry_time.strftime('%Y-%m-%d %H:%M:%S')} - Strategy {strategy}")
        print(f"  Asset: {coin} {option}")
        print(f"  Entry Price: ${entry_price:.2f}")
        print(f"  Period End: {period_end.strftime('%H:%M')}")
        print(f"  Time to End: {((period_end - entry_time).total_seconds() / 60):.1f} minutes")
        print(f"{'='*80}\n")

        # Save to JSON immediately
        self.save_results()

    def close_positions(self, data: Dict[str, Dict], current_time: datetime):
        """Close positions that have reached their period end"""
        period_end = self.get_next_period_end(current_time)

        # Find positions to close (within 30 seconds of period end)
        positions_to_close = [pos for pos in self.positions
                             if pos.period_end == period_end and
                             (period_end - current_time).total_seconds() <= 30]

        if not positions_to_close:
            return

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

            print(f"\n{'='*80}")
            print(f"[CLOSE] {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Strategy {pos.strategy}")
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
                    total_pnl=pos1.pnl + pos2.pnl
                )
                self.trades.append(trade)

                print(f"\n{'*'*80}")
                print(f"TRADE PAIR COMPLETED - Strategy {trade.strategy} - {pair_type}")
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
        results = {
            'last_updated': datetime.now().isoformat(),
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

        with open('trading_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def run(self):
        """Main trading loop"""
        print("="*80)
        print("Binary Options Trading Simulator - DRY RUN")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nCoin Pairs: BTC-ETH, BTC-SOL, ETH-SOL")
        print("\nStrategies:")
        print("  A:  $0.30-$0.44, >10 min")
        print("  A2: $0.30-$0.44, >7 min")
        print("  B:  $0.24-$0.47, >10 min")
        print("  C:  Combined <$0.80, >10 min")
        print("\nAll strategies can run simultaneously per period for each pair")
        print("Exit price unavailable = Total loss ($0.00)")
        print("="*80)

        last_status_time = time.time()

        try:
            while True:
                current_time = datetime.now()

                # Read market data
                data = self.read_market_data()

                # Check for position closes first
                self.close_positions(data, current_time)

                # Check for new trade opportunities (returns list of signals)
                trade_signals = self.check_trade_conditions(data, current_time)

                for signal in trade_signals:
                    asset1, price1, asset2, price2, period_end, strategy, pair_type = signal

                    # Generate unique pair_id for this trade pair
                    coin1 = asset1.split('_')[0]
                    coin2 = asset2.split('_')[0]
                    pair_id = self.get_pair_id(strategy, period_end, coin1, coin2)

                    # Open both positions
                    self.open_position(asset1, price1, current_time, period_end, strategy, pair_id)
                    self.open_position(asset2, price2, current_time, period_end, strategy, pair_id)

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
