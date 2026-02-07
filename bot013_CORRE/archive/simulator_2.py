import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

class Strategy(Enum):
    A = "A"      # 0.30-0.44, >10 min
    A2 = "A2"    # 0.30-0.44, >7 min
    B = "B"      # 0.24-0.47, >10 min
    C = "C"      # Combined <0.8, >10 min

@dataclass
class Position:
    """Represents an open position"""
    asset_type: str  # "BTC_PUT", "ETH_CALL", etc.
    entry_price: float
    entry_time: datetime
    period_end: datetime
    strategy: str
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
    btc_asset: str
    btc_entry: float
    btc_exit: float
    eth_asset: str
    eth_entry: float
    eth_exit: float
    btc_pnl: float
    eth_pnl: float
    total_pnl: float

class BinaryOptionsTrader:
    def __init__(self):
        self.data_files = {
            'BTC_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
            'BTC_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
            'ETH_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_ETH.json',
            'ETH_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_ETH.json'
        }
        
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.total_pnl = {strategy.value: 0.0 for strategy in Strategy}
        self.total_pnl['ALL'] = 0.0
        
        # Track which strategies have been used for each period
        self.period_strategies = {}
        
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
    
    def check_strategy_a(self, btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                        time_to_end, period_end) -> Optional[tuple]:
        """Strategy A: 0.30-0.44, >10 minutes"""
        if time_to_end <= 10:
            return None
        
        if self.is_strategy_used(period_end, Strategy.A):
            return None
        
        # Strategy A1: BTC PUT + ETH CALL
        if (btc_put_ask and eth_call_ask and
            0.30 <= btc_put_ask <= 0.44 and
            0.30 <= eth_call_ask <= 0.44):
            return ('BTC_PUT', btc_put_ask, 'ETH_CALL', eth_call_ask, period_end, Strategy.A.value)
        
        # Strategy A2: BTC CALL + ETH PUT
        if (btc_call_ask and eth_put_ask and
            0.30 <= btc_call_ask <= 0.44 and
            0.30 <= eth_put_ask <= 0.44):
            return ('BTC_CALL', btc_call_ask, 'ETH_PUT', eth_put_ask, period_end, Strategy.A.value)
        
        return None
    
    def check_strategy_a2(self, btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                         time_to_end, period_end) -> Optional[tuple]:
        """Strategy A2: 0.30-0.44, >7 minutes"""
        if time_to_end <= 7:
            return None
        
        if self.is_strategy_used(period_end, Strategy.A2):
            return None
        
        # Strategy A2.1: BTC PUT + ETH CALL
        if (btc_put_ask and eth_call_ask and
            0.30 <= btc_put_ask <= 0.44 and
            0.30 <= eth_call_ask <= 0.44):
            return ('BTC_PUT', btc_put_ask, 'ETH_CALL', eth_call_ask, period_end, Strategy.A2.value)
        
        # Strategy A2.2: BTC CALL + ETH PUT
        if (btc_call_ask and eth_put_ask and
            0.30 <= btc_call_ask <= 0.44 and
            0.30 <= eth_put_ask <= 0.44):
            return ('BTC_CALL', btc_call_ask, 'ETH_PUT', eth_put_ask, period_end, Strategy.A2.value)
        
        return None
    
    def check_strategy_b(self, btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                        time_to_end, period_end) -> Optional[tuple]:
        """Strategy B: 0.24-0.47, >10 minutes"""
        if time_to_end <= 10:
            return None
        
        if self.is_strategy_used(period_end, Strategy.B):
            return None
        
        # Strategy B1: BTC PUT + ETH CALL
        if (btc_put_ask and eth_call_ask and
            0.24 <= btc_put_ask <= 0.47 and
            0.24 <= eth_call_ask <= 0.47):
            return ('BTC_PUT', btc_put_ask, 'ETH_CALL', eth_call_ask, period_end, Strategy.B.value)
        
        # Strategy B2: BTC CALL + ETH PUT
        if (btc_call_ask and eth_put_ask and
            0.24 <= btc_call_ask <= 0.47 and
            0.24 <= eth_put_ask <= 0.47):
            return ('BTC_CALL', btc_call_ask, 'ETH_PUT', eth_put_ask, period_end, Strategy.B.value)
        
        return None
    
    def check_strategy_c(self, btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                        time_to_end, period_end) -> Optional[tuple]:
        """Strategy C: Combined price <0.8, >10 minutes"""
        if time_to_end <= 10:
            return None
        
        if self.is_strategy_used(period_end, Strategy.C):
            return None
        
        # Strategy C1: BTC PUT + ETH CALL (combined < 0.8)
        if (btc_put_ask and eth_call_ask and
            (btc_put_ask + eth_call_ask) < 0.8):
            return ('BTC_PUT', btc_put_ask, 'ETH_CALL', eth_call_ask, period_end, Strategy.C.value)
        
        # Strategy C2: BTC CALL + ETH PUT (combined < 0.8)
        if (btc_call_ask and eth_put_ask and
            (btc_call_ask + eth_put_ask) < 0.8):
            return ('BTC_CALL', btc_call_ask, 'ETH_PUT', eth_put_ask, period_end, Strategy.C.value)
        
        return None
    
    def is_strategy_used(self, period_end: datetime, strategy: Strategy) -> bool:
        """Check if a strategy has already been used for this period"""
        period_key = period_end.isoformat()
        if period_key not in self.period_strategies:
            self.period_strategies[period_key] = set()
        return strategy.value in self.period_strategies[period_key]
    
    def mark_strategy_used(self, period_end: datetime, strategy: str):
        """Mark a strategy as used for this period"""
        period_key = period_end.isoformat()
        if period_key not in self.period_strategies:
            self.period_strategies[period_key] = set()
        self.period_strategies[period_key].add(strategy)
    
    def check_trade_conditions(self, data: Dict[str, Dict], current_time: datetime) -> Optional[tuple]:
        """Check all strategies for trade conditions"""
        period_end = self.get_next_period_end(current_time)
        time_to_end = (period_end - current_time).total_seconds() / 60
        
        # Extract prices
        try:
            btc_put_ask = data['BTC_PUT']['best_ask']['price'] if data['BTC_PUT'] else None
            btc_call_ask = data['BTC_CALL']['best_ask']['price'] if data['BTC_CALL'] else None
            eth_put_ask = data['ETH_PUT']['best_ask']['price'] if data['ETH_PUT'] else None
            eth_call_ask = data['ETH_CALL']['best_ask']['price'] if data['ETH_CALL'] else None
        except (KeyError, TypeError):
            return None
        
        # Check strategies in priority order: A, A2, B, C
        signal = self.check_strategy_a(btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                                       time_to_end, period_end)
        if signal:
            return signal
        
        signal = self.check_strategy_a2(btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                                        time_to_end, period_end)
        if signal:
            return signal
        
        signal = self.check_strategy_b(btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                                       time_to_end, period_end)
        if signal:
            return signal
        
        signal = self.check_strategy_c(btc_put_ask, btc_call_ask, eth_put_ask, eth_call_ask, 
                                       time_to_end, period_end)
        if signal:
            return signal
        
        return None
    
    def open_position(self, asset_type: str, entry_price: float, entry_time: datetime, 
                     period_end: datetime, strategy: str):
        """Open a new position"""
        position = Position(
            asset_type=asset_type,
            entry_price=entry_price,
            entry_time=entry_time,
            period_end=period_end,
            strategy=strategy
        )
        self.positions.append(position)
        
        print(f"\n{'='*80}")
        print(f"[OPEN] {entry_time.strftime('%Y-%m-%d %H:%M:%S')} - Strategy {strategy}")
        print(f"  Asset: {asset_type}")
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
        
        # Group by strategy and period for trade recording
        trade_pairs = {}
        
        for pos in positions_to_close:
            asset_type = pos.asset_type.split('_')[0]  # BTC or ETH
            
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
            
            print(f"\n{'='*80}")
            print(f"[CLOSE] {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Strategy {pos.strategy}")
            print(f"  Asset: {pos.asset_type}")
            print(f"  Entry: ${pos.entry_price:.2f} @ {pos.entry_time.strftime('%H:%M:%S')}")
            print(f"  Exit:  ${pos.exit_price:.2f} @ {pos.exit_time.strftime('%H:%M:%S')}")
            print(f"  PNL: ${pos.pnl:+.2f}")
            print(f"{'='*80}\n")
            
            # Group positions by strategy for trade pair creation
            key = (pos.strategy, pos.period_end)
            if key not in trade_pairs:
                trade_pairs[key] = {}
            trade_pairs[key][asset_type] = pos
        
        # Record trade pairs
        for (strategy, period), positions_dict in trade_pairs.items():
            if 'BTC' in positions_dict and 'ETH' in positions_dict:
                btc_pos = positions_dict['BTC']
                eth_pos = positions_dict['ETH']
                
                trade = Trade(
                    timestamp=btc_pos.entry_time,
                    period_end=btc_pos.period_end,
                    strategy=strategy,
                    btc_asset=btc_pos.asset_type,
                    btc_entry=btc_pos.entry_price,
                    btc_exit=btc_pos.exit_price,
                    eth_asset=eth_pos.asset_type,
                    eth_entry=eth_pos.entry_price,
                    eth_exit=eth_pos.exit_price,
                    btc_pnl=btc_pos.pnl,
                    eth_pnl=eth_pos.pnl,
                    total_pnl=btc_pos.pnl + eth_pos.pnl
                )
                self.trades.append(trade)
                
                print(f"\n{'*'*80}")
                print(f"TRADE PAIR COMPLETED - Strategy {strategy}")
                print(f"  Total PNL: ${trade.total_pnl:+.2f}")
                print(f"  Strategy {strategy} PNL: ${self.total_pnl[strategy]:+.2f}")
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
        
        if self.positions:
            print("\nOpen Positions:")
            for pos in self.positions:
                time_remaining = (pos.period_end - datetime.now()).total_seconds() / 60
                print(f"  [{pos.strategy}] {pos.asset_type}: Entry ${pos.entry_price:.2f}, Closes in {time_remaining:.1f} min")
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
        print("\nStrategies:")
        print("  A:  BTC PUT/CALL + ETH CALL/PUT, $0.30-$0.44, >10 min")
        print("  A2: BTC PUT/CALL + ETH CALL/PUT, $0.30-$0.44, >7 min")
        print("  B:  BTC PUT/CALL + ETH CALL/PUT, $0.24-$0.47, >10 min")
        print("  C:  BTC PUT/CALL + ETH CALL/PUT, combined <$0.80, >10 min")
        print("\nAll strategies can run simultaneously per period")
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
                
                # Check for new trade opportunities
                trade_signal = self.check_trade_conditions(data, current_time)
                if trade_signal:
                    btc_asset, btc_price, eth_asset, eth_price, period_end, strategy = trade_signal
                    self.open_position(btc_asset, btc_price, current_time, period_end, strategy)
                    self.open_position(eth_asset, eth_price, current_time, period_end, strategy)
                    self.mark_strategy_used(period_end, strategy)
                
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

if __name__ == "__main__":
    trader = BinaryOptionsTrader()
    trader.run()
