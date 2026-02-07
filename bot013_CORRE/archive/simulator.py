import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Position:
    """Represents an open position"""
    asset_type: str  # "BTC_PUT", "ETH_CALL", etc.
    entry_price: float
    entry_time: datetime
    period_end: datetime
    size: float = 1.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None

@dataclass
class Trade:
    """Represents a completed trade pair"""
    timestamp: datetime
    period_end: datetime
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
        self.total_pnl = 0.0
        
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
    
    def check_trade_conditions(self, data: Dict[str, Dict], current_time: datetime) -> Optional[tuple]:
        """Check if trade conditions are met"""
        period_end = self.get_next_period_end(current_time)
        time_to_end = (period_end - current_time).total_seconds() / 60
        
        # Must be more than 10 minutes to end
        if time_to_end <= 10:
            return None
        
        # Check if we already have open positions for this period
        if any(pos.period_end == period_end for pos in self.positions):
            return None
        
        # Extract prices
        try:
            btc_put_ask = data['BTC_PUT']['best_ask']['price'] if data['BTC_PUT'] else None
            btc_call_ask = data['BTC_CALL']['best_ask']['price'] if data['BTC_CALL'] else None
            eth_put_ask = data['ETH_PUT']['best_ask']['price'] if data['ETH_PUT'] else None
            eth_call_ask = data['ETH_CALL']['best_ask']['price'] if data['ETH_CALL'] else None
        except (KeyError, TypeError):
            return None
        
        # Strategy 1: BTC PUT + ETH CALL
        if (btc_put_ask and eth_call_ask and
            0.30 <= btc_put_ask <= 0.44 and
            0.30 <= eth_call_ask <= 0.44):
            return ('BTC_PUT', btc_put_ask, 'ETH_CALL', eth_call_ask, period_end)
        
        # Strategy 2: BTC CALL + ETH PUT
        if (btc_call_ask and eth_put_ask and
            0.30 <= btc_call_ask <= 0.44 and
            0.30 <= eth_put_ask <= 0.44):
            return ('BTC_CALL', btc_call_ask, 'ETH_PUT', eth_put_ask, period_end)
        
        return None
    
    def open_position(self, asset_type: str, entry_price: float, entry_time: datetime, period_end: datetime):
        """Open a new position"""
        position = Position(
            asset_type=asset_type,
            entry_price=entry_price,
            entry_time=entry_time,
            period_end=period_end
        )
        self.positions.append(position)
        print(f"\n{'='*80}")
        print(f"[OPEN] {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Asset: {asset_type}")
        print(f"  Entry Price: ${entry_price:.2f}")
        print(f"  Period End: {period_end.strftime('%H:%M')}")
        print(f"  Time to End: {((period_end - entry_time).total_seconds() / 60):.1f} minutes")
        print(f"{'='*80}\n")
    
    def close_positions(self, data: Dict[str, Dict], current_time: datetime):
        """Close positions that have reached their period end"""
        period_end = self.get_next_period_end(current_time)
        
        # Find positions to close (within 30 seconds of period end)
        positions_to_close = [pos for pos in self.positions 
                             if pos.period_end == period_end and 
                             (period_end - current_time).total_seconds() <= 30]
        
        if not positions_to_close:
            return
        
        # Group by period for trade recording
        btc_pos = None
        eth_pos = None
        
        for pos in positions_to_close:
            asset_type = pos.asset_type.split('_')[0]  # BTC or ETH
            option_type = pos.asset_type.split('_')[1]  # PUT or CALL
            
            try:
                # Get exit price from best bid
                exit_price = data[pos.asset_type]['best_bid']['price']
            except (KeyError, TypeError):
                print(f"Warning: Could not get exit price for {pos.asset_type}, using entry price")
                exit_price = pos.entry_price
            
            pos.exit_price = exit_price
            pos.exit_time = current_time
            pos.pnl = (exit_price - pos.entry_price) * pos.size
            self.total_pnl += pos.pnl
            
            print(f"\n{'='*80}")
            print(f"[CLOSE] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Asset: {pos.asset_type}")
            print(f"  Entry: ${pos.entry_price:.2f} @ {pos.entry_time.strftime('%H:%M:%S')}")
            print(f"  Exit:  ${pos.exit_price:.2f} @ {pos.exit_time.strftime('%H:%M:%S')}")
            print(f"  PNL: ${pos.pnl:+.2f}")
            print(f"{'='*80}\n")
            
            if asset_type == 'BTC':
                btc_pos = pos
            else:
                eth_pos = pos
        
        # Record trade pair
        if btc_pos and eth_pos:
            trade = Trade(
                timestamp=btc_pos.entry_time,
                period_end=btc_pos.period_end,
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
            print(f"TRADE PAIR COMPLETED")
            print(f"  Total PNL: ${trade.total_pnl:+.2f}")
            print(f"  Cumulative PNL: ${self.total_pnl:+.2f}")
            print(f"  Total Trades: {len(self.trades)}")
            print(f"{'*'*80}\n")
        
        # Remove closed positions
        self.positions = [pos for pos in self.positions if pos not in positions_to_close]
    
    def print_status(self):
        """Print current status"""
        print(f"\n--- Status Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Completed Trades: {len(self.trades)}")
        print(f"Total PNL: ${self.total_pnl:+.2f}")
        if self.positions:
            print("\nOpen Positions:")
            for pos in self.positions:
                time_remaining = (pos.period_end - datetime.now()).total_seconds() / 60
                print(f"  {pos.asset_type}: Entry ${pos.entry_price:.2f}, Closes in {time_remaining:.1f} min")
        print()
    
    def save_results(self):
        """Save trading results to file"""
        results = {
            'total_pnl': self.total_pnl,
            'total_trades': len(self.trades),
            'trades': [asdict(trade) for trade in self.trades]
        }
        
        # Convert datetime objects to strings for JSON serialization
        for trade in results['trades']:
            trade['timestamp'] = trade['timestamp'].isoformat()
            trade['period_end'] = trade['period_end'].isoformat()
        
        with open('trading_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to trading_results.json")
    
    def run(self):
        """Main trading loop"""
        print("="*80)
        print("Binary Options Trading Simulator - DRY RUN")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nStrategy:")
        print("  - BTC PUT ($0.30-$0.44) + ETH CALL ($0.30-$0.44)")
        print("  - BTC CALL ($0.30-$0.44) + ETH PUT ($0.30-$0.44)")
        print("  - Entry: >10 min before period end")
        print("  - Exit: At period end (00, 15, 30, 45)")
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
                    btc_asset, btc_price, eth_asset, eth_price, period_end = trade_signal
                    self.open_position(btc_asset, btc_price, current_time, period_end)
                    self.open_position(eth_asset, eth_price, current_time, period_end)
                
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
            print(f"  Total PNL: ${self.total_pnl:+.2f}")
            if self.trades:
                winning_trades = sum(1 for t in self.trades if t.total_pnl > 0)
                print(f"  Win Rate: {winning_trades}/{len(self.trades)} ({100*winning_trades/len(self.trades):.1f}%)")

if __name__ == "__main__":
    trader = BinaryOptionsTrader()
    trader.run()
