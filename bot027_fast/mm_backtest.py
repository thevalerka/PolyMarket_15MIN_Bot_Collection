#!/usr/bin/env python3
"""
Market Maker Strategy Backtester
================================

Backtests the advanced MM strategies against historical data to:
1. Validate strategy parameters
2. Find optimal configuration
3. Measure expected P&L and Sharpe ratio
4. Test regime detection accuracy

Usage:
    python3 mm_backtest.py --data /path/to/historical_data.csv
    python3 mm_backtest.py --generate-sample  # Create sample data for testing
    python3 mm_backtest.py --live-data        # Load from live data collector output
"""

import json
import csv
import random
import statistics
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mm_engine import (
    AdvancedMarketMaker,
    MMConfig,
    MarketState,
    PriceData,
    OptionsData,
    MarketRegime,
    TradePeriodPhase,
)

# Path to collected live data
LIVE_DATA_DIR = '/home/ubuntu/013_2025_polymarket/backtest_data'


@dataclass
class HistoricalTick:
    """Single historical data point."""
    timestamp: int
    btc_price: float
    btc_price_bybit: Optional[float] = None
    btc_price_chainlink: Optional[float] = None
    btc_price_coinbase: Optional[float] = None
    call_bid: float = 0.0
    call_ask: float = 0.0
    put_bid: float = 0.0
    put_ask: float = 0.0
    strike: float = 0.0
    period_start: int = 0
    bybit_direction: Optional[str] = None


@dataclass
class Trade:
    """Record of an executed trade."""
    timestamp: int
    asset: str
    side: str
    price: float
    size: float
    pnl: float = 0.0


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_spread_captured: float = 0.0
    quotes_generated: int = 0
    quotes_pulled: int = 0
    regime_accuracy: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


class MarketSimulator:
    """
    Simulates market fills for backtesting.
    """
    
    def __init__(
        self,
        fill_probability: float = 0.3,  # Probability of getting filled
        adverse_selection_rate: float = 0.2,  # Rate of fills being adverse
        slippage_bps: float = 5,  # Slippage in basis points
    ):
        self.fill_probability = fill_probability
        self.adverse_selection_rate = adverse_selection_rate
        self.slippage_bps = slippage_bps
    
    def simulate_fills(
        self,
        quotes: Dict[str, list],
        tick: HistoricalTick,
        next_tick: Optional[HistoricalTick],
    ) -> List[Trade]:
        """
        Simulate which quotes get filled.
        """
        fills = []
        
        for asset_type, quote_list in quotes.items():
            for quote in quote_list:
                # Determine fill probability based on quote competitiveness
                if asset_type == 'CALL':
                    market_bid = tick.call_bid
                    market_ask = tick.call_ask
                else:
                    market_bid = tick.put_bid
                    market_ask = tick.put_ask
                
                # More aggressive quotes have higher fill probability
                if quote.side == 'bid':
                    aggressiveness = (quote.price - market_bid) / max(market_bid, 0.01)
                    fill_prob = self.fill_probability * (1 + aggressiveness * 2)
                else:
                    aggressiveness = (market_ask - quote.price) / max(market_ask, 0.01)
                    fill_prob = self.fill_probability * (1 + aggressiveness * 2)
                
                # Check for fill
                if random.random() < fill_prob:
                    # Simulate adverse selection
                    is_adverse = random.random() < self.adverse_selection_rate
                    
                    # Calculate fill price with slippage
                    slippage = quote.price * (self.slippage_bps / 10000)
                    if quote.side == 'bid':
                        fill_price = quote.price + slippage if is_adverse else quote.price
                    else:
                        fill_price = quote.price - slippage if is_adverse else quote.price
                    
                    fills.append(Trade(
                        timestamp=tick.timestamp,
                        asset=asset_type,
                        side=quote.side,
                        price=fill_price,
                        size=quote.size,
                    ))
        
        return fills


class Backtester:
    """
    Backtests MM strategies against historical data.
    """
    
    def __init__(self, config: Optional[MMConfig] = None):
        self.config = config or MMConfig()
        self.mm = AdvancedMarketMaker(self.config)
        self.simulator = MarketSimulator()
        
        # Position tracking
        self.positions: Dict[str, float] = {'CALL': 0, 'PUT': 0}
        self.avg_entry: Dict[str, float] = {'CALL': 0, 'PUT': 0}
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        
    def create_market_state(self, tick: HistoricalTick) -> MarketState:
        """Create MarketState from historical tick."""
        state = MarketState()
        
        # Simulate price sources (all from same price in backtest)
        state.bybit = PriceData(
            source='bybit',
            price=tick.btc_price,
            timestamp=tick.timestamp,
            direction='UP' if random.random() > 0.5 else 'DOWN',
        )
        state.chainlink = PriceData(
            source='chainlink',
            price=tick.btc_price,
            timestamp=tick.timestamp,
        )
        
        # Options
        state.call = OptionsData(
            asset_type='CALL',
            best_bid=tick.call_bid,
            best_ask=tick.call_ask,
            mid=(tick.call_bid + tick.call_ask) / 2,
            spread=tick.call_ask - tick.call_bid,
            timestamp=tick.timestamp,
        )
        state.put = OptionsData(
            asset_type='PUT',
            best_bid=tick.put_bid,
            best_ask=tick.put_ask,
            mid=(tick.put_bid + tick.put_ask) / 2,
            spread=tick.put_ask - tick.put_bid,
            timestamp=tick.timestamp,
        )
        
        state.strike_price = tick.strike
        state.strike_timestamp = tick.period_start
        
        # Time to expiry
        period_end = tick.period_start + 15 * 60 * 1000
        state.time_to_expiry_seconds = max(0, (period_end - tick.timestamp) / 1000)
        state.timestamp = tick.timestamp
        
        return state
    
    def process_fills(self, fills: List[Trade], tick: HistoricalTick):
        """Process filled trades and update positions."""
        for fill in fills:
            # Update position
            if fill.side == 'bid':
                # We bought
                old_pos = self.positions[fill.asset]
                old_avg = self.avg_entry[fill.asset]
                
                new_pos = old_pos + fill.size
                if new_pos != 0:
                    self.avg_entry[fill.asset] = (old_pos * old_avg + fill.size * fill.price) / new_pos
                self.positions[fill.asset] = new_pos
                
            else:
                # We sold
                old_pos = self.positions[fill.asset]
                
                if old_pos > 0:
                    # Closing long position
                    close_size = min(old_pos, fill.size)
                    pnl = (fill.price - self.avg_entry[fill.asset]) * close_size
                    fill.pnl = pnl
                    self.realized_pnl += pnl
                
                self.positions[fill.asset] = old_pos - fill.size
            
            self.trades.append(fill)
    
    def settle_period(self, final_tick: HistoricalTick):
        """Settle positions at end of period."""
        # Determine settlement prices
        price_vs_strike = final_tick.btc_price - final_tick.strike
        
        # CALL pays 1.0 if price > strike, else 0
        call_settlement = 1.0 if price_vs_strike > 0 else 0.0
        # PUT pays 1.0 if price < strike, else 0
        put_settlement = 1.0 if price_vs_strike < 0 else 0.0
        
        # Settle CALL position
        if self.positions['CALL'] != 0:
            pnl = (call_settlement - self.avg_entry['CALL']) * self.positions['CALL']
            self.realized_pnl += pnl
            self.trades.append(Trade(
                timestamp=final_tick.timestamp,
                asset='CALL',
                side='settlement',
                price=call_settlement,
                size=abs(self.positions['CALL']),
                pnl=pnl,
            ))
            self.positions['CALL'] = 0
            self.avg_entry['CALL'] = 0
        
        # Settle PUT position
        if self.positions['PUT'] != 0:
            pnl = (put_settlement - self.avg_entry['PUT']) * self.positions['PUT']
            self.realized_pnl += pnl
            self.trades.append(Trade(
                timestamp=final_tick.timestamp,
                asset='PUT',
                side='settlement',
                price=put_settlement,
                size=abs(self.positions['PUT']),
                pnl=pnl,
            ))
            self.positions['PUT'] = 0
            self.avg_entry['PUT'] = 0
        
        self.equity_curve.append(self.realized_pnl)
    
    def calculate_unrealized_pnl(self, tick: HistoricalTick) -> float:
        """Calculate unrealized P&L at current tick."""
        unrealized = 0.0
        
        if self.positions['CALL'] != 0:
            mark = tick.call_bid if self.positions['CALL'] > 0 else tick.call_ask
            unrealized += (mark - self.avg_entry['CALL']) * self.positions['CALL']
        
        if self.positions['PUT'] != 0:
            mark = tick.put_bid if self.positions['PUT'] > 0 else tick.put_ask
            unrealized += (mark - self.avg_entry['PUT']) * self.positions['PUT']
        
        return unrealized
    
    def run(self, data: List[HistoricalTick]) -> BacktestResults:
        """Run backtest on historical data."""
        results = BacktestResults()
        
        current_period = None
        
        for i, tick in enumerate(data):
            # Check for new period
            if tick.period_start != current_period:
                if current_period is not None:
                    # Settle previous period
                    self.settle_period(data[i-1])
                current_period = tick.period_start
            
            # Create market state
            state = self.create_market_state(tick)
            
            # Generate quotes
            quotes = self.mm.generate_all_quotes(state)
            
            # Track quote stats
            quote_count = len(quotes.get('CALL', [])) + len(quotes.get('PUT', []))
            results.quotes_generated += quote_count
            if quote_count == 0:
                results.quotes_pulled += 1
            
            # Simulate fills
            next_tick = data[i+1] if i < len(data) - 1 else None
            fills = self.simulator.simulate_fills(quotes, tick, next_tick)
            
            # Process fills
            self.process_fills(fills, tick)
            
            # Track equity
            total_equity = self.realized_pnl + self.calculate_unrealized_pnl(tick)
            self.equity_curve.append(total_equity)
        
        # Final settlement
        if data:
            self.settle_period(data[-1])
        
        # Calculate results
        results.total_pnl = self.realized_pnl
        results.trades = self.trades
        results.equity_curve = self.equity_curve
        results.total_trades = len([t for t in self.trades if t.side != 'settlement'])
        results.winning_trades = len([t for t in self.trades if t.pnl > 0])
        results.losing_trades = len([t for t in self.trades if t.pnl < 0])
        
        # Calculate max drawdown
        peak = 0
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        results.max_drawdown = max_dd
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = [self.equity_curve[i] - self.equity_curve[i-1] 
                      for i in range(1, len(self.equity_curve))]
            if returns and statistics.stdev(returns) > 0:
                results.sharpe_ratio = (statistics.mean(returns) / statistics.stdev(returns)) * (252 ** 0.5)
        
        return results


def generate_sample_data(
    num_periods: int = 100,
    ticks_per_period: int = 180,  # Every 5 seconds for 15 min
) -> List[HistoricalTick]:
    """Generate synthetic historical data for testing."""
    data = []
    
    base_price = 67000
    base_time = int(datetime.now().timestamp() * 1000) - (num_periods * 15 * 60 * 1000)
    
    for period in range(num_periods):
        period_start = base_time + period * 15 * 60 * 1000
        strike = base_price + random.gauss(0, 100)
        
        price = base_price + random.gauss(0, 50)
        
        for tick_num in range(ticks_per_period):
            timestamp = period_start + tick_num * 5000  # 5 second intervals
            
            # Random walk price
            price += random.gauss(0, 10)
            
            # Calculate option prices based on distance from strike
            time_remaining = (ticks_per_period - tick_num) / ticks_per_period
            prob_above = 0.5 + (price - strike) / (200 * (1 + time_remaining))
            prob_above = max(0.05, min(0.95, prob_above))
            
            # Add spread
            spread = 0.01 + 0.02 * (1 - time_remaining)  # Spread widens near expiry
            
            call_mid = prob_above
            put_mid = 1 - prob_above
            
            tick = HistoricalTick(
                timestamp=timestamp,
                btc_price=price,
                call_bid=max(0.01, call_mid - spread/2),
                call_ask=min(0.99, call_mid + spread/2),
                put_bid=max(0.01, put_mid - spread/2),
                put_ask=min(0.99, put_mid + spread/2),
                strike=strike,
                period_start=period_start,
            )
            data.append(tick)
        
        # Update base price with some drift
        base_price = price
    
    return data


def save_sample_data(data: List[HistoricalTick], filepath: str):
    """Save sample data to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'btc_price', 'call_bid', 'call_ask', 
                        'put_bid', 'put_ask', 'strike', 'period_start'])
        for tick in data:
            writer.writerow([
                tick.timestamp, tick.btc_price, tick.call_bid, tick.call_ask,
                tick.put_bid, tick.put_ask, tick.strike, tick.period_start
            ])


def load_data(filepath: str) -> List[HistoricalTick]:
    """Load historical data from CSV."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tick = HistoricalTick(
                timestamp=int(row['timestamp']),
                btc_price=float(row['btc_price']),
                call_bid=float(row['call_bid']),
                call_ask=float(row['call_ask']),
                put_bid=float(row['put_bid']),
                put_ask=float(row['put_ask']),
                strike=float(row['strike']),
                period_start=int(row['period_start']),
            )
            data.append(tick)
    return data


def load_live_collected_data(data_dir: str = LIVE_DATA_DIR) -> List[HistoricalTick]:
    """
    Load data from the live data collector output files.
    Combines all CSV files in the directory.
    """
    data = []
    
    # Find all market_data_*.csv files
    pattern = os.path.join(data_dir, 'market_data_*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"⚠️ No data files found in {data_dir}")
        return data
    
    print(f"📁 Found {len(files)} data files:")
    for f in files:
        print(f"   - {os.path.basename(f)}")
    
    for filepath in files:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            file_ticks = 0
            
            for row in reader:
                try:
                    # Get BTC price (prefer chainlink as that's what Polymarket uses)
                    btc_price = None
                    btc_price_chainlink = None
                    btc_price_bybit = None
                    btc_price_coinbase = None
                    
                    if row.get('chainlink_price') and row['chainlink_price'] != '':
                        btc_price_chainlink = float(row['chainlink_price'])
                        btc_price = btc_price_chainlink
                    
                    if row.get('bybit_price') and row['bybit_price'] != '':
                        btc_price_bybit = float(row['bybit_price'])
                        if btc_price is None:
                            btc_price = btc_price_bybit
                    
                    if row.get('coinbase_price') and row['coinbase_price'] != '':
                        btc_price_coinbase = float(row['coinbase_price'])
                        if btc_price is None:
                            btc_price = btc_price_coinbase
                    
                    # Skip if missing critical data
                    if btc_price is None:
                        continue
                    if not row.get('call_bid') or row['call_bid'] == '':
                        continue
                    if not row.get('strike_price') or row['strike_price'] == '':
                        continue
                    
                    tick = HistoricalTick(
                        timestamp=int(row['collection_time']),
                        btc_price=btc_price,
                        btc_price_bybit=btc_price_bybit,
                        btc_price_chainlink=btc_price_chainlink,
                        btc_price_coinbase=btc_price_coinbase,
                        call_bid=float(row['call_bid']),
                        call_ask=float(row['call_ask']),
                        put_bid=float(row['put_bid']),
                        put_ask=float(row['put_ask']),
                        strike=float(row['strike_price']),
                        period_start=int(row['period_start']) if row.get('period_start') and row['period_start'] != '' else 0,
                        bybit_direction=row.get('bybit_direction'),
                    )
                    data.append(tick)
                    file_ticks += 1
                    
                except (ValueError, KeyError) as e:
                    continue  # Skip malformed rows
            
            print(f"   Loaded {file_ticks:,} ticks from {os.path.basename(filepath)}")
    
    # Sort by timestamp
    data.sort(key=lambda x: x.timestamp)
    
    # Print summary
    if data:
        periods = len(set(t.period_start for t in data))
        duration_mins = (data[-1].timestamp - data[0].timestamp) / 1000 / 60
        print(f"\n📊 Total: {len(data):,} ticks | {periods} periods | {duration_mins:.1f} minutes")
    
    return data


def print_results(results: BacktestResults, config_name: str = "Default"):
    """Print backtest results."""
    print(f"\n{'='*60}")
    print(f"   BACKTEST RESULTS - {config_name}")
    print(f"{'='*60}")
    
    print(f"\n--- P&L Summary ---")
    print(f"   Total P&L: ${results.total_pnl:,.2f}")
    print(f"   Max Drawdown: ${results.max_drawdown:,.2f}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
    
    print(f"\n--- Trade Statistics ---")
    print(f"   Total Trades: {results.total_trades}")
    print(f"   Winning Trades: {results.winning_trades}")
    print(f"   Losing Trades: {results.losing_trades}")
    print(f"   Win Rate: {results.win_rate:.1%}")
    
    print(f"\n--- Quote Statistics ---")
    print(f"   Quotes Generated: {results.quotes_generated}")
    print(f"   Quotes Pulled: {results.quotes_pulled}")
    pull_rate = results.quotes_pulled / max(1, results.quotes_generated + results.quotes_pulled)
    print(f"   Pull Rate: {pull_rate:.1%}")
    
    print(f"\n{'='*60}")


def run_optimization(data: List[HistoricalTick]) -> Dict[str, BacktestResults]:
    """Run optimization across different configurations."""
    configs = {
        'Default': MMConfig(),
        'Aggressive': MMConfig(
            base_spread=0.015,
            min_spread=0.008,
            inventory_skew_factor=0.0005,
        ),
        'Conservative': MMConfig(
            base_spread=0.03,
            min_spread=0.02,
            inventory_skew_factor=0.002,
        ),
        'Wide Tiers': MMConfig(
            tier_sizes=[25, 50, 100, 200],
            tier_spreads=[0.02, 0.04, 0.06, 0.08],
        ),
        'Tight Tiers': MMConfig(
            tier_sizes=[100, 200, 400, 800],
            tier_spreads=[0.005, 0.01, 0.015, 0.02],
        ),
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\nRunning backtest with {name} config...")
        backtester = Backtester(config)
        results[name] = backtester.run(data.copy())
        print_results(results[name], name)
    
    # Find best config
    best_name = max(results.keys(), key=lambda k: results[k].sharpe_ratio)
    print(f"\n{'='*60}")
    print(f"   BEST CONFIG: {best_name}")
    print(f"   Sharpe: {results[best_name].sharpe_ratio:.2f}")
    print(f"   P&L: ${results[best_name].total_pnl:,.2f}")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MM Strategy Backtester')
    parser.add_argument('--data', type=str, help='Path to historical data CSV')
    parser.add_argument('--live-data', action='store_true',
                       help='Load from live data collector output')
    parser.add_argument('--data-dir', type=str, default=LIVE_DATA_DIR,
                       help=f'Directory with collected data (default: {LIVE_DATA_DIR})')
    parser.add_argument('--generate-sample', action='store_true', 
                       help='Generate sample data for testing')
    parser.add_argument('--optimize', action='store_true',
                       help='Run optimization across configs')
    parser.add_argument('--periods', type=int, default=100,
                       help='Number of periods for sample data')
    
    args = parser.parse_args()
    
    if args.generate_sample:
        print(f"Generating {args.periods} periods of sample data...")
        data = generate_sample_data(num_periods=args.periods)
        filepath = 'sample_data.csv'
        save_sample_data(data, filepath)
        print(f"Saved to {filepath}")
        print(f"Total ticks: {len(data)}")
    
    elif args.live_data:
        print(f"Loading live collected data from {args.data_dir}...")
        data = load_live_collected_data(args.data_dir)
        
        if not data:
            print("❌ No data loaded. Run data_collector.py first to collect data.")
            return
        
        print(f"Loaded {len(data):,} ticks")
        
        if args.optimize:
            run_optimization(data)
        else:
            backtester = Backtester()
            results = backtester.run(data)
            print_results(results)
    
    elif args.data:
        print(f"Loading data from {args.data}...")
        data = load_data(args.data)
        print(f"Loaded {len(data)} ticks")
        
        if args.optimize:
            run_optimization(data)
        else:
            backtester = Backtester()
            results = backtester.run(data)
            print_results(results)
    else:
        # Demo with generated data
        print("Running demo with synthetic data...")
        data = generate_sample_data(num_periods=50)
        
        if args.optimize:
            run_optimization(data)
        else:
            backtester = Backtester()
            results = backtester.run(data)
            print_results(results)


if __name__ == '__main__':
    main()
