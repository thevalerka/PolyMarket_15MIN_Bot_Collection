#!/usr/bin/env python3
"""
Binary Options Arbitrage Bot - DRY RUN TESTER
Tests three strategies simultaneously without actual trading
Tracks PNL for each strategy across restarts
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import logging
import warnings

# Suppress numpy warnings
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PriceSnapshot:
    """Single price observation"""
    timestamp: float
    btc_price: float
    put_bid: float
    put_ask: float
    call_bid: float
    call_ask: float
    put_spread: float
    call_spread: float


@dataclass
class SimulatedPosition:
    """Simulated position tracker"""
    token_type: str  # 'PUT' or 'CALL'
    entry_price: float
    entry_time: float
    quantity: float
    entry_btc_price: float
    strategy_name: str


@dataclass
class SimulatedTrade:
    """Completed simulated trade"""
    timestamp: str
    strategy: str
    token_type: str
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    btc_price: float
    pnl: Optional[float] = None
    reason: str = ""


class StrategyTester:
    """Tests a single strategy configuration"""
    
    def __init__(self, name: str, discrepancy_threshold: float, position_size: float = 5.0):
        self.name = name
        self.discrepancy_threshold = discrepancy_threshold
        self.position_size = position_size
        
        # Position tracking
        self.position: Optional[SimulatedPosition] = None
        self.trades_today: List[SimulatedTrade] = []
        self.daily_pnl: float = 0.0
        
        # Statistics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.largest_win: float = 0.0
        self.largest_loss: float = 0.0
    
    def check_entry_signal(self, snapshot: PriceSnapshot, 
                           expected_put: Optional[float], 
                           expected_call: Optional[float],
                           has_spike: bool,
                           spike_pct: float) -> Optional[Tuple[str, float, str]]:
        """Check for entry signals (lag-based only, no spike detection)"""
        if self.position is not None:
            return None
        
        if expected_put is None or expected_call is None:
            return None
        
        # Don't trade extreme prices
        if snapshot.put_ask >= 0.97 or snapshot.put_ask <= 0.03:
            expected_put = None
        if snapshot.call_ask >= 0.97 or snapshot.call_ask <= 0.03:
            expected_call = None
        
        put_lag = expected_put - snapshot.put_ask if expected_put else -999
        call_lag = expected_call - snapshot.call_ask if expected_call else -999
        
        # Simple lag-based entry only
        if expected_put and put_lag >= self.discrepancy_threshold:
            return ('PUT', snapshot.put_ask, 
                   f"PUT lag ${put_lag:.3f}")
        
        if expected_call and call_lag >= self.discrepancy_threshold:
            return ('CALL', snapshot.call_ask,
                   f"CALL lag ${call_lag:.3f}")
        
        return None
    
    def check_exit_signal(self, snapshot: PriceSnapshot,
                          expected_put: Optional[float], 
                          expected_call: Optional[float],
                          seconds_remaining: float) -> Optional[Tuple[float, str]]:
        """
        Check for exit signals:
        1. Bid >= 0.99
        2. Opposite opportunity detected (exit, don't flip)
        3. Near expiry
        """
        if not self.position:
            return None
        
        if self.position.token_type == 'PUT':
            # ALWAYS SELL at 0.99
            if snapshot.put_bid >= 0.99:
                return (0.99, "Hit max price 0.99")
            
            # Exit if CALL opportunity detected (but don't flip)
            if expected_call:
                call_lag = expected_call - snapshot.call_ask
                if call_lag >= self.discrepancy_threshold and snapshot.call_ask <= 0.97:
                    return (snapshot.put_bid, f"Opposite opportunity (CALL lag ${call_lag:.3f})")
        
        else:  # CALL
            # ALWAYS SELL at 0.99
            if snapshot.call_bid >= 0.99:
                return (0.99, "Hit max price 0.99")
            
            # Exit if PUT opportunity detected (but don't flip)
            if expected_put:
                put_lag = expected_put - snapshot.put_ask
                if put_lag >= self.discrepancy_threshold and snapshot.put_ask <= 0.97:
                    return (snapshot.call_bid, f"Opposite opportunity (PUT lag ${put_lag:.3f})")
        
        # Near expiry
        if seconds_remaining < 60:
            exit_price = snapshot.put_bid if self.position.token_type == 'PUT' else snapshot.call_bid
            return (exit_price, f"Near expiry ({seconds_remaining:.0f}s)")
        
        return None
    
    def execute_simulated_buy(self, token_type: str, entry_price: float, 
                              btc_price: float, reason: str):
        """Execute simulated buy"""
        self.position = SimulatedPosition(
            token_type=token_type,
            entry_price=entry_price,
            entry_time=time.time(),
            quantity=self.position_size,
            entry_btc_price=btc_price,
            strategy_name=self.name
        )
        
        trade = SimulatedTrade(
            timestamp=datetime.now().isoformat(),
            strategy=self.name,
            token_type=token_type,
            action='BUY',
            price=entry_price,
            quantity=self.position_size,
            btc_price=btc_price,
            reason=reason
        )
        
        self.trades_today.append(trade)
        logger.info(f"[{self.name}] 🟢 SIMULATED BUY: {token_type} @ ${entry_price:.4f} | {reason}")
    
    def execute_simulated_sell(self, exit_price: float, btc_price: float, reason: str):
        """Execute simulated sell"""
        if not self.position:
            return
        
        pnl = (exit_price - self.position.entry_price) * self.position.quantity
        self.daily_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.largest_win = max(self.largest_win, pnl)
        else:
            self.losing_trades += 1
            self.largest_loss = min(self.largest_loss, pnl)
        
        trade = SimulatedTrade(
            timestamp=datetime.now().isoformat(),
            strategy=self.name,
            token_type=self.position.token_type,
            action='SELL',
            price=exit_price,
            quantity=self.position.quantity,
            btc_price=btc_price,
            pnl=pnl,
            reason=reason
        )
        
        self.trades_today.append(trade)
        
        logger.info(f"[{self.name}] 🔴 SIMULATED SELL: {self.position.token_type} @ ${exit_price:.4f} | "
                   f"PNL: ${pnl:+.2f} | {reason}")
        
        self.position = None
    
    def get_stats(self) -> Dict:
        """Get strategy statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'name': self.name,
            'threshold': self.discrepancy_threshold,
            'daily_pnl': round(self.daily_pnl, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 1),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'open_position': self.position.token_type if self.position else None
        }


class MultiStrategyTester:
    """Tests multiple strategies simultaneously"""
    
    def __init__(
        self,
        put_file: str = "/home/ubuntu/013_2025_polymarket/15M_PUT.json",
        call_file: str = "/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        btc_file: str = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json",
        results_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/test_results"
    ):
        self.put_file = put_file
        self.call_file = call_file
        self.btc_file = btc_file
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize three strategies
        self.strategies = {
            'A1': StrategyTester('A1', discrepancy_threshold=0.03),
            'A2': StrategyTester('A2', discrepancy_threshold=0.02),
            'A4': StrategyTester('A4', discrepancy_threshold=0.04)
        }
        
        # Price history (shared across strategies)
        self.price_history_5s = deque(maxlen=50)
        self.price_history_15s = deque(maxlen=150)
        self.price_history_30s = deque(maxlen=300)
        
        # Period tracking
        self.period_start_minute: Optional[int] = None
        
        # Load existing results
        self.load_daily_results()
        
        logger.info("="*80)
        logger.info("🧪 MULTI-STRATEGY DRY RUN TESTER")
        logger.info("="*80)
        logger.info("Testing 3 strategies simultaneously:")
        logger.info("  A1: threshold = 0.03")
        logger.info("  A2: threshold = 0.02 (more aggressive)")
        logger.info("  A4: threshold = 0.04 (more conservative)")
        logger.info("Algorithm: Delta-based lag detection (not regression)")
        logger.info("NO ACTUAL TRADES - Simulation only")
        logger.info("="*80)
    
    def get_results_path(self) -> Path:
        """Get path for today's results file"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.results_dir / f"test_results_{today}.json"
    
    def load_daily_results(self):
        """Load today's results if exists"""
        results_path = self.get_results_path()
        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                
                for strategy_name, strategy_data in data.get('strategies', {}).items():
                    if strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]
                        strategy.daily_pnl = strategy_data.get('daily_pnl', 0.0)
                        strategy.total_trades = strategy_data.get('total_trades', 0)
                        strategy.winning_trades = strategy_data.get('winning_trades', 0)
                        strategy.losing_trades = strategy_data.get('losing_trades', 0)
                        strategy.largest_win = strategy_data.get('largest_win', 0.0)
                        strategy.largest_loss = strategy_data.get('largest_loss', 0.0)
                        
                        # Reload trades
                        for trade_data in strategy_data.get('trades', []):
                            trade = SimulatedTrade(**trade_data)
                            strategy.trades_today.append(trade)
                
                logger.info(f"📥 Loaded existing results from {results_path.name}")
                for name, strategy in self.strategies.items():
                    logger.info(f"   {name}: ${strategy.daily_pnl:+.2f} PNL, {strategy.total_trades} trades")
            
            except Exception as e:
                logger.error(f"Error loading results: {e}")
    
    def save_daily_results(self):
        """Save current results"""
        results_path = self.get_results_path()
        
        data = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'last_updated': datetime.now().isoformat(),
            'strategies': {}
        }
        
        for name, strategy in self.strategies.items():
            stats = strategy.get_stats()
            stats['trades'] = [asdict(t) for t in strategy.trades_today]
            data['strategies'][name] = stats
        
        try:
            with open(results_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def read_json_file(self, filepath: str) -> Optional[dict]:
        """Safely read JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    
    def get_current_prices(self) -> Optional[PriceSnapshot]:
        """Read current prices"""
        put_data = self.read_json_file(self.put_file)
        call_data = self.read_json_file(self.call_file)
        btc_data = self.read_json_file(self.btc_file)
        
        if not all([put_data, call_data, btc_data]):
            return None
        
        try:
            put_bid = put_data.get('best_bid')
            put_ask = put_data.get('best_ask')
            call_bid = call_data.get('best_bid')
            call_ask = call_data.get('best_ask')
            
            if not put_bid or not put_ask or not call_bid or not call_ask:
                return None
            
            snapshot = PriceSnapshot(
                timestamp=time.time(),
                btc_price=btc_data['price'],
                put_bid=put_bid.get('price', 0.0),
                put_ask=put_ask.get('price', 0.0),
                call_bid=call_bid.get('price', 0.0),
                call_ask=call_ask.get('price', 0.0),
                put_spread=put_data.get('spread', 0.0),
                call_spread=call_data.get('spread', 0.0)
            )
            
            if not (0 <= snapshot.put_bid <= 1 and 0 <= snapshot.put_ask <= 1 and
                    0 <= snapshot.call_bid <= 1 and 0 <= snapshot.call_ask <= 1):
                return None
            
            return snapshot
        except (KeyError, TypeError) as e:
            return None
    
    def check_period_expiry(self) -> Tuple[bool, float]:
        """Check if period expired"""
        now = datetime.now()
        current_minute = now.minute
        
        period_start_minutes = [0, 15, 30, 45]
        for start_min in period_start_minutes:
            if current_minute >= start_min and current_minute < start_min + 15:
                seconds_into_period = (current_minute - start_min) * 60 + now.second
                seconds_remaining = 900 - seconds_into_period
                
                # Detect new period
                if self.period_start_minute != start_min:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD DETECTED - Clearing positions")
                    logger.info(f"{'='*80}")
                    
                    # Close all open positions
                    for strategy in self.strategies.values():
                        if strategy.position:
                            logger.info(f"[{strategy.name}] Marking position as CLOSED (period expired)")
                            strategy.position = None
                    
                    # Clear price history
                    self.price_history_5s.clear()
                    self.price_history_15s.clear()
                    self.price_history_30s.clear()
                    
                    self.period_start_minute = start_min
                    
                    # Save results
                    self.save_daily_results()
                
                is_expired = seconds_remaining <= 0
                return is_expired, seconds_remaining
        
        return True, 0
    
    def detect_btc_spike(self) -> Tuple[bool, float]:
        """Detect BTC spike"""
        if len(self.price_history_5s) < 20:
            return False, 0.0
        
        recent_prices = list(self.price_history_5s)
        current_price = recent_prices[-1].btc_price
        price_2s_ago = recent_prices[-20].btc_price
        
        spike_pct = ((current_price - price_2s_ago) / price_2s_ago) * 100
        has_spike = abs(spike_pct) > 0.05
        
        return has_spike, spike_pct
    
    def calculate_btc_option_sensitivity(self, token_type: str) -> Optional[float]:
        """
        Calculate real-time sensitivity: how much option price changes per $1 BTC move
        Uses recent actual observations
        """
        if len(self.price_history_5s) < 10:
            return None
        
        recent = list(self.price_history_5s)[-10:]  # Last 1 second
        
        if token_type == 'PUT':
            prices = [(s.btc_price, (s.put_bid + s.put_ask) / 2) for s in recent]
        else:
            prices = [(s.btc_price, (s.call_bid + s.call_ask) / 2) for s in recent]
        
        # Calculate average sensitivity from recent changes
        sensitivities = []
        for i in range(1, len(prices)):
            btc_change = prices[i][0] - prices[i-1][0]
            option_change = prices[i][1] - prices[i-1][1]
            
            if abs(btc_change) > 0.1:  # Only when BTC actually moved
                sensitivity = option_change / btc_change
                sensitivities.append(sensitivity)
        
        if not sensitivities:
            # Default sensitivity based on token type
            return -0.00001 if token_type == 'PUT' else 0.00001
        
        # Use median to avoid outliers
        return np.median(sensitivities)
    
    def calculate_expected_price_delta_based(self, current_btc: float, token_type: str) -> Optional[float]:
        """
        Delta-based lag detection: detect when market hasn't caught up to BTC movement
        """
        if len(self.price_history_5s) < 20:
            return None
        
        recent = list(self.price_history_5s)
        
        # Get BTC movement over last 2 seconds (20 samples @ 10Hz)
        btc_2s_ago = recent[-20].btc_price
        btc_now = recent[-1].btc_price
        btc_change = btc_now - btc_2s_ago
        
        # Get last known option price (from 2s ago when BTC was stable)
        if token_type == 'PUT':
            baseline_price = (recent[-20].put_bid + recent[-20].put_ask) / 2
            current_price = (recent[-1].put_bid + recent[-1].put_ask) / 2
        else:
            baseline_price = (recent[-20].call_bid + recent[-20].call_ask) / 2
            current_price = (recent[-1].call_bid + recent[-1].call_ask) / 2
        
        # Only calculate if BTC actually moved significantly
        if abs(btc_change) < 1.0:  # Less than $1 move - use current price
            return current_price
        
        # Get current sensitivity
        sensitivity = self.calculate_btc_option_sensitivity(token_type)
        if sensitivity is None:
            return None
        
        # Calculate expected price based on BTC movement
        expected_change = btc_change * sensitivity
        expected_price = baseline_price + expected_change
        
        # Clamp to valid range
        expected_price = max(0.01, min(0.99, expected_price))
        
        return round(expected_price, 2)
    
    def run(self):
        """Main test loop"""
        logger.info("\n🚀 Starting Multi-Strategy Tester")
        
        iteration = 0
        last_save = time.time()
        last_status = time.time()
        
        try:
            while True:
                # Check period
                is_expired, seconds_remaining = self.check_period_expiry()
                
                if is_expired:
                    time.sleep(5)
                    continue
                
                # Read prices
                snapshot = self.get_current_prices()
                
                if snapshot:
                    # Add to history
                    self.price_history_5s.append(snapshot)
                    self.price_history_15s.append(snapshot)
                    self.price_history_30s.append(snapshot)
                    
                    # Calculate expected prices using delta-based approach
                    if len(self.price_history_5s) >= 20:
                        expected_put = self.calculate_expected_price_delta_based(snapshot.btc_price, 'PUT')
                        expected_call = self.calculate_expected_price_delta_based(snapshot.btc_price, 'CALL')
                        has_spike, spike_pct = self.detect_btc_spike()
                        
                        # Test each strategy
                        for strategy in self.strategies.values():
                            if strategy.position is None:
                                # Check entry
                                entry_signal = strategy.check_entry_signal(
                                    snapshot, expected_put, expected_call, has_spike, spike_pct
                                )
                                if entry_signal:
                                    token_type, entry_price, reason = entry_signal
                                    strategy.execute_simulated_buy(token_type, entry_price, 
                                                                  snapshot.btc_price, reason)
                            else:
                                # Check exit
                                exit_signal = strategy.check_exit_signal(
                                    snapshot, expected_put, expected_call, seconds_remaining
                                )
                                if exit_signal:
                                    exit_price, reason = exit_signal
                                    strategy.execute_simulated_sell(exit_price, snapshot.btc_price, reason)
                        
                        # Status update every 30 seconds
                        if time.time() - last_status >= 30:
                            logger.info(f"\n{'='*80}")
                            logger.info(f"📊 STATUS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
                            logger.info(f"{'='*80}")
                            logger.info(f"BTC: ${snapshot.btc_price:.2f} | Expiry: {seconds_remaining/60:.1f}m")
                            logger.info(f"PUT: {snapshot.put_bid:.2f}/{snapshot.put_ask:.2f} | "
                                      f"CALL: {snapshot.call_bid:.2f}/{snapshot.call_ask:.2f}")
                            
                            for name, strategy in self.strategies.items():
                                stats = strategy.get_stats()
                                pos = f"📍 {stats['open_position']}" if stats['open_position'] else "⭕"
                                logger.info(f"[{name}] ${stats['daily_pnl']:+7.2f} | "
                                          f"{stats['total_trades']:2d} trades | "
                                          f"{stats['win_rate']:5.1f}% WR | {pos}")
                            
                            logger.info(f"{'='*80}\n")
                            last_status = time.time()
                
                # Auto-save every 60 seconds
                if time.time() - last_save >= 60:
                    self.save_daily_results()
                    last_save = time.time()
                
                iteration += 1
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Test stopped by user")
            logger.info("\n" + "="*80)
            logger.info("📊 FINAL RESULTS")
            logger.info("="*80)
            
            for name, strategy in self.strategies.items():
                stats = strategy.get_stats()
                logger.info(f"\n[{name}] Threshold: ${stats['threshold']}")
                logger.info(f"  Daily PNL:     ${stats['daily_pnl']:+.2f}")
                logger.info(f"  Total Trades:  {stats['total_trades']}")
                logger.info(f"  Win/Loss:      {stats['winning_trades']}/{stats['losing_trades']}")
                logger.info(f"  Win Rate:      {stats['win_rate']:.1f}%")
                logger.info(f"  Largest Win:   ${stats['largest_win']:+.2f}")
                logger.info(f"  Largest Loss:  ${stats['largest_loss']:+.2f}")
            
            self.save_daily_results()
            logger.info(f"\n💾 Results saved to: {self.get_results_path()}")


def main():
    """Entry point"""
    tester = MultiStrategyTester()
    tester.run()


if __name__ == "__main__":
    main()
