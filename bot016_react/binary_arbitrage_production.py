#!/usr/bin/env python3
"""
Binary Options Regression Arbitrage Bot - PRODUCTION VERSION
Exploits market lag after BTC price spikes using polynomial regression
Integrated with Polymarket API for real trading
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
import logging
import warnings
import os
import sys

# Suppress numpy warnings
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# Import Polymarket trading core
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

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
class Position:
    """Open position tracker"""
    token_type: str  # 'PUT' or 'CALL'
    token_id: str
    entry_price: float
    entry_time: float
    quantity: float
    entry_btc_price: float


@dataclass
class Trade:
    """Completed trade record"""
    timestamp: str
    token_type: str
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    btc_price: float
    pnl: Optional[float] = None
    reason: str = ""


class BinaryArbitrageProduction:
    """Production binary options arbitrage bot"""
    
    def __init__(
        self,
        credentials: dict,
        put_file: str = "/home/ubuntu/013_2025_polymarket/15M_PUT.json",
        call_file: str = "/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        btc_file: str = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json",
        discrepancy_threshold: float = 0.03,
        position_size: float = 5.0,  # 5 shares max
        daily_log_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react",
        sensitivity_data_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data"
    ):
        self.put_file = put_file
        self.call_file = call_file
        self.btc_file = btc_file
        self.discrepancy_threshold = discrepancy_threshold
        self.position_size = position_size
        self.daily_log_dir = Path(daily_log_dir)
        self.daily_log_dir.mkdir(exist_ok=True)
        self.sensitivity_data_dir = Path(sensitivity_data_dir)
        
        # Load sensitivity data
        self.sensitivity_data = self.load_sensitivity_data()
        
        # Initialize Polymarket trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )
        
        # Price history buffers
        self.price_history_5s = deque(maxlen=50)
        self.price_history_15s = deque(maxlen=150)
        self.price_history_30s = deque(maxlen=300)
        
        # Position tracking
        self.position: Optional[Position] = None
        self.trades_today: List[Trade] = []
        self.daily_pnl: float = 0.0
        
        # Asset IDs (will be reloaded each period)
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None
        self.period_start_minute: Optional[int] = None
        self.period_start_btc: Optional[float] = None  # BTC price at period start (strike)
        
        # Position verification
        self.last_position_check: float = 0.0
        self.position_check_interval: float = 60.0  # Check every 60 seconds
        
        # Load today's log if exists
        self.load_daily_log()
        
        logger.info("="*80)
        logger.info("🚀 BINARY ARBITRAGE BOT - PRODUCTION MODE")
        logger.info("="*80)
        logger.info(f"Discrepancy threshold: ${self.discrepancy_threshold}")
        logger.info(f"Position size: {self.position_size} shares")
        
        if self.sensitivity_data:
            num_bins = len(self.sensitivity_data.get('bins', {}))
            logger.info(f"Strategy: Sensitivity-based lag detection")
            logger.info(f"✅ Sensitivity data loaded: {num_bins} bins")
        else:
            logger.info(f"Strategy: Delta-based lag detection (fallback)")
            logger.info(f"⚠️  No sensitivity data available")
        logger.info("="*80)
    
    def get_daily_log_path(self) -> Path:
        """Get path for today's log file"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_log_dir / f"trading_log_{today}.json"
    
    def load_daily_log(self):
        """Load today's trading log if it exists"""
        log_path = self.get_daily_log_path()
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    data = json.load(f)
                    self.trades_today = [Trade(**t) for t in data.get('trades', [])]
                    self.daily_pnl = data.get('daily_pnl', 0.0)
                    logger.info(f"Loaded {len(self.trades_today)} trades from today")
                    logger.info(f"Current daily PNL: ${self.daily_pnl:.2f}")
            except Exception as e:
                logger.error(f"Error loading daily log: {e}")
    
    def load_sensitivity_data(self) -> Optional[dict]:
        """Load cumulative sensitivity data from master file"""
        try:
            # Use master file (not daily file)
            data_file = self.sensitivity_data_dir / "sensitivity_master.json"
            
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # Log statistics
                num_bins = len(data.get('bins', {}))
                total_measurements = data.get('total_measurements', 0)
                logger.info(f"📥 Loaded sensitivity master file:")
                logger.info(f"   Bins: {num_bins}")
                logger.info(f"   Total measurements: {total_measurements:,}")
                
                return data
            else:
                logger.warning(f"⚠️  Sensitivity master file not found: {data_file}")
                logger.warning(f"   Run sensitivity monitor first to build data")
                return None
        except Exception as e:
            logger.warning(f"Could not load sensitivity data: {e}")
            return None
    
    def get_bin_key(self, distance: float, seconds_to_expiry: float, volatility: float) -> str:
        """Get bin key for current state"""
        # Distance bins
        distance_bins = [
            (0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
            (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
            (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
            (1280, float('inf'), "1280+")
        ]
        
        # Time bins
        time_bins = [
            (13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
            (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
            (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
            (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
            (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
            (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
            (2, 5, "5s-2s"), (0, 2, "last-2s")
        ]
        
        # Volatility bins
        vol_bins = [
            (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
            (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"),
            (120, float('inf'), "120+")
        ]
        
        def get_bin_label(value, bins):
            for min_val, max_val, label in bins:
                if min_val <= value < max_val:
                    return label
            return bins[-1][2]
        
        dist_label = get_bin_label(distance, distance_bins)
        time_label = get_bin_label(seconds_to_expiry, time_bins)
        vol_label = get_bin_label(volatility, vol_bins)
        
        return f"{dist_label}|{time_label}|{vol_label}"
    
    def get_expected_sensitivity(self, bin_key: str, token_type: str) -> Optional[float]:
        """Get expected sensitivity from historical data"""
        if not self.sensitivity_data:
            return None
        
        bins = self.sensitivity_data.get('bins', {})
        bin_data = bins.get(bin_key)
        
        if not bin_data:
            return None
        
        # Use median sensitivity (more robust than average)
        if token_type == 'PUT':
            median_sens = bin_data.get('put_sensitivity', {}).get('median')
        else:
            median_sens = bin_data.get('call_sensitivity', {}).get('median')
        
        # If median is exactly 0, the data is not useful (too much noise)
        # Return None so bot won't trade on this bin
        if median_sens is not None and abs(median_sens) < 0.000001:
            return None
        
        return median_sens
    
    def save_daily_log(self):
        """Save today's trading activity"""
        log_path = self.get_daily_log_path()
        data = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'daily_pnl': round(self.daily_pnl, 2),
            'total_trades': len(self.trades_today),
            'trades': [asdict(t) for t in self.trades_today],
            'current_position': asdict(self.position) if self.position else None
        }
        
        try:
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving daily log: {e}")
    
    def get_usdc_balance(self) -> float:
        """Get USDC balance from Polymarket"""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            
            response = self.trader.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL
                )
            )
            
            balance_raw = int(response.get('balance', 0))
            balance_usdc = balance_raw / 10**6
            
            return balance_usdc
        except Exception as e:
            logger.error(f"Error getting USDC balance: {e}")
            return 0.0
    
    def check_token_balance(self, token_id: str) -> float:
        """Check balance of specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance for {token_id[:12]}...: {e}")
            return 0.0
    
    def read_json_file(self, filepath: str) -> Optional[dict]:
        """Safely read JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
            return None
    
    def reload_asset_ids(self):
        """Reload PUT and CALL asset IDs from data files"""
        put_data = self.read_json_file(self.put_file)
        call_data = self.read_json_file(self.call_file)
        
        if put_data and call_data:
            new_put_id = put_data.get('asset_id')
            new_call_id = call_data.get('asset_id')
            
            # Check if IDs changed
            put_changed = new_put_id != self.current_put_id
            call_changed = new_call_id != self.current_call_id
            
            if put_changed or call_changed:
                logger.info(f"   Asset IDs {'CHANGED' if (put_changed or call_changed) else 'confirmed'}:")
                
                if put_changed:
                    logger.info(f"   PUT:  {self.current_put_id[:16] if self.current_put_id else 'None'}... → {new_put_id[:16]}...")
                else:
                    logger.info(f"   PUT:  {new_put_id[:16]}... (unchanged)")
                
                if call_changed:
                    logger.info(f"   CALL: {self.current_call_id[:16] if self.current_call_id else 'None'}... → {new_call_id[:16]}...")
                else:
                    logger.info(f"   CALL: {new_call_id[:16]}... (unchanged)")
                
                self.current_put_id = new_put_id
                self.current_call_id = new_call_id
            else:
                logger.info(f"   Asset IDs confirmed (no change):")
                logger.info(f"   PUT:  {new_put_id[:16]}...")
                logger.info(f"   CALL: {new_call_id[:16]}...")
        else:
            logger.warning(f"   ⚠️  Could not read asset IDs from files")
    
    def check_period_expiry(self) -> Tuple[bool, float]:
        """
        Check if current period has expired
        Returns: (is_expired, seconds_remaining)
        """
        now = datetime.now()
        current_minute = now.minute
        
        # Find which period we're in (0-14, 15-29, 30-44, 45-59)
        period_start_minutes = [0, 15, 30, 45]
        for start_min in period_start_minutes:
            if current_minute >= start_min and current_minute < start_min + 15:
                seconds_into_period = (current_minute - start_min) * 60 + now.second
                seconds_remaining = 900 - seconds_into_period  # 15 minutes = 900 seconds
                
                is_expired = seconds_remaining <= 0
                return is_expired, seconds_remaining
        
        return True, 0
    
    def get_strike_price_from_bybit(self, start_min: int) -> Optional[float]:
        """
        Get the exact opening price of the 15-minute candle from Bybit API
        This is the true strike price for the period
        """
        try:
            import requests
            from datetime import datetime, timezone
            
            # Calculate the timestamp for this period's start
            now = datetime.now(timezone.utc)
            period_start = now.replace(minute=start_min, second=0, microsecond=0)
            start_timestamp = int(period_start.timestamp() * 1000)  # milliseconds
            
            # Bybit API endpoint
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
                    result = data.get('result', {})
                    kline_list = result.get('list', [])
                    
                    if kline_list and len(kline_list) > 0:
                        # list[1] is the open price
                        open_price = float(kline_list[0][1])
                        return open_price
            
            logger.warning(f"⚠️  Bybit API returned invalid data")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching strike price from Bybit: {e}")
            return None
    
    def handle_new_period(self, start_min: int):
        """
        Handle transition to new period:
        1. Mark old positions as closed
        2. Capture BTC price at period start (strike)
        3. Load asset IDs immediately
        4. Wait 20 seconds for market to settle AND accumulate price data
        5. Reload asset IDs to confirm
        """
        logger.info("="*80)
        logger.info(f"🔄 NEW PERIOD DETECTED - Starting at :{start_min:02d}")
        logger.info("="*80)
        
        # STEP 1: Close old positions
        if self.position:
            logger.warning(f"⚠️  Marking previous period position as CLOSED")
            logger.warning(f"   Token: {self.position.token_type} @ ${self.position.entry_price:.2f}")
            logger.warning(f"   Position will settle at expiry")
            self.position = None
        
        # STEP 2: Capture BTC price at period start (strike price) from Bybit API
        logger.info("📍 STEP 2: Getting strike price from Bybit API...")
        strike_from_api = self.get_strike_price_from_bybit(start_min)
        
        if strike_from_api:
            self.period_start_btc = strike_from_api
            logger.info(f"   ✅ Strike Price (15m candle open): ${self.period_start_btc:,.2f}")
        else:
            # Fallback to current price from local file
            logger.warning(f"   ⚠️  Could not get strike from Bybit API, using fallback")
            btc_data = self.read_json_file(self.btc_file)
            if btc_data:
                self.period_start_btc = btc_data['price']
                logger.info(f"   📍 Strike Price (fallback): ${self.period_start_btc:,.2f}")
            else:
                logger.error(f"   ❌ Could not determine strike price!")
        
        # STEP 3: Load asset IDs immediately
        logger.info("📥 STEP 3: Loading asset IDs for new period...")
        self.reload_asset_ids()
        
        # STEP 4: Wait 20 seconds for market to stabilize AND accumulate data
        logger.info("⏳ STEP 2: Waiting 20 seconds for markets to stabilize and data accumulation...")
        logger.info("   (Need sufficient price history for sensitivity-based trading)")
        for i in range(20, 0, -1):
            logger.info(f"   🚫 BLACKOUT: {i} seconds remaining...")
            time.sleep(1)
        
        # STEP 5: Reload asset IDs to confirm they're stable
        logger.info("🔄 STEP 3: Reloading asset IDs to confirm...")
        self.reload_asset_ids()
        
        # STEP 6: Clear price history to start fresh
        logger.info("🧹 STEP 4: Clearing old price history...")
        self.price_history_5s.clear()
        self.price_history_15s.clear()
        self.price_history_30s.clear()
        logger.info("   Price buffers reset - will accumulate new period data")
        
        logger.info("✅ New period ready - Trading resumed")
        logger.info("   Note: Sensitivity-based trading needs ~2 seconds of data before first trade")
        logger.info("="*80)
    
    def is_in_period_blackout(self) -> Tuple[bool, float]:
        """
        Check if we're in the 5-second blackout at start of new period
        Returns: (in_blackout, seconds_into_period)
        
        Note: Blackout is now handled in handle_new_period() with blocking sleep
        This function just checks if we need to trigger the period change
        """
        now = datetime.now()
        current_minute = now.minute
        
        period_start_minutes = [0, 15, 30, 45]
        for start_min in period_start_minutes:
            if current_minute >= start_min and current_minute < start_min + 15:
                seconds_into_period = (current_minute - start_min) * 60 + now.second
                
                # Check if this is a new period we haven't processed yet
                if self.period_start_minute != start_min:
                    self.period_start_minute = start_min
                    # Trigger period change handling
                    self.handle_new_period(start_min)
                    # After handling, we're past the blackout
                    return False, seconds_into_period
                
                return False, seconds_into_period
        
        return False, 0
    
    def get_current_prices(self) -> Optional[PriceSnapshot]:
        """Read current prices from all three files"""
        put_data = self.read_json_file(self.put_file)
        call_data = self.read_json_file(self.call_file)
        btc_data = self.read_json_file(self.btc_file)
        
        if not all([put_data, call_data, btc_data]):
            return None
        
        try:
            # Handle None/null best_bid and best_ask
            put_bid = put_data.get('best_bid')
            put_ask = put_data.get('best_ask')
            call_bid = call_data.get('best_bid')
            call_ask = call_data.get('best_ask')
            
            # Skip if any critical price is missing
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
            
            # Sanity check
            if not (0 <= snapshot.put_bid <= 1 and 0 <= snapshot.put_ask <= 1 and
                    0 <= snapshot.call_bid <= 1 and 0 <= snapshot.call_ask <= 1):
                return None
            
            return snapshot
        except (KeyError, TypeError) as e:
            logger.debug(f"Error parsing price data: {e}")
            return None
    
    def check_data_freshness(self, snapshot: PriceSnapshot) -> bool:
        """Check if data is fresh (not stale)"""
        current_time = time.time()
        data_age = current_time - snapshot.timestamp
        
        if data_age > 5.0:
            logger.warning(f"⚠️  Data is stale ({data_age:.1f}s old)")
            return False
        
        return True
    
    def verify_position_from_wallet(self):
        """
        Verify position by checking actual token holdings in wallet
        Called every 1 minute to sync position tracking with reality
        ALSO validates that asset IDs are correct for current period
        """
        logger.info("\n" + "="*70)
        logger.info("🔍 POSITION VERIFICATION - Checking wallet holdings")
        logger.info("="*70)
        
        # STEP 1: Validate current asset IDs by trying to check balances
        if self.current_put_id and self.current_call_id:
            try:
                put_balance = self.check_token_balance(self.current_put_id)
                call_balance = self.check_token_balance(self.current_call_id)
                logger.info(f"   PUT balance:  {put_balance:.2f} shares")
                logger.info(f"   CALL balance: {call_balance:.2f} shares")
            except Exception as e:
                # If balance check fails, asset IDs might be stale
                error_str = str(e).lower()
                if 'does not exist' in error_str or 'orderbook' in error_str:
                    logger.error(f"❌ ASSET ID ERROR: {e}")
                    logger.error(f"   Current PUT ID:  {self.current_put_id[:20]}...")
                    logger.error(f"   Current CALL ID: {self.current_call_id[:20]}...")
                    logger.warning(f"⚠️  Asset IDs appear to be stale - FORCE RELOADING")
                    
                    # Force reload from files
                    self.reload_asset_ids()
                    
                    # Try again with new IDs
                    try:
                        put_balance = self.check_token_balance(self.current_put_id)
                        call_balance = self.check_token_balance(self.current_call_id)
                        logger.info(f"   ✅ Reloaded - PUT balance:  {put_balance:.2f} shares")
                        logger.info(f"   ✅ Reloaded - CALL balance: {call_balance:.2f} shares")
                    except Exception as e2:
                        logger.error(f"❌ Still failing after reload: {e2}")
                        logger.error(f"   Data files may not be updated yet")
                        logger.info("="*70 + "\n")
                        return
                else:
                    # Different error, just log and return
                    logger.error(f"❌ Error checking balance: {e}")
                    logger.info("="*70 + "\n")
                    return
        else:
            logger.warning(f"   ⚠️  Asset IDs not loaded yet")
            logger.info("="*70 + "\n")
            return
        
        # STEP 2: Determine what position we actually have
        has_put = put_balance >= 0.5
        has_call = call_balance >= 0.5
        
        # Case 1: We have PUT tokens
        if has_put and not has_call:
            if self.position is None or self.position.token_type != 'PUT':
                logger.warning(f"⚠️  WALLET SYNC: Found PUT position not in tracking!")
                logger.warning(f"   Creating position record for {put_balance:.2f} PUT shares")
                
                # Get current PUT price for entry price estimate
                put_data = self.read_json_file(self.put_file)
                current_price = 0.50  # Default
                if put_data and put_data.get('best_bid'):
                    current_price = (put_data['best_bid'].get('price', 0.50) + 
                                   put_data['best_ask'].get('price', 0.50)) / 2
                
                self.position = Position(
                    token_type='PUT',
                    token_id=self.current_put_id,
                    entry_price=current_price,
                    entry_time=time.time(),
                    quantity=put_balance,
                    entry_btc_price=0  # Unknown
                )
                logger.info(f"   ✅ Position tracking updated: PUT @ ${current_price:.2f}")
            elif abs(self.position.quantity - put_balance) > 0.5:
                logger.warning(f"   ⚠️  Position quantity mismatch!")
                logger.warning(f"   Tracked: {self.position.quantity:.2f}, Actual: {put_balance:.2f}")
                logger.warning(f"   Updating to actual balance")
                self.position.quantity = put_balance
        
        # Case 2: We have CALL tokens
        elif has_call and not has_put:
            if self.position is None or self.position.token_type != 'CALL':
                logger.warning(f"⚠️  WALLET SYNC: Found CALL position not in tracking!")
                logger.warning(f"   Creating position record for {call_balance:.2f} CALL shares")
                
                # Get current CALL price for entry price estimate
                call_data = self.read_json_file(self.call_file)
                current_price = 0.50  # Default
                if call_data and call_data.get('best_bid'):
                    current_price = (call_data['best_bid'].get('price', 0.50) + 
                                   call_data['best_ask'].get('price', 0.50)) / 2
                
                self.position = Position(
                    token_type='CALL',
                    token_id=self.current_call_id,
                    entry_price=current_price,
                    entry_time=time.time(),
                    quantity=call_balance,
                    entry_btc_price=0  # Unknown
                )
                logger.info(f"   ✅ Position tracking updated: CALL @ ${current_price:.2f}")
            elif abs(self.position.quantity - call_balance) > 0.5:
                logger.warning(f"   ⚠️  Position quantity mismatch!")
                logger.warning(f"   Tracked: {self.position.quantity:.2f}, Actual: {call_balance:.2f}")
                logger.warning(f"   Updating to actual balance")
                self.position.quantity = call_balance
        
        # Case 3: We have both (shouldn't happen)
        elif has_put and has_call:
            logger.error(f"❌ UNEXPECTED: Both PUT and CALL tokens in wallet!")
            logger.error(f"   PUT: {put_balance:.2f}, CALL: {call_balance:.2f}")
            logger.error(f"   Keeping current tracking, manual intervention may be needed")
        
        # Case 4: We have neither
        else:
            if self.position is not None:
                logger.warning(f"⚠️  WALLET SYNC: Position tracking shows {self.position.token_type} "
                             f"but wallet is empty")
                logger.warning(f"   Clearing position tracking")
                self.position = None
            else:
                logger.info(f"   ✅ Wallet empty, no position (as expected)")
        
        logger.info("="*70 + "\n")
    
    def detect_btc_spike(self) -> Tuple[bool, float]:
        """Detect if BTC just had a significant price spike"""
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
        Uses recent actual observations, not regression
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
    
    def calculate_expected_price_with_sensitivity(self, current_btc: float, token_type: str,
                                                    baseline_price: float, btc_change: float,
                                                    distance: float, seconds_to_expiry: float,
                                                    volatility: float) -> Optional[float]:
        """
        Calculate expected price using ONLY sensitivity data
        Returns None if:
        1. Bin doesn't exist
        2. Sensitivity is None
        3. Sensitivity median is 0 (no meaningful data)
        
        NO FALLBACKS - Only trade with proven data
        """
        # Get bin key for current state
        bin_key = self.get_bin_key(distance, seconds_to_expiry, volatility)
        
        # Get sensitivity from historical data
        expected_sensitivity = self.get_expected_sensitivity(bin_key, token_type)
        
        # Reject if:
        # 1. No sensitivity data (bin doesn't exist or sensitivity is None)
        # 2. Sensitivity is zero (no meaningful reaction)
        if expected_sensitivity is None or abs(expected_sensitivity) < 0.000001:
            return None
        
        # Use historical sensitivity
        expected_change = btc_change * expected_sensitivity
        expected_price = baseline_price + expected_change
        
        # Clamp to valid range
        expected_price = max(0.01, min(0.99, expected_price))
        
        return round(expected_price, 2)
    
    def calculate_expected_price_delta_based(self, current_btc: float, token_type: str) -> Optional[float]:
        """
        Sensitivity-based lag detection - ONLY trades with historical data
        Returns None if no sensitivity data available
        """
        if len(self.price_history_5s) < 20:
            return None
        
        # Check if we have sensitivity data loaded
        if not self.sensitivity_data:
            return None
        
        recent = list(self.price_history_5s)
        
        # Get BTC movement over last 2 seconds
        btc_2s_ago = recent[-20].btc_price
        btc_now = recent[-1].btc_price
        btc_change = btc_now - btc_2s_ago
        
        # Get baseline option price
        if token_type == 'PUT':
            baseline_price = (recent[-20].put_bid + recent[-20].put_ask) / 2
            current_price = (recent[-1].put_bid + recent[-1].put_ask) / 2
        else:
            baseline_price = (recent[-20].call_bid + recent[-20].call_ask) / 2
            current_price = (recent[-1].call_bid + recent[-1].call_ask) / 2
        
        # Only calculate if BTC moved significantly
        if abs(btc_change) < 1.0:
            return None  # Don't trade on small movements
        
        # Must have period start BTC to calculate distance
        if not self.period_start_btc:
            return None
        
        # Calculate current state for bin lookup
        distance = abs(btc_now - self.period_start_btc)
        is_expired, seconds_remaining = self.check_period_expiry()
        
        # Calculate volatility
        if len(self.price_history_5s) >= 60:
            last_minute_prices = [s.btc_price for s in list(self.price_history_5s)[-60:]]
            volatility = max(last_minute_prices) - min(last_minute_prices)
        else:
            volatility = 0.0
        
        # Calculate expected price using ONLY sensitivity data
        expected_price = self.calculate_expected_price_with_sensitivity(
            btc_now, token_type, baseline_price, btc_change,
            distance, seconds_remaining, volatility
        )
        
        # Return None if bin doesn't exist (no fallback)
        return expected_price
    
    def calculate_expected_prices(self, current_btc: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate expected PUT and CALL prices using delta-based approach"""
        expected_put = self.calculate_expected_price_delta_based(current_btc, 'PUT')
        expected_call = self.calculate_expected_price_delta_based(current_btc, 'CALL')
        return expected_put, expected_call
    
    def check_entry_signal(self, snapshot: PriceSnapshot, 
                           expected_put: Optional[float], expected_call: Optional[float]) -> Optional[Tuple[str, float, str]]:
        """Check for entry signals based on market lag (NO SPIKE DETECTION)"""
        # CRITICAL: Reject if no valid sensitivity data (bin missing or median=0)
        if expected_put is None and expected_call is None:
            return None  # No bin data at all
        
        # Don't trade extreme prices
        if snapshot.put_ask >= 0.97 or snapshot.put_ask <= 0.03:
            expected_put = None
        if snapshot.call_ask >= 0.97 or snapshot.call_ask <= 0.03:
            expected_call = None
        
        # If both are None after filtering, no trade
        if expected_put is None and expected_call is None:
            return None
        
        put_lag = expected_put - snapshot.put_ask if expected_put else -999
        call_lag = expected_call - snapshot.call_ask if expected_call else -999
        
        # Simple lag-based entry (removed spike detection)
        if expected_put and put_lag >= self.discrepancy_threshold:
            return ('PUT', snapshot.put_ask, 
                   f"PUT lag ${put_lag:.3f}")
        
        if expected_call and call_lag >= self.discrepancy_threshold:
            return ('CALL', snapshot.call_ask,
                   f"CALL lag ${call_lag:.3f}")
        
        return None
    
    def check_exit_signal(self, snapshot: PriceSnapshot,
                          expected_put: Optional[float], expected_call: Optional[float]) -> Optional[Tuple[float, str]]:
        """
        Check for exit signals:
        1. Bid >= 0.99 (max profit)
        2. Opposite opportunity detected (exit and wait)
        3. Near expiry (< 1 min)
        """
        if not self.position:
            return None
        
        if self.position.token_type == 'PUT':
            # ALWAYS SELL at 0.99
            if snapshot.put_bid >= 0.99:
                return (0.99, "Hit max price 0.99")
            
            # Exit if CALL opportunity detected (but don't flip - just exit)
            if expected_call:
                call_lag = expected_call - snapshot.call_ask
                if call_lag >= self.discrepancy_threshold and snapshot.call_ask <= 0.97:
                    return (snapshot.put_bid, f"Opposite opportunity detected (CALL lag ${call_lag:.3f})")
        
        else:  # CALL
            # ALWAYS SELL at 0.99
            if snapshot.call_bid >= 0.99:
                return (0.99, "Hit max price 0.99")
            
            # Exit if PUT opportunity detected (but don't flip - just exit)
            if expected_put:
                put_lag = expected_put - snapshot.put_ask
                if put_lag >= self.discrepancy_threshold and snapshot.put_ask <= 0.97:
                    return (snapshot.call_bid, f"Opposite opportunity detected (PUT lag ${put_lag:.3f})")
        
        # Near expiry exit
        is_expired, seconds_remaining = self.check_period_expiry()
        if seconds_remaining < 60:  # < 1 minute
            exit_price = snapshot.put_bid if self.position.token_type == 'PUT' else snapshot.call_bid
            return (exit_price, f"Near expiry ({seconds_remaining:.0f}s)")
        
        return None
    
    def execute_buy(self, token_type: str, token_id: str, ask_price: float, 
                    btc_price: float, reason: str) -> bool:
        """Execute buy order (battle-tested from bot_C2_2)"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING BUY ORDER")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {self.position_size} shares")
            logger.info(f"💰 Ask Price: ${ask_price:.4f}")
            logger.info(f"📝 Reason: {reason}")
            
            required = ask_price * self.position_size
            logger.info(f"💵 Expected Cost: ${required:.2f}")
            
            # Check balance
            initial_usdc = self.get_usdc_balance()
            logger.info(f"💰 USDC Balance: ${initial_usdc:.2f}")
            
            # CRITICAL: Minimum balance check
            MIN_BALANCE = 4.90
            if initial_usdc < MIN_BALANCE:
                logger.error(f"❌ INSUFFICIENT BALANCE: ${initial_usdc:.2f} < ${MIN_BALANCE:.2f} minimum")
                logger.error(f"   Cannot trade with less than ${MIN_BALANCE:.2f} USDC")
                return False
            
            if initial_usdc < required:
                logger.error(f"❌ Insufficient balance for this trade")
                logger.error(f"   Need: ${required:.2f}, Have: ${initial_usdc:.2f}")
                return False
            
            # PHASE 1: Place order
            logger.info(f"\n🚀 PHASE 1: Placing buy order...")
            
            try:
                order_id = self.trader.place_buy_order(
                    token_id=token_id,
                    price=ask_price,
                    quantity=self.position_size
                )
            except Exception as order_error:
                error_str = str(order_error).lower()
                if 'does not exist' in error_str or 'orderbook' in error_str:
                    logger.error(f"❌ Error placing BUY order: {order_error}")
                    logger.error(f"   Token ID appears to be stale: {token_id[:20]}...")
                    logger.warning(f"⚠️  FORCE RELOADING ASSET IDs")
                    
                    # Reload asset IDs
                    self.reload_asset_ids()
                    
                    logger.info(f"   Retrying with new token ID...")
                    # Get new token ID
                    new_token_id = self.current_put_id if token_type == 'PUT' else self.current_call_id
                    
                    if new_token_id != token_id:
                        logger.info(f"   Old: {token_id[:20]}...")
                        logger.info(f"   New: {new_token_id[:20]}...")
                        
                        # Retry with new ID
                        try:
                            order_id = self.trader.place_buy_order(
                                token_id=new_token_id,
                                price=ask_price,
                                quantity=self.position_size
                            )
                            token_id = new_token_id  # Use new ID going forward
                            logger.info(f"   ✅ Retry successful with new token ID")
                        except Exception as retry_error:
                            logger.error(f"❌ Retry also failed: {retry_error}")
                            return False
                    else:
                        logger.error(f"   Token ID unchanged after reload - data files may be stale")
                        return False
                else:
                    # Different error, just propagate
                    raise
            
            if not order_id:
                logger.error(f"❌ Failed to place order")
                return False
            
            logger.info(f"✅ Order placed: {order_id[:16]}...")
            
            # PHASE 2: Verify (1s, 5s, 10s, 20s)
            logger.info("\n🔍 PHASE 2: Verifying...")
            
            verification_times = [1, 5, 10, 20]
            position_confirmed = False
            
            for wait_time in verification_times:
                time.sleep(wait_time)
                
                current_usdc = self.get_usdc_balance()
                usdc_spent = initial_usdc - current_usdc
                balance = self.check_token_balance(token_id)
                
                logger.info(f"   {wait_time}s: USDC spent ${usdc_spent:.2f}, Balance {balance:.2f}")
                
                if balance >= self.position_size * 0.95:
                    position_confirmed = True
                    logger.info(f"   ✅ Position confirmed")
                    break
                elif usdc_spent > (required * 0.8):
                    position_confirmed = True
                    logger.info(f"   💡 USDC suggests position open")
                    break
            
            # PHASE 3: Final check
            final_balance = self.check_token_balance(token_id)
            
            if position_confirmed and final_balance >= 0.5:
                logger.info(f"\n✅ SUCCESS: Position opened")
                
                self.position = Position(
                    token_type=token_type,
                    token_id=token_id,
                    entry_price=ask_price,
                    entry_time=time.time(),
                    quantity=final_balance,
                    entry_btc_price=btc_price
                )
                
                trade = Trade(
                    timestamp=datetime.now().isoformat(),
                    token_type=token_type,
                    action='BUY',
                    price=ask_price,
                    quantity=final_balance,
                    btc_price=btc_price,
                    reason=reason
                )
                
                self.trades_today.append(trade)
                self.save_daily_log()
                
                logger.info(f"📊 Position: {final_balance:.2f} @ ${ask_price:.4f}")
                return True
            else:
                logger.warning(f"\n⚠️  Position not confirmed")
                self.trader.cancel_all_orders()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def execute_sell(self, token_id: str, bid_price: float, reason: str) -> bool:
        """Execute sell order (battle-tested from bot_C2_2)"""
        try:
            # Verify we have tokens
            actual_balance = self.check_token_balance(token_id)
            
            if actual_balance < 0.5:
                logger.warning(f"\n⚠️  SELL ABORTED: No tokens in wallet")
                self.position = None
                return False
            
            size = actual_balance
            
            logger.info(f"\n{'='*60}")
            logger.info(f"💰 EXECUTING SELL ORDER - {reason}")
            logger.info(f"{'='*60}")
            logger.info(f"📦 Size: {size:.2f} shares")
            logger.info(f"💰 Bid: ${bid_price:.4f}")
            
            # Calculate P&L
            if self.position:
                pnl = (bid_price - self.position.entry_price) * size
                pnl_pct = ((bid_price / self.position.entry_price) - 1) * 100
                logger.info(f"📈 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            # Set sell price
            sell_price = max(0.01, bid_price - 0.001) if bid_price > 0 else 0.01
            logger.info(f"💰 Sell Price: ${sell_price:.4f}")
            
            # Place order
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=size
            )
            
            if not order_id:
                logger.error(f"❌ Failed to place sell order")
                return False
            
            logger.info(f"✅ Sell order placed: {order_id[:16]}...")
            
            # Wait and verify
            time.sleep(3)
            balance = self.check_token_balance(token_id)
            
            if balance < 0.5:
                logger.info(f"✅ Sell FILLED - Position CLOSED")
                
                if self.position:
                    pnl = (bid_price - self.position.entry_price) * size
                    self.daily_pnl += pnl
                    
                    trade = Trade(
                        timestamp=datetime.now().isoformat(),
                        token_type=self.position.token_type,
                        action='SELL',
                        price=bid_price,
                        quantity=size,
                        btc_price=0,  # Will be updated
                        pnl=pnl,
                        reason=reason
                    )
                    
                    self.trades_today.append(trade)
                    self.save_daily_log()
                
                self.position = None
                return True
            else:
                logger.warning(f"⚠️  May not be filled - Remaining: {balance:.2f}")
                self.trader.cancel_all_orders()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing sell: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Binary Arbitrage Bot - PRODUCTION MODE")
        
        iteration = 0
        last_freshness_check = time.time()
        
        try:
            while True:
                # Check for period expiry
                is_expired, seconds_remaining = self.check_period_expiry()
                
                if is_expired:
                    if self.position:
                        logger.warning("⏰ PERIOD EXPIRED - Position will settle at expiry")
                        self.position = None
                    time.sleep(5)
                    continue
                
                # Check for period change (handles blackout internally with blocking sleep)
                self.is_in_period_blackout()
                
                # Read prices
                snapshot = self.get_current_prices()
                
                if snapshot:
                    # Check freshness every 5s
                    if time.time() - last_freshness_check > 5.0:
                        if not self.check_data_freshness(snapshot):
                            logger.error("❌ Data stale - skipping")
                            time.sleep(1)
                            last_freshness_check = time.time()
                            continue
                        last_freshness_check = time.time()
                    
                    # Verify position from wallet every 60s
                    if time.time() - self.last_position_check > self.position_check_interval:
                        self.verify_position_from_wallet()
                        self.last_position_check = time.time()
                    
                    # Add to history
                    self.price_history_5s.append(snapshot)
                    self.price_history_15s.append(snapshot)
                    self.price_history_30s.append(snapshot)
                    
                    # Calculate expected prices
                    if len(self.price_history_5s) >= 20:
                        expected_put, expected_call = self.calculate_expected_prices(snapshot.btc_price)
                        has_spike, spike_pct = self.detect_btc_spike()
                        
                        # CRITICAL: Always check for 0.99 sell opportunity first
                        # Even if position tracking is off, sell if we have tokens at 0.99
                        if snapshot.put_bid >= 0.99 or snapshot.call_bid >= 0.99:
                            # Check what we actually have in wallet
                            if self.current_put_id and snapshot.put_bid >= 0.99:
                                put_balance = self.check_token_balance(self.current_put_id)
                                if put_balance >= 0.5:
                                    logger.warning(f"💎 BID @ 0.99 DETECTED - PUT balance: {put_balance:.2f}")
                                    if self.position and self.position.token_type == 'PUT':
                                        # Normal exit
                                        self.execute_sell(self.position.token_id, 0.99, "Hit max price 0.99")
                                    else:
                                        # Position not tracked but we have tokens!
                                        logger.warning(f"⚠️  PUT @ 0.99 but position not tracked - selling anyway!")
                                        self.execute_sell(self.current_put_id, 0.99, "Emergency sell @ 0.99 (untracked position)")
                            
                            if self.current_call_id and snapshot.call_bid >= 0.99:
                                call_balance = self.check_token_balance(self.current_call_id)
                                if call_balance >= 0.5:
                                    logger.warning(f"💎 BID @ 0.99 DETECTED - CALL balance: {call_balance:.2f}")
                                    if self.position and self.position.token_type == 'CALL':
                                        # Normal exit
                                        self.execute_sell(self.position.token_id, 0.99, "Hit max price 0.99")
                                    else:
                                        # Position not tracked but we have tokens!
                                        logger.warning(f"⚠️  CALL @ 0.99 but position not tracked - selling anyway!")
                                        self.execute_sell(self.current_call_id, 0.99, "Emergency sell @ 0.99 (untracked position)")
                        
                        # Log periodically
                        if iteration % 50 == 0:
                            spike_str = f"⚡ {spike_pct:+.3f}%" if has_spike else ""
                            exp_put_str = f"{expected_put:.2f}" if expected_put else "N/A (no bin)"
                            exp_call_str = f"{expected_call:.2f}" if expected_call else "N/A (no bin)"
                            
                            pos_str = "⭕ NO POSITION"
                            if self.position:
                                pos_str = f"📍 OPEN {self.position.token_type} @ ${self.position.entry_price:.2f}"
                            
                            # Show current bin status and sensitivity
                            if self.period_start_btc:
                                distance = abs(snapshot.btc_price - self.period_start_btc)
                                if len(self.price_history_5s) >= 60:
                                    last_minute_prices = [s.btc_price for s in list(self.price_history_5s)[-60:]]
                                    volatility = max(last_minute_prices) - min(last_minute_prices)
                                else:
                                    volatility = 0.0
                                current_bin = self.get_bin_key(distance, seconds_remaining, volatility)
                                
                                # Get sensitivity for this bin
                                put_sensitivity = self.get_expected_sensitivity(current_bin, 'PUT')
                                call_sensitivity = self.get_expected_sensitivity(current_bin, 'CALL')
                                
                                bin_exists = self.sensitivity_data and current_bin in self.sensitivity_data.get('bins', {})
                                
                                if bin_exists and put_sensitivity is not None and call_sensitivity is not None:
                                    bin_str = f"✅ {current_bin}"
                                    sens_str = f"\n   Sensitivity: PUT {put_sensitivity:+.6f}, CALL {call_sensitivity:+.6f} per $1"
                                else:
                                    bin_str = f"❌ {current_bin} (no data)"
                                    sens_str = ""
                            else:
                                bin_str = "N/A"
                                sens_str = ""
                            
                            logger.info(f"\n📊 BTC: ${snapshot.btc_price:.2f} {spike_str}")
                            logger.info(f"   PUT: {snapshot.put_bid:.2f}/{snapshot.put_ask:.2f} (exp: {exp_put_str})")
                            logger.info(f"   CALL: {snapshot.call_bid:.2f}/{snapshot.call_ask:.2f} (exp: {exp_call_str})")
                            logger.info(f"   Bin: {bin_str}{sens_str}")
                            logger.info(f"   Expiry: {seconds_remaining/60:.1f}m")
                            logger.info(f"💰 PNL: ${self.daily_pnl:.2f} | Trades: {len(self.trades_today)} | {pos_str}")
                        
                        # Check signals
                        if self.position is None:
                            entry_signal = self.check_entry_signal(snapshot, expected_put, expected_call)
                            if entry_signal:
                                token_type, entry_price, reason = entry_signal
                                token_id = self.current_put_id if token_type == 'PUT' else self.current_call_id
                                if token_id:
                                    self.execute_buy(token_type, token_id, entry_price, snapshot.btc_price, reason)
                        else:
                            # Look for exit (NO FLIPPING - just simple exit)
                            exit_signal = self.check_exit_signal(snapshot, expected_put, expected_call)
                            if exit_signal:
                                exit_price, reason = exit_signal
                                self.execute_sell(self.position.token_id, exit_price, reason)
                    else:
                        # Not enough data for regression yet
                        if iteration % 50 == 0:
                            logger.info(f"⏳ Accumulating price data... "
                                      f"({len(self.price_history_5s)}/20 samples needed for regression)")
                
                iteration += 1
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("\n⏸️  Bot stopped by user")
            if self.position:
                logger.warning(f"⚠️  Open position: {self.position.token_type} @ ${self.position.entry_price:.2f}")
            self.save_daily_log()
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            self.save_daily_log()


def main():
    """Entry point"""
    try:
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'
        credentials = load_credentials_from_env(env_path)
        logger.info(f"✅ Credentials loaded from {env_path}")
    except Exception as e:
        logger.error(f"❌ Error loading credentials: {e}")
        return
    
    if not credentials:
        logger.error("❌ Failed to load credentials")
        sys.exit(1)
    
    bot = BinaryArbitrageProduction(credentials)
    bot.run()


if __name__ == "__main__":
    main()
