# spike_config.py - Enhanced Configuration for BTC Spike Trading Bot
"""
Enhanced Configuration System for BTC Spike Trading

New Features:
- Spike detection parameters
- Protected sell configuration
- WebSocket connection settings
- Risk management parameters
- Logging and monitoring settings
"""

import os
import time
import json
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv('/home/ubuntu/013_2025_polymarket/keys/keys_ovh13.env')

# API Configuration (unchanged)
CLOB_API_URL = os.getenv("CLOB_API_URL")
PRIVATE_KEY = os.getenv("PK")
API_KEY = os.getenv("CLOB_API_KEY")
API_SECRET = os.getenv("CLOB_SECRET")
API_PASSPHRASE = os.getenv("CLOB_PASS_PHRASE")
CHAIN_ID = 137

# Token Management (unchanged)
def get_token_id():
    """Read TOKEN_ID from PUT.json file."""
    try:
        with open('/home/ubuntu/013_2025_polymarket/PUT.json', 'r') as f:
            data = json.load(f)
            return data.get('asset_id', '')
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading TOKEN_ID from PUT.json: {e}")
        return "37244842009101165723544499378199128671041778300958345638920066113343892104627"

def reload_token_id():
    """Reload TOKEN_ID and update global variable."""
    global TOKEN_ID
    new_token_id = get_token_id()
    if new_token_id != TOKEN_ID:
        old_token_id = TOKEN_ID
        TOKEN_ID = new_token_id
        print(f"🔄 TOKEN_ID UPDATED!")
        print(f"   Old: {old_token_id[:20]}...")
        print(f"   New: {TOKEN_ID[:20]}...")
        return True
    return False

TOKEN_ID = get_token_id()

# File Paths
BOOK_DATA_PATH = "/home/ubuntu/013_2025_polymarket/PUT.json"
BTC_PRICE_PATH = "/home/ubuntu/013_2025_polymarket/btc_price.json"

# === NEW SPIKE TRADING PARAMETERS ===

# Spike Detection Configuration
SPIKE_DETECTION = {
    'lookback_minutes': 5,           # Candlestick analysis window
    'min_spike_threshold': 0.003,    # 0.3% minimum spike threshold
    'spike_cooldown_seconds': 120,   # 2 minutes between spike buys
    'volatility_adjustment': True,   # Adjust threshold based on volatility
    'candlestick_period': 60,        # 1-minute candlesticks
    'min_data_points': 10,           # Minimum price points for analysis
}

# Protected Sell Configuration
PROTECTED_SELL = {
    'duration_seconds': 60,          # 1-minute protection period
    'min_ask_size': 1000,           # Minimum shares required on ASK level
    'preferred_ask_size': 2000,     # Preferred shares for high protection
    'max_levels_to_check': 5,       # Check top 5 ASK levels
    'price_buffer': 0.001,          # Small buffer above protected level
}

# WebSocket Configuration
WEBSOCKET_CONFIG = {
    'ping_interval': 30,            # Ping every 30 seconds
    'ping_timeout': 10,             # Wait 10 seconds for pong
    'max_reconnect_attempts': 10,   # Maximum reconnection tries
    'reconnect_delay': 5,           # Seconds between reconnection attempts
    'message_rate_limit': 1000,     # Max messages per minute
    'spike_check_interval': 1.0,    # Check spikes every second
}

# Risk Management (enhanced)
RISK_MANAGEMENT = {
    'max_spike_buys_per_hour': 5,   # Limit spike buys per hour
    'max_daily_trades': 50,         # Daily trade limit
    'emergency_stop_loss': 0.05,    # 5% emergency stop loss
    'connection_timeout_action': 'cancel_orders',  # Action on disconnect
    'max_position_size': 0.10,      # 10% of total capital
    'btc_volatility_filter': 0.20,  # Stop trading if BTC vol > 20%
}

# Trading Parameters (existing + enhanced)
MAX_TRADE_AMOUNT = 4.5              # USD per spike buy
MAX_TOKEN_VALUE_USD = 5             # Maximum total USD value of tokens
MIN_BALANCE_SELL = 100000           # Minimum balance for sells (0.1 tokens)
MIN_QUANTITY_FILTER = 200           # Minimum order size to consider

# Enhanced Spread Parameters
SPREAD_CONFIG = {
    'min_spread': 0.02,             # Minimum spread in USD
    'max_spread': 0.20,             # Maximum spread in USD
    'spread_buffer': 0.01,          # Additional buffer for spread sells
    'market_impact_buffer': 0.002,  # Buffer for market impact
    'spread_sell_threshold': 0.015, # Minimum spread for spread sells
}

# Backward compatibility constants
MIN_SPREAD = SPREAD_CONFIG['min_spread']
MAX_SPREAD = SPREAD_CONFIG['max_spread']
SPREAD_BUFFER = SPREAD_CONFIG['spread_buffer']
SPREAD_FORMULA_PATH = "/home/ubuntu/013_2025_polymarket/spread_formula.json"

# Mandatory Sell (unchanged)
MANDATORY_SELL_PRICE = 0.99         # Emergency exit at $0.99

# Risk Management Limits
MAX_DELAY_MS = 2000                 # Maximum acceptable data delay

# === NEW MONITORING & LOGGING ===

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',            # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,            # Save logs to file
    'log_file_path': '/home/ubuntu/013_2025_polymarket/logs/spike_bot.log',
    'max_log_size_mb': 50,          # Rotate logs at 50MB
    'keep_log_files': 7,            # Keep 7 log files
    'log_trades': True,             # Log all trades to separate file
    'trade_log_path': '/home/ubuntu/013_2025_polymarket/logs/trades.json',
}

# Performance Monitoring
MONITORING = {
    'save_performance_stats': True,
    'stats_file_path': '/home/ubuntu/013_2025_polymarket/logs/performance.json',
    'stats_save_interval': 300,     # Save every 5 minutes
    'spike_detection_stats': True,
    'websocket_stats': True,
    'trading_stats': True,
}

# Alert Configuration
ALERTS = {
    'enable_alerts': True,
    'alert_on_spike_detection': True,
    'alert_on_trade_execution': True,
    'alert_on_connection_issues': True,
    'alert_on_error_threshold': 5,   # Alert after 5 consecutive errors
    'alert_methods': ['console'],    # console, file, webhook
}

# === ENHANCED STATE CLASS ===

class SpikeState:
    """Enhanced state management for spike trading."""
    
    def __init__(self):
        # Basic state (from original)
        self.current_analysis = None
        self.last_update_time = 0
        self.api_calls_count = 0
        self.last_balance_check = 0.0
        self.last_known_balance = 0.0
        
        # Spike trading state
        self.spike_detection_active = True
        self.last_spike_time = 0
        self.spike_count_today = 0
        self.spike_history = []         # List of detected spikes
        
        # Protected sell state
        self.protected_sell_active = False
        self.protected_sell_start_time = 0
        self.protected_sell_end_time = 0
        
        # Trading performance
        self.trades_today = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit_loss = 0.0
        
        # Connection monitoring
        self.websocket_connected = False
        self.last_price_update = 0
        self.connection_errors = 0
        
        # Price history for spike detection
        self.btc_price_history = []
        self.price_update_count = 0
        
        # Performance metrics
        self.spike_detection_accuracy = 0.0
        self.avg_trade_execution_time = 0.0
        self.last_performance_save = 0
        
        print("✅ Enhanced spike trading state initialized")
    
    def add_spike_detection(self, spike_info: dict):
        """Add spike detection to history."""
        spike_record = {
            'timestamp': time.time(),
            'price_change': spike_info.get('price_change', 0),
            'strength': spike_info.get('strength', 0),
            'threshold': spike_info.get('threshold', 0),
            'successful_trade': False  # Will be updated if trade executes
        }
        
        self.spike_history.append(spike_record)
        self.last_spike_time = time.time()
        
        # Keep only last 100 spikes
        self.spike_history = self.spike_history[-100:]
        
        # Update daily count
        today = datetime.now().date()
        if not hasattr(self, 'last_spike_date') or self.last_spike_date != today:
            self.spike_count_today = 1
            self.last_spike_date = today
        else:
            self.spike_count_today += 1
    
    def update_trade_result(self, success: bool, profit_loss: float = 0.0):
        """Update trade execution results."""
        if success:
            self.successful_trades += 1
        else:
            self.failed_trades += 1
        
        self.trades_today += 1
        self.total_profit_loss += profit_loss
        
        # Update last spike record if exists
        if self.spike_history:
            self.spike_history[-1]['successful_trade'] = success
    
    def update_websocket_status(self, connected: bool, error_count: int = 0):
        """Update WebSocket connection status."""
        self.websocket_connected = connected
        self.connection_errors = error_count
        
        if connected:
            self.last_price_update = time.time()
    
    def add_btc_price(self, price: float, timestamp: float):
        """Add BTC price point."""
        self.btc_price_history.append((timestamp, price))
        self.price_update_count += 1
        
        # Keep only last 1000 price points
        if len(self.btc_price_history) > 1000:
            self.btc_price_history = self.btc_price_history[-1000:]
    
    def is_new_hour(self) -> bool:
        """Check if we're in a new hour (binary option reset)."""
        current_hour = datetime.now().hour
        if hasattr(self, 'last_hour'):
            if current_hour != self.last_hour:
                self.last_hour = current_hour
                # Reset daily counters if new day
                if current_hour == 0:
                    self.spike_count_today = 0
                    self.trades_today = 0
                return True
        else:
            self.last_hour = current_hour
        return False
    
    def can_place_spike_buy(self) -> bool:
        """Check if spike buy is allowed based on risk limits."""
        # Check hourly limit
        current_time = time.time()
        hour_ago = current_time - 3600
        recent_spikes = [s for s in self.spike_history if s['timestamp'] > hour_ago and s['successful_trade']]
        
        if len(recent_spikes) >= RISK_MANAGEMENT['max_spike_buys_per_hour']:
            return False
        
        # Check daily limit
        if self.trades_today >= RISK_MANAGEMENT['max_daily_trades']:
            return False
        
        # Check cooldown
        if current_time - self.last_spike_time < SPIKE_DETECTION['spike_cooldown_seconds']:
            return False
        
        return True
    
    def get_performance_summary(self) -> dict:
        """Get comprehensive performance summary."""
        total_trades = self.successful_trades + self.failed_trades
        
        return {
            'trading_performance': {
                'total_trades': total_trades,
                'successful_trades': self.successful_trades,
                'failed_trades': self.failed_trades,
                'success_rate': self.successful_trades / total_trades if total_trades > 0 else 0,
                'total_profit_loss': self.total_profit_loss,
                'trades_today': self.trades_today
            },
            'spike_detection': {
                'total_spikes_detected': len(self.spike_history),
                'spikes_today': self.spike_count_today,
                'spike_accuracy': self.spike_detection_accuracy,
                'last_spike_ago_seconds': time.time() - self.last_spike_time if self.last_spike_time > 0 else None
            },
            'connection_status': {
                'websocket_connected': self.websocket_connected,
                'price_updates_received': self.price_update_count,
                'connection_errors': self.connection_errors,
                'last_price_update_ago': time.time() - self.last_price_update if self.last_price_update > 0 else None
            },
            'system_status': {
                'api_calls_made': self.api_calls_count,
                'last_update_time': self.last_update_time,
                'protected_sell_active': self.protected_sell_active
            }
        }
    
    def save_performance_stats(self):
        """Save performance statistics to file."""
        if not MONITORING['save_performance_stats']:
            return
        
        current_time = time.time()
        if current_time - self.last_performance_save < MONITORING['stats_save_interval']:
            return
        
        try:
            stats = self.get_performance_summary()
            stats['timestamp'] = datetime.now().isoformat()
            stats['unix_timestamp'] = current_time
            
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(MONITORING['stats_file_path']), exist_ok=True)
            
            # Save stats
            with open(MONITORING['stats_file_path'], 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.last_performance_save = current_time
            
        except Exception as e:
            print(f"❌ Error saving performance stats: {e}")

# Global enhanced state instance
state = SpikeState()

# === UTILITY FUNCTIONS ===

def create_log_directories():
    """Create necessary log directories."""
    if LOGGING_CONFIG['log_to_file']:
        os.makedirs(os.path.dirname(LOGGING_CONFIG['log_file_path']), exist_ok=True)
    
    if LOGGING_CONFIG['log_trades']:
        os.makedirs(os.path.dirname(LOGGING_CONFIG['trade_log_path']), exist_ok=True)
    
    if MONITORING['save_performance_stats']:
        os.makedirs(os.path.dirname(MONITORING['stats_file_path']), exist_ok=True)

def validate_configuration():
    """Validate configuration parameters."""
    errors = []
    
    # Validate spike detection parameters
    if SPIKE_DETECTION['lookback_minutes'] < 1:
        errors.append("Spike lookback_minutes must be >= 1")
    
    if SPIKE_DETECTION['min_spike_threshold'] <= 0:
        errors.append("Min spike threshold must be > 0")
    
    # Validate protected sell parameters
    if PROTECTED_SELL['min_ask_size'] < 100:
        errors.append("Min ASK size should be >= 100")
    
    # Validate risk management
    if RISK_MANAGEMENT['max_spike_buys_per_hour'] < 1:
        errors.append("Max spike buys per hour must be >= 1")
    
    # Validate trading parameters
    if MAX_TRADE_AMOUNT <= 0:
        errors.append("Max trade amount must be > 0")
    
    if MAX_TOKEN_VALUE_USD <= MAX_TRADE_AMOUNT:
        errors.append("Max token value must be > max trade amount")
    
    return errors

def get_config_summary():
    """Get configuration summary for display."""
    return {
        'spike_detection': SPIKE_DETECTION,
        'protected_sell': PROTECTED_SELL,
        'risk_management': RISK_MANAGEMENT,
        'trading_limits': {
            'max_trade_amount': MAX_TRADE_AMOUNT,
            'max_token_value': MAX_TOKEN_VALUE_USD,
            'mandatory_sell_price': MANDATORY_SELL_PRICE
        },
        'monitoring': {
            'logging_enabled': LOGGING_CONFIG['log_to_file'],
            'performance_tracking': MONITORING['save_performance_stats'],
            'alerts_enabled': ALERTS['enable_alerts']
        }
    }

# Initialize on import
create_log_directories()

# Validate configuration
config_errors = validate_configuration()
if config_errors:
    print("⚠️ Configuration Errors:")
    for error in config_errors:
        print(f"   ❌ {error}")
    print("Please fix configuration before starting the bot.")
else:
    print("✅ Configuration validated successfully")
