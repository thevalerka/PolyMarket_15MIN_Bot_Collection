# collar_config.py - Simplified Configuration for Collar Strategy Bot
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/ubuntu/013_2025_polymarket/keys/keys_ovh13.env')

# API Configuration
CLOB_API_URL = os.getenv("CLOB_API_URL")
PRIVATE_KEY = os.getenv("PK")
API_KEY = os.getenv("CLOB_API_KEY")
API_SECRET = os.getenv("CLOB_SECRET")
API_PASSPHRASE = os.getenv("CLOB_PASS_PHRASE")
CHAIN_ID = 137

# Function to read TOKEN_ID from PUT.json
def get_token_id():
    """Read TOKEN_ID from PUT.json file."""
    try:
        with open('/home/ubuntu/013_2025_polymarket/PUT.json', 'r') as f:
            data = json.load(f)
            return data.get('asset_id', '')
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading TOKEN_ID from PUT.json: {e}")
        # Fallback to original hardcoded value if file read fails
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

# Token Configuration - Read from PUT.json
TOKEN_ID = get_token_id()

# File Paths
BOOK_DATA_PATH = "/home/ubuntu/013_2025_polymarket/PUT.json"
BTC_PRICE_PATH = "/home/ubuntu/013_2025_polymarket/btc_price.json"

# Trading Parameters
MAX_TRADE_AMOUNT = 4.5  # USD per buy order
MAX_TOKEN_VALUE_USD = 5  # Maximum total USD value of tokens to hold
MIN_BALANCE_SELL = 100000  # Minimum balance for sells (0.1 tokens)
MIN_QUANTITY_FILTER = 200  # Minimum order size to consider

# Dynamic Spread Parameters
MIN_SPREAD = 0.02  # Minimum spread in USD
MAX_SPREAD = 0.20  # Maximum spread in USD
SPREAD_BUFFER = 0.01  # Additional buffer added to ideal spread
SPREAD_FORMULA_PATH = "/home/ubuntu/013_2025_polymarket/spread_formula.json"

# Mandatory Sell Feature
# When token reaches $0.99, it indicates the binary option market is effectively over
# The bot will execute immediate market sell to capture maximum value before expiry
MANDATORY_SELL_PRICE = 0.99  # Price level that triggers mandatory sell

# Risk Management
MAX_DELAY_MS = 2000  # Maximum acceptable data delay

# Global State Class (Enhanced for Dynamic Spreads)
class CollarState:
    def __init__(self):
        # Current market data
        self.current_analysis = None
        self.last_update_time = 0

        # API call tracking
        self.api_calls_count = 0

        # Balance tracking
        self.last_balance_check = 0.0
        self.last_known_balance = 0.0

        # Price history for spread calculation
        self.btc_price_history = []  # (timestamp, price)
        self.token_price_history = []  # (timestamp, price)
        self.market_spread_history = []  # (timestamp, spread, mid_price)

        # Spread formula tracking
        self.current_spread_formula = None
        self.formula_accuracy_history = []  # (timestamp, predicted_spread, actual_spread, accuracy)
        self.formula_last_updated = 0
        self.formula_validation_count = 0

        # Market condition detection
        self.manipulation_alerts = []
        self.formula_performance_score = 1.0  # 1.0 = perfect, 0.0 = terrible

    def add_price_data(self, btc_price: float, token_mid: float, market_spread: float):
        """Add new price data for analysis."""
        current_time = time.time()

        self.btc_price_history.append((current_time, btc_price))
        if token_mid is not None:
            self.token_price_history.append((current_time, token_mid))
        if market_spread is not None:
            self.market_spread_history.append((current_time, market_spread, token_mid))

        # Keep only last 2 hours of data
        cutoff_time = current_time - 7200
        self.btc_price_history = [(t, p) for t, p in self.btc_price_history if t > cutoff_time]
        self.token_price_history = [(t, p) for t, p in self.token_price_history if t > cutoff_time]
        self.market_spread_history = [(t, s, m) for t, s, m in self.market_spread_history if t > cutoff_time]

    def validate_formula_accuracy(self, predicted_spread: float, actual_spread: float):
        """Track formula accuracy over time."""
        current_time = time.time()

        if actual_spread > 0 and predicted_spread > 0:
            accuracy = 1.0 - abs(predicted_spread - actual_spread) / actual_spread
            accuracy = max(0.0, min(1.0, accuracy))  # Clamp to 0-1

            self.formula_accuracy_history.append((current_time, predicted_spread, actual_spread, accuracy))

            # Keep only last hour of accuracy data
            cutoff_time = current_time - 3600
            self.formula_accuracy_history = [(t, p, a, acc) for t, p, a, acc in self.formula_accuracy_history if t > cutoff_time]

            # Update performance score (rolling average of last 20 predictions)
            recent_accuracies = [acc for _, _, _, acc in self.formula_accuracy_history[-20:]]
            if recent_accuracies:
                self.formula_performance_score = sum(recent_accuracies) / len(recent_accuracies)

    def should_recalibrate_formula(self) -> bool:
        """Determine if formula needs recalibration."""
        # Recalibrate if performance drops below 70% or every 30 minutes
        return (self.formula_performance_score < 0.70 or
                time.time() - self.formula_last_updated > 1800)

    def is_new_hour(self) -> bool:
        """Check if we're in a new hour (binary option reset)."""
        import datetime
        current_hour = datetime.datetime.now().hour
        if hasattr(self, 'last_hour'):
            if current_hour != self.last_hour:
                self.last_hour = current_hour
                return True
        else:
            self.last_hour = current_hour
        return False

    def update_balance_tracking(self, balance_tokens: float):
        """Update balance tracking."""
        self.last_known_balance = balance_tokens
        self.last_balance_check = time.time()

# Global state instance
state = CollarState()
