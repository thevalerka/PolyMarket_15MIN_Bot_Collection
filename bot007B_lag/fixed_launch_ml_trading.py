#!/usr/bin/env python3
"""
FIXED ML Trading Launcher

Launches ML arbitrage trading with corrected configuration parameters.
Uses the improved ML trading executor with proper parameter names.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append('/home/ubuntu/013_2025_polymarket')

# Load your existing environment configuration
load_dotenv('/home/ubuntu/013_2025_polymarket/keys/keys_ovh13.env')

# FIXED: Import from the improved ML trading executor
from improved_ml_executor import ImprovedMLTradingExecutor, ImprovedMLTradingConfig

def create_fixed_ml_trading_config():
    """Create ML trading configuration with CORRECTED parameter names"""

    # FIXED: Use ImprovedMLTradingConfig with correct parameter names
    config = ImprovedMLTradingConfig(
        # API Configuration (from your existing setup)
        clob_api_url=os.getenv("CLOB_API_URL"),
        private_key=os.getenv("PK"),
        api_key=os.getenv("CLOB_API_KEY"),
        api_secret=os.getenv("CLOB_SECRET"),
        api_passphrase=os.getenv("CLOB_PASS_PHRASE"),
        chain_id=137,

        # File paths (your existing paths)
        btc_file='/home/ubuntu/013_2025_polymarket/btc_price.json',
        call_file='/home/ubuntu/013_2025_polymarket/CALL.json',
        put_file='/home/ubuntu/013_2025_polymarket/PUT.json',

        # FIXED: Conservative trading parameters with CORRECT names
        max_position_size_usd=5.0,        # Small positions for testing
        max_total_exposure_usd=15.0,      # Limited total exposure
        min_confidence_threshold=0.75,    # High ML confidence required
        min_profit_threshold=0.025,       # 2.5 cent minimum profit
        max_trades_per_hour=3,            # Very conservative rate limiting

        # FIXED: Risk management with correct parameter names
        stop_loss_threshold=-0.025,       # 2.5 cent stop loss
        take_profit_threshold=0.075,      # 7.5 cent take profit
        max_hold_time_minutes=15,         # 15 minute max hold

        # FIXED: Order execution with corrected parameter names
        order_timeout_seconds=30,
        price_buffer_pct=0.0005,           # FIXED: Changed from price_buffer_cents to price_buffer_pct
        min_order_size=1.0,
        max_price_impact=0.03,            # FIXED: Added missing parameter
    )

    return config

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = [
        "CLOB_API_URL",
        "PK",
        "CLOB_API_KEY",
        "CLOB_SECRET",
        "CLOB_PASS_PHRASE"
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   {var}")
        print("\nPlease check your .env file at:")
        print("   /home/ubuntu/013_2025_polymarket/keys/keys_ovh13.env")
        return False

    print("✅ Environment variables validated")
    return True

def check_file_accessibility():
    """Check if required JSON files are accessible"""
    files_to_check = [
        '/home/ubuntu/013_2025_polymarket/btc_price.json',
        '/home/ubuntu/013_2025_polymarket/CALL.json',
        '/home/ubuntu/013_2025_polymarket/PUT.json'
    ]

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False

        try:
            with open(file_path, 'r') as f:
                import json
                json.load(f)
            print(f"✅ {os.path.basename(file_path)}: OK")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return False

    return True

def check_ml_detector_availability():
    """Check if the improved ML detector is available"""
    try:
        # Try importing the fixed ML detector
        from fixed_ml_detector import FixedMLArbitrageDetector
        print("✅ Fixed ML Detector: Available")
        return True
    except ImportError:
        print("⚠️ Fixed ML Detector: Not found")
        print("   Will use improved ML executor with enhanced training logic")
        return True  # Continue anyway, improved executor has better training

def main():
    """Main launcher function with fixed configuration"""
    print("🚀 FIXED ML ARBITRAGE TRADING LAUNCHER")
    print("=" * 60)

    # Validate environment
    if not validate_environment():
        return 1

    # Check file accessibility
    if not check_file_accessibility():
        return 1

    # Check ML detector availability
    check_ml_detector_availability()

    # Create configuration with FIXED parameters
    try:
        config = create_fixed_ml_trading_config()
        print("✅ Configuration created successfully")
        print("🔧 Using IMPROVED parameters (fixes training loop issues)")
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        print("💡 Make sure you're using the improved ML executor")
        return 1

    # Display configuration summary
    print("\n📊 FIXED TRADING CONFIGURATION:")
    print(f"├─ Max Position Size: ${config.max_position_size_usd}")
    print(f"├─ Max Total Exposure: ${config.max_total_exposure_usd}")
    print(f"├─ Min ML Confidence: {config.min_confidence_threshold:.0%}")
    print(f"├─ Min Profit Threshold: ${config.min_profit_threshold:.3f}")
    print(f"├─ Max Trades/Hour: {config.max_trades_per_hour}")
    print(f"├─ Stop Loss: ${config.stop_loss_threshold:.3f}")
    print(f"├─ Take Profit: ${config.take_profit_threshold:.3f}")
    print(f"├─ Max Hold Time: {config.max_hold_time_minutes}m")
    print(f"├─ Price Buffer: {config.price_buffer_pct:.1%} (FIXED: percentage-based)")
    print(f"└─ Max Price Impact: {config.max_price_impact:.1%}")

    # Enhanced safety notice
    print("\n⚠️ IMPORTANT SAFETY NOTICE:")
    print("This system will place REAL trades with REAL money.")
    print("FIXES APPLIED:")
    print("• No more huge position sizes from small prices")
    print("• No more negative target prices")
    print("• No more training loop issues")
    print("• Conservative position sizing and risk management")
    print(f"Maximum loss exposure: ${config.max_total_exposure_usd}")

    confirmation = input("\nDo you want to start FIXED ML trading? (yes/no): ").lower().strip()

    if confirmation != 'yes':
        print("🛑 Trading cancelled by user")
        return 0

    # Start improved trading
    try:
        print("\n🎯 Starting FIXED ML Arbitrage Trading...")
        print("🔧 Using improved configuration (resolves all known issues)")
        print("💡 Press Ctrl+C to stop and see final performance report")
        print("=" * 60)

        # FIXED: Use ImprovedMLTradingExecutor
        executor = ImprovedMLTradingExecutor(config)
        executor.run_trading_loop()

    except KeyboardInterrupt:
        print("\n✅ Trading stopped gracefully")
        return 0
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Make sure the improved_ml_executor.py file is in the same directory")
        print("💡 Or update the import path at the top of this script")
        return 1
    except Exception as e:
        print(f"\n❌ Trading error: {e}")
        return 1

def create_compatibility_launcher():
    """Create a version that works with your existing files"""
    print("\n🔄 ALTERNATIVE: Creating compatibility launcher...")

    try:
        # Try to use your original ML trading executor but with fixed parameters
        from ml_trading_executor import MLTradingExecutor, MLTradingConfig

        # Create config with only parameters that exist in original
        compat_config = MLTradingConfig(
            clob_api_url=os.getenv("CLOB_API_URL"),
            private_key=os.getenv("PK"),
            api_key=os.getenv("CLOB_API_KEY"),
            api_secret=os.getenv("CLOB_SECRET"),
            api_passphrase=os.getenv("CLOB_PASS_PHRASE"),

            # Use original parameter names but safer values
            max_position_size_usd=5.0,       # Much smaller
            max_total_exposure_usd=15.0,     # Much smaller
            min_confidence_threshold=0.80,   # Higher threshold
            min_profit_threshold=0.030,      # Higher threshold
            max_trades_per_hour=2,           # Much lower

            # Risk management
            stop_loss_threshold=-0.020,
            take_profit_threshold=0.060,
            max_hold_time_minutes=10,

            # Order execution - use original parameter names
            price_buffer_cents=0.0005,       # Very small buffer to avoid negative prices
            min_order_size=1.0,
        )

        print("✅ Compatibility configuration created")
        print("⚠️ Using original executor with SAFER parameters")
        print("💡 This should avoid the negative price and large position issues")

        return compat_config, MLTradingExecutor

    except Exception as e:
        print(f"❌ Compatibility launcher failed: {e}")
        return None, None

if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as e:
        print(f"\n❌ Launch failed: {e}")
        print("\n🔄 Trying compatibility mode...")

        # Try compatibility mode
        compat_config, compat_executor_class = create_compatibility_launcher()

        if compat_config and compat_executor_class:
            print("Starting in compatibility mode...")
            try:
                executor = compat_executor_class(compat_config)
                executor.run_trading_loop()
                exit_code = 0
            except Exception as compat_error:
                print(f"❌ Compatibility mode also failed: {compat_error}")
                exit_code = 1
        else:
            exit_code = 1

    sys.exit(exit_code)
