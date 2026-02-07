#!/usr/bin/env python3
"""
ML Trading Launcher

Launches ML arbitrage trading with your existing configuration setup.
Uses the same environment variables and API setup as your spike trading bot.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append('/home/ubuntu/013_2025_polymarket')

# Load your existing environment configuration
load_dotenv('/home/ubuntu/013_2025_polymarket/keys/keys_ovh13.env')

# FIXED: Import from your EXISTING files (no renames!)
from ml_trading_executor import MLTradingExecutor, MLTradingConfig

def create_ml_trading_config():
    """Create ML trading configuration using your existing environment setup"""

    config = MLTradingConfig(
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

        # Trading parameters - SAFE VALUES
        max_position_size_usd=5.0,        # Small positions
        max_total_exposure_usd=7.0,      # Limited exposure
        min_confidence_threshold=0.75,    # High confidence requirement
        min_profit_threshold=0.025,       # 2.5 cent minimum profit
        max_trades_per_hour=3,            # Conservative rate

        # Risk management
        stop_loss_threshold=-0.025,       # 2.5 cent stop loss
        take_profit_threshold=0.075,      # 7.5 cent take profit
        max_hold_time_minutes=15,         # 15 minute max hold

        # Order execution - USING CORRECT PARAMETER NAMES
        order_timeout_seconds=30,
        price_buffer_pct=0.002,           # 0.2% buffer (much safer than cents)
        min_order_size=1.0,
        max_price_impact=0.03,            # 3% max price impact
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

def main():
    """Main launcher function"""
    print("🚀 ML ARBITRAGE TRADING LAUNCHER")
    print("=" * 60)

    # Validate environment
    if not validate_environment():
        return 1

    # Check file accessibility
    if not check_file_accessibility():
        return 1

    # Create configuration
    try:
        config = create_ml_trading_config()
        print("✅ Configuration created successfully")
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return 1

    # Display configuration summary
    print("\n📊 TRADING CONFIGURATION:")
    print(f"├─ Max Position Size: ${config.max_position_size_usd}")
    print(f"├─ Max Total Exposure: ${config.max_total_exposure_usd}")
    print(f"├─ Min ML Confidence: {config.min_confidence_threshold:.0%}")
    print(f"├─ Min Profit Threshold: ${config.min_profit_threshold:.3f}")
    print(f"├─ Max Trades/Hour: {config.max_trades_per_hour}")
    print(f"├─ Stop Loss: ${config.stop_loss_threshold:.3f}")
    print(f"├─ Take Profit: ${config.take_profit_threshold:.3f}")
    print(f"├─ Max Hold Time: {config.max_hold_time_minutes}m")


    # Safety confirmation
    print("\n⚠️ IMPORTANT SAFETY NOTICE:")
    print("This system will place REAL trades with REAL money.")
    print("Make sure you understand the risks and have tested thoroughly.")
    print(f"Maximum loss exposure: ${config.max_total_exposure_usd}")

    #confirmation = input("\nDo you want to start ML trading? (yes/no): ").lower().strip()
    confirmation = 'yes'

    if confirmation != 'yes':
        print("🛑 Trading cancelled by user")
        return 0

    # Start trading
    try:
        print("\n🎯 Starting ML Arbitrage Trading...")
        print("💡 Press Ctrl+C to stop and see final performance report")
        print("=" * 60)

        executor = MLTradingExecutor(config)
        executor.run_trading_loop()

    except KeyboardInterrupt:
        print("\n✅ Trading stopped gracefully")
        return 0
    except Exception as e:
        print(f"\n❌ Trading error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
