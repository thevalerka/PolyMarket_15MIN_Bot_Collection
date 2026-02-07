#!/usr/bin/env python3
"""
Advanced Market Maker Bot
=========================

Production-ready bot that integrates the MM engine with live data feeds
and generates actionable quote recommendations.

Usage:
    python3 mm_bot.py                    # Run with default settings
    python3 mm_bot.py --dry-run          # Simulation mode
    python3 mm_bot.py --aggressive       # Tighter spreads, more quoting
    python3 mm_bot.py --conservative     # Wider spreads, more pulling
"""

import json
import time
import os
import sys
import argparse
from datetime import datetime
from typing import Optional
import signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mm_engine import (
    AdvancedMarketMaker,
    MMConfig,
    MarketState,
    create_market_state_from_files,
    QuoteAction,
    TradePeriodPhase,
    MarketRegime,
    OrderFlowToxicity,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default file paths - adjust these for your setup
DEFAULT_PATHS = {
    'bybit': '/home/ubuntu/013_2025_polymarket/bybit_btc_price.json',
    'chainlink': '/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json',
    'coinbase': '/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json',
    'call': '/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json',
    'put': '/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json',
}

# Output path for quote recommendations
QUOTE_OUTPUT_PATH = '/home/ubuntu/013_2025_polymarket/bot027_fast/mm_quotes.json'
STATUS_OUTPUT_PATH = '/home/ubuntu/013_2025_polymarket/bot027_fast/mm_status.json'


# =============================================================================
# BOT CLASS
# =============================================================================

class MMBot:
    """
    Market Maker Bot that runs continuously and outputs quote recommendations.
    """

    def __init__(self, config: MMConfig, paths: dict, dry_run: bool = False):
        self.config = config
        self.paths = paths
        self.dry_run = dry_run
        self.mm = AdvancedMarketMaker(config)

        self.running = False
        self.iteration = 0
        self.last_period_start: Optional[int] = None

        # Performance tracking
        self.total_quotes = 0
        self.total_pulls = 0
        self.start_time = time.time()

    def get_current_period_start(self) -> int:
        """
        Calculate the start timestamp of the current 15-minute period.
        Polymarket periods start on 15-minute boundaries.
        """
        now_ms = int(time.time() * 1000)
        period_ms = 15 * 60 * 1000  # 15 minutes in milliseconds

        # Floor to nearest 15-minute boundary
        period_start = (now_ms // period_ms) * period_ms
        return period_start

    def load_market_state(self) -> Optional[MarketState]:
        """Load current market state from files."""
        try:
            period_start = self.get_current_period_start()

            state = create_market_state_from_files(
                bybit_path=self.paths['bybit'],
                chainlink_path=self.paths['chainlink'],
                coinbase_path=self.paths['coinbase'],
                call_path=self.paths['call'],
                put_path=self.paths['put'],
                period_start_timestamp=period_start,
            )

            return state

        except Exception as e:
            print(f"❌ Error loading market state: {e}")
            return None

    def output_quotes(self, quotes: dict, state: MarketState):
        """Output quote recommendations to file."""
        output = {
            'timestamp': int(time.time() * 1000),
            'timestamp_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'time_to_expiry_seconds': state.time_to_expiry_seconds,
            'phase': state.phase.value,
            'regime': state.regime.value,
            'quotes': {
                'CALL': [
                    {
                        'side': q.side,
                        'price': q.price,
                        'size': q.size,
                        'layer': q.layer,
                    }
                    for q in quotes.get('CALL', [])
                ],
                'PUT': [
                    {
                        'side': q.side,
                        'price': q.price,
                        'size': q.size,
                        'layer': q.layer,
                    }
                    for q in quotes.get('PUT', [])
                ],
            }
        }

        if not self.dry_run:
            try:
                with open(QUOTE_OUTPUT_PATH, 'w') as f:
                    json.dump(output, f, indent=2)
            except Exception as e:
                print(f"❌ Error writing quotes: {e}")

        return output

    def output_status(self, state: MarketState):
        """Output MM status to file."""
        status = self.mm.get_status_report(state)
        status['iteration'] = self.iteration
        status['uptime_seconds'] = time.time() - self.start_time
        status['dry_run'] = self.dry_run

        if not self.dry_run:
            try:
                with open(STATUS_OUTPUT_PATH, 'w') as f:
                    json.dump(status, f, indent=2)
            except:
                pass

        return status

    def print_dashboard(self, state: MarketState, quotes: dict, status: dict):
        """Print live dashboard to terminal."""
        os.system('clear' if os.name == 'posix' else 'cls')

        mode_str = "🔵 DRY RUN" if self.dry_run else "🟢 LIVE"

        print("="*80)
        print(f"   ADVANCED MARKET MAKER - {mode_str}")
        print("="*80)
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Iteration: {self.iteration}")
        print(f"   Uptime: {time.time() - self.start_time:.0f}s | Quotes: {self.total_quotes} | Pulls: {self.total_pulls}")
        print("="*80)

        # Market State
        print(f"\n{'─'*30} MARKET STATE {'─'*30}")

        phase_colors = {
            TradePeriodPhase.EARLY: "🟢",
            TradePeriodPhase.MID: "🟡",
            TradePeriodPhase.LATE: "🟠",
            TradePeriodPhase.FINAL: "🔴",
        }

        regime_colors = {
            MarketRegime.LOW_VOLATILITY: "🔵",
            MarketRegime.MEAN_REVERTING: "🟢",
            MarketRegime.TRENDING_UP: "📈",
            MarketRegime.TRENDING_DOWN: "📉",
            MarketRegime.HIGH_VOLATILITY: "🔴",
            MarketRegime.UNKNOWN: "⚪",
        }

        toxicity_colors = {
            OrderFlowToxicity.LOW: "🟢",
            OrderFlowToxicity.MEDIUM: "🟡",
            OrderFlowToxicity.HIGH: "🟠",
            OrderFlowToxicity.EXTREME: "🔴",
        }

        phase_icon = phase_colors.get(state.phase, "⚪")
        regime_icon = regime_colors.get(state.regime, "⚪")
        toxicity_icon = toxicity_colors.get(state.toxicity, "⚪")

        print(f"   Phase: {phase_icon} {state.phase.value.upper()} ({state.time_to_expiry_seconds:.0f}s to expiry)")
        print(f"   Regime: {regime_icon} {state.regime.value.upper()}")
        print(f"   Toxicity: {toxicity_icon} {state.toxicity.value.upper()}")
        print(f"   Predicted Direction: {status.get('predicted_direction', 'N/A')}")
        print(f"   Vol Spike: {'⚠️ YES' if status.get('vol_spike') else '✅ No'}")

        # Prices
        print(f"\n{'─'*30} PRICE SOURCES {'─'*30}")
        print(f"   {'Source':<12} {'Price':>14} {'Status'}")
        print(f"   {'─'*50}")

        if state.bybit:
            dir_icon = "📈" if state.bybit.direction == 'UP' else "📉" if state.bybit.direction == 'DOWN' else "➡️"
            print(f"   {'Bybit':<12} ${state.bybit.price:>13,.2f} {dir_icon}")
        if state.chainlink:
            print(f"   {'Chainlink':<12} ${state.chainlink.price:>13,.2f}")
        if state.coinbase:
            print(f"   {'Coinbase':<12} ${state.coinbase.price:>13,.2f}")

        if state.strike_price:
            diff = state.chainlink.price - state.strike_price if state.chainlink else 0
            diff_icon = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            print(f"\n   Strike: ${state.strike_price:,.2f} | Current: {diff_icon} ${diff:+,.2f}")

        # Options
        print(f"\n{'─'*30} OPTIONS {'─'*30}")
        print(f"   {'Type':<8} {'Bid':>10} {'Ask':>10} {'Mid':>10} {'Spread':>10}")
        print(f"   {'─'*50}")

        if state.call:
            print(f"   {'CALL':<8} {state.call.best_bid:>10.4f} {state.call.best_ask:>10.4f} "
                  f"{state.call.mid:>10.4f} {state.call.spread:>10.4f}")
        if state.put:
            print(f"   {'PUT':<8} {state.put.best_bid:>10.4f} {state.put.best_ask:>10.4f} "
                  f"{state.put.mid:>10.4f} {state.put.spread:>10.4f}")

        # Our Quotes
        print(f"\n{'─'*30} OUR QUOTES {'─'*30}")

        call_quotes = quotes.get('CALL', [])
        put_quotes = quotes.get('PUT', [])

        if not call_quotes and not put_quotes:
            print("   ⚠️  ALL QUOTES PULLED")
        else:
            print(f"   {'Asset':<8} {'Side':<6} {'Price':>10} {'Size':>10} {'Layer':>8}")
            print(f"   {'─'*50}")

            for q in call_quotes[:4]:  # Show top 4 layers
                print(f"   {'CALL':<8} {q.side:<6} {q.price:>10.4f} {q.size:>10.0f} {q.layer:>8}")

            for q in put_quotes[:4]:
                print(f"   {'PUT':<8} {q.side:<6} {q.price:>10.4f} {q.size:>10.0f} {q.layer:>8}")

        # Leadership
        print(f"\n{'─'*30} LEADERSHIP SCORES {'─'*30}")
        scores = status.get('leadership_scores', {})
        total = sum(scores.values())

        if total > 0:
            for source, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                pct = (score / total) * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"   {source.upper():<12} {bar} {pct:>5.1f}%")
        else:
            print("   Collecting data...")

        # Hedge Recommendations
        hedge = self.mm.evaluate_hedge(state)
        if hedge:
            print(f"\n{'─'*30} HEDGE RECOMMENDATIONS {'─'*30}")
            for rec in hedge:
                print(f"   💡 {rec['action'].upper()}: {rec['side']} {rec['size']:.0f} {rec['asset']} @ {rec['price']:.4f}")
                print(f"      Reason: {rec['reason']}")

        print("\n" + "="*80)
        print("   Press Ctrl+C to stop")
        print("="*80)

    def run_iteration(self) -> bool:
        """Run a single iteration of the MM loop."""
        # Load market state
        state = self.load_market_state()
        if not state:
            return False

        # Check for new period
        period_start = self.get_current_period_start()
        if period_start != self.last_period_start:
            print(f"\n🔄 New period started: {datetime.fromtimestamp(period_start/1000)}")
            self.last_period_start = period_start

        # Generate quotes
        quotes = self.mm.generate_all_quotes(state)

        # Track stats
        quote_count = len(quotes.get('CALL', [])) + len(quotes.get('PUT', []))
        self.total_quotes += quote_count
        if quote_count == 0:
            self.total_pulls += 1

        # Output
        self.output_quotes(quotes, state)
        status = self.output_status(state)

        # Display
        self.print_dashboard(state, quotes, status)

        self.iteration += 1
        return True

    def run(self, interval_ms: int = 100):
        """Run the bot continuously."""
        self.running = True

        # Setup signal handler
        def signal_handler(sig, frame):
            print("\n\n⚠️ Stopping bot...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        print("🚀 Starting Advanced Market Maker Bot...")
        print(f"   Interval: {interval_ms}ms")
        print(f"   Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print("")

        while self.running:
            try:
                self.run_iteration()
                time.sleep(interval_ms / 1000)
            except Exception as e:
                print(f"❌ Error in iteration: {e}")
                time.sleep(1)

        print("\n✅ Bot stopped cleanly")
        print(f"   Total iterations: {self.iteration}")
        print(f"   Total quotes: {self.total_quotes}")
        print(f"   Total pulls: {self.total_pulls}")


# =============================================================================
# CONFIG PRESETS
# =============================================================================

def get_aggressive_config() -> MMConfig:
    """Configuration for aggressive market making (tighter spreads)."""
    config = MMConfig()
    config.base_spread = 0.015
    config.min_spread = 0.008
    config.inventory_skew_factor = 0.0005
    config.momentum_skew_factor = 0.003
    config.vol_spike_threshold = 0.003  # Higher threshold before pulling
    config.tier_sizes = [100, 200, 400, 800]
    return config


def get_conservative_config() -> MMConfig:
    """Configuration for conservative market making (wider spreads)."""
    config = MMConfig()
    config.base_spread = 0.03
    config.min_spread = 0.02
    config.max_spread = 0.15
    config.inventory_skew_factor = 0.002
    config.momentum_skew_factor = 0.01
    config.vol_spike_threshold = 0.001  # Lower threshold, pull earlier
    config.gamma_danger_threshold = 180  # Pull earlier near expiry
    config.tier_sizes = [25, 50, 100, 200]
    return config


def get_scalping_config() -> MMConfig:
    """Configuration for high-frequency scalping."""
    config = MMConfig()
    config.base_spread = 0.01
    config.min_spread = 0.005
    config.max_spread = 0.05
    config.inventory_skew_factor = 0.001
    config.momentum_skew_factor = 0.002
    config.max_position_size = 200
    config.tier_sizes = [20, 40, 60, 80]
    config.tier_spreads = [0.005, 0.01, 0.015, 0.02]
    return config


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Advanced Market Maker Bot')
    parser.add_argument('--dry-run', action='store_true', help='Run without outputting files')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive config')
    parser.add_argument('--conservative', action='store_true', help='Use conservative config')
    parser.add_argument('--scalping', action='store_true', help='Use scalping config')
    parser.add_argument('--interval', type=int, default=100, help='Update interval in ms')

    args = parser.parse_args()

    # Select config
    if args.aggressive:
        config = get_aggressive_config()
        print("📊 Using AGGRESSIVE config")
    elif args.conservative:
        config = get_conservative_config()
        print("📊 Using CONSERVATIVE config")
    elif args.scalping:
        config = get_scalping_config()
        print("📊 Using SCALPING config")
    else:
        config = MMConfig()
        print("📊 Using DEFAULT config")

    # Verify paths exist
    print("\n📁 Checking data files...")
    all_exist = True
    for name, path in DEFAULT_PATHS.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")
        if not exists:
            all_exist = False

    if not all_exist and not args.dry_run:
        print("\n⚠️ Some files missing. Use --dry-run to test without live data.")
        # Continue anyway for testing

    # Create and run bot
    bot = MMBot(config, DEFAULT_PATHS, dry_run=args.dry_run)
    bot.run(interval_ms=args.interval)


if __name__ == '__main__':
    main()
