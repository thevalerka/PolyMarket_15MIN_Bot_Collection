#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime, timezone
from collections import defaultdict
import copy

class OptimalExitSimulation:
    def __init__(self, base_path="/home/ubuntu/013_2025_polymarket/"):
        self.base_path = base_path
        self.results_file = os.path.join(base_path, "bot004_blackScholes/optimal_exit_results_base.json")

        # Trading parameters
        self.position_size = 1.0  # $1 per trade
        self.trade_frequency = 60  # seconds between trades

        # Exit strategies to test
        self.exit_strategies = {
            "hold_to_expiry": {
                "description": "Hold all positions until expiration",
                "total_invested": 0,
                "total_pnl": 0,
                "trades": 0,
                "wins": 0,
                "positions": []
            },
            "sell_profit_20pct": {
                "description": "Sell when profit reaches 20%",
                "total_invested": 0,
                "total_pnl": 0,
                "trades": 0,
                "wins": 0,
                "positions": []
            },
            "sell_profit_50pct": {
                "description": "Sell when profit reaches 50%",
                "total_invested": 0,
                "total_pnl": 0,
                "trades": 0,
                "wins": 0,
                "positions": []
            },
            "stop_loss_20pct": {
                "description": "Stop loss at -20%, otherwise hold",
                "total_invested": 0,
                "total_pnl": 0,
                "trades": 0,
                "wins": 0,
                "positions": []
            },
            "discrepancy_reversion": {
                "description": "Sell when discrepancy reverses by 50%",
                "total_invested": 0,
                "total_pnl": 0,
                "trades": 0,
                "wins": 0,
                "positions": []
            },
            "smart_exit": {
                "description": "Take profit at 30% OR stop loss at -15%",
                "total_invested": 0,
                "total_pnl": 0,
                "trades": 0,
                "wins": 0,
                "positions": []
            },
            "always_trade_both": {
                "description": "Buy both CALL and PUT every minute ($0.50 each)",
                "total_invested": 0,
                "total_pnl": 0,
                "trades": 0,
                "wins": 0,
                "positions": []
            }
        }

        # Entry strategies
        self.entry_strategies = ["follow_market", "fade_market", "high_discrepancy"]

        self.load_existing_data()

    def load_existing_data(self):
        """Load existing simulation data"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    if 'exit_strategies' in data:
                        for strategy_name, strategy_data in data['exit_strategies'].items():
                            if strategy_name in self.exit_strategies:
                                self.exit_strategies[strategy_name].update(strategy_data)
            except Exception as e:
                print(f"Error loading existing data: {e}")

    def save_results(self):
        """Save simulation results"""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        results = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "simulation_summary": self.get_performance_summary(),
            "exit_strategies": self.exit_strategies,
            "optimal_exits_analysis": self.analyze_optimal_exits(),
            "trade_frequency_minutes": self.trade_frequency / 60
        }

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def load_market_data(self):
        """Load current market data"""
        try:
            prob_file = os.path.join(self.base_path, "option_probabilities.json")
            call_file = os.path.join(self.base_path, "CALL.json")
            put_file = os.path.join(self.base_path, "PUT.json")

            with open(prob_file, 'r') as f:
                prob_data = json.load(f)
            with open(call_file, 'r') as f:
                call_data = json.load(f)
            with open(put_file, 'r') as f:
                put_data = json.load(f)

            return {
                "probabilities": prob_data,
                "call_market": call_data,
                "put_market": put_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            print(f"Error loading market data: {e}")
            return None

    def calculate_discrepancies(self, market_data):
        """Calculate market vs BS discrepancies"""
        prob_data = market_data["probabilities"]
        call_data = market_data["call_market"]
        put_data = market_data["put_market"]

        # Extract data
        call_prob_bs = prob_data["call_probability"]
        put_prob_bs = prob_data["put_probability"]

        call_bid = call_data["best_bid"]["price"]
        call_ask = call_data["best_ask"]["price"]
        put_bid = put_data["best_bid"]["price"]
        put_ask = put_data["best_ask"]["price"]

        call_mid = (call_bid + call_ask) / 2
        put_mid = (put_bid + put_ask) / 2

        return {
            "call_prob_bs": call_prob_bs,
            "put_prob_bs": put_prob_bs,
            "call_bid": call_bid,
            "call_ask": call_ask,
            "call_mid": call_mid,
            "put_bid": put_bid,
            "put_ask": put_ask,
            "put_mid": put_mid,
            "call_discrepancy": call_mid - call_prob_bs,
            "put_discrepancy": put_mid - put_prob_bs,
            "call_discrepancy_pct": ((call_mid - call_prob_bs) / call_prob_bs) * 100 if call_prob_bs > 0 else 0,
            "put_discrepancy_pct": ((put_mid - put_prob_bs) / put_prob_bs) * 100 if put_prob_bs > 0 else 0,
            "total_discrepancy": abs(call_mid - call_prob_bs) + abs(put_mid - put_prob_bs),
            "current_price": prob_data.get("current_price"),
            "strike_price": prob_data.get("strike_price"),
            "time_to_expiry": prob_data.get("time_to_expiry_hours", 1)
        }

    def determine_entry_signal(self, discrepancies):
        """Determine what to buy based on discrepancies"""
        call_disc = discrepancies["call_discrepancy"]
        put_disc = discrepancies["put_discrepancy"]
        total_disc = discrepancies["total_discrepancy"]

        # More aggressive entry criteria - trade more frequently
        min_threshold = 0.02  # Reduced from 0.10 to 0.02 (2%)

        # Always try to find a trade opportunity
        if total_disc >= min_threshold:
            # Follow market strategy: buy what market favors more than BS
            if call_disc > put_disc and call_disc > 0.01:
                return {"option": "CALL", "strategy": "follow_market", "entry_discrepancy": call_disc}
            elif put_disc > call_disc and put_disc > 0.01:
                return {"option": "PUT", "strategy": "follow_market", "entry_discrepancy": put_disc}

        # If discrepancy is small, use fade strategy
        if abs(call_disc) > abs(put_disc):
            if call_disc < 0:  # Market undervalues CALL vs BS
                return {"option": "CALL", "strategy": "fade_market", "entry_discrepancy": call_disc}
            else:  # Market overvalues CALL vs BS
                return {"option": "PUT", "strategy": "contrarian", "entry_discrepancy": put_disc}
        else:
            if put_disc < 0:  # Market undervalues PUT vs BS
                return {"option": "PUT", "strategy": "fade_market", "entry_discrepancy": put_disc}
            else:  # Market overvalues PUT vs BS
                return {"option": "CALL", "strategy": "contrarian", "entry_discrepancy": call_disc}

        # Fallback: if no clear signal, buy the option with higher BS probability
        if discrepancies["call_prob_bs"] > discrepancies["put_prob_bs"]:
            return {"option": "CALL", "strategy": "bs_follow", "entry_discrepancy": call_disc}
        else:
            return {"option": "PUT", "strategy": "bs_follow", "entry_discrepancy": put_disc}

    def calculate_unrealized_pnl(self, position, current_discrepancies):
        """Calculate current unrealized P&L for a position"""
        option_type = position["option_type"]
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        cost = position["cost"]  # Use actual cost instead of fixed position size

        if option_type == "CALL":
            current_bid = current_discrepancies["call_bid"]  # What we can sell for
        else:
            current_bid = current_discrepancies["put_bid"]

        # Current value if we sell at bid
        current_value = current_bid * quantity
        # P&L = current value - what we paid
        unrealized_pnl = current_value - cost
        unrealized_pnl_pct = (unrealized_pnl / cost) * 100

        return {
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "current_value": current_value,
            "current_exit_price": current_bid
        }

    def check_exit_conditions(self, position, current_discrepancies, strategy_name):
        """Check if position should be exited based on strategy"""
        pnl_data = self.calculate_unrealized_pnl(position, current_discrepancies)
        pnl_pct = pnl_data["unrealized_pnl_pct"]

        current_discrepancy = (current_discrepancies["call_discrepancy"]
                             if position["option_type"] == "CALL"
                             else current_discrepancies["put_discrepancy"])

        entry_discrepancy = position["entry_discrepancy"]

        # Check various exit conditions
        if strategy_name == "hold_to_expiry":
            # Only exit on expiration
            return current_discrepancies["time_to_expiry"] <= 0.0167  # 1 minute

        elif strategy_name == "always_trade_both":
            # For always trade both strategy, use smart exit logic
            return pnl_pct >= 25 or pnl_pct <= -20 or current_discrepancies["time_to_expiry"] <= 0.0167

        elif strategy_name == "sell_profit_20pct":
            return pnl_pct >= 20

        elif strategy_name == "sell_profit_50pct":
            return pnl_pct >= 50

        elif strategy_name == "stop_loss_20pct":
            return pnl_pct <= -20

        elif strategy_name == "discrepancy_reversion":
            # Exit if discrepancy has reversed by 50%
            discrepancy_change_pct = ((current_discrepancy - entry_discrepancy) / entry_discrepancy) * 100 if entry_discrepancy != 0 else 0
            return abs(discrepancy_change_pct) >= 50

        elif strategy_name == "smart_exit":
            # Take profit at 30% OR stop loss at -15%
            return pnl_pct >= 30 or pnl_pct <= -15

        return False

    def execute_exit(self, position, current_discrepancies, exit_reason):
        """Execute exit trade"""
        pnl_data = self.calculate_unrealized_pnl(position, current_discrepancies)

        position["exit_timestamp"] = datetime.now(timezone.utc).isoformat()
        position["exit_price"] = pnl_data["current_exit_price"]
        position["exit_value"] = pnl_data["current_value"]
        position["realized_pnl"] = pnl_data["unrealized_pnl"]
        position["realized_pnl_pct"] = pnl_data["unrealized_pnl_pct"]
        position["exit_reason"] = exit_reason
        position["exit_discrepancy"] = (current_discrepancies["call_discrepancy"]
                                      if position["option_type"] == "CALL"
                                      else current_discrepancies["put_discrepancy"])
        position["closed"] = True

        return position["realized_pnl"]

    def check_expiration(self, current_discrepancies):
        """Check if options have expired and resolve positions"""
        if current_discrepancies["time_to_expiry"] <= 0.0167:  # Less than 1 minute
            current_price = current_discrepancies["current_price"]
            strike_price = current_discrepancies["strike_price"]

            if current_price and strike_price:
                if current_price >= strike_price:
                    return "CALL_WON"
                else:
                    return "PUT_WON"
        return None

    def resolve_expired_positions(self, resolution):
        """Resolve all positions at expiration"""
        for strategy_name, strategy in self.exit_strategies.items():
            for position in strategy["positions"]:
                if not position.get("closed", False):
                    # Calculate final payout
                    if resolution == "CALL_WON":
                        final_value = 1.0 if position["option_type"] == "CALL" else 0.0
                    else:  # PUT_WON
                        final_value = 1.0 if position["option_type"] == "PUT" else 0.0

                    cost = position["cost"]

                    position["exit_timestamp"] = datetime.now(timezone.utc).isoformat()
                    position["exit_price"] = final_value / position["quantity"] if position["quantity"] > 0 else 0
                    position["exit_value"] = final_value
                    position["realized_pnl"] = final_value - cost
                    position["realized_pnl_pct"] = ((final_value - cost) / cost) * 100
                    position["exit_reason"] = f"EXPIRATION_{resolution}"
                    position["closed"] = True

                    strategy["total_pnl"] += position["realized_pnl"]
                    if position["realized_pnl"] > 0:
                        strategy["wins"] += 1

    def update_positions(self, current_discrepancies):
        """Update all open positions and check exit conditions"""
        expiration = self.check_expiration(current_discrepancies)

        if expiration:
            print(f"🏁 OPTIONS EXPIRED: {expiration}")
            self.resolve_expired_positions(expiration)
            return

        # Update each strategy's positions
        for strategy_name, strategy in self.exit_strategies.items():
            positions_to_close = []

            for i, position in enumerate(strategy["positions"]):
                if position.get("closed", False):
                    continue

                # Update unrealized P&L
                pnl_data = self.calculate_unrealized_pnl(position, current_discrepancies)
                position["current_unrealized_pnl"] = pnl_data["unrealized_pnl"]
                position["current_unrealized_pnl_pct"] = pnl_data["unrealized_pnl_pct"]

                # Check exit conditions
                if self.check_exit_conditions(position, current_discrepancies, strategy_name):
                    realized_pnl = self.execute_exit(position, current_discrepancies, f"STRATEGY_{strategy_name}")
                    strategy["total_pnl"] += realized_pnl
                    if realized_pnl > 0:
                        strategy["wins"] += 1

                    print(f"💰 EXITED {position['option_type']} | Strategy: {strategy_name} | P&L: ${realized_pnl:+.3f} ({position['realized_pnl_pct']:+.1f}%)")

    def execute_new_trades(self, current_discrepancies):
        """Execute new trades based on entry signals"""

        # Strategy 1: Always trade both (ensures activity every minute)
        # Buy both CALL and PUT with $0.50 each for this strategy
        call_price = current_discrepancies["call_ask"]
        put_price = current_discrepancies["put_ask"]

        # Create CALL position for always_trade_both strategy
        call_position = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "option_type": "CALL",
            "entry_strategy": "always_both",
            "entry_price": call_price,
            "quantity": 0.5 / call_price,  # $0.50 position
            "cost": 0.5,
            "entry_discrepancy": current_discrepancies["call_discrepancy"],
            "closed": False,
            "exit_strategy": "always_trade_both"
        }

        # Create PUT position for always_trade_both strategy
        put_position = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "option_type": "PUT",
            "entry_strategy": "always_both",
            "entry_price": put_price,
            "quantity": 0.5 / put_price,  # $0.50 position
            "cost": 0.5,
            "entry_discrepancy": current_discrepancies["put_discrepancy"],
            "closed": False,
            "exit_strategy": "always_trade_both"
        }

        # Add both positions to always_trade_both strategy
        self.exit_strategies["always_trade_both"]["positions"].extend([call_position, put_position])
        self.exit_strategies["always_trade_both"]["total_invested"] += 1.0  # $0.50 + $0.50
        self.exit_strategies["always_trade_both"]["trades"] += 2

        print(f"📈 ALWAYS TRADE: CALL @ ${call_price:.4f} + PUT @ ${put_price:.4f} ($0.50 each)")

        # Strategy 2: Selective trading based on signals
        entry_signal = self.determine_entry_signal(current_discrepancies)

        if entry_signal:
            option_type = entry_signal["option"]
            strategy = entry_signal["strategy"]
            entry_discrepancy = entry_signal["entry_discrepancy"]

            # Get entry price (buy at ask)
            if option_type == "CALL":
                entry_price = current_discrepancies["call_ask"]
            else:
                entry_price = current_discrepancies["put_ask"]

            quantity = self.position_size / entry_price

            # Create position for all other exit strategies (except always_trade_both)
            base_position = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "option_type": option_type,
                "entry_strategy": strategy,
                "entry_price": entry_price,
                "quantity": quantity,
                "cost": self.position_size,
                "entry_discrepancy": entry_discrepancy,
                "closed": False
            }

            # Add to all exit strategies except always_trade_both
            strategies_to_add = [name for name in self.exit_strategies.keys() if name != "always_trade_both"]

            for strategy_name in strategies_to_add:
                position = copy.deepcopy(base_position)
                position["exit_strategy"] = strategy_name

                self.exit_strategies[strategy_name]["positions"].append(position)
                self.exit_strategies[strategy_name]["total_invested"] += self.position_size
                self.exit_strategies[strategy_name]["trades"] += 1

            print(f"📈 SELECTIVE TRADE: {option_type} @ ${entry_price:.4f} | Strategy: {strategy} | Discrepancy: {entry_discrepancy:+.4f}")

        else:
            # If no selective signal, still trade based on BS probabilities to ensure activity
            if current_discrepancies["call_prob_bs"] > current_discrepancies["put_prob_bs"]:
                option_type = "CALL"
                entry_price = current_discrepancies["call_ask"]
                entry_discrepancy = current_discrepancies["call_discrepancy"]
            else:
                option_type = "PUT"
                entry_price = current_discrepancies["put_ask"]
                entry_discrepancy = current_discrepancies["put_discrepancy"]

            quantity = self.position_size / entry_price

            base_position = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "option_type": option_type,
                "entry_strategy": "fallback_bs",
                "entry_price": entry_price,
                "quantity": quantity,
                "cost": self.position_size,
                "entry_discrepancy": entry_discrepancy,
                "closed": False
            }

            # Add to all exit strategies except always_trade_both
            strategies_to_add = [name for name in self.exit_strategies.keys() if name != "always_trade_both"]

            for strategy_name in strategies_to_add:
                position = copy.deepcopy(base_position)
                position["exit_strategy"] = strategy_name

                self.exit_strategies[strategy_name]["positions"].append(position)
                self.exit_strategies[strategy_name]["total_invested"] += self.position_size
                self.exit_strategies[strategy_name]["trades"] += 1

            print(f"📈 FALLBACK TRADE: {option_type} @ ${entry_price:.4f} | Following BS probability | Discrepancy: {entry_discrepancy:+.4f}")

    def analyze_optimal_exits(self):
        """Analyze which exit conditions were most profitable"""
        analysis = {}

        for strategy_name, strategy in self.exit_strategies.items():
            closed_positions = [p for p in strategy["positions"] if p.get("closed", False)]

            if not closed_positions:
                continue

            # Calculate metrics
            total_trades = len(closed_positions)
            profitable_trades = len([p for p in closed_positions if p["realized_pnl"] > 0])
            avg_pnl = sum(p["realized_pnl"] for p in closed_positions) / total_trades
            avg_pnl_pct = sum(p["realized_pnl_pct"] for p in closed_positions) / total_trades

            # Exit reason analysis
            exit_reasons = {}
            for position in closed_positions:
                reason = position.get("exit_reason", "UNKNOWN")
                if reason not in exit_reasons:
                    exit_reasons[reason] = {"count": 0, "total_pnl": 0}
                exit_reasons[reason]["count"] += 1
                exit_reasons[reason]["total_pnl"] += position["realized_pnl"]

            analysis[strategy_name] = {
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "win_rate": (profitable_trades / total_trades) * 100,
                "avg_pnl": avg_pnl,
                "avg_pnl_pct": avg_pnl_pct,
                "exit_reasons": exit_reasons,
                "roi": (strategy["total_pnl"] / strategy["total_invested"]) * 100 if strategy["total_invested"] > 0 else 0
            }

        return analysis

    def get_performance_summary(self):
        """Get performance summary for all exit strategies"""
        summary = {}

        for strategy_name, strategy in self.exit_strategies.items():
            open_positions = len([p for p in strategy["positions"] if not p.get("closed", False)])
            closed_positions = len([p for p in strategy["positions"] if p.get("closed", False)])

            summary[strategy_name] = {
                "description": strategy["description"],
                "total_invested": strategy["total_invested"],
                "total_pnl": strategy["total_pnl"],
                "roi_percent": (strategy["total_pnl"] / strategy["total_invested"]) * 100 if strategy["total_invested"] > 0 else 0,
                "total_trades": strategy["trades"],
                "wins": strategy["wins"],
                "win_rate_percent": (strategy["wins"] / closed_positions) * 100 if closed_positions > 0 else 0,
                "open_positions": open_positions,
                "closed_positions": closed_positions
            }

        return summary

    def print_performance_report(self, cycle_count=None):
        """Print detailed performance report"""
        print("\n" + "="*80)
        print("OPTIMAL EXIT STRATEGY ANALYSIS")
        print("="*80)
        if cycle_count:
            print(f"Cycle: {cycle_count} | Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        else:
            print(f"Report Time: {datetime.now(timezone.utc).isoformat()}")

        summary = self.get_performance_summary()

        # Show active positions summary first
        total_open = sum(s["open_positions"] for s in summary.values())
        total_invested = sum(s["total_invested"] for s in summary.values())
        total_pnl = sum(s["total_pnl"] for s in summary.values())
        total_closed = sum(s["closed_positions"] for s in summary.values())

        print(f"📊 LIVE STATUS: {total_open} open positions | {total_closed} completed trades | ${total_invested:.0f} invested | P&L: ${total_pnl:+.2f}")

        # Sort strategies by ROI (only show those with trades)
        sorted_strategies = [(name, stats) for name, stats in summary.items() if stats["total_invested"] > 0]
        sorted_strategies.sort(key=lambda x: x[1]["roi_percent"], reverse=True)

        print(f"\n📈 EXIT STRATEGY PERFORMANCE (Sorted by ROI):")
        print("-" * 80)

        for strategy_name, stats in sorted_strategies:
            print(f"🎯 {strategy_name.upper()}: ROI {stats['roi_percent']:+.1f}% | "
                  f"${stats['total_pnl']:+.2f} | {stats['open_positions']} open | "
                  f"{stats['win_rate_percent']:.0f}% wins ({stats['wins']}/{stats['closed_positions']})")

        # Show best strategy
        if sorted_strategies:
            best = sorted_strategies[0]
            print(f"\n🏆 CURRENT BEST: {best[0]} ({best[1]['roi_percent']:+.1f}% ROI)")

    def run_simulation_cycle(self):
        """Run one cycle of the simulation"""
        market_data = self.load_market_data()
        if not market_data:
            return False

        current_discrepancies = self.calculate_discrepancies(market_data)

        # Update existing positions first
        self.update_positions(current_discrepancies)

        # Execute new trades
        self.execute_new_trades(current_discrepancies)

        # Save results
        self.save_results()

        return True

def main():
    """Main simulation loop"""
    print("🚀 Starting OPTIMAL EXIT STRATEGY SIMULATION")
    print("💰 Position Size: $1 per selective trade, $0.50 each for always-trade")
    print("⏰ Trading Frequency: GUARANTEED every 1 minute")
    print("📊 Performance Reports: Every 1 minute")
    print("🎯 Testing 7 different exit strategies simultaneously")
    print("🔄 Always trades BOTH options ($0.50 each) + selective trades ($1)")
    print("📊 Tracking optimal exit conditions for maximum profitability")
    print("Press Ctrl+C to stop and view final report")

    simulation = OptimalExitSimulation()
    cycle_count = 0

    try:
        while True:
            cycle_count += 1

            if simulation.run_simulation_cycle():
                simulation.print_performance_report(cycle_count)
                print(f"⏳ Next cycle in 60 seconds...")

                time.sleep(60)  # Trade every minute
            else:
                print("⚠️  Failed to load market data, retrying in 30 seconds...")
                time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n🛑 SIMULATION STOPPED BY USER")

        # Show detailed final report
        print("\n" + "="*80)
        print("FINAL DETAILED PERFORMANCE REPORT")
        print("="*80)

        summary = simulation.get_performance_summary()
        sorted_strategies = sorted(summary.items(), key=lambda x: x[1]["roi_percent"], reverse=True)

        for strategy_name, stats in sorted_strategies:
            if stats["total_invested"] == 0:
                continue

            print(f"\n🎯 {strategy_name.upper()}")
            print(f"   Strategy: {stats['description']}")
            print(f"   Total Invested: ${stats['total_invested']:.2f}")
            print(f"   Total P&L: ${stats['total_pnl']:+.2f}")
            print(f"   ROI: {stats['roi_percent']:+.2f}%")
            print(f"   Closed Trades: {stats['closed_positions']} | Wins: {stats['wins']} | Win Rate: {stats['win_rate_percent']:.1f}%")
            print(f"   Open Positions: {stats['open_positions']}")

        # Show detailed exit analysis
        analysis = simulation.analyze_optimal_exits()
        print(f"\n📈 DETAILED EXIT ANALYSIS:")
        for strategy, data in analysis.items():
            if data['total_trades'] == 0:
                continue
            print(f"\n{strategy}: ROI {data['roi']:+.1f}%, Win Rate {data['win_rate']:.1f}%")
            for reason, reason_data in data['exit_reasons'].items():
                avg_pnl = reason_data['total_pnl'] / reason_data['count']
                print(f"  {reason}: {reason_data['count']} trades, avg P&L: ${avg_pnl:+.3f}")

        print(f"\n📁 Results saved to: {simulation.results_file}")

if __name__ == "__main__":
    main()
