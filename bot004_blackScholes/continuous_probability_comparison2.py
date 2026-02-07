#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime, timezone
from collections import defaultdict

def load_historical_tracking():
    """Load historical tracking data"""
    base_path = "/home/ubuntu/013_2025_polymarket/"
    tracking_file = os.path.join(base_path, "bot004_blackScholes/market_vs_bs_tracking.json")

    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                return json.load(f)
        except:
            pass

    # Initialize empty tracking data
    return {
        "total_predictions": 0,
        "market_wins": 0,
        "bs_wins": 0,
        "ties": 0,
        "current_session": {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "predictions": []
        },
        "historical_sessions": [],
        "accuracy_stats": {
            "market_accuracy": 0.0,
            "bs_accuracy": 0.0,
            "market_win_rate": 0.0
        }
    }

def save_historical_tracking(tracking_data):
    """Save historical tracking data"""
    base_path = "/home/ubuntu/013_2025_polymarket/"
    tracking_file = os.path.join(base_path, "bot004_blackScholes/market_vs_bs_tracking.json")

    os.makedirs(os.path.dirname(tracking_file), exist_ok=True)

    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)

def detect_option_resolution(current_price, strike_price, time_to_expiry):
    """Detect if option has been resolved based on price and time"""
    # If time to expiry is very small (< 1 minute) or zero, consider it expired
    if time_to_expiry <= 0.0167:  # 1/60 hour = 1 minute
        if current_price >= strike_price:
            return "CALL_WON"
        else:
            return "PUT_WON"
    return None

def update_prediction_tracking(tracking_data, results, resolution=None):
    """Update tracking with current prediction and optionally resolve previous predictions"""
    current_prediction = {
        "timestamp": results["analysis_timestamp"],
        "market_call_prob": results["market_prices"]["call"]["mid_price"],
        "bs_call_prob": results["theoretical_probabilities"]["call_probability"],
        "market_put_prob": results["market_prices"]["put"]["mid_price"],
        "bs_put_prob": results["theoretical_probabilities"]["put_probability"],
        "current_price": results.get("current_price"),
        "strike_price": results.get("strike_price"),
        "time_to_expiry": results.get("time_to_expiry_hours"),
        "resolved": False,
        "resolution": None,
        "market_correct": None,
        "bs_correct": None
    }

    # Add current prediction to session
    tracking_data["current_session"]["predictions"].append(current_prediction)

    # If we have a resolution, resolve the most recent unresolved prediction
    if resolution:
        # Find the most recent unresolved prediction
        for pred in reversed(tracking_data["current_session"]["predictions"]):
            if not pred["resolved"]:
                pred["resolved"] = True
                pred["resolution"] = resolution

                # Determine who was more accurate
                if resolution == "CALL_WON":
                    # CALL option won
                    pred["market_correct"] = pred["market_call_prob"] > pred["market_put_prob"]
                    pred["bs_correct"] = pred["bs_call_prob"] > pred["bs_put_prob"]
                else:  # PUT_WON
                    # PUT option won
                    pred["market_correct"] = pred["market_put_prob"] > pred["market_call_prob"]
                    pred["bs_correct"] = pred["bs_put_prob"] > pred["bs_call_prob"]

                # Update overall stats
                tracking_data["total_predictions"] += 1

                if pred["market_correct"] and pred["bs_correct"]:
                    tracking_data["ties"] += 1
                elif pred["market_correct"] and not pred["bs_correct"]:
                    tracking_data["market_wins"] += 1
                elif pred["bs_correct"] and not pred["market_correct"]:
                    tracking_data["bs_wins"] += 1

                # Update accuracy stats
                total_resolved = tracking_data["market_wins"] + tracking_data["bs_wins"] + tracking_data["ties"]
                if total_resolved > 0:
                    tracking_data["accuracy_stats"]["market_win_rate"] = tracking_data["market_wins"] / total_resolved
                    tracking_data["accuracy_stats"]["market_accuracy"] = (tracking_data["market_wins"] + tracking_data["ties"]) / total_resolved
                    tracking_data["accuracy_stats"]["bs_accuracy"] = (tracking_data["bs_wins"] + tracking_data["ties"]) / total_resolved

                break

    return tracking_data

def analyze_prediction_patterns(tracking_data):
    """Analyze patterns in prediction accuracy"""
    resolved_predictions = []
    for pred in tracking_data["current_session"]["predictions"]:
        if pred["resolved"]:
            resolved_predictions.append(pred)

    if len(resolved_predictions) < 2:
        return {}

    # Analyze patterns
    patterns = {
        "recent_accuracy": {},
        "confidence_analysis": {},
        "divergence_analysis": {}
    }

    # Recent 10 predictions accuracy
    recent_10 = resolved_predictions[-10:] if len(resolved_predictions) >= 10 else resolved_predictions
    market_recent_correct = sum(1 for p in recent_10 if p["market_correct"])
    bs_recent_correct = sum(1 for p in recent_10 if p["bs_correct"])

    patterns["recent_accuracy"] = {
        "sample_size": len(recent_10),
        "market_recent_rate": market_recent_correct / len(recent_10),
        "bs_recent_rate": bs_recent_correct / len(recent_10)
    }

    # Analyze by confidence levels
    high_divergence_preds = [p for p in resolved_predictions if abs(p["market_call_prob"] - p["bs_call_prob"]) > 0.15]
    if high_divergence_preds:
        market_high_div_correct = sum(1 for p in high_divergence_preds if p["market_correct"])
        patterns["divergence_analysis"] = {
            "high_divergence_count": len(high_divergence_preds),
            "market_accuracy_high_div": market_high_div_correct / len(high_divergence_preds),
            "threshold": 0.15
        }

    return patterns

def load_json_file(filepath):
    """Load JSON data from file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file - {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    """Load JSON data from file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file - {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_differences():
    """Calculate differences between theoretical probabilities and market prices"""

    # Define file paths
    base_path = "/home/ubuntu/013_2025_polymarket/"
    prob_file = os.path.join(base_path, "option_probabilities.json")
    call_file = os.path.join(base_path, "CALL.json")
    put_file = os.path.join(base_path, "PUT.json")
    output_file = os.path.join(base_path, "bot004_blackScholes/callput_diff.json")
    output_file_www = "/var/www/html/polymarket/callput_diff.json"


    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file_www), exist_ok=True)

    # Load data from files
    prob_data = load_json_file(prob_file)
    call_data = load_json_file(call_file)
    put_data = load_json_file(put_file)

    if not all([prob_data, call_data, put_data]):
        print("Error: Could not load all required data files")
        return None

    try:
        # Extract probabilities
        call_probability = prob_data["call_probability"]
        put_probability = prob_data["put_probability"]

        # Extract market prices
        call_best_bid = call_data["best_bid"]["price"]
        call_best_ask = call_data["best_ask"]["price"]
        put_best_bid = put_data["best_bid"]["price"]
        put_best_ask = put_data["best_ask"]["price"]

        # Calculate mid prices
        call_mid_price = (call_best_bid + call_best_ask) / 2
        put_mid_price = (put_best_bid + put_best_ask) / 2

        # Calculate differences for CALL option
        call_bid_diff_dollars = call_probability - call_best_bid
        call_ask_diff_dollars = call_probability - call_best_ask
        call_mid_diff_dollars = call_probability - call_mid_price

        call_bid_diff_percent = (call_bid_diff_dollars / call_best_bid) * 100 if call_best_bid != 0 else 0
        call_ask_diff_percent = (call_ask_diff_dollars / call_best_ask) * 100 if call_best_ask != 0 else 0
        call_mid_diff_percent = (call_mid_diff_dollars / call_mid_price) * 100 if call_mid_price != 0 else 0

        # Calculate differences for PUT option
        put_bid_diff_dollars = put_probability - put_best_bid
        put_ask_diff_dollars = put_probability - put_best_ask
        put_mid_diff_dollars = put_probability - put_mid_price

        put_bid_diff_percent = (put_bid_diff_dollars / put_best_bid) * 100 if put_best_bid != 0 else 0
        put_ask_diff_percent = (put_ask_diff_dollars / put_best_ask) * 100 if put_best_ask != 0 else 0
        put_mid_diff_percent = (put_mid_diff_dollars / put_mid_price) * 100 if put_mid_price != 0 else 0

        # Prepare results
        results = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_timestamps": {
                "probabilities": prob_data.get("timestamp", prob_data.get("datetime")),
                "call_market": call_data.get("timestamp", call_data.get("timestamp_readable")),
                "put_market": put_data.get("timestamp", put_data.get("timestamp_readable"))
            },
            "theoretical_probabilities": {
                "call_probability": round(call_probability, 8),
                "put_probability": round(put_probability, 8),
                "probability_sum": round(call_probability + put_probability, 8)
            },
            "market_prices": {
                "call": {
                    "best_bid": round(call_best_bid, 6),
                    "best_ask": round(call_best_ask, 6),
                    "mid_price": round(call_mid_price, 6),
                    "spread": round(call_best_ask - call_best_bid, 6)
                },
                "put": {
                    "best_bid": round(put_best_bid, 6),
                    "best_ask": round(put_best_ask, 6),
                    "mid_price": round(put_mid_price, 6),
                    "spread": round(put_best_ask - put_best_bid, 6)
                }
            },
            "differences": {
                "call": {
                    "vs_bid": {
                        "dollars": round(call_bid_diff_dollars, 8),
                        "percent": round(call_bid_diff_percent, 4)
                    },
                    "vs_ask": {
                        "dollars": round(call_ask_diff_dollars, 8),
                        "percent": round(call_ask_diff_percent, 4)
                    },
                    "vs_mid": {
                        "dollars": round(call_mid_diff_dollars, 8),
                        "percent": round(call_mid_diff_percent, 4)
                    }
                },
                "put": {
                    "vs_bid": {
                        "dollars": round(put_bid_diff_dollars, 8),
                        "percent": round(put_bid_diff_percent, 4)
                    },
                    "vs_ask": {
                        "dollars": round(put_ask_diff_dollars, 8),
                        "percent": round(put_ask_diff_percent, 4)
                    },
                    "vs_mid": {
                        "dollars": round(put_mid_diff_dollars, 8),
                        "percent": round(put_mid_diff_percent, 4)
                    }
                }
            },
            "market_analysis": {
                "call_overpriced_vs_theory": call_mid_price > call_probability,
                "put_overpriced_vs_theory": put_mid_price > put_probability,
                "total_market_price_sum": round(call_mid_price + put_mid_price, 6),
                "arbitrage_opportunity": abs((call_mid_price + put_mid_price) - 1.0) > 0.01,
                "market_vs_theory_divergence": {
                    "call_divergence_abs": abs(call_mid_diff_dollars),
                    "put_divergence_abs": abs(put_mid_diff_dollars),
                    "total_divergence": abs(call_mid_diff_dollars) + abs(put_mid_diff_dollars)
                },
                "market_bias_analysis": {
                    "market_favors_call": call_mid_price > call_probability,
                    "market_favors_put": put_mid_price > put_probability,
                    "market_confidence_vs_theory": {
                        "call_market_confidence": round((call_mid_price - 0.5) * 2, 4) if call_mid_price > 0.5 else round((0.5 - call_mid_price) * -2, 4),
                        "call_theory_confidence": round((call_probability - 0.5) * 2, 4) if call_probability > 0.5 else round((0.5 - call_probability) * -2, 4),
                        "confidence_gap": round(((call_mid_price - 0.5) * 2) - ((call_probability - 0.5) * 2), 4)
                    }
                }
            }
        }

        # Save results to file
        with open(output_file_www, 'w') as f:
            json.dump(results, f, indent=2)

        return results


        # Save results to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    except KeyError as e:
        print(f"Error: Missing key in data - {e}")
        return None
    except Exception as e:
        print(f"Error calculating differences: {e}")
        return None

def print_analysis(results):
    """Print formatted analysis results"""
    if not results:
        return

    print("\n" + "="*70)
    print("PROBABILITY vs MARKET PRICE ANALYSIS")
    print("="*70)
    print(f"Analysis Time: {results['analysis_timestamp']}")

    # Resolution status
    if results.get("resolution"):
        print(f"🏁 OPTION RESOLVED: {results['resolution']}")
    elif results.get("time_to_expiry_hours", 0) > 0:
        print(f"⏰ Time to Expiry: {results['time_to_expiry_hours']:.4f} hours")

    # Theoretical probabilities
    probs = results['theoretical_probabilities']
    print(f"\nTheoretical Probabilities (Black-Scholes):")
    print(f"  Call: {probs['call_probability']:.6f} ({probs['call_probability']*100:.4f}%)")
    print(f"  Put:  {probs['put_probability']:.6f} ({probs['put_probability']*100:.4f}%)")
    print(f"  Sum:  {probs['probability_sum']:.6f}")

    # Market prices
    call_market = results['market_prices']['call']
    put_market = results['market_prices']['put']
    print(f"\nMarket Prices:")
    print(f"  CALL - Bid: {call_market['best_bid']:.4f}, Ask: {call_market['best_ask']:.4f}, Mid: {call_market['mid_price']:.4f}")
    print(f"  PUT  - Bid: {put_market['best_bid']:.4f}, Ask: {put_market['best_ask']:.4f}, Mid: {put_market['mid_price']:.4f}")

    # Differences
    call_diffs = results['differences']['call']
    put_diffs = results['differences']['put']

    print(f"\nCALL Option Differences (Theory vs Market):")
    print(f"  vs Bid: ${call_diffs['vs_bid']['dollars']:+.6f} ({call_diffs['vs_bid']['percent']:+.2f}%)")
    print(f"  vs Ask: ${call_diffs['vs_ask']['dollars']:+.6f} ({call_diffs['vs_ask']['percent']:+.2f}%)")
    print(f"  vs Mid: ${call_diffs['vs_mid']['dollars']:+.6f} ({call_diffs['vs_mid']['percent']:+.2f}%)")

    print(f"\nPUT Option Differences (Theory vs Market):")
    print(f"  vs Bid: ${put_diffs['vs_bid']['dollars']:+.6f} ({put_diffs['vs_bid']['percent']:+.2f}%)")
    print(f"  vs Ask: ${put_diffs['vs_ask']['dollars']:+.6f} ({put_diffs['vs_ask']['percent']:+.2f}%)")
    print(f"  vs Mid: ${put_diffs['vs_mid']['dollars']:+.6f} ({put_diffs['vs_mid']['percent']:+.2f}%)")

    # Market analysis
    analysis = results['market_analysis']
    print(f"\nMarket Analysis:")
    print(f"  Call Overpriced: {analysis['call_overpriced_vs_theory']}")
    print(f"  Put Overpriced:  {analysis['put_overpriced_vs_theory']}")
    print(f"  Market Sum: {analysis['total_market_price_sum']:.6f} (should be ~1.0)")
    print(f"  Arbitrage Opportunity: {analysis['arbitrage_opportunity']}")
    print(f"  Total Divergence: {analysis['market_vs_theory_divergence']['total_divergence']:.6f}")

    # Tracking statistics
    if 'tracking_stats' in results:
        tracking = results['tracking_stats']
        print(f"\n🏆 PERFORMANCE TRACKING:")
        print(f"  Total Resolved Predictions: {tracking['total_predictions']}")

        if tracking['total_predictions'] > 0:
            print(f"  Market Wins: {tracking['market_wins']} ({tracking['accuracy_stats']['market_win_rate']*100:.1f}%)")
            print(f"  Black-Scholes Wins: {tracking['bs_wins']}")
            print(f"  Ties: {tracking['ties']}")
            print(f"  Market Accuracy: {tracking['accuracy_stats']['market_accuracy']*100:.1f}%")
            print(f"  Black-Scholes Accuracy: {tracking['accuracy_stats']['bs_accuracy']*100:.1f}%")

            # Show who's winning
            if tracking['market_wins'] > tracking['bs_wins']:
                print(f"  🥇 MARKET is outperforming Black-Scholes!")
            elif tracking['bs_wins'] > tracking['market_wins']:
                print(f"  🥇 BLACK-SCHOLES is outperforming Market!")
            else:
                print(f"  🤝 Currently tied!")

        print(f"  Current Session Predictions: {tracking['current_session_predictions']}")

        # Pattern analysis
        if 'patterns' in tracking and tracking['patterns']:
            patterns = tracking['patterns']
            print(f"\n📊 PATTERN ANALYSIS:")

            if 'recent_accuracy' in patterns:
                recent = patterns['recent_accuracy']
                print(f"  Recent {recent['sample_size']} predictions:")
                print(f"    Market: {recent['market_recent_rate']*100:.1f}% accurate")
                print(f"    Black-Scholes: {recent['bs_recent_rate']*100:.1f}% accurate")

            if 'divergence_analysis' in patterns:
                div = patterns['divergence_analysis']
                print(f"  High Divergence (>{div['threshold']*100:.0f}%) Analysis:")
                print(f"    Sample size: {div['high_divergence_count']}")
                print(f"    Market accuracy: {div['market_accuracy_high_div']*100:.1f}%")

    # Current prediction
    market_call_favored = call_market['mid_price'] > put_market['mid_price']
    bs_call_favored = probs['call_probability'] > probs['put_probability']

    print(f"\n🔮 CURRENT PREDICTION:")
    print(f"  Market favors: {'CALL' if market_call_favored else 'PUT'} ({call_market['mid_price']*100:.1f}% vs {put_market['mid_price']*100:.1f}%)")
    print(f"  Black-Scholes favors: {'CALL' if bs_call_favored else 'PUT'} ({probs['call_probability']*100:.1f}% vs {probs['put_probability']*100:.1f}%)")

    if market_call_favored == bs_call_favored:
        print(f"  ✅ AGREEMENT: Both favor the same option")
    else:
        print(f"  ⚡ DISAGREEMENT: Market vs Theory conflict!")

        # Show historical context
        if 'tracking_stats' in results and results['tracking_stats']['total_predictions'] > 0:
            market_win_rate = results['tracking_stats']['accuracy_stats']['market_win_rate']
            if market_win_rate > 0.6:
                print(f"    📈 Market has {market_win_rate*100:.1f}% win rate - consider following market")
            elif market_win_rate < 0.4:
                print(f"    📉 Market has {market_win_rate*100:.1f}% win rate - consider following Black-Scholes")
            else:
                print(f"    ⚖️  Close race at {market_win_rate*100:.1f}% - monitor more data")

def main():
    """Main function to run continuous analysis"""
    print("Starting Continuous Probability vs Market Price Analysis")
    print("Press Ctrl+C to stop")

    iteration = 0

    try:
        while True:
            iteration += 1
            print(f"\n[Iteration {iteration}] Running analysis...")

            results = calculate_differences()

            if results:
                print_analysis(results)
                print(f"Results saved to: /home/ubuntu/013_2025_polymarket/bot004_blackScholes/callput_diff.json")
            else:
                print("Failed to calculate differences - retrying in next iteration")

            # Wait before next iteration (adjust as needed)
            time.sleep(10)  # 10 seconds between updates

    except KeyboardInterrupt:
        print("\n\nAnalysis stopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()
