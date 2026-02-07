#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime, timezone

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

def calculate_differences():
    """Calculate differences between theoretical probabilities and market prices"""
    
    # Define file paths
    base_path = "/home/ubuntu/013_2025_polymarket/"
    prob_file = os.path.join(base_path, "option_probabilities.json")
    call_file = os.path.join(base_path, "CALL.json")
    put_file = os.path.join(base_path, "PUT.json")
    output_file = os.path.join(base_path, "bot004_blackScholes/callput_diff.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
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
                }
            }
        }
        
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
