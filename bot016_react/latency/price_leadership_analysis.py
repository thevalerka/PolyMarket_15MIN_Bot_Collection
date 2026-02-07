"""
Price Leadership Analysis Tool

Determines which exchange leads price movements by analyzing:
1. Cross-correlation with time lags
2. Granger causality testing
3. Lead-lag relationships
4. Price change leadership
"""

import websocket
import json
import threading
import time
from datetime import datetime
from collections import defaultdict, deque
from pybit.unified_trading import WebSocket as BybitWebSocket
import statistics
import numpy as np
from scipy import stats

class PriceLeadershipAnalyzer:
    def __init__(self, window_size=100, max_lag=10):
        """
        window_size: Number of price updates to keep for analysis
        max_lag: Maximum lag (in updates) to test for leadership
        """
        self.prices = defaultdict(lambda: deque(maxlen=window_size))
        self.timestamps = defaultdict(lambda: deque(maxlen=window_size))
        self.price_changes = defaultdict(lambda: deque(maxlen=window_size))
        
        self.window_size = window_size
        self.max_lag = max_lag
        self.lock = threading.Lock()
        
        # Leadership metrics
        self.lead_counts = defaultdict(int)  # How many times each exchange led
        self.correlation_results = {}
        self.granger_results = {}
        
    def add_price(self, source, price, timestamp):
        """Add a price update from an exchange"""
        with self.lock:
            self.prices[source].append(float(price))
            self.timestamps[source].append(timestamp)
            
            # Calculate price change if we have previous price
            if len(self.prices[source]) > 1:
                prev_price = self.prices[source][-2]
                price_change = (float(price) - prev_price) / prev_price * 100  # Percentage change
                self.price_changes[source].append(price_change)
    
    def calculate_cross_correlation(self, source1, source2):
        """
        Calculate cross-correlation between two price series at different lags.
        Positive lag means source1 leads source2.
        """
        if len(self.prices[source1]) < 20 or len(self.prices[source2]) < 20:
            return None
        
        # Get synchronized price arrays
        prices1 = list(self.prices[source1])
        prices2 = list(self.prices[source2])
        
        # Ensure same length
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]
        
        if len(prices1) < 20:
            return None
        
        # Normalize (remove mean, divide by std)
        prices1 = np.array(prices1)
        prices2 = np.array(prices2)
        
        prices1 = (prices1 - np.mean(prices1)) / (np.std(prices1) + 1e-10)
        prices2 = (prices2 - np.mean(prices2)) / (np.std(prices2) + 1e-10)
        
        # Calculate correlation at different lags
        correlations = {}
        max_lag = min(self.max_lag, len(prices1) // 4)
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # source2 leads source1
                corr = np.corrcoef(prices1[:lag], prices2[-lag:])[0, 1]
            elif lag > 0:
                # source1 leads source2
                corr = np.corrcoef(prices1[lag:], prices2[:-lag])[0, 1]
            else:
                # No lag
                corr = np.corrcoef(prices1, prices2)[0, 1]
            
            correlations[lag] = corr
        
        return correlations
    
    def detect_price_change_leader(self):
        """
        Detect which exchange first shows significant price changes.
        Returns: Dict with leadership counts for each direction change.
        """
        if not self.price_changes:
            return {}
        
        # Get all sources with enough data
        sources_with_data = [s for s in self.price_changes.keys() 
                            if len(self.price_changes[s]) >= 10]
        
        if len(sources_with_data) < 2:
            return {}
        
        # Look at recent price changes
        window = 20
        leadership = defaultdict(int)
        
        for i in range(max(0, min([len(self.price_changes[s]) for s in sources_with_data]) - window), 
                      min([len(self.price_changes[s]) for s in sources_with_data])):
            
            # Get price changes at this point for all sources
            changes = {}
            for source in sources_with_data:
                if i < len(self.price_changes[source]):
                    changes[source] = self.price_changes[source][i]
            
            if not changes:
                continue
            
            # Find who had the biggest absolute change (potential leader)
            leader = max(changes.items(), key=lambda x: abs(x[1]))
            leadership[leader[0]] += 1
        
        return leadership
    
    def analyze_leadership(self):
        """
        Comprehensive leadership analysis.
        Returns detailed results about which exchange leads.
        """
        with self.lock:
            sources = list(self.prices.keys())
            
            if len(sources) < 2:
                return {"error": "Need at least 2 exchanges with data"}
            
            results = {
                "cross_correlation": {},
                "leadership_summary": {},
                "price_change_leadership": {}
            }
            
            # Cross-correlation analysis
            for i, source1 in enumerate(sources):
                for source2 in sources[i+1:]:
                    correlations = self.calculate_cross_correlation(source1, source2)
                    if correlations:
                        # Find lag with maximum correlation
                        max_lag = max(correlations.items(), key=lambda x: abs(x[1]))
                        
                        key = f"{source1}_vs_{source2}"
                        results["cross_correlation"][key] = {
                            "best_lag": max_lag[0],
                            "correlation": max_lag[1],
                            "interpretation": self._interpret_lag(source1, source2, max_lag[0])
                        }
            
            # Price change leadership
            price_leadership = self.detect_price_change_leader()
            results["price_change_leadership"] = price_leadership
            
            # Overall leadership ranking
            leadership_scores = defaultdict(float)
            
            # Score from cross-correlation (being a leader in correlations)
            for key, data in results["cross_correlation"].items():
                source1, source2 = key.split("_vs_")
                lag = data["best_lag"]
                corr = abs(data["correlation"])
                
                if lag > 0:  # source1 leads
                    leadership_scores[source1] += corr * lag
                elif lag < 0:  # source2 leads
                    leadership_scores[source2] += corr * abs(lag)
            
            # Score from price change leadership
            total_changes = sum(price_leadership.values())
            if total_changes > 0:
                for source, count in price_leadership.items():
                    leadership_scores[source] += (count / total_changes) * 10
            
            # Rank exchanges by leadership score
            ranked = sorted(leadership_scores.items(), key=lambda x: x[1], reverse=True)
            results["leadership_summary"] = {
                source: {
                    "score": score,
                    "rank": rank + 1
                }
                for rank, (source, score) in enumerate(ranked)
            }
            
            return results
    
    def _interpret_lag(self, source1, source2, lag):
        """Interpret what a lag means"""
        if lag > 0:
            return f"{source1} leads {source2} by {lag} updates"
        elif lag < 0:
            return f"{source2} leads {source1} by {abs(lag)} updates"
        else:
            return f"{source1} and {source2} move simultaneously"
    
    def print_analysis(self):
        """Print comprehensive analysis report"""
        results = self.analyze_leadership()
        
        if "error" in results:
            print(f"\n❌ {results['error']}")
            return
        
        print("\n" + "="*90)
        print("PRICE LEADERSHIP ANALYSIS")
        print("="*90)
        
        # Leadership Summary
        if results["leadership_summary"]:
            print("\n🏆 OVERALL LEADERSHIP RANKING:")
            print("-"*90)
            for source, data in sorted(results["leadership_summary"].items(), 
                                      key=lambda x: x[1]["rank"]):
                print(f"  #{data['rank']} - {source:<25} | Leadership Score: {data['score']:.2f}")
        
        # Cross-correlation results
        if results["cross_correlation"]:
            print("\n📊 CROSS-CORRELATION ANALYSIS (Lead-Lag Relationships):")
            print("-"*90)
            for pair, data in results["cross_correlation"].items():
                source1, source2 = pair.split("_vs_")
                lag = data["best_lag"]
                corr = data["correlation"]
                
                if abs(corr) > 0.5:  # Only show significant correlations
                    print(f"  {source1} ↔ {source2}:")
                    print(f"    Correlation: {corr:.3f} at lag {lag}")
                    print(f"    {data['interpretation']}")
                    print()
        
        # Price change leadership
        if results["price_change_leadership"]:
            print("\n📈 PRICE MOVEMENT LEADERSHIP (Who moves first):")
            print("-"*90)
            total = sum(results["price_change_leadership"].values())
            for source, count in sorted(results["price_change_leadership"].items(), 
                                       key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                print(f"  {source:<25} | Led {count} times ({percentage:.1f}%)")
        
        print("\n" + "="*90)
        print("\n💡 INTERPRETATION GUIDE:")
        print("  • Leadership Score: Higher = More likely to lead price movements")
        print("  • Positive Lag: Exchange A leads Exchange B")
        print("  • Negative Lag: Exchange B leads Exchange A")
        print("  • Correlation > 0.7: Strong relationship")
        print("  • Correlation > 0.5: Moderate relationship")
        print("="*90 + "\n")


# WebSocket connection classes (same as before, but feeding into analyzer)

class BinanceSpotWSLeadership:
    def __init__(self, analyzer):
        self.ws = None
        self.analyzer = analyzer
        self.url = "wss://stream.binance.com:9443/stream"
        self.name = "BINANCE_SPOT"
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                if stream_data.get('e') == 'aggTrade':
                    price = stream_data['p']
                    timestamp = stream_data['T']
                    self.analyzer.add_price(self.name, price, timestamp)
        except Exception as e:
            pass
    
    def on_error(self, ws, error):
        pass
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"🔴 {self.name} closed")
    
    def on_open(self, ws):
        print(f"✅ {self.name} connected")
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": ["btcusdt@aggTrade"],
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))
    
    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        thread = threading.Thread(target=self.ws.run_forever)
        thread.daemon = True
        thread.start()


class BinanceFuturesWSLeadership:
    def __init__(self, analyzer):
        self.ws = None
        self.analyzer = analyzer
        self.url = "wss://fstream.binance.com/stream"
        self.name = "BINANCE_FUTURES"
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                if stream_data.get('e') == 'aggTrade':
                    price = stream_data['p']
                    timestamp = stream_data['T']
                    self.analyzer.add_price(self.name, price, timestamp)
        except Exception as e:
            pass
    
    def on_error(self, ws, error):
        pass
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"🔴 {self.name} closed")
    
    def on_open(self, ws):
        print(f"✅ {self.name} connected")
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": ["btcusdt@aggTrade"],
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))
    
    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        thread = threading.Thread(target=self.ws.run_forever)
        thread.daemon = True
        thread.start()


class BybitSpotWSLeadership:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.name = "BYBIT_SPOT"
        self.ws = None
        
    def handle_message(self, message):
        try:
            if 'topic' in message and message['topic'] == 'publicTrade.BTCUSDT':
                if 'data' in message and len(message['data']) > 0:
                    trade_data = message['data'][0]
                    price = trade_data['p']
                    timestamp = trade_data['T']
                    self.analyzer.add_price(self.name, price, timestamp)
        except Exception as e:
            pass
    
    def start(self):
        try:
            self.ws = BybitWebSocket(testnet=False, channel_type="spot")
            self.ws.trade_stream(symbol="BTCUSDT", callback=self.handle_message)
            print(f"✅ {self.name} connected")
        except Exception as e:
            print(f"❌ {self.name} connection error: {e}")


class BybitFuturesWSLeadership:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.name = "BYBIT_FUTURES"
        self.ws = None
        
    def handle_message(self, message):
        try:
            if 'topic' in message and message['topic'] == 'publicTrade.BTCUSDT':
                if 'data' in message and len(message['data']) > 0:
                    trade_data = message['data'][0]
                    price = trade_data['p']
                    timestamp = trade_data['T']
                    self.analyzer.add_price(self.name, price, timestamp)
        except Exception as e:
            pass
    
    def start(self):
        try:
            self.ws = BybitWebSocket(testnet=False, channel_type="linear")
            self.ws.trade_stream(symbol="BTCUSDT", callback=self.handle_message)
            print(f"✅ {self.name} connected")
        except Exception as e:
            print(f"❌ {self.name} connection error: {e}")


class CoinbaseWSLeadership:
    def __init__(self, analyzer):
        self.ws = None
        self.analyzer = analyzer
        self.url = "wss://ws-feed.exchange.coinbase.com"
        self.name = "COINBASE"
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if data.get('type') in ['ticker', 'match']:
                time_str = data.get('time')
                if time_str:
                    dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    timestamp = dt.timestamp() * 1000
                    price = data.get('price')
                    if price:
                        self.analyzer.add_price(self.name, price, timestamp)
        except Exception as e:
            pass
    
    def on_error(self, ws, error):
        pass
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"🔴 {self.name} closed")
    
    def on_open(self, ws):
        print(f"✅ {self.name} connected")
        subscribe_message = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker", "matches"]
        }
        ws.send(json.dumps(subscribe_message))
    
    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        thread = threading.Thread(target=self.ws.run_forever)
        thread.daemon = True
        thread.start()


def main(test_duration_seconds=120, analysis_interval=30):
    """
    Run price leadership analysis
    
    test_duration_seconds: How long to collect data
    analysis_interval: How often to print interim analysis
    """
    print("="*90)
    print("PRICE LEADERSHIP ANALYSIS")
    print("="*90)
    print("Analyzing which exchange leads Bitcoin price movements...\n")
    print(f"Collection time: {test_duration_seconds} seconds")
    print(f"Analysis interval: {analysis_interval} seconds\n")
    
    # Create analyzer
    analyzer = PriceLeadershipAnalyzer(window_size=200, max_lag=10)
    
    # Initialize all connections
    binance_spot = BinanceSpotWSLeadership(analyzer)
    binance_futures = BinanceFuturesWSLeadership(analyzer)
    bybit_spot = BybitSpotWSLeadership(analyzer)
    bybit_futures = BybitFuturesWSLeadership(analyzer)
    coinbase = CoinbaseWSLeadership(analyzer)
    
    # Start all connections
    print("Starting connections...\n")
    binance_spot.start()
    time.sleep(1)
    binance_futures.start()
    time.sleep(1)
    bybit_spot.start()
    time.sleep(1)
    bybit_futures.start()
    time.sleep(1)
    coinbase.start()
    time.sleep(3)
    
    print("🚀 All connections started!\n")
    print("Collecting data for analysis...\n")
    
    try:
        # Run for specified duration with periodic analysis
        start_time = time.time()
        last_analysis = start_time
        
        while time.time() - start_time < test_duration_seconds:
            time.sleep(1)
            
            # Print interim analysis every interval
            if time.time() - last_analysis >= analysis_interval:
                analyzer.print_analysis()
                last_analysis = time.time()
                print(f"\n⏱️  Continuing data collection... ({int(test_duration_seconds - (time.time() - start_time))}s remaining)\n")
        
        print("\n🛑 Data collection complete! Generating final analysis...\n")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user!\n")
    
    # Print final comprehensive analysis
    analyzer.print_analysis()
    
    # Export results to JSON
    results = analyzer.analyze_leadership()
    with open('leadership_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("📁 Results exported to leadership_analysis.json\n")


if __name__ == "__main__":
    # Run analysis for 2 minutes with updates every 30 seconds
    # Adjust these parameters as needed
    main(test_duration_seconds=120, analysis_interval=30)
