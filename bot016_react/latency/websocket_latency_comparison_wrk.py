import websocket
import json
import threading
import time
from datetime import datetime
from collections import defaultdict
from pybit.unified_trading import WebSocket as BybitWebSocket
import statistics

class LatencyTracker:
    def __init__(self):
        self.latencies = defaultdict(list)
        self.counts = defaultdict(int)
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def add_latency(self, source, latency_ms):
        """Add a latency measurement"""
        with self.lock:
            self.latencies[source].append(latency_ms)
            self.counts[source] += 1
    
    def get_stats(self):
        """Get statistics for all sources"""
        with self.lock:
            stats = {}
            for source, latencies in self.latencies.items():
                if latencies:
                    stats[source] = {
                        'count': len(latencies),
                        'avg_ms': statistics.mean(latencies),
                        'median_ms': statistics.median(latencies),
                        'min_ms': min(latencies),
                        'max_ms': max(latencies),
                        'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0
                    }
            return stats
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        print("\n" + "="*80)
        print("LATENCY COMPARISON RESULTS")
        print("="*80)
        
        # Sort by average latency
        sorted_sources = sorted(stats.items(), key=lambda x: x[1]['avg_ms'])
        
        for rank, (source, data) in enumerate(sorted_sources, 1):
            print(f"\n#{rank} - {source}")
            print(f"  Count: {data['count']} messages")
            print(f"  Average: {data['avg_ms']:.2f} ms")
            print(f"  Median: {data['median_ms']:.2f} ms")
            print(f"  Min: {data['min_ms']:.2f} ms | Max: {data['max_ms']:.2f} ms")
            print(f"  Std Dev: {data['std_dev']:.2f} ms")
        
        print("\n" + "="*80)
        print(f"Test Duration: {time.time() - self.start_time:.1f} seconds")
        print("="*80 + "\n")


class BinanceSpotWS:
    def __init__(self, tracker):
        self.ws = None
        self.tracker = tracker
        self.url = "wss://stream.binance.com:9443/stream"
        self.name = "BINANCE_SPOT"
        
    def on_message(self, ws, message):
        try:
            receive_time = time.time() * 1000  # Current time in ms
            data = json.loads(message)
            
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                if stream_data.get('e') == 'aggTrade':
                    # Trade time from exchange
                    trade_time = stream_data['T']
                    # Calculate latency
                    latency = receive_time - trade_time
                    
                    self.tracker.add_latency(self.name, latency)
                    print(f"🟢 {self.name}: {stream_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            print(f"❌ {self.name} Error: {e}")
    
    def on_error(self, ws, error):
        print(f"❌ {self.name} Error: {error}")
    
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


class BinanceFuturesWS:
    def __init__(self, tracker):
        self.ws = None
        self.tracker = tracker
        self.url = "wss://fstream.binance.com/stream"
        self.name = "BINANCE_FUTURES"
        
    def on_message(self, ws, message):
        try:
            receive_time = time.time() * 1000
            data = json.loads(message)
            
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                if stream_data.get('e') == 'aggTrade':
                    trade_time = stream_data['T']
                    latency = receive_time - trade_time
                    
                    self.tracker.add_latency(self.name, latency)
                    print(f"🟡 {self.name}: {stream_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            print(f"❌ {self.name} Error: {e}")
    
    def on_error(self, ws, error):
        print(f"❌ {self.name} Error: {error}")
    
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


class BybitSpotWS:
    def __init__(self, tracker):
        self.tracker = tracker
        self.name = "BYBIT_SPOT"
        self.ws = None
        
    def handle_message(self, message):
        try:
            receive_time = time.time() * 1000
            
            if 'topic' in message and message['topic'] == 'publicTrade.BTCUSDT':
                if 'data' in message and len(message['data']) > 0:
                    trade_data = message['data'][0]
                    trade_time = trade_data['T']
                    latency = receive_time - trade_time
                    
                    self.tracker.add_latency(self.name, latency)
                    print(f"🔵 {self.name}: {trade_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            print(f"❌ {self.name} Error: {e}")
    
    def start(self):
        try:
            self.ws = BybitWebSocket(
                testnet=False,
                channel_type="spot"
            )
            self.ws.trade_stream(
                symbol="BTCUSDT",
                callback=self.handle_message
            )
            print(f"✅ {self.name} connected")
        except Exception as e:
            print(f"❌ {self.name} connection error: {e}")


class BybitFuturesWS:
    def __init__(self, tracker):
        self.tracker = tracker
        self.name = "BYBIT_FUTURES"
        self.ws = None
        
    def handle_message(self, message):
        try:
            receive_time = time.time() * 1000
            
            if 'topic' in message and message['topic'] == 'publicTrade.BTCUSDT':
                if 'data' in message and len(message['data']) > 0:
                    trade_data = message['data'][0]
                    trade_time = trade_data['T']
                    latency = receive_time - trade_time
                    
                    self.tracker.add_latency(self.name, latency)
                    print(f"🟣 {self.name}: {trade_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            print(f"❌ {self.name} Error: {e}")
    
    def start(self):
        try:
            self.ws = BybitWebSocket(
                testnet=False,
                channel_type="linear"
            )
            self.ws.trade_stream(
                symbol="BTCUSDT",
                callback=self.handle_message
            )
            print(f"✅ {self.name} connected")
        except Exception as e:
            print(f"❌ {self.name} connection error: {e}")


def main():
    print("Starting WebSocket Latency Comparison Test...")
    print("Comparing 4 WebSocket connections for BTCUSDT\n")
    
    # Create tracker
    tracker = LatencyTracker()
    
    # Initialize all WebSocket connections
    binance_spot = BinanceSpotWS(tracker)
    binance_futures = BinanceFuturesWS(tracker)
    bybit_spot = BybitSpotWS(tracker)
    bybit_futures = BybitFuturesWS(tracker)
    
    # Start all connections
    binance_spot.start()
    time.sleep(1)
    binance_futures.start()
    time.sleep(1)
    bybit_spot.start()
    time.sleep(1)
    bybit_futures.start()
    
    print("\n🚀 All WebSocket connections started!")
    print("Press Ctrl+C to stop and see results...\n")
    
    try:
        # Run for a specified duration or until interrupted
        while True:
            time.sleep(10)
            # Print interim stats every 10 seconds
            print("\n📊 Interim Stats:")
            tracker.print_stats()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping test...")
        tracker.print_stats()


if __name__ == "__main__":
    main()
