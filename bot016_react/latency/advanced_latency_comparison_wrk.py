"""
Advanced WebSocket Latency Comparison Tool

This implements multiple methods for measuring latency:
1. Event-to-Receive Latency (using exchange timestamps)
2. Round-Trip Time (RTT) - Ping/Pong
3. Clock Drift Compensation
4. One-way delay estimation using NTP sync
"""

import websocket
import json
import threading
import time
from datetime import datetime
from collections import defaultdict
from pybit.unified_trading import WebSocket as BybitWebSocket
import statistics
import ntplib
from typing import Dict, List, Optional

class EnhancedLatencyTracker:
    def __init__(self):
        # Different types of latency measurements
        self.event_latencies = defaultdict(list)  # Exchange timestamp to receive time
        self.rtt_latencies = defaultdict(list)    # Round-trip time (ping-pong)
        self.clock_offset = 0                      # Local clock offset from NTP
        self.counts = defaultdict(int)
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # Try to sync clock with NTP
        self._sync_clock()
        
    def _sync_clock(self):
        """Synchronize local clock with NTP server"""
        try:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request('pool.ntp.org', version=3)
            self.clock_offset = response.offset * 1000  # Convert to ms
            print(f"🕐 Clock synchronized. Offset: {self.clock_offset:.2f}ms")
        except Exception as e:
            print(f"⚠️  NTP sync failed: {e}. Using system clock.")
            self.clock_offset = 0
    
    def get_corrected_time_ms(self):
        """Get current time in ms, corrected for clock offset"""
        return (time.time() * 1000) - self.clock_offset
    
    def add_event_latency(self, source, latency_ms):
        """Add event-to-receive latency measurement"""
        with self.lock:
            self.event_latencies[source].append(latency_ms)
            self.counts[source] += 1
    
    def add_rtt_latency(self, source, rtt_ms):
        """Add round-trip time measurement"""
        with self.lock:
            self.rtt_latencies[source].append(rtt_ms)
    
    def get_stats(self):
        """Get comprehensive statistics"""
        with self.lock:
            stats = {}
            
            for source in set(list(self.event_latencies.keys()) + list(self.rtt_latencies.keys())):
                stats[source] = {}
                
                # Event latency stats
                if source in self.event_latencies and self.event_latencies[source]:
                    latencies = self.event_latencies[source]
                    stats[source]['event'] = {
                        'count': len(latencies),
                        'avg_ms': statistics.mean(latencies),
                        'median_ms': statistics.median(latencies),
                        'min_ms': min(latencies),
                        'max_ms': max(latencies),
                        'p95_ms': self._percentile(latencies, 95),
                        'p99_ms': self._percentile(latencies, 99),
                        'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0
                    }
                
                # RTT stats
                if source in self.rtt_latencies and self.rtt_latencies[source]:
                    rtts = self.rtt_latencies[source]
                    stats[source]['rtt'] = {
                        'count': len(rtts),
                        'avg_ms': statistics.mean(rtts),
                        'median_ms': statistics.median(rtts),
                        'min_ms': min(rtts),
                        'max_ms': max(rtts),
                        'std_dev': statistics.stdev(rtts) if len(rtts) > 1 else 0,
                        'estimated_one_way': statistics.mean(rtts) / 2
                    }
            
            return stats
    
    def _percentile(self, data, percentile):
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_comprehensive_stats(self):
        """Print detailed statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*100)
        print("COMPREHENSIVE LATENCY ANALYSIS")
        print("="*100)
        print(f"Test Duration: {time.time() - self.start_time:.1f} seconds")
        print(f"Clock Offset (NTP): {self.clock_offset:.2f}ms")
        print("="*100)
        
        # Rank by event latency
        sources_with_event = [(s, d['event']['avg_ms']) for s, d in stats.items() if 'event' in d]
        sorted_sources = sorted(sources_with_event, key=lambda x: x[1])
        
        for rank, (source, avg_latency) in enumerate(sorted_sources, 1):
            source_stats = stats[source]
            
            print(f"\n{'#' + str(rank)} - {source}")
            print("-" * 100)
            
            # Event Latency (Exchange Timestamp → Receive)
            if 'event' in source_stats:
                e = source_stats['event']
                print(f"  📨 Event-to-Receive Latency:")
                print(f"     Samples: {e['count']}")
                print(f"     Average: {e['avg_ms']:.2f}ms | Median: {e['median_ms']:.2f}ms")
                print(f"     Min: {e['min_ms']:.2f}ms | Max: {e['max_ms']:.2f}ms")
                print(f"     P95: {e['p95_ms']:.2f}ms | P99: {e['p99_ms']:.2f}ms")
                print(f"     Std Dev: {e['std_dev']:.2f}ms")
            
            # RTT Latency
            if 'rtt' in source_stats:
                r = source_stats['rtt']
                print(f"\n  🔄 Round-Trip Time (RTT):")
                print(f"     Samples: {r['count']}")
                print(f"     Average RTT: {r['avg_ms']:.2f}ms")
                print(f"     Estimated One-Way: {r['estimated_one_way']:.2f}ms")
                print(f"     Min: {r['min_ms']:.2f}ms | Max: {r['max_ms']:.2f}ms")
                print(f"     Std Dev: {r['std_dev']:.2f}ms")
        
        print("\n" + "="*100)
        
        # Winner summary
        if sorted_sources:
            winner = sorted_sources[0][0]
            winner_latency = sorted_sources[0][1]
            print(f"🏆 FASTEST: {winner} with {winner_latency:.2f}ms average latency")
            print("="*100 + "\n")
    
    def export_to_json(self, filename='latency_results.json'):
        """Export results to JSON file"""
        stats = self.get_stats()
        with open(filename, 'w') as f:
            json.dump({
                'test_duration_seconds': time.time() - self.start_time,
                'clock_offset_ms': self.clock_offset,
                'statistics': stats
            }, f, indent=2)
        print(f"📁 Results exported to {filename}")


class BinanceSpotWSEnhanced:
    def __init__(self, tracker):
        self.ws = None
        self.tracker = tracker
        self.url = "wss://stream.binance.com:9443/stream"
        self.name = "BINANCE_SPOT"
        self.ping_sent_time = None
        
    def on_message(self, ws, message):
        try:
            receive_time = self.tracker.get_corrected_time_ms()
            data = json.loads(message)
            
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                if stream_data.get('e') == 'aggTrade':
                    trade_time = stream_data['T']
                    latency = receive_time - trade_time
                    
                    self.tracker.add_event_latency(self.name, latency)
                    print(f"🟢 {self.name}: ${stream_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            pass  # Silent to reduce noise
    
    def on_pong(self, ws, message):
        """Handle pong response to measure RTT"""
        if self.ping_sent_time:
            rtt = (time.time() * 1000) - self.ping_sent_time
            self.tracker.add_rtt_latency(self.name, rtt)
            print(f"🏓 {self.name} RTT: {rtt:.2f}ms")
            self.ping_sent_time = None
    
    def send_ping(self):
        """Send ping to measure RTT"""
        if self.ws:
            self.ping_sent_time = time.time() * 1000
            self.ws.send_ping()
    
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
            on_close=self.on_close,
            on_pong=self.on_pong
        )
        thread = threading.Thread(target=self.ws.run_forever, kwargs={'ping_interval': 30})
        thread.daemon = True
        thread.start()


class BinanceFuturesWSEnhanced:
    def __init__(self, tracker):
        self.ws = None
        self.tracker = tracker
        self.url = "wss://fstream.binance.com/stream"
        self.name = "BINANCE_FUTURES"
        self.ping_sent_time = None
        
    def on_message(self, ws, message):
        try:
            receive_time = self.tracker.get_corrected_time_ms()
            data = json.loads(message)
            
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                if stream_data.get('e') == 'aggTrade':
                    trade_time = stream_data['T']
                    latency = receive_time - trade_time
                    
                    self.tracker.add_event_latency(self.name, latency)
                    print(f"🟡 {self.name}: ${stream_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            pass
    
    def on_pong(self, ws, message):
        if self.ping_sent_time:
            rtt = (time.time() * 1000) - self.ping_sent_time
            self.tracker.add_rtt_latency(self.name, rtt)
            print(f"🏓 {self.name} RTT: {rtt:.2f}ms")
            self.ping_sent_time = None
    
    def send_ping(self):
        if self.ws:
            self.ping_sent_time = time.time() * 1000
            self.ws.send_ping()
    
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
            on_close=self.on_close,
            on_pong=self.on_pong
        )
        thread = threading.Thread(target=self.ws.run_forever, kwargs={'ping_interval': 30})
        thread.daemon = True
        thread.start()


class BybitSpotWSEnhanced:
    def __init__(self, tracker):
        self.tracker = tracker
        self.name = "BYBIT_SPOT"
        self.ws = None
        
    def handle_message(self, message):
        try:
            receive_time = self.tracker.get_corrected_time_ms()
            
            if 'topic' in message and message['topic'] == 'publicTrade.BTCUSDT':
                if 'data' in message and len(message['data']) > 0:
                    trade_data = message['data'][0]
                    trade_time = trade_data['T']
                    latency = receive_time - trade_time
                    
                    self.tracker.add_event_latency(self.name, latency)
                    print(f"🔵 {self.name}: ${trade_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            pass
    
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


class BybitFuturesWSEnhanced:
    def __init__(self, tracker):
        self.tracker = tracker
        self.name = "BYBIT_FUTURES"
        self.ws = None
        
    def handle_message(self, message):
        try:
            receive_time = self.tracker.get_corrected_time_ms()
            
            if 'topic' in message and message['topic'] == 'publicTrade.BTCUSDT':
                if 'data' in message and len(message['data']) > 0:
                    trade_data = message['data'][0]
                    trade_time = trade_data['T']
                    latency = receive_time - trade_time
                    
                    self.tracker.add_event_latency(self.name, latency)
                    print(f"🟣 {self.name}: ${trade_data['p']} | Latency: {latency:.2f}ms")
                    
        except Exception as e:
            pass
    
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


def periodic_ping(connections, interval=30):
    """Send periodic pings to measure RTT"""
    while True:
        time.sleep(interval)
        for conn in connections:
            if hasattr(conn, 'send_ping'):
                try:
                    conn.send_ping()
                except:
                    pass


def main(test_duration_seconds=60):
    print("="*100)
    print("ADVANCED WEBSOCKET LATENCY COMPARISON")
    print("="*100)
    print("Measuring multiple latency metrics:")
    print("  1. Event-to-Receive: Time from exchange timestamp to local receipt")
    print("  2. Round-Trip Time (RTT): Ping-pong latency")
    print("  3. NTP Clock Synchronization for accuracy")
    print(f"\nTest will run for {test_duration_seconds} seconds...\n")
    
    # Create tracker
    tracker = EnhancedLatencyTracker()
    
    # Initialize all WebSocket connections
    binance_spot = BinanceSpotWSEnhanced(tracker)
    binance_futures = BinanceFuturesWSEnhanced(tracker)
    bybit_spot = BybitSpotWSEnhanced(tracker)
    bybit_futures = BybitFuturesWSEnhanced(tracker)
    
    # Start all connections
    print("Starting connections...\n")
    binance_spot.start()
    time.sleep(1)
    binance_futures.start()
    time.sleep(1)
    bybit_spot.start()
    time.sleep(1)
    bybit_futures.start()
    time.sleep(2)
    
    # Start periodic ping thread
    ping_connections = [binance_spot, binance_futures]
    ping_thread = threading.Thread(target=periodic_ping, args=(ping_connections,))
    ping_thread.daemon = True
    ping_thread.start()
    
    print("🚀 All connections started!\n")
    
    try:
        # Run for specified duration
        end_time = time.time() + test_duration_seconds
        
        while time.time() < end_time:
            remaining = int(end_time - time.time())
            print(f"\r⏱️  Time remaining: {remaining}s", end='', flush=True)
            time.sleep(1)
        
        print("\n\n🛑 Test complete! Generating results...\n")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user!\n")
    
    # Print final results
    tracker.print_comprehensive_stats()
    tracker.export_to_json()


if __name__ == "__main__":
    # Run test for 60 seconds (you can change this)
    main(test_duration_seconds=60)
