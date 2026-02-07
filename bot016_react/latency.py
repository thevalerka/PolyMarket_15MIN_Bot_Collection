import json
from datetime import datetime

class LatencyAnalyzer:
    def __init__(self):
        self.latencies = []

    def analyze(self, websocket1_messages, websocket2_messages, websocket3_messages, websocket4_messages):
        for message in websocket1_messages:
            if "T" in message:
                timestamp_ws1 = int(message["T"])
                server_timestamp = int(message["E"])
                latency = timestamp_ws1 - server_timestamp
                self.latencies.append(latency)

        for message in websocket2_messages:
            if "ts" in message and "T" in message:
                timestamp_ws2 = int(message["T"])
                server_timestamp = int(message["ts"])
                latency = timestamp_ws2 - server_timestamp
                self.latencies.append(latency)

        for message in websocket3_messages:
            if "T" in message:
                timestamp_ws3 = int(message["T"])
                server_timestamp = int(message["E"])
                latency = timestamp_ws3 - server_timestamp
                self.latencies.append(latency)

        for message in websocket4_messages:
            if "ts" in message and "T" in message:
                timestamp_ws4 = int(message["T"])
                server_timestamp = int(message["ts"])
                latency = timestamp_ws4 - server_timestamp
                self.latencies.append(latency)

        total_latency = sum(self.latencies)
        average_latency = total_latency / len(self.latencies)

        print(f"Average latency: {average_latency} milliseconds")
        print(f"Total latency: {total_latency} milliseconds")

# Usage:
analyzer = LatencyAnalyzer()
websocket1_messages = [... your WebSocket messages from websocket1 ...]
websocket2_messages = [... your WebSocket messages from websocket2 ...]
websocket3_messages = [... your WebSocket messages from websocket3 ...]
websocket4_messages = [... your WebSocket messages from websocket4 ...]

analyzer.analyze(websocket1_messages, websocket2_messages, websocket3_messages, websocket4_messages)
