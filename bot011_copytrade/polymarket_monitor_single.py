import requests
import json
import time
import os
from datetime import datetime
from typing import Set, Dict, List

class PolymarketTradeMonitor:
    def __init__(self):
        self.api_url = "https://data-api.polymarket.com/activity?user=0x118ed6e1e562ec54f1fa6e7f0a2741fe0d32ea4b&limit=10&offset=0"
        self.probabilities_file = "/home/ubuntu/013_2025_polymarket/option_probabilities.json"
        self.output_file = "bitcoin_trades.json"
        self.seen_transactions: Set[str] = set()
        
        # Load existing trades if file exists
        self.load_existing_trades()
        
    def load_existing_trades(self):
        """Load existing trades to avoid duplicates"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    existing_trades = json.load(f)
                    # Extract transaction hashes from existing trades
                    self.seen_transactions = {trade.get('transactionHash', '') for trade in existing_trades}
                    print(f"Loaded {len(existing_trades)} existing trades")
            except Exception as e:
                print(f"Error loading existing trades: {e}")
                self.seen_transactions = set()
    
    def read_probabilities(self) -> Dict:
        """Read call/put probabilities from local file"""
        try:
            with open(self.probabilities_file, 'r') as f:
                data = json.load(f)
                return {
                    'call_probability': data.get('call_probability', 0),
                    'put_probability': data.get('put_probability', 0),
                    'timestamp': data.get('timestamp', 0)
                }
        except Exception as e:
            print(f"Error reading probabilities file: {e}")
            return {'call_probability': 0, 'put_probability': 0, 'timestamp': 0}
    
    def fetch_trades(self) -> List[Dict]:
        """Fetch latest trades from Polymarket API"""
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return []
    
    def is_bitcoin_trade(self, trade: Dict) -> bool:
        """Check if trade is Bitcoin-related"""
        title = trade.get('title', '').upper()
        return 'BITCOIN' in title
    
    def process_new_trade(self, trade: Dict, probabilities: Dict) -> Dict:
        """Process a new Bitcoin trade and format for saving"""
        processed_trade = {
            'timestamp': trade.get('timestamp'),
            'datetime': datetime.fromtimestamp(trade.get('timestamp', 0)).isoformat(),
            'transactionHash': trade.get('transactionHash'),
            'title': trade.get('title'),
            'size': trade.get('size'),
            'usdcSize': trade.get('usdcSize'),
            'price': trade.get('price'),
            'outcome': trade.get('outcome'),
            'side': trade.get('side'),
            'call_probability': probabilities.get('call_probability'),
            'put_probability': probabilities.get('put_probability'),
            'probabilities_timestamp': probabilities.get('timestamp'),
            'detected_at': datetime.now().isoformat()
        }
        return processed_trade
    
    def save_trade(self, trade_data: Dict):
        """Save trade data to JSON file"""
        try:
            # Load existing data
            existing_trades = []
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r') as f:
                    existing_trades = json.load(f)
            
            # Append new trade
            existing_trades.append(trade_data)
            
            # Save back to file
            with open(self.output_file, 'w') as f:
                json.dump(existing_trades, f, indent=2)
            
            print(f"Saved new Bitcoin trade: {trade_data['title']} - {trade_data['outcome']} - Size: {trade_data['size']}")
            
        except Exception as e:
            print(f"Error saving trade: {e}")
    
    def monitor_trades(self):
        """Main monitoring loop"""
        print("Starting Polymarket Bitcoin trade monitor...")
        print(f"Monitoring user: 0x118ed6e1e562ec54f1fa6e7f0a2741fe0d32ea4b")
        print(f"Output file: {self.output_file}")
        print(f"Probabilities file: {self.probabilities_file}")
        print("Checking for Bitcoin trades every 10 seconds...")
        print("-" * 60)
        
        while True:
            try:
                # Fetch latest trades
                trades = self.fetch_trades()
                
                if not trades:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No trades found or API error")
                    time.sleep(10)
                    continue
                
                # Check for new Bitcoin trades
                new_bitcoin_trades = 0
                for trade in trades:
                    transaction_hash = trade.get('transactionHash', '')
                    
                    # Skip if we've already seen this trade
                    if transaction_hash in self.seen_transactions:
                        continue
                    
                    # Add to seen transactions
                    self.seen_transactions.add(transaction_hash)
                    
                    # Check if it's a Bitcoin trade
                    if self.is_bitcoin_trade(trade):
                        # Read current probabilities
                        probabilities = self.read_probabilities()
                        
                        # Process and save the trade
                        trade_data = self.process_new_trade(trade, probabilities)
                        self.save_trade(trade_data)
                        new_bitcoin_trades += 1
                
                if new_bitcoin_trades == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No new Bitcoin trades found")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {new_bitcoin_trades} new Bitcoin trade(s)")
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in monitoring loop: {e}")
            
            # Wait 10 seconds before next check
            time.sleep(10)

def main():
    """Run the trade monitor"""
    monitor = PolymarketTradeMonitor()
    monitor.monitor_trades()

if __name__ == "__main__":
    main()
