import requests
import json
import time
from datetime import datetime
from collections import defaultdict

class PolymarketPositionAnalyzer:
    def __init__(self, call_data_file=None, put_data_file=None):
        """
        Initialize the analyzer with CALL and PUT token data

        Args:
            call_data_file: Path to CALL.json file
            put_data_file: Path to PUT.json file
        """
        self.call_data = None
        self.put_data = None

        # Load CALL and PUT data if files are provided
        if call_data_file:
            try:
                with open(call_data_file, 'r') as f:
                    self.call_data = json.load(f)
            except Exception as e:
                print(f"Error loading CALL data: {e}")

        if put_data_file:
            try:
                with open(put_data_file, 'r') as f:
                    self.put_data = json.load(f)
            except Exception as e:
                print(f"Error loading PUT data: {e}")

    def set_call_put_data_from_content(self, call_content, put_content):
        """
        Set CALL and PUT data from JSON content strings

        Args:
            call_content: JSON string content for CALL data
            put_content: JSON string content for PUT data
        """
        try:
            self.call_data = json.loads(call_content)
            self.put_data = json.loads(put_content)
            print("✅ CALL and PUT data loaded successfully")
        except Exception as e:
            print(f"❌ Error parsing CALL/PUT data: {e}")

    def get_trader_positions(self, user_address):
        """
        Fetch trader positions from Polymarket API

        Args:
            user_address: Trader's wallet address

        Returns:
            List of positions or None if error
        """
        url = f"https://data-api.polymarket.com/positions?user={user_address}&limit=500"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching positions for {user_address}: {e}")
            return None

    def analyze_positions(self, positions, trader_address):
        """
        Analyze positions for UP/CALL and DOWN/PUT tokens

        Args:
            positions: List of position data
            trader_address: Trader's address for display

        Returns:
            Dict with analysis results
        """
        if not self.call_data or not self.put_data:
            print("❌ CALL and PUT data not loaded")
            return None

        call_asset_id = self.call_data['asset_id']
        put_asset_id = self.put_data['asset_id']

        call_best_bid = self.call_data['best_bid']['price']
        put_best_bid = self.put_data['best_bid']['price']

        relevant_positions = []

        for position in positions:
            asset_id = position['asset']

            # Check if this position matches our CALL or PUT assets
            if asset_id == call_asset_id:
                # This is a CALL position
                position_type = "UP/CALL"
                current_market_price = call_best_bid
                relevant_positions.append({
                    'type': position_type,
                    'position': position,
                    'current_market_price': current_market_price
                })
            elif asset_id == put_asset_id:
                # This is a PUT position
                position_type = "DOWN/PUT"
                current_market_price = put_best_bid
                relevant_positions.append({
                    'type': position_type,
                    'position': position,
                    'current_market_price': current_market_price
                })

        return relevant_positions

    def display_trader_analysis(self, trader_address, relevant_positions, nickname=None):
        """
        Display analysis for a single trader

        Args:
            trader_address: Trader's address
            relevant_positions: List of relevant positions
            nickname: Optional nickname for the trader
        """
        if not relevant_positions:
            trader_label = f"{nickname} ({trader_address[:10]}...)" if nickname else trader_address
            print(f"🔍 No UP/CALL or DOWN/PUT positions found for {trader_label}")
            return

        print(f"\n{'='*80}")
        if nickname:
            print(f"📊 TRADER ANALYSIS: {nickname}")
            print(f"📍 Address: {trader_address}")
        else:
            print(f"📊 TRADER ANALYSIS: {trader_address}")
        print(f"{'='*80}")

        total_positions = len(relevant_positions)
        total_value = sum(pos['position']['currentValue'] for pos in relevant_positions)
        total_pnl = sum(pos['position']['cashPnl'] for pos in relevant_positions)

        print(f"Total Relevant Positions: {total_positions}")
        print(f"Total Current Value: ${total_value:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"-" * 80)

        for i, pos_data in enumerate(relevant_positions, 1):
            pos_type = pos_data['type']
            position = pos_data['position']
            market_price = pos_data['current_market_price']

            # Determine if bullish or bearish
            direction = "🟢 BULLISH" if pos_type == "UP/CALL" else "🔴 BEARISH"

            print(f"\n{i}. {direction} - {pos_type}")
            print(f"   Asset: {position['asset']}")
            print(f"   Title: {position['title']}")
            print(f"   Outcome: {position['outcome']}")
            print(f"   📏 Position Size: {position['size']:.6f}")
            print(f"   💰 Average Price: ${position['avgPrice']:.6f}")
            print(f"   📈 Current Market Price: ${market_price:.6f}")
            print(f"   💵 Current Value: ${position['currentValue']:.2f}")
            print(f"   📊 P&L: ${position['cashPnl']:.2f} ({position['percentPnl']:.1f}%)")

            # Price comparison
            if position['avgPrice'] < market_price:
                print(f"   ✅ Position is in profit (bought at ${position['avgPrice']:.6f}, market at ${market_price:.6f})")
            else:
                print(f"   ❌ Position is at loss (bought at ${position['avgPrice']:.6f}, market at ${market_price:.6f})")

    def get_trader_nickname(self, trader_address, trader_data):
        """
        Get nickname for a trader address

        Args:
            trader_address: The trader's address
            trader_data: Dict or list of trader data

        Returns:
            Nickname string
        """
        if isinstance(trader_data, dict):
            return trader_data.get(trader_address, f"Unknown_{trader_address[:6]}")
        else:
            return f"Trader_{trader_address[:6]}"

    def analyze_multiple_traders(self, trader_data):
        """
        Analyze multiple traders for UP/CALL and DOWN/PUT positions

        Args:
            trader_data: Dict with trader addresses as keys and nicknames as values,
                        or list of addresses (nicknames will be auto-generated)
        """
        # Handle both dict and list inputs
        if isinstance(trader_data, list):
            trader_dict = {addr: f"Trader_{i+1}" for i, addr in enumerate(trader_data)}
        else:
            trader_dict = trader_data

        print(f"\n🚀 ANALYZING {len(trader_dict)} TRADERS FOR UP/CALL & DOWN/PUT POSITIONS")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.call_data and self.put_data:
            print(f"🎯 Target Assets:")
            print(f"   CALL: {self.call_data['asset_name']} (${self.call_data['best_bid']['price']:.6f})")
            print(f"   PUT: {self.put_data['asset_name']} (${self.put_data['best_bid']['price']:.6f})")

        all_results = {}

        for trader_address, nickname in trader_dict.items():
            print(f"\n📡 Fetching positions for {nickname} ({trader_address[:10]}...)...")
            positions = self.get_trader_positions(trader_address)

            if positions:
                relevant_positions = self.analyze_positions(positions, trader_address)
                all_results[trader_address] = {
                    'nickname': nickname,
                    'positions': relevant_positions
                }
                self.display_trader_analysis(trader_address, relevant_positions, nickname)
            else:
                print(f"❌ Failed to fetch positions for {nickname}")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        # Summary
        self.display_summary(all_results)

    def display_summary(self, all_results):
        """
        Display summary across all traders

        Args:
            all_results: Dict with all trader results including nicknames
        """
        print(f"\n{'='*80}")
        print(f"📋 SUMMARY ACROSS ALL TRADERS")
        print(f"{'='*80}")

        total_traders_with_positions = 0
        total_call_positions = 0
        total_put_positions = 0
        total_call_value = 0
        total_put_value = 0

        trader_breakdown = []

        for trader_address, trader_info in all_results.items():
            nickname = trader_info['nickname']
            positions = trader_info['positions']

            if positions:
                total_traders_with_positions += 1
                trader_calls = 0
                trader_puts = 0
                trader_call_value = 0
                trader_put_value = 0

                for pos_data in positions:
                    if pos_data['type'] == "UP/CALL":
                        total_call_positions += 1
                        trader_calls += 1
                        pos_value = pos_data['position']['currentValue']
                        total_call_value += pos_value
                        trader_call_value += pos_value
                    else:
                        total_put_positions += 1
                        trader_puts += 1
                        pos_value = pos_data['position']['currentValue']
                        total_put_value += pos_value
                        trader_put_value += pos_value

                # Store trader breakdown for detailed summary
                trader_breakdown.append({
                    'nickname': nickname,
                    'address': trader_address,
                    'calls': trader_calls,
                    'puts': trader_puts,
                    'call_value': trader_call_value,
                    'put_value': trader_put_value
                })

        print(f"Traders with relevant positions: {total_traders_with_positions}/{len(all_results)}")
        print(f"🟢 Total UP/CALL positions: {total_call_positions} (${total_call_value:.2f})")
        print(f"🔴 Total DOWN/PUT positions: {total_put_positions} (${total_put_value:.2f})")

        if total_call_positions + total_put_positions > 0:
            call_percentage = (total_call_positions / (total_call_positions + total_put_positions)) * 100
            put_percentage = (total_put_positions / (total_call_positions + total_put_positions)) * 100
            print(f"📊 Sentiment: {call_percentage:.1f}% BULLISH, {put_percentage:.1f}% BEARISH")

        # Detailed trader breakdown
        if trader_breakdown:
            print(f"\n📈 TRADER BREAKDOWN:")
            print(f"-" * 80)
            for trader in trader_breakdown:
                stance = ""
                if trader['calls'] > 0 and trader['puts'] == 0:
                    stance = "🟢 BULLISH"
                elif trader['puts'] > 0 and trader['calls'] == 0:
                    stance = "🔴 BEARISH"
                elif trader['calls'] > 0 and trader['puts'] > 0:
                    stance = "⚪ MIXED"

                total_value = trader['call_value'] + trader['put_value']
                print(f"{trader['nickname']}: {stance} | CALLS: {trader['calls']} | PUTS: {trader['puts']} | Total: ${total_value:.2f}")


# Example usage
def main():
    # Initialize analyzer with real data files
    analyzer = PolymarketPositionAnalyzer(
        call_data_file='/home/ubuntu/013_2025_polymarket/CALL.json',
        put_data_file='/home/ubuntu/013_2025_polymarket/PUT.json'
    )

    # If files are not available, use the uploaded data directly
    if not analyzer.call_data or not analyzer.put_data:
        # Use real CALL and PUT data from uploaded files
        call_content = """{
  "asset_name": "CALL",
  "asset_id": "43009338837374304741695824038486955879079948473184084933307215546114897553526",
  "market": "0xe9fac6cb411bb9e9c0a5e55afbf528328e9a34789f94cf369d370160d0433148",
  "timestamp": "1756291597563",
  "timestamp_readable": "2025-08-27 10:46:37.563",
  "delay_ms": 68,
  "hash": "a686bc5fc2b3aae6234ee7545ae09269a640e2ed",
  "best_bid": {
    "price": 0.89,
    "size": 350.0
  },
  "best_ask": {
    "price": 0.92,
    "size": 1200.0
  },
  "spread": 0.030000000000000027,
  "total_bids": 33,
  "total_asks": 8,
  "complete_book": {
    "bids": [
      {
        "price": "0.01",
        "size": "1522"
      },
      {
        "price": "0.03",
        "size": "71"
      },
      {
        "price": "0.05",
        "size": "25"
      },
      {
        "price": "0.07",
        "size": "15"
      },
      {
        "price": "0.1",
        "size": "10"
      },
      {
        "price": "0.21",
        "size": "485.83"
      },
      {
        "price": "0.25",
        "size": "10"
      },
      {
        "price": "0.31",
        "size": "5"
      },
      {
        "price": "0.33",
        "size": "5"
      },
      {
        "price": "0.34",
        "size": "253.65"
      },
      {
        "price": "0.4",
        "size": "415"
      },
      {
        "price": "0.43",
        "size": "100"
      },
      {
        "price": "0.44",
        "size": "20"
      },
      {
        "price": "0.45",
        "size": "10"
      },
      {
        "price": "0.47",
        "size": "5"
      },
      {
        "price": "0.5",
        "size": "25"
      },
      {
        "price": "0.52",
        "size": "5"
      },
      {
        "price": "0.6",
        "size": "5"
      },
      {
        "price": "0.7",
        "size": "17.5"
      },
      {
        "price": "0.75",
        "size": "35"
      },
      {
        "price": "0.77",
        "size": "101"
      },
      {
        "price": "0.78",
        "size": "101"
      },
      {
        "price": "0.79",
        "size": "106"
      },
      {
        "price": "0.8",
        "size": "15101"
      },
      {
        "price": "0.81",
        "size": "1101"
      },
      {
        "price": "0.82",
        "size": "1101"
      },
      {
        "price": "0.83",
        "size": "3101"
      },
      {
        "price": "0.84",
        "size": "101"
      },
      {
        "price": "0.85",
        "size": "1101"
      },
      {
        "price": "0.86",
        "size": "1901"
      },
      {
        "price": "0.87",
        "size": "6601"
      },
      {
        "price": "0.88",
        "size": "1309.83"
      },
      {
        "price": "0.89",
        "size": "350"
      }
    ],
    "asks": [
      {
        "price": "0.99",
        "size": "8452.4"
      },
      {
        "price": "0.98",
        "size": "1051"
      },
      {
        "price": "0.97",
        "size": "194"
      },
      {
        "price": "0.96",
        "size": "658"
      },
      {
        "price": "0.95",
        "size": "159"
      },
      {
        "price": "0.94",
        "size": "37.9"
      },
      {
        "price": "0.93",
        "size": "30"
      },
      {
        "price": "0.92",
        "size": "1200"
      }
    ]
  },
  "updated_at": 1756291597631
}"""

        put_content = """{
  "asset_name": "PUT",
  "asset_id": "85718987114555999631880959300753089573612894158613817268878391760118188940864",
  "market": "0xe9fac6cb411bb9e9c0a5e55afbf528328e9a34789f94cf369d370160d0433148",
  "timestamp": "1756291597559",
  "timestamp_readable": "2025-08-27 10:46:37.559",
  "delay_ms": 74,
  "hash": "5cdf17878496f5bae8344a42c3fdf509cc89d270",
  "best_bid": {
    "price": 0.08,
    "size": 1200.0
  },
  "best_ask": {
    "price": 0.11,
    "size": 350.0
  },
  "spread": 0.03,
  "total_bids": 8,
  "total_asks": 33,
  "complete_book": {
    "bids": [
      {
        "price": "0.01",
        "size": "8452.4"
      },
      {
        "price": "0.02",
        "size": "1051"
      },
      {
        "price": "0.03",
        "size": "194"
      },
      {
        "price": "0.04",
        "size": "658"
      },
      {
        "price": "0.05",
        "size": "159"
      },
      {
        "price": "0.06",
        "size": "5"
      },
      {
        "price": "0.07",
        "size": "30"
      },
      {
        "price": "0.08",
        "size": "1200"
      }
    ],
    "asks": [
      {
        "price": "0.99",
        "size": "1522"
      },
      {
        "price": "0.97",
        "size": "71"
      },
      {
        "price": "0.95",
        "size": "25"
      },
      {
        "price": "0.93",
        "size": "15"
      },
      {
        "price": "0.9",
        "size": "10"
      },
      {
        "price": "0.79",
        "size": "485.83"
      },
      {
        "price": "0.75",
        "size": "10"
      },
      {
        "price": "0.69",
        "size": "5"
      },
      {
        "price": "0.67",
        "size": "5"
      },
      {
        "price": "0.66",
        "size": "253.65"
      },
      {
        "price": "0.6",
        "size": "415"
      },
      {
        "price": "0.57",
        "size": "100"
      },
      {
        "price": "0.56",
        "size": "20"
      },
      {
        "price": "0.55",
        "size": "10"
      },
      {
        "price": "0.53",
        "size": "5"
      },
      {
        "price": "0.5",
        "size": "25"
      },
      {
        "price": "0.48",
        "size": "5"
      },
      {
        "price": "0.4",
        "size": "5"
      },
      {
        "price": "0.3",
        "size": "17.5"
      },
      {
        "price": "0.25",
        "size": "35"
      },
      {
        "price": "0.23",
        "size": "101"
      },
      {
        "price": "0.22",
        "size": "101"
      },
      {
        "price": "0.21",
        "size": "106"
      },
      {
        "price": "0.2",
        "size": "15101"
      },
      {
        "price": "0.19",
        "size": "1101"
      },
      {
        "price": "0.18",
        "size": "1101"
      },
      {
        "price": "0.17",
        "size": "3101"
      },
      {
        "price": "0.16",
        "size": "101"
      },
      {
        "price": "0.15",
        "size": "1101"
      },
      {
        "price": "0.14",
        "size": "1901"
      },
      {
        "price": "0.13",
        "size": "6601"
      },
      {
        "price": "0.12",
        "size": "1309.83"
      },
      {
        "price": "0.11",
        "size": "350"
      }
    ]
  },
  "updated_at": 1756291597633
}"""

        analyzer.set_call_put_data_from_content(call_content, put_content)

    # Real trader addresses with nicknames
    traders_with_nicknames = {
        "0x88712ac5d0f65592fcccb4708523c8fa6ee5830a": "NoGambleNoCrypto",
        "0x834ea21b0b55e1fa3804e2aaf5c15fe0b5648015": "XnoXno",
        "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e": "londonBridge",
        "0xc0ffa24eb2bfaeaafe11b5c1143281f7c590136b": "Abacaxi",
        "0x5487559b207c980f660774d06cd5ca144377a014":"RichlittleBoy",
        "0xd98635c6bafebb11787bdbfd80688612d1bc1320":"NorrisFan 3%",
        "0xb563eb0184543459596fd1011d013b7451600115":"anon90-merging",
        "0xb077e7482c6bea18f75e33bf823cbbe59cda8fe8":"0XNuttawut 0.7%",
        "0xeffcc79a8572940cee2238b44eac89f2c48fda88":"firstorder 2.2%",
        "0x8749194e5105c97c3d134e974e103b44eea44ea4":"0x066 1.4%",
        "0x7485d661b858b117a66e1b4fcbecfaea87ac1393":"1TickWonder 1.4%"
    }

    # Analyze all traders with nicknames
    analyzer.analyze_multiple_traders(traders_with_nicknames)

if __name__ == "__main__":
    main()
