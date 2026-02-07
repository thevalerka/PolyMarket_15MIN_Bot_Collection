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
        directional_traders = []  # For final clean summary

        for trader_address, trader_info in all_results.items():
            nickname = trader_info['nickname']
            positions = trader_info['positions']

            if positions:
                total_traders_with_positions += 1
                trader_calls = 0
                trader_puts = 0
                trader_call_value = 0
                trader_put_value = 0
                trader_call_size = 0
                trader_put_size = 0
                trader_call_avg_price = 0
                trader_put_avg_price = 0

                for pos_data in positions:
                    if pos_data['type'] == "UP/CALL":
                        total_call_positions += 1
                        trader_calls += 1
                        pos_value = pos_data['position']['currentValue']
                        pos_size = pos_data['position']['size']
                        pos_avg_price = pos_data['position']['avgPrice']
                        total_call_value += pos_value
                        trader_call_value += pos_value
                        trader_call_size += pos_size
                        trader_call_avg_price = pos_avg_price  # For single positions
                    else:
                        total_put_positions += 1
                        trader_puts += 1
                        pos_value = pos_data['position']['currentValue']
                        pos_size = pos_data['position']['size']
                        pos_avg_price = pos_data['position']['avgPrice']
                        total_put_value += pos_value
                        trader_put_value += pos_value
                        trader_put_size += pos_size
                        trader_put_avg_price = pos_avg_price  # For single positions

                # Store trader breakdown for detailed summary
                trader_breakdown.append({
                    'nickname': nickname,
                    'address': trader_address,
                    'calls': trader_calls,
                    'puts': trader_puts,
                    'call_value': trader_call_value,
                    'put_value': trader_put_value,
                    'call_size': trader_call_size,
                    'put_size': trader_put_size,
                    'call_avg_price': trader_call_avg_price,
                    'put_avg_price': trader_put_avg_price
                })

                # Add to directional traders (exclude MIXED)
                if trader_calls > 0 and trader_puts == 0:
                    # Pure BULLISH trader
                    directional_traders.append({
                        'nickname': nickname,
                        'direction': 'UP',
                        'avg_price': trader_call_avg_price,
                        'total_size': trader_call_size,
                        'emoji': '🟢'
                    })
                elif trader_puts > 0 and trader_calls == 0:
                    # Pure BEARISH trader
                    directional_traders.append({
                        'nickname': nickname,
                        'direction': 'DOWN',
                        'avg_price': trader_put_avg_price,
                        'total_size': trader_put_size,
                        'emoji': '🔴'
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

        # FINAL DIRECTIONAL SUMMARY (excluding MIXED traders)
        if directional_traders:
            print(f"\n{'🎯 DIRECTIONAL TRADERS ONLY (NO MIXED POSITIONS)'}")
            print(f"{'='*80}")

            # Sort by direction (UP first, then DOWN) and then by size
            directional_traders.sort(key=lambda x: (x['direction'] == 'DOWN', -x['total_size']))

            print(f"{'TRADER':<20} {'DIRECTION':<12} {'AVG PRICE':<12} {'SIZE':<15}")
            print(f"-" * 80)

            up_traders = []
            down_traders = []

            for trader in directional_traders:
                direction_display = f"{trader['emoji']} {trader['direction']}"
                print(f"{trader['nickname']:<20} {direction_display:<12} ${trader['avg_price']:<11.6f} {trader['total_size']:<15.6f}")

                if trader['direction'] == 'UP':
                    up_traders.append(trader)
                else:
                    down_traders.append(trader)

            print(f"-" * 80)
            print(f"🟢 PURE BULLISH TRADERS: {len(up_traders)}")
            print(f"🔴 PURE BEARISH TRADERS: {len(down_traders)}")

            if len(up_traders) + len(down_traders) > 0:
                pure_bull_pct = (len(up_traders) / (len(up_traders) + len(down_traders))) * 100
                pure_bear_pct = (len(down_traders) / (len(up_traders) + len(down_traders))) * 100
                print(f"📊 PURE DIRECTIONAL SENTIMENT: {pure_bull_pct:.1f}% UP vs {pure_bear_pct:.1f}% DOWN")

            # Show total sizes
            total_up_size = sum(t['total_size'] for t in up_traders)
            total_down_size = sum(t['total_size'] for t in down_traders)
            print(f"📏 TOTAL UP SIZE: {total_up_size:.6f}")
            print(f"📏 TOTAL DOWN SIZE: {total_down_size:.6f}")

        else:
            print(f"\n❌ NO PURE DIRECTIONAL TRADERS FOUND (All traders have mixed positions)")



# Example usage
def main():
    # Initialize analyzer with real data files only
    analyzer = PolymarketPositionAnalyzer(
        call_data_file='/home/ubuntu/013_2025_polymarket/CALL.json',
        put_data_file='/home/ubuntu/013_2025_polymarket/PUT.json'
    )

    # Check if data loaded successfully
    if not analyzer.call_data or not analyzer.put_data:
        print("❌ ERROR: Could not load CALL.json or PUT.json files!")
        print("Make sure the files exist at:")
        print("  - /home/ubuntu/013_2025_polymarket/CALL.json")
        print("  - /home/ubuntu/013_2025_polymarket/PUT.json")
        return

    print("✅ Successfully loaded real CALL and PUT data")

    # Real trader addresses with nicknames
    traders_with_nicknames = {
        "0x88712ac5d0f65592fcccb4708523c8fa6ee5830a": "NoGambleNoCrypto",
        "0x834ea21b0b55e1fa3804e2aaf5c15fe0b5648015": "XnoXno",
        "0xfb1c3c1ab4fb2d0cbcbb9538c8d4d357dd95963e": "londonBridge 0.56%",
        "0xc0ffa24eb2bfaeaafe11b5c1143281f7c590136b": "Abacaxi",
        "0x5487559b207c980f660774d06cd5ca144377a014":"RichlittleBoy 3%",
        "0xd98635c6bafebb11787bdbfd80688612d1bc1320":"NorrisFan 3%",
        "0xb563eb0184543459596fd1011d013b7451600115":"anon90-merging",
        "0xb077e7482c6bea18f75e33bf823cbbe59cda8fe8":"0XNuttawut 0.7%",
        "0xeffcc79a8572940cee2238b44eac89f2c48fda88":"firstorder 2.2%",
        "0x8749194e5105c97c3d134e974e103b44eea44ea4":"0x066 1.4%",
        "0x7485d661b858b117a66e1b4fcbecfaea87ac1393":"1TickWonder 1.4%",
        "0x118ed6e1e562ec54f1fa6e7f0a2741fe0d32ea4b":"Selllowbuyhigh 4,4%",
        "0x1a4249cd596a8e51b267dfe3c56cacc25815a00b":"Lambsauce 2.8%",
        "0xc631d9d610b9939f0b915b1916864e9b806876f6" : "EGH 1.75%",
        "0x136f87c54565a35c42b9af5b7cffe678425468c3": "Wulala 0x136",
        "0x8aa57ac47693dd9576d90ecc3a960ef00bc63c24": "GregSharp 2.8%"
            }

    # Analyze all traders with nicknames
    analyzer.analyze_multiple_traders(traders_with_nicknames)

if __name__ == "__main__":
    main()
