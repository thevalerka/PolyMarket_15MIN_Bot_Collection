import requests
import time
from datetime import datetime
from collections import defaultdict

def monitor_trader_activity(user_address, hours_back=1):
    """
    Monitor trader activity and calculate weighted average costs for recent trades
    
    Args:
        user_address: Trader's wallet address
        hours_back: How many hours back to look for trades
    """
    url = f"https://data-api.polymarket.com/activity?user={user_address}&limit=500&offset=0"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Get current timestamp and calculate cutoff time
        current_time = int(time.time())
        cutoff_time = current_time - (hours_back * 3600)  # 1 hour = 3600 seconds
        
        # Filter trades from last hour
        recent_trades = [trade for trade in data if trade['timestamp'] >= cutoff_time]
        
        if not recent_trades:
            print(f"No trades found in the last {hours_back} hour(s)")
            return
        
        # Group trades by asset
        asset_trades = defaultdict(list)
        for trade in recent_trades:
            asset_id = trade['asset']
            asset_trades[asset_id].append(trade)
        
        # Calculate weighted average cost for each asset
        print(f"\nTrader Activity Analysis (Last {hours_back} hour(s)) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        
        for asset_id, trades in asset_trades.items():
            total_weighted_price = 0
            total_size = 0
            
            # Get asset info from first trade
            asset_info = trades[0]
            outcome = asset_info['outcome']
            title = asset_info['title']
            
            # Separate buys and sells for additional info
            buy_trades = [t for t in trades if t['side'] == 'BUY']
            sell_trades = [t for t in trades if t['side'] == 'SELL']
            
            # Calculate weighted average price
            for trade in trades:
                price = trade['price']
                size = trade['size']
                
                # For weighted average: sum(price * size) / sum(size)
                total_weighted_price += price * size
                total_size += size
            
            avg_weighted_cost = total_weighted_price / total_size if total_size > 0 else 0
            
            print(f"Asset ID: {asset_id}")
            print(f"Outcome: {outcome}")
            print(f"Title: {title}")
            print(f"Average Weighted Cost: ${avg_weighted_cost:.6f}")
            print(f"Total Size: {total_size:.6f}")
            print(f"Trades: {len(buy_trades)} BUY, {len(sell_trades)} SELL")
            
            # Show individual trades for transparency
            print("Individual Trades:")
            for i, trade in enumerate(trades, 1):
                trade_time = datetime.fromtimestamp(trade['timestamp']).strftime('%H:%M:%S')
                print(f"  {i}. {trade_time} | {trade['side']} | Size: {trade['size']:.6f} | Price: ${trade['price']:.6f}")
            
            print("-" * 120)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")

def continuous_monitoring(user_address, hours_back=1, check_interval=60):
    """
    Continuously monitor trader activity
    
    Args:
        user_address: Trader's wallet address
        hours_back: How many hours back to look for trades
        check_interval: How often to check (in seconds)
    """
    print(f"Starting continuous monitoring of trader: {user_address}")
    print(f"Checking every {check_interval} seconds for trades in last {hours_back} hour(s)")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            monitor_trader_activity(user_address, hours_back)
            print(f"Next check in {check_interval} seconds...")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def analyze_sample_data():
    """Analyze the provided sample data"""
    sample_data = [
        {
            "proxyWallet": "0x88712ac5d0f65592fcccb4708523c8fa6ee5830a",
            "timestamp": 1756206165,
            "conditionId": "0x389f7f0d34dba9245f4f90d7e737ae50d187cfc18f41fe70d0c850e3519ab9ef",
            "type": "TRADE",
            "size": 290.060908,
            "usdcSize": 130.527409,
            "transactionHash": "0x801384c09ec2c73ed95cb6c9c1c33f787ba856099e4b2c195329a3ce1bb2a978",
            "price": 0.450000001379021,
            "asset": "80372852584045417851755567636954939923916051546785297109561908351424375263922",
            "side": "BUY",
            "outcomeIndex": 1,
            "title": "XRP Up or Down - August 26, 7AM ET",
            "slug": "xrp-up-or-down-august-26-7am-et",
            "outcome": "Down",
            "name": "nogamblenocrypto",
            "pseudonym": "Slushy-Context"
        },
        {
            "proxyWallet": "0x88712ac5d0f65592fcccb4708523c8fa6ee5830a",
            "timestamp": 1756205807,
            "conditionId": "0xa2395804eb2898abf5941b71e707506a6dd2da6791948253e09dfbb709226281",
            "type": "TRADE",
            "size": 1.11111,
            "usdcSize": 0.111111,
            "transactionHash": "0x1a70221adb54dfebbe848091c111c9cb39147bc2409a94e14ca5ee748ea9c121",
            "price": 0.1,
            "asset": "78121632813808110132497415626479909058858808849906162782260147379361950333630",
            "side": "BUY",
            "outcomeIndex": 0,
            "title": "XRP Up or Down - August 26, 6AM ET",
            "slug": "xrp-up-or-down-august-26-6am-et",
            "outcome": "Up",
            "name": "nogamblenocrypto",
            "pseudonym": "Slushy-Context"
        }
    ]
    
    print("Analysis of Sample Data:")
    print("=" * 120)
    
    # Group by asset
    asset_trades = defaultdict(list)
    for trade in sample_data:
        asset_id = trade['asset']
        asset_trades[asset_id].append(trade)
    
    for asset_id, trades in asset_trades.items():
        total_weighted_price = 0
        total_size = 0
        
        asset_info = trades[0]
        outcome = asset_info['outcome']
        title = asset_info['title']
        
        for trade in trades:
            price = trade['price']
            size = trade['size']
            total_weighted_price += price * size
            total_size += size
        
        avg_weighted_cost = total_weighted_price / total_size if total_size > 0 else 0
        
        print(f"Asset ID: {asset_id}")
        print(f"Outcome: {outcome}")
        print(f"Title: {title}")
        print(f"Average Weighted Cost: ${avg_weighted_cost:.6f}")
        print("-" * 120)

# Example usage
if __name__ == "__main__":
    trader_address = "0x88712ac5d0f65592fcccb4708523c8fa6ee5830a"
    
    # Analyze sample data first
    analyze_sample_data()
    
    print("\n" + "="*50 + " LIVE MONITORING " + "="*50)
    
    # Single check for live data
    monitor_trader_activity(trader_address, hours_back=1)
    
    # Uncomment below for continuous monitoring (checks every 60 seconds)
    # continuous_monitoring(trader_address, hours_back=1, check_interval=60)