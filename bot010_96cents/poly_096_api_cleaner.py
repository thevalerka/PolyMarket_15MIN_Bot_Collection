#!/usr/bin/env python3
"""
Polymarket Token Cleaner - API VERSION

Monitors wallet activity via API and sells tokens at $1.00
- Fetches recent trades from Polymarket API
- Extracts unique asset IDs from trade history
- Checks balance for each token
- Sells at $1.00 if balance >= 5.0
- Checks every 5 minutes
"""

import os
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging

# Import the core trading functions
from polymarket_trading_core import PolymarketTrader, load_credentials_from_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolymarketAPITokenCleaner:
    """Token cleaner that gets data from Polymarket API and sells at $1.00"""

    def __init__(self):
        """Initialize the token cleaner"""
        self.trader = self._initialize_trader()
        self.api_url = 'https://data-api.polymarket.com/activity?user=0x4b43d2a40699a9a48bfadef6d85f3440817a65ab&limit=50'
        self.sell_price = 0.999  # Always sell at $1.00
        self.min_order_size = 5.0  # Polymarket minimum order size
        self.check_interval = 300  # Check every 5 minutes (300 seconds)

        logger.info("🧹 Polymarket API Token Cleaner initialized")
        logger.info(f"🌐 API URL: {self.api_url}")
        logger.info(f"💰 Sell price: ${self.sell_price:.2f}")
        logger.info(f"⏰ Check interval: {self.check_interval} seconds")

    def _initialize_trader(self) -> PolymarketTrader:
        """Initialize the Polymarket trader"""
        try:
            # Use same credentials as main bot - Updated to keys_ovh38.env
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env')
            return PolymarketTrader(
                clob_api_url=creds['clob_api_url'],
                private_key=creds['private_key'],
                api_key=creds['api_key'],
                api_secret=creds['api_secret'],
                api_passphrase=creds['api_passphrase']
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize trader: {e}")
            raise

    async def fetch_activity_data(self) -> List[Dict]:
        """Fetch activity data from Polymarket API"""
        try:
            logger.info("🌐 Fetching activity data from Polymarket API...")

            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"❌ API request failed with status {response.status}")
                        return []

                    data = await response.json()

                    if not isinstance(data, list):
                        logger.error("❌ API response is not a list")
                        return []

                    logger.info(f"✅ Fetched {len(data)} activity records")
                    return data

        except asyncio.TimeoutError:
            logger.error("❌ API request timeout")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"❌ HTTP client error: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error fetching activity data: {e}")
            return []

    def extract_asset_ids(self, activity_data: List[Dict]) -> Set[str]:
        """Extract unique asset IDs from activity data"""
        try:
            asset_ids = set()
            
            for record in activity_data:
                if not isinstance(record, dict):
                    continue

                # Extract asset ID from the record
                asset_id = record.get('asset')
                
                if asset_id and isinstance(asset_id, str):
                    asset_ids.add(asset_id)
                    
                    # Log some info about this asset
                    title = record.get('title', 'Unknown')
                    outcome = record.get('outcome', 'Unknown')
                    side = record.get('side', 'Unknown')
                    price = record.get('price', 0)
                    timestamp = record.get('timestamp', 0)
                    
                    try:
                        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        readable_time = 'Unknown time'
                    
                    logger.debug(f"📊 Found asset: {title} - {outcome} | {side} @ ${price:.4f} | {readable_time}")

            logger.info(f"📋 Extracted {len(asset_ids)} unique asset IDs from activity data")
            
            # Log first few asset IDs for debugging
            for i, asset_id in enumerate(list(asset_ids)[:3]):
                logger.info(f"🎯 Asset {i+1}: {asset_id[:20]}...")
            
            if len(asset_ids) > 3:
                logger.info(f"    ... and {len(asset_ids) - 3} more assets")

            return asset_ids

        except Exception as e:
            logger.error(f"❌ Error extracting asset IDs: {e}")
            return set()

    def check_token_balance(self, token_id: str) -> float:
        """Check current balance of a specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance

        except Exception as e:
            logger.error(f"❌ Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def sell_token(self, token_id: str, current_balance: float, token_info: str = "") -> bool:
        """Sell token at $1.00 price"""
        try:
            if current_balance < self.min_order_size:
                logger.debug(f"⏭️ Token {token_id[:12]}... balance {current_balance:.2f} below minimum ({self.min_order_size})")
                return False

            # Sell all available balance (cap at reasonable amount)
            sell_amount = min(current_balance, 1000.0)  # Cap at 1000 tokens per order

            logger.info(f"💸 Selling Token {token_id[:12]}...: {sell_amount:.2f} tokens @ ${self.sell_price:.2f}")
            if token_info:
                logger.info(f"   Token info: {token_info}")

            order_id = self.trader.place_sell_order(token_id, self.sell_price, sell_amount)

            if order_id:
                logger.info(f"✅ SELL ORDER PLACED - Order ID: {order_id}")
                logger.info(f"   Amount: {sell_amount:.2f} tokens")
                logger.info(f"   Price: ${self.sell_price:.2f}")
                logger.info(f"   Total Value: ${sell_amount * self.sell_price:.2f}")
                return True
            else:
                logger.error(f"❌ SELL ORDER FAILED for token {token_id[:12]}...")
                return False

        except Exception as e:
            logger.error(f"❌ Error selling token {token_id[:12]}...: {e}")
            return False

    def get_token_info_from_activity(self, token_id: str, activity_data: List[Dict]) -> str:
        """Get human-readable token info from activity data"""
        try:
            for record in activity_data:
                if record.get('asset') == token_id:
                    title = record.get('title', 'Unknown')
                    outcome = record.get('outcome', 'Unknown')
                    return f"{title} - {outcome}"
            return f"Token {token_id[:12]}..."
        except:
            return f"Token {token_id[:12]}..."

    async def process_tokens(self) -> int:
        """Process all tokens from API data"""
        try:
            # Fetch activity data from API
            activity_data = await self.fetch_activity_data()

            if not activity_data:
                logger.info("🔍 No activity data available")
                return 0

            # Extract unique asset IDs
            asset_ids = self.extract_asset_ids(activity_data)

            if not asset_ids:
                logger.info("🔍 No asset IDs found in activity data")
                return 0

            tokens_processed = 0

            logger.info(f"🔍 Processing {len(asset_ids)} unique tokens from API data...")

            for token_id in asset_ids:
                try:
                    # Get human-readable token info
                    token_info = self.get_token_info_from_activity(token_id, activity_data)

                    logger.info(f"📊 Checking balance for: {token_info}")

                    # Check current balance
                    current_balance = self.check_token_balance(token_id)

                    if current_balance >= self.min_order_size:
                        logger.info(f"💰 {token_info} has {current_balance:.2f} tokens - attempting to sell")
                        
                        # Try to sell the token
                        sell_success = self.sell_token(token_id, current_balance, token_info)

                        if sell_success:
                            tokens_processed += 1
                            logger.info(f"✅ Successfully placed sell order for {token_info}")
                            
                            # Add small delay between sells to avoid rate limits
                            await asyncio.sleep(2)

                    elif current_balance > 0:
                        logger.info(f"📊 {token_info} has {current_balance:.2f} tokens (below minimum {self.min_order_size})")
                    else:
                        logger.debug(f"📊 {token_info} has zero balance")

                except Exception as e:
                    logger.error(f"❌ Error processing token {token_id[:12]}...: {e}")
                    continue

            return tokens_processed

        except Exception as e:
            logger.error(f"❌ Error processing tokens: {e}")
            return 0

    async def run(self):
        """Main cleaner loop"""
        logger.info("🧹 Starting Polymarket API Token Cleaner")
        logger.info(f"⏰ Check interval: {self.check_interval} seconds ({self.check_interval/60:.1f} minutes)")

        while True:
            try:
                current_time = datetime.now().strftime('%H:%M:%S')
                logger.info(f"🔄 Starting token processing cycle... ({current_time})")

                # Process tokens from API
                tokens_processed = await self.process_tokens()

                if tokens_processed > 0:
                    logger.info(f"✅ Processed {tokens_processed} tokens successfully")
                else:
                    logger.info("ℹ️ No tokens were processed this cycle")

                # Wait before next cycle
                logger.info(f"⏳ Waiting {self.check_interval} seconds until next check...")
                await asyncio.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("🛑 Cleaner shutdown requested")
                break
            except Exception as e:
                logger.error(f"❌ Error in main cleaner loop: {e}")
                logger.info("⏳ Waiting 60 seconds before retry...")
                await asyncio.sleep(60)  # Wait 1 minute on error

        logger.info("🧹 Token cleaner stopped")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    try:
        # Initialize token cleaner
        cleaner = PolymarketAPITokenCleaner()

        # Run the cleaner
        await cleaner.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    print("🧹 POLYMARKET API TOKEN CLEANER")
    print("=" * 50)
    print("FUNCTIONALITY:")
    print("• Fetches wallet activity from Polymarket API")
    print("• Extracts asset IDs from recent trade history")
    print("• Checks balance for each asset (ERC1155 NFT)")
    print("• Sells at $1.00 if balance >= 5 tokens")
    print("• Processes all unique tokens from API data")
    print("• Checks every 5 minutes")
    print()
    print("PROCESS:")
    print("• Fetch activity data from API endpoint")
    print("• Extract unique asset IDs from trades")
    print("• Check balance for each asset")
    print("• Sell tokens with sufficient balance")
    print("• Wait and repeat")
    print()
    print("API ENDPOINT:")
    print("• https://data-api.polymarket.com/activity")
    print("• user=0x4b43d2a40699a9a48bfadef6d85f3440817a65ab")
    print("• limit=50 (last 50 trades)")
    print()
    print("FILES:")
    print("• Credentials: keys/keys_ovh38.env")
    print("• Tokens are ERC1155 NFTs")
    print("• No local file storage needed")
    print()
    print("⚠️  WARNING: This will sell all tokens with balance >= 5 at $1.00!")
    print("⚠️  Make sure the API endpoint is accessible!")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Cleaner stopped by user")
    except Exception as e:
        print(f"\n❌ Cleaner failed: {e}")
