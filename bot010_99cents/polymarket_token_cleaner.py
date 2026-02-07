#!/usr/bin/env python3
"""
Polymarket Token Cleaner - SIMPLIFIED VERSION

Monitors traded_tokens.json and sells tokens at $1.00
- Scans all tokens in the file
- Checks balance for each token
- Sells at $1.00 if balance >= 5.0
- Only deletes tokens older than 24 hours
- Checks every 1 minute if tokens exist, sleeps 10 minutes if none
"""

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import the core trading functions
from polymarket_trading_core import PolymarketTrader, load_credentials_from_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolymarketTokenCleaner:
    """Token cleaner that sells all tracked tokens at $1.00"""

    def __init__(self):
        """Initialize the token cleaner"""
        self.trader = self._initialize_trader()
        self.tokens_file = '/home/ubuntu/013_2025_polymarket/bot010_99cents/traded_tokens.json'
        self.sell_price = 0.999  # Always sell at $1.00
        self.min_order_size = 5.0  # Polymarket minimum order size
        self.token_expiry_hours = 24  # Delete tokens after 24 hours

        logger.info("🧹 Polymarket Token Cleaner initialized (SIMPLIFIED)")
        logger.info(f"📁 Monitoring file: {self.tokens_file}")
        logger.info(f"💰 Sell price: ${self.sell_price:.2f}")
        logger.info(f"⏰ Token expiry: {self.token_expiry_hours} hours")

    def _initialize_trader(self) -> PolymarketTrader:
        """Initialize the Polymarket trader"""
        try:
            # Use same credentials as main bot
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env')
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

    def load_tokens(self) -> List[Dict]:
        """Load tokens from JSON file"""
        try:
            if not os.path.exists(self.tokens_file):
                logger.warning(f"⚠️ Tokens file not found: {self.tokens_file}")
                return []

            with open(self.tokens_file, 'r') as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"⚠️ Empty tokens file: {self.tokens_file}")
                return []

            data = json.loads(content)

            if not data:
                logger.warning(f"⚠️ No data in tokens file: {self.tokens_file}")
                return []

            tokens = data.get('tokens', [])
            logger.info(f"📄 Loaded {len(tokens)} token records")
            return tokens

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in tokens file: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading tokens: {e}")
            return []

    def save_tokens(self, tokens: List[Dict]):
        """Save tokens back to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.tokens_file), exist_ok=True)

            data = {'tokens': tokens}

            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"💾 Saved {len(tokens)} token records")

        except Exception as e:
            logger.error(f"❌ Error saving tokens: {e}")

    def is_token_expired(self, token_timestamp: float) -> bool:
        """Check if token is older than 24 hours"""
        try:
            current_time = time.time()
            token_age_hours = (current_time - token_timestamp) / 3600  # Convert to hours
            return token_age_hours > self.token_expiry_hours
        except Exception as e:
            logger.error(f"❌ Error checking token expiry: {e}")
            return False

    def check_token_balance(self, token_id: str, token_name: str) -> float:
        """Check current balance of a specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            logger.info(f"💰 {token_name} ({token_id[:12]}...) balance: {balance:.2f} tokens")
            return balance

        except Exception as e:
            logger.error(f"❌ Error checking balance for {token_name} ({token_id[:12]}...): {e}")
            return 0.0

    def sell_token(self, token_id: str, token_name: str, current_balance: float) -> bool:
        """Sell token at $1.00 price"""
        try:
            if current_balance < self.min_order_size:
                logger.info(f"⚠️ {token_name} balance {current_balance:.2f} below minimum order size ({self.min_order_size})")
                return False

            # Sell all available balance (cap at reasonable amount)
            sell_amount = min(current_balance, 1000.0)  # Cap at 1000 tokens per order

            logger.info(f"💸 Selling {token_name}: {sell_amount:.2f} tokens @ ${self.sell_price:.2f}")
            logger.info(f"   Token ID: {token_id[:20]}...")

            order_id = self.trader.place_sell_order(token_id, self.sell_price, sell_amount)

            if order_id:
                logger.info(f"✅ SELL ORDER PLACED: {token_name} - Order ID: {order_id}")
                logger.info(f"   Amount: {sell_amount:.2f} tokens")
                logger.info(f"   Price: ${self.sell_price:.2f}")
                logger.info(f"   Total Value: ${sell_amount * self.sell_price:.2f}")
                return True
            else:
                logger.error(f"❌ SELL ORDER FAILED: {token_name}")
                return False

        except Exception as e:
            logger.error(f"❌ Error selling {token_name}: {e}")
            return False

    def process_tokens(self) -> int:
        """Process all tokens in the file"""
        try:
            # Load current tokens
            tokens = self.load_tokens()

            if not tokens:
                logger.info("📭 No tokens to process")
                return 0

            tokens_processed = 0
            tokens_to_remove = []
            current_time = time.time()

            logger.info(f"🔍 Processing {len(tokens)} tracked tokens...")

            for i, token_info in enumerate(tokens):
                try:
                    token_id = token_info.get('token_id')
                    token_name = token_info.get('token_name', 'UNKNOWN')
                    token_timestamp = token_info.get('timestamp', 0)
                    token_datetime = token_info.get('datetime', 'unknown')

                    if not token_id:
                        logger.warning(f"⚠️ Skipping token {i} - missing token_id")
                        tokens_to_remove.append(i)
                        continue

                    # Check if token is expired (older than 24 hours)
                    if self.is_token_expired(token_timestamp):
                        token_age_hours = (current_time - token_timestamp) / 3600
                        logger.info(f"🗑️ {token_name} expired ({token_age_hours:.1f}h old), removing from tracking")
                        tokens_to_remove.append(i)
                        continue

                    logger.info(f"📊 Processing {token_name} (added: {token_datetime})")

                    # Check current balance
                    current_balance = self.check_token_balance(token_id, token_name)

                    # Try to sell the token if we have enough balance
                    if current_balance >= self.min_order_size:
                        sell_success = self.sell_token(token_id, token_name, current_balance)

                        if sell_success:
                            tokens_processed += 1
                            logger.info(f"✅ Successfully placed sell order for {token_name}")

                    elif current_balance > 0:
                        logger.info(f"📊 {token_name} has {current_balance:.2f} tokens (below minimum {self.min_order_size})")
                    else:
                        logger.info(f"📊 {token_name} has zero balance")

                except Exception as e:
                    logger.error(f"❌ Error processing token {i}: {e}")
                    continue

            # Remove expired tokens from the list (in reverse order to maintain indices)
            if tokens_to_remove:
                logger.info(f"🗑️ Removing {len(tokens_to_remove)} expired tokens...")
                for i in sorted(tokens_to_remove, reverse=True):
                    try:
                        removed_token = tokens.pop(i)
                        logger.info(f"🗑️ Removed: {removed_token.get('token_name', 'UNKNOWN')}")
                    except (IndexError, KeyError) as e:
                        logger.error(f"❌ Error removing token at index {i}: {e}")

                # Save updated tokens list
                self.save_tokens(tokens)
                logger.info(f"🧹 Cleaned up {len(tokens_to_remove)} expired tokens")

            return tokens_processed

        except Exception as e:
            logger.error(f"❌ Error processing tokens: {e}")
            return 0

    async def run(self):
        """Main cleaner loop"""
        logger.info("🧹 Starting Polymarket Token Cleaner (SIMPLIFIED)")
        logger.info(f"⏰ Check interval: 1 minute")
        logger.info(f"😴 Sleep interval (if no tokens): 10 minutes")
        logger.info(f"🗑️ Token cleanup: Remove after {self.token_expiry_hours} hours")

        while True:
            try:
                current_time = datetime.now().strftime('%H:%M:%S')
                logger.info(f"🔄 Checking for tokens to process... ({current_time})")

                # Process tokens
                tokens_processed = self.process_tokens()

                if tokens_processed > 0:
                    logger.info(f"✅ Processed {tokens_processed} tokens, checking again in 1 minute")
                    await asyncio.sleep(60)  # Check again in 1 minute if we did work
                else:
                    # Load tokens to see if there are any left
                    tokens = self.load_tokens()

                    if not tokens:
                        logger.info("😴 No tokens to process, sleeping for 10 minutes...")
                        await asyncio.sleep(600)  # Sleep 10 minutes if no tokens
                    else:
                        logger.info(f"⏳ {len(tokens)} tokens tracked but none processable now, checking again in 1 minute")
                        await asyncio.sleep(60)  # Check again in 1 minute

            except KeyboardInterrupt:
                logger.info("🛑 Cleaner shutdown requested")
                break
            except Exception as e:
                logger.error(f"❌ Error in main cleaner loop: {e}")
                logger.info("⏳ Waiting 1 minute before retry...")
                await asyncio.sleep(60)  # Wait 1 minute on error

        logger.info("🧹 Token cleaner stopped")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    try:
        # Initialize token cleaner
        cleaner = PolymarketTokenCleaner()

        # Run the cleaner
        await cleaner.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    print("🧹 POLYMARKET TOKEN CLEANER - SIMPLIFIED")
    print("=" * 50)
    print("FUNCTIONALITY:")
    print("• Monitors bot010_99cents/traded_tokens.json file")
    print("• Scans ALL tokens in the file")
    print("• Checks balance for each token (ERC1155 NFT)")
    print("• Sells at $1.00 if balance >= 5 tokens")
    print("• Removes tokens older than 24 hours only")
    print("• Checks every 1 minute if tokens exist")
    print("• Sleeps 10 minutes if no tokens to process")
    print()
    print("PROCESS:")
    print("• Load all token records from JSON")
    print("• Check balance for each token")
    print("• Sell tokens with sufficient balance")
    print("• Clean up expired tokens (>24h old)")
    print("• Save updated JSON file")
    print()
    print("FILES:")
    print("• Input: bot010_99cents/traded_tokens.json")
    print("• Credentials: keys/keys_ovh39.env")
    print("• Tokens are ERC1155 NFTs")
    print()
    print("⚠️  WARNING: This will sell all tokens with balance >= 5 at $1.00!")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Cleaner stopped by user")
    except Exception as e:
        print(f"\n❌ Cleaner failed: {e}")
