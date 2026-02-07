# Polymarket Position Redeemer - Asset List Version
# Redeems specific asset IDs from ASSETLIST.json regardless of value
"""
Requirements: Python 3.12+, pip install polymarket_apis python-dotenv
Usage: Add PRIVATE_KEY to .env file, then run python redeemer40_assetlist.py
"""

import os
import json
from dotenv import load_dotenv
from polymarket_apis import PolymarketWeb3Client, PolymarketDataClient
import time

ASSET_LIST_PATH = '/home/ubuntu/013_2025_polymarket/ASSETLIST.json'
ENV_PATH = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'


def load_asset_list():
    """Load asset IDs from JSON file"""
    with open(ASSET_LIST_PATH, 'r') as f:
        return json.load(f)


def main():
    load_dotenv(ENV_PATH)
    private_key = os.getenv("PK")
    if not private_key:
        raise Exception("PRIVATE_KEY not found in .env")

    print("💰 Polymarket Position Redeemer (Asset List Version)")
    print("=" * 50)

    # Load target asset IDs
    asset_ids = load_asset_list()
    print(f"📋 Loaded {len(asset_ids)} asset IDs from ASSETLIST.json")

    # Initialize clients
    web3 = PolymarketWeb3Client(private_key=private_key, signature_type=0)
    data = PolymarketDataClient()

    print(f"👤 Wallet: {web3.address}")

    # Get all positions to find condition_ids for our asset_ids
    positions = data.get_positions(web3.address, limit=1000)

    if not positions:
        print("❌ No positions found for wallet")
        return

    print(f"📊 Found {len(positions)} total positions in wallet")

    # Filter positions that match our asset IDs
    target_positions = []
    for pos in positions:
        # Check if position's asset_id matches any in our list
        asset_id = str(pos.asset_id) if hasattr(pos, 'asset_id') else None
        token_id = str(pos.token_id) if hasattr(pos, 'token_id') else None
        
        if asset_id in asset_ids or token_id in asset_ids:
            target_positions.append(pos)
            print(f"   ✓ Found matching position: {asset_id or token_id}")

    if not target_positions:
        print("❌ No positions matching asset list found")
        print("\n🔍 Attempting direct redemption with asset IDs as condition IDs...")
        
        # Try redeeming directly using asset IDs (they might be condition IDs)
        success_count = 0
        for i, asset_id in enumerate(asset_ids, 1):
            print(f"\n{i}. Attempting to redeem asset: {asset_id[:20]}...")
            
            try:
                # Try with default amounts - redemption typically uses the full balance
                # For binary markets, try both outcomes
                for neg_risk in [False, True]:
                    try:
                        result = web3.redeem_position(
                            condition_id=asset_id,
                            amounts=[1, 1],  # Will be adjusted by contract
                            neg_risk=neg_risk
                        )
                        if result:
                            print(f"   ✅ SUCCESS! Transaction: {result}")
                            success_count += 1
                            break
                    except Exception as e:
                        if "revert" not in str(e).lower():
                            print(f"   ⚠️ Attempt with neg_risk={neg_risk}: {e}")
                        continue
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
        
        print(f"\n🎉 Direct redemption completed! {success_count}/{len(asset_ids)} successful")
        return

    print(f"\n🎯 Found {len(target_positions)} positions matching asset list:")

    for i, pos in enumerate(target_positions, 1):
        try:
            value = pos.current_value if hasattr(pos, 'current_value') else 0
            size = pos.size if hasattr(pos, 'size') else 'N/A'
            asset_id = pos.asset_id if hasattr(pos, 'asset_id') else pos.token_id
            print(f"   {i}. Asset: {str(asset_id)[:20]}... | Value: ${value:.2f} | Size: {size}")
        except Exception as e:
            print(f"   {i}. Error displaying position: {e}")

    print("\n💰 Starting redemption process...")

    success_count = 0
    for i, pos in enumerate(target_positions, 1):
        asset_id = pos.asset_id if hasattr(pos, 'asset_id') else pos.token_id
        value = pos.current_value if hasattr(pos, 'current_value') else 0
        print(f"\n{i}. Redeeming asset {str(asset_id)[:20]}... (value: ${value:.2f})")

        try:
            # Prepare amounts array
            amounts = [0, 0]
            if hasattr(pos, 'outcome_index') and hasattr(pos, 'size'):
                amounts[pos.outcome_index] = pos.size

            print(f"   📊 Amounts: {amounts}")
            print(f"   🎯 Condition: {pos.condition_id}")

            neg_risk = pos.negative_risk if hasattr(pos, 'negative_risk') else False
            print(f"   ⚠️ Neg Risk: {neg_risk}")

            # Redeem position
            result = web3.redeem_position(
                condition_id=pos.condition_id,
                amounts=amounts,
                neg_risk=neg_risk
            )

            if result:
                print(f"   ✅ SUCCESS! Transaction: {result}")
                success_count += 1
            else:
                print(f"   ❌ FAILED! No transaction returned")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    print(f"\n🎉 Redemption completed! {success_count}/{len(target_positions)} successful")


if __name__ == "__main__":
    print("🔄 Starting asset list redeemer (runs every 60 seconds)")
    print("Press Ctrl+C to stop")

    while True:
        try:
            print(f"\n{'='*50}")
            print(f"⏰ Run started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            main()
            print(f"⏰ Waiting 60 seconds until next run...")
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n👋 Redeemer stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            print(f"⏰ Waiting 60 seconds before retry...")
            time.sleep(60)
