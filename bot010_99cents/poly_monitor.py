#!/usr/bin/env python3
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryOptionsMonitor:
    def __init__(self):
        self.data_dir = "/home/ubuntu/013_2025_polymarket/bot010_99cents/data"
        self.risk_analysis_file = os.path.join(self.data_dir, "95risk_analysis.json")
        self.put_ohlc_file = os.path.join(self.data_dir, "PUT_OHLC.json")
        self.call_ohlc_file = os.path.join(self.data_dir, "CALL_OHLC.json")

        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Tracking data for current hour
        self.current_assets = {"PUT": None, "CALL": None}
        self.threshold_trackers = {
            "PUT": {0.95: None, 0.96: None, 0.97: None, 0.98: None, 0.99: None},
            "CALL": {0.95: None, 0.96: None, 0.97: None, 0.98: None, 0.99: None}
        }

        # OHLC data for current minute
        self.current_minute_data = {
            "PUT": {"timestamp": None, "prices": [], "open": None, "high": None, "low": None, "close": None},
            "CALL": {"timestamp": None, "prices": [], "open": None, "high": None, "low": None, "close": None}
        }

        # Load existing data
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing analysis and OHLC data"""
        try:
            if os.path.exists(self.risk_analysis_file):
                with open(self.risk_analysis_file, 'r') as f:
                    self.risk_analysis_data = json.load(f)
            else:
                self.risk_analysis_data = []

            # Load OHLC data
            for asset_type, filename in [("PUT", self.put_ohlc_file), ("CALL", self.call_ohlc_file)]:
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        setattr(self, f"{asset_type.lower()}_ohlc_data", data)
                else:
                    setattr(self, f"{asset_type.lower()}_ohlc_data", [])

        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            self.risk_analysis_data = []
            self.put_ohlc_data = []
            self.call_ohlc_data = []

    def read_json_file(self, filepath: str) -> Optional[Dict]:
        """Safely read JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {e}")
            return None

    def get_minute_timestamp(self, timestamp_ms: int) -> int:
        """Get minute-level timestamp"""
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return int(dt.replace(second=0, microsecond=0).timestamp())

    def update_ohlc(self, asset_type: str, price: float, timestamp_ms: int):
        """Update OHLC data for 1-minute candles"""
        minute_ts = self.get_minute_timestamp(timestamp_ms)
        current_data = self.current_minute_data[asset_type]

        # If new minute, finalize previous minute and start new one
        if current_data["timestamp"] is None or current_data["timestamp"] != minute_ts:
            # Save previous minute if it exists
            if current_data["timestamp"] is not None and current_data["prices"]:
                ohlc_entry = {
                    "timestamp": current_data["timestamp"],
                    "datetime": datetime.fromtimestamp(current_data["timestamp"]).isoformat(),
                    "open": current_data["open"],
                    "high": current_data["high"],
                    "low": current_data["low"],
                    "close": current_data["close"]
                }

                ohlc_data = getattr(self, f"{asset_type.lower()}_ohlc_data")
                ohlc_data.append(ohlc_entry)

                # Save to file
                filename = self.put_ohlc_file if asset_type == "PUT" else self.call_ohlc_file
                self.save_json_file(filename, ohlc_data)

            # Start new minute
            current_data["timestamp"] = minute_ts
            current_data["prices"] = [price]
            current_data["open"] = price
            current_data["high"] = price
            current_data["low"] = price
            current_data["close"] = price
        else:
            # Update current minute
            current_data["prices"].append(price)
            current_data["high"] = max(current_data["high"], price)
            current_data["low"] = min(current_data["low"], price)
            current_data["close"] = price

    def check_threshold_hit(self, asset_type: str, current_price: float, timestamp_ms: int):
        """Check if any threshold is hit and start tracking"""
        thresholds = self.threshold_trackers[asset_type]

        for threshold in [0.95, 0.96, 0.97, 0.98, 0.99]:
            if thresholds[threshold] is None and current_price >= threshold:
                thresholds[threshold] = {
                    "hit_time": timestamp_ms,
                    "hit_time_readable": datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
                    "hit_price": current_price,
                    "max_drop_amount": 0.0,
                    "max_drop_percentage": 0.0,
                    "lowest_price": current_price
                }
                logger.info(f"{asset_type} hit ${threshold} threshold at price ${current_price}")

    def update_threshold_tracking(self, asset_type: str, current_price: float):
        """Update tracking for active thresholds"""
        thresholds = self.threshold_trackers[asset_type]

        for threshold, tracker in thresholds.items():
            if tracker is not None:
                # Update lowest price seen
                tracker["lowest_price"] = min(tracker["lowest_price"], current_price)

                # Calculate drop from threshold
                drop_amount = threshold - tracker["lowest_price"]
                drop_percentage = (drop_amount / threshold) * 100

                # Update max drops
                tracker["max_drop_amount"] = max(tracker["max_drop_amount"], drop_amount)
                tracker["max_drop_percentage"] = max(tracker["max_drop_percentage"], drop_percentage)

    def save_hourly_analysis(self, asset_type: str, asset_id: str, hour_end_time: int):
        """Save analysis data for completed hour"""
        thresholds = self.threshold_trackers[asset_type]

        analysis_entry = {
            "asset_type": asset_type,
            "asset_id": asset_id,
            "hour_end_time": hour_end_time,
            "hour_end_readable": datetime.fromtimestamp(hour_end_time / 1000).isoformat(),
            "thresholds": {}
        }

        # Save data for each threshold that was hit
        for threshold, tracker in thresholds.items():
            if tracker is not None:
                analysis_entry["thresholds"][str(threshold)] = {
                    "hit_time": tracker["hit_time"],
                    "hit_time_readable": tracker["hit_time_readable"],
                    "hit_price": tracker["hit_price"],
                    "lowest_price": tracker["lowest_price"],
                    "max_drop_amount": tracker["max_drop_amount"],
                    "max_drop_percentage": tracker["max_drop_percentage"]
                }

        # Only save if at least one threshold was hit
        if analysis_entry["thresholds"]:
            self.risk_analysis_data.append(analysis_entry)
            self.save_json_file(self.risk_analysis_file, self.risk_analysis_data)
            logger.info(f"Saved hourly analysis for {asset_type} asset {asset_id}")

    def reset_tracking(self, asset_type: str):
        """Reset tracking for new hour"""
        self.threshold_trackers[asset_type] = {
            0.95: None, 0.96: None, 0.97: None, 0.98: None, 0.99: None
        }

    def save_json_file(self, filepath: str, data):
        """Safely save JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")

    def process_asset_data(self, asset_type: str, data: Dict):
        """Process data for PUT or CALL asset"""
        if not data:
            return

        asset_id = data.get("asset_id")
        timestamp_ms = int(data.get("timestamp", 0))
        best_bid = data.get("best_bid", {}).get("price", 0)

        if not asset_id or not timestamp_ms or not best_bid:
            return

        # Check if this is a new asset (new hour)
        current_asset = self.current_assets[asset_type]
        if current_asset and current_asset != asset_id:
            # Save analysis for previous asset
            self.save_hourly_analysis(asset_type, current_asset, timestamp_ms)
            # Reset tracking
            self.reset_tracking(asset_type)
            logger.info(f"New {asset_type} asset detected: {asset_id}")

        self.current_assets[asset_type] = asset_id

        # Update OHLC data
        self.update_ohlc(asset_type, best_bid, timestamp_ms)

        # Check for threshold hits
        self.check_threshold_hit(asset_type, best_bid, timestamp_ms)

        # Update existing threshold tracking
        self.update_threshold_tracking(asset_type, best_bid)

    def run(self):
        """Main monitoring loop"""
        logger.info("Starting Binary Options Risk Analysis Monitor")
        logger.info(f"Monitoring files: PUT.json and CALL.json")
        logger.info(f"Saving analysis to: {self.risk_analysis_file}")
        logger.info(f"Saving OHLC to: {self.put_ohlc_file} and {self.call_ohlc_file}")

        while True:
            try:
                # Read PUT data
                put_data = self.read_json_file("/home/ubuntu/013_2025_polymarket/PUT.json")
                self.process_asset_data("PUT", put_data)

                # Read CALL data
                call_data = self.read_json_file("/home/ubuntu/013_2025_polymarket/CALL.json")
                self.process_asset_data("CALL", call_data)

                # Sleep for a short interval
                time.sleep(1)  # Check every second

            except KeyboardInterrupt:
                logger.info("Stopping monitor...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    monitor = BinaryOptionsMonitor()
    monitor.run()
