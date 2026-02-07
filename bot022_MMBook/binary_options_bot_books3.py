#!/usr/bin/env python3
"""
15-Minute Binary Options Simulation Bot v2
===========================================
Strategy: Order Book Imbalance with Period Management

Rules:
- BUY when: sum(top4_bids) > sum(top4_asks) × 2
- SELL when: sum(top4_asks) > sum(top4_bids) × 2 OR price >= 0.99
- NO BUY if: price > 0.95 OR price < 0.05
- Period ends at minute 00, 15, 30, 45
- Last 10 seconds: SELL all positions, NO new buys
- Reset everything at period end
- Track PNL per period and overall
"""

import json
import time
import os
import math
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from pathlib import Path

# Directory for trade logs
TRADES_LOG_DIR = "/home/ubuntu/013_2025_polymarket/bot022_MMBook/trades_logs3"


@dataclass
class Position:
    asset_type: str  # 'CALL' or 'PUT'
    entry_price: float
    entry_time: datetime
    size: float = 100.0
    choppiness: Optional[float] = None  # Choppiness at entry
    volatility: Optional[float] = None  # Volatility at entry


@dataclass
class Trade:
    trade_id: int
    period_id: str
    asset_type: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'signal', 'price_limit', 'period_end', 'last_10s'
    choppiness: Optional[float] = None  # Choppiness index at entry
    volatility: Optional[float] = None  # Volatility (ATR%) at entry


@dataclass
class PeriodStats:
    period_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    trades: List[Trade] = field(default_factory=list)
    pnl: float = 0.0
    wins: int = 0
    losses: int = 0


class BinaryOptionsBot:
    def __init__(self, call_file: str, put_file: str,
                 imbalance_ratio: float = 2.6,
                 position_size: float = 100.0,
                 trades_log_dir: str = TRADES_LOG_DIR):
        self.call_file = call_file
        self.put_file = put_file
        self.imbalance_ratio = imbalance_ratio
        self.position_size = position_size
        self.trades_log_dir = trades_log_dir

        # Create trades log directory
        os.makedirs(self.trades_log_dir, exist_ok=True)

        # Positions
        self.call_position: Optional[Position] = None
        self.put_position: Optional[Position] = None

        # Tracking
        self.all_trades: List[Trade] = []
        self.periods: Dict[str, PeriodStats] = {}
        self.current_period_id: Optional[str] = None
        self.trade_counter = 0
        self.iteration = 0

        # Price limits
        self.MAX_BUY_PRICE = 0.95
        self.MIN_BUY_PRICE = 0.05
        self.AUTO_SELL_PRICE = 0.99
        self.PERIOD_END_BUFFER_SECONDS = 10

    def calculate_choppiness(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Choppiness Index"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return 0.0

        highs = highs[-period:]
        lows = lows[-period:]
        closes = closes[-period:]

        high_max = max(highs)
        low_min = min(lows)

        atr_sum = 0.0
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            atr_sum += true_range

        if high_max - low_min == 0:
            return 0.0

        chop = 100 * math.log10(atr_sum / (high_max - low_min)) / math.log10(period)
        return chop

    def calculate_volatility(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate volatility as ATR percentage"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return 0.0

        atr_sum = 0.0
        count = 0
        for i in range(1, min(period + 1, len(closes))):
            high_low = highs[-i] - lows[-i]
            high_close = abs(highs[-i] - closes[-i-1]) if i < len(closes) else 0
            low_close = abs(lows[-i] - closes[-i-1]) if i < len(closes) else 0
            true_range = max(high_low, high_close, low_close)
            atr_sum += true_range
            count += 1

        if count == 0:
            return 0.0

        atr = atr_sum / count
        current_price = closes[-1]

        if current_price == 0:
            return 0.0

        # Return ATR as percentage of current price
        volatility_pct = (atr / current_price) * 100
        return volatility_pct

    def get_market_metrics(self) -> tuple[Optional[float], Optional[float]]:
        """Get choppiness and volatility from Binance historical data"""
        try:
            # Get historical data for choppiness from Binance
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'limit': 30  # Need at least 14 for choppiness
            }
            response = requests.get(url, params=params, timeout=5)

            if response.status_code != 200:
                return None, None

            klines = response.json()

            if len(klines) < 14:
                return None, None

            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]

            choppiness = self.calculate_choppiness(highs, lows, closes)
            volatility = self.calculate_volatility(highs, lows, closes)

            return choppiness, volatility

        except Exception as e:
            print(f"  ⚠️  Failed to get market metrics: {e}")
            return None, None

    def get_period_id(self, dt: datetime) -> str:
        """Get period identifier (e.g., '2026-01-15_10:15')"""
        minute = (dt.minute // 15) * 15
        period_start = dt.replace(minute=minute, second=0, microsecond=0)
        return period_start.strftime('%Y-%m-%d_%H:%M')

    def get_period_end(self, dt: datetime) -> datetime:
        """Get the end time of current period"""
        minute = (dt.minute // 15) * 15
        period_start = dt.replace(minute=minute, second=0, microsecond=0)
        return period_start + timedelta(minutes=15)

    def get_seconds_to_period_end(self, dt: datetime) -> float:
        """Get seconds remaining until period ends"""
        period_end = self.get_period_end(dt)
        delta = period_end - dt
        return delta.total_seconds()

    def is_last_10_seconds(self, dt: datetime) -> bool:
        """Check if we're in the last 10 seconds of period"""
        return self.get_seconds_to_period_end(dt) <= self.PERIOD_END_BUFFER_SECONDS

    def is_new_period(self, dt: datetime) -> bool:
        """Check if this is a new period"""
        period_id = self.get_period_id(dt)
        return period_id != self.current_period_id

    def save_period_trades(self, period_id: str):
        """Save trades for a period to JSON file"""
        if period_id not in self.periods:
            return

        period = self.periods[period_id]

        # Build the data structure
        period_data = {
            "period_id": period_id,
            "start_time": period.start_time.isoformat() if period.start_time else None,
            "end_time": period.end_time.isoformat() if period.end_time else None,
            "summary": {
                "total_trades": len(period.trades),
                "wins": period.wins,
                "losses": period.losses,
                "pnl": round(period.pnl, 2),
                "win_rate": round(period.wins / len(period.trades) * 100, 1) if period.trades else 0
            },
            "trades": []
        }

        for trade in period.trades:
            trade_data = {
                "trade_id": trade.trade_id,
                "asset_type": trade.asset_type,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "size": trade.size,
                "pnl": round(trade.pnl, 2),
                "pnl_percent": round(trade.pnl_percent, 2),
                "exit_reason": trade.exit_reason,
                "choppiness": round(trade.choppiness, 2) if trade.choppiness is not None else None,
                "volatility": round(trade.volatility, 3) if trade.volatility is not None else None
            }
            period_data["trades"].append(trade_data)

        # Create filename: trades_2026-01-15_10-00.json
        safe_period_id = period_id.replace(":", "-")
        filename = f"trades_{safe_period_id}.json"
        filepath = os.path.join(self.trades_log_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(period_data, f, indent=2)
            print(f"\n  💾 Trades saved to: {filepath}")
        except Exception as e:
            print(f"\n  ❌ Failed to save trades: {e}")

        return filepath

    def start_new_period(self, dt: datetime):
        """Initialize a new trading period"""
        period_id = self.get_period_id(dt)

        # Close previous period and save trades
        if self.current_period_id and self.current_period_id in self.periods:
            self.periods[self.current_period_id].end_time = dt
            self.save_period_trades(self.current_period_id)

        # Start new period
        self.current_period_id = period_id
        period_start = dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)

        self.periods[period_id] = PeriodStats(
            period_id=period_id,
            start_time=period_start
        )

        # Reset positions for new period
        self.call_position = None
        self.put_position = None

        print(f"\n{'='*70}")
        print(f"🆕 NEW PERIOD: {period_id}")
        print(f"   Period ends at: {self.get_period_end(dt).strftime('%H:%M:%S')}")
        print(f"{'='*70}")

    def load_order_book(self, filepath: str) -> Optional[Dict]:
        """Load and parse order book from JSON file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            depth = 0
            end = 0
            for i, char in enumerate(content):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            return json.loads(content[:end])
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def analyze_book(self, book: Dict) -> Dict:
        """Analyze order book and return metrics"""
        complete_book = book.get('complete_book', {})

        # Get top 4 bids (highest prices)
        bids = sorted(
            [(float(l['price']), float(l['size'])) for l in complete_book.get('bids', [])],
            key=lambda x: x[0], reverse=True
        )[:4]

        # Get top 4 asks (lowest prices)
        asks = sorted(
            [(float(l['price']), float(l['size'])) for l in complete_book.get('asks', [])],
            key=lambda x: x[0]
        )[:4]

        sum_bids = sum(size for _, size in bids)
        sum_asks = sum(size for _, size in asks)

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 1

        return {
            'bids': bids,
            'asks': asks,
            'sum_bids': sum_bids,
            'sum_asks': sum_asks,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid,
            'bid_ask_ratio': sum_bids / sum_asks if sum_asks > 0 else float('inf'),
            'ask_bid_ratio': sum_asks / sum_bids if sum_bids > 0 else float('inf'),
            'timestamp': book.get('timestamp_readable', '')
        }

    def can_buy(self, price: float, current_time: datetime) -> tuple[bool, str]:
        """Check if buying is allowed"""
        if self.is_last_10_seconds(current_time):
            return False, f"Last {self.PERIOD_END_BUFFER_SECONDS}s of period"
        if price > self.MAX_BUY_PRICE:
            return False, f"Price {price:.2f} > {self.MAX_BUY_PRICE}"
        if price < self.MIN_BUY_PRICE:
            return False, f"Price {price:.2f} < {self.MIN_BUY_PRICE}"
        return True, "OK"

    def should_buy_signal(self, analysis: Dict) -> bool:
        """Check if buy signal is present"""
        return analysis['sum_bids'] > analysis['sum_asks'] * self.imbalance_ratio

    def should_sell_signal(self, analysis: Dict) -> bool:
        """Check if sell signal is present"""
        return analysis['sum_asks'] > analysis['sum_bids'] * self.imbalance_ratio

    def should_sell_price_limit(self, price: float) -> bool:
        """Check if price hit auto-sell limit"""
        return price >= self.AUTO_SELL_PRICE

    def execute_buy(self, asset_type: str, price: float, current_time: datetime) -> Position:
        """Execute a buy and return position with market metrics"""
        # Get market metrics at entry
        choppiness, volatility = self.get_market_metrics()

        position = Position(
            asset_type=asset_type,
            entry_price=price,
            entry_time=current_time,
            size=self.position_size,
            choppiness=choppiness,
            volatility=volatility
        )

        metrics_str = ""
        if choppiness is not None and volatility is not None:
            metrics_str = f" [Chop: {choppiness:.1f}, Vol: {volatility:.2f}%]"

        print(f"\n  🟢 BUY {asset_type} @ {price:.2f}{metrics_str}")
        return position

    def execute_sell(self, position: Position, exit_price: float,
                     current_time: datetime, reason: str,
                     choppiness: Optional[float] = None,
                     volatility: Optional[float] = None) -> Trade:
        """Execute a sell and record trade"""
        self.trade_counter += 1

        pnl = (exit_price - position.entry_price) * position.size
        pnl_pct = ((exit_price / position.entry_price) - 1) * 100 if position.entry_price > 0 else 0

        trade = Trade(
            trade_id=self.trade_counter,
            period_id=self.current_period_id,
            asset_type=position.asset_type,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=current_time,
            size=position.size,
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=reason,
            choppiness=choppiness,
            volatility=volatility
        )

        # Update tracking
        self.all_trades.append(trade)
        if self.current_period_id in self.periods:
            period = self.periods[self.current_period_id]
            period.trades.append(trade)
            period.pnl += pnl
            if pnl > 0:
                period.wins += 1
            else:
                period.losses += 1

        emoji = "✅" if pnl >= 0 else "❌"
        print(f"\n  {emoji} SELL {position.asset_type} @ {exit_price:.2f} [{reason}]")
        print(f"     Entry: {position.entry_price:.2f} → Exit: {exit_price:.2f}")
        print(f"     PNL: ${pnl:.2f} ({pnl_pct:+.2f}%)")

        return trade

    def force_close_positions(self, call_analysis: Dict, put_analysis: Dict,
                               current_time: datetime, reason: str):
        """Force close all open positions"""
        if self.call_position:
            self.execute_sell(
                self.call_position,
                call_analysis['best_bid'],
                current_time,
                reason,
                choppiness=self.call_position.choppiness,
                volatility=self.call_position.volatility
            )
            self.call_position = None

        if self.put_position:
            self.execute_sell(
                self.put_position,
                put_analysis['best_bid'],
                current_time,
                reason,
                choppiness=self.put_position.choppiness,
                volatility=self.put_position.volatility
            )
            self.put_position = None

    def print_book_summary(self, asset_type: str, analysis: Dict):
        """Print compact order book summary"""
        print(f"\n  📊 {asset_type}: Bid={analysis['best_bid']:.2f} | Ask={analysis['best_ask']:.2f}")
        print(f"     Σ Bids: {analysis['sum_bids']:,.0f} | Σ Asks: {analysis['sum_asks']:,.0f} | Ratio: {analysis['bid_ask_ratio']:.2f}x")

    def print_period_summary(self):
        """Print summary for current period"""
        if self.current_period_id not in self.periods:
            return

        period = self.periods[self.current_period_id]
        print(f"\n  📈 Period {self.current_period_id}:")
        print(f"     Trades: {len(period.trades)} | Wins: {period.wins} | Losses: {period.losses}")
        print(f"     Period PNL: ${period.pnl:.2f}")

    def run_iteration(self, current_time: datetime = None):
        """Run a single iteration"""
        self.iteration += 1

        # Load order books
        call_book = self.load_order_book(self.call_file)
        put_book = self.load_order_book(self.put_file)

        if not call_book or not put_book:
            print(f"[{self.iteration}] ❌ Failed to load order books")
            return

        # Parse timestamp or use provided time
        if current_time is None:
            try:
                ts = call_book.get('timestamp_readable', '')
                current_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
            except:
                current_time = datetime.now()

        # Check for new period
        if self.is_new_period(current_time):
            self.start_new_period(current_time)

        # Analyze books
        call_analysis = self.analyze_book(call_book)
        put_analysis = self.analyze_book(put_book)

        # Calculate time remaining
        seconds_remaining = self.get_seconds_to_period_end(current_time)
        in_last_10s = self.is_last_10_seconds(current_time)

        print(f"\n{'─'*70}")
        print(f"[{self.iteration}] {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Period: {self.current_period_id}")
        print(f"    ⏱️  {seconds_remaining:.0f}s remaining {'⚠️ CLOSING WINDOW!' if in_last_10s else ''}")

        self.print_book_summary('CALL', call_analysis)
        self.print_book_summary('PUT', put_analysis)

        # === TRADING LOGIC ===
        print(f"\n  🎯 TRADING:")

        # Check if in last 10 seconds - force close and no new trades
        if in_last_10s:
            print(f"     ⚠️  Last {self.PERIOD_END_BUFFER_SECONDS}s - Closing positions, no new trades")
            self.force_close_positions(call_analysis, put_analysis, current_time, "last_10s")
            self.print_period_summary()
            return

        # === CALL LOGIC ===
        if self.call_position:
            # Check sell conditions
            current_bid = call_analysis['best_bid']
            unrealized = (current_bid - self.call_position.entry_price) * self.position_size
            print(f"     CALL: Holding @ {self.call_position.entry_price:.2f} | Bid: {current_bid:.2f} | Unrealized: ${unrealized:+.2f}")

            if self.should_sell_price_limit(current_bid):
                self.execute_sell(self.call_position, current_bid, current_time, "price_0.99",
                                choppiness=self.call_position.choppiness,
                                volatility=self.call_position.volatility)
                self.call_position = None
            elif self.should_sell_signal(call_analysis):
                self.execute_sell(self.call_position, current_bid, current_time, "signal",
                                choppiness=self.call_position.choppiness,
                                volatility=self.call_position.volatility)
                self.call_position = None
        else:
            # Check buy conditions
            ask_price = call_analysis['best_ask']
            can_buy, reason = self.can_buy(ask_price, current_time)

            if self.should_buy_signal(call_analysis):
                if can_buy:
                    self.call_position = self.execute_buy("CALL", ask_price, current_time)
                else:
                    print(f"     CALL: BUY signal but blocked - {reason}")
            else:
                print(f"     CALL: No signal (ratio {call_analysis['bid_ask_ratio']:.2f}x < {self.imbalance_ratio}x)")

        # === PUT LOGIC ===
        if self.put_position:
            # Check sell conditions
            current_bid = put_analysis['best_bid']
            unrealized = (current_bid - self.put_position.entry_price) * self.position_size
            print(f"     PUT:  Holding @ {self.put_position.entry_price:.2f} | Bid: {current_bid:.2f} | Unrealized: ${unrealized:+.2f}")

            if self.should_sell_price_limit(current_bid):
                self.execute_sell(self.put_position, current_bid, current_time, "price_0.99",
                                choppiness=self.put_position.choppiness,
                                volatility=self.put_position.volatility)
                self.put_position = None
            elif self.should_sell_signal(put_analysis):
                self.execute_sell(self.put_position, current_bid, current_time, "signal",
                                choppiness=self.put_position.choppiness,
                                volatility=self.put_position.volatility)
                self.put_position = None
        else:
            # Check buy conditions
            ask_price = put_analysis['best_ask']
            can_buy, reason = self.can_buy(ask_price, current_time)

            if self.should_buy_signal(put_analysis):
                if can_buy:
                    self.put_position = self.execute_buy("PUT", ask_price, current_time)
                else:
                    print(f"     PUT:  BUY signal but blocked - {reason}")
            else:
                print(f"     PUT:  No signal (ratio {put_analysis['bid_ask_ratio']:.2f}x < {self.imbalance_ratio}x)")

        self.print_period_summary()

    def print_final_summary(self):
        """Print comprehensive final summary"""
        # Save the last period if exists
        if self.current_period_id and self.current_period_id in self.periods:
            self.periods[self.current_period_id].end_time = datetime.now()
            self.save_period_trades(self.current_period_id)

        # Save overall summary
        self.save_overall_summary()

        print(f"\n{'='*70}")
        print(f"📋 FINAL SIMULATION SUMMARY")
        print(f"{'='*70}")

    def save_overall_summary(self):
        """Save overall trading summary to JSON"""
        total_pnl = sum(t.pnl for t in self.all_trades)
        total_wins = sum(1 for t in self.all_trades if t.pnl > 0)
        total_losses = len(self.all_trades) - total_wins

        # Exit reasons breakdown
        exit_reasons = {}
        for t in self.all_trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        summary_data = {
            "generated_at": datetime.now().isoformat(),
            "overall": {
                "total_trades": len(self.all_trades),
                "total_pnl": round(total_pnl, 2),
                "wins": total_wins,
                "losses": total_losses,
                "win_rate": round(total_wins / len(self.all_trades) * 100, 1) if self.all_trades else 0,
                "avg_pnl_per_trade": round(total_pnl / len(self.all_trades), 2) if self.all_trades else 0
            },
            "exit_reasons": exit_reasons,
            "periods": {}
        }

        # Add per-period summaries
        for period_id, period in sorted(self.periods.items()):
            summary_data["periods"][period_id] = {
                "trades": len(period.trades),
                "wins": period.wins,
                "losses": period.losses,
                "pnl": round(period.pnl, 2)
            }

        # Save to file
        filepath = os.path.join(self.trades_log_dir, "summary_overall.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"\n  💾 Overall summary saved to: {filepath}")
        except Exception as e:
            print(f"\n  ❌ Failed to save overall summary: {e}")

        # Overall stats
        total_pnl = sum(t.pnl for t in self.all_trades)
        total_wins = sum(1 for t in self.all_trades if t.pnl > 0)
        total_losses = len(self.all_trades) - total_wins

        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Trades: {len(self.all_trades)}")
        print(f"   Total PNL: ${total_pnl:.2f}")
        print(f"   Win Rate: {total_wins}/{len(self.all_trades)} ({total_wins/len(self.all_trades)*100:.1f}%)" if self.all_trades else "   No trades")

        if self.all_trades:
            avg_pnl = total_pnl / len(self.all_trades)
            print(f"   Avg PNL/Trade: ${avg_pnl:.2f}")

            wins = [t for t in self.all_trades if t.pnl > 0]
            losses = [t for t in self.all_trades if t.pnl <= 0]
            if wins:
                print(f"   Avg Win: ${sum(t.pnl for t in wins)/len(wins):.2f}")
            if losses:
                print(f"   Avg Loss: ${sum(t.pnl for t in losses)/len(losses):.2f}")

        # Per-period breakdown
        print(f"\n📅 PNL BY PERIOD:")
        print(f"   {'Period':<20} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'PNL':>12}")
        print(f"   {'-'*20} {'-'*8} {'-'*6} {'-'*8} {'-'*12}")

        for period_id, period in sorted(self.periods.items()):
            emoji = "✅" if period.pnl >= 0 else "❌"
            print(f"   {period_id:<20} {len(period.trades):>8} {period.wins:>6} {period.losses:>8} {emoji}${period.pnl:>10.2f}")

        print(f"   {'-'*20} {'-'*8} {'-'*6} {'-'*8} {'-'*12}")
        print(f"   {'TOTAL':<20} {len(self.all_trades):>8} {total_wins:>6} {total_losses:>8} ${total_pnl:>11.2f}")

        # Exit reason breakdown
        if self.all_trades:
            print(f"\n📝 EXIT REASONS:")
            reasons = {}
            for t in self.all_trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            for reason, count in sorted(reasons.items()):
                print(f"   {reason}: {count} trades")

        # Trade log
        if self.all_trades:
            print(f"\n📝 FULL TRADE LOG:")
            print(f"   {'#':>4} {'Period':<18} {'Type':<6} {'Entry':>7} {'Exit':>7} {'PNL':>10} {'Reason':<12}")
            print(f"   {'-'*4} {'-'*18} {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*12}")

            for t in self.all_trades:
                emoji = "✅" if t.pnl >= 0 else "❌"
                print(f"   {emoji}{t.trade_id:>3} {t.period_id:<18} {t.asset_type:<6} {t.entry_price:>7.2f} {t.exit_price:>7.2f} ${t.pnl:>8.2f} {t.exit_reason:<12}")

    def run_continuous(self, interval: float = 1.0, max_iterations: int = None):
        """Run continuously monitoring files"""
        print(f"\n{'*'*70}")
        print(f"* BINARY OPTIONS BOT v2 - CONTINUOUS MODE")
        print(f"* Imbalance Ratio: {self.imbalance_ratio}x")
        print(f"* Price Limits: No buy if >{self.MAX_BUY_PRICE} or <{self.MIN_BUY_PRICE}")
        print(f"* Auto-sell at: {self.AUTO_SELL_PRICE}")
        print(f"* Period buffer: {self.PERIOD_END_BUFFER_SECONDS}s before period end")
        print(f"{'*'*70}")

        try:
            while True:
                self.run_iteration()

                if max_iterations and self.iteration >= max_iterations:
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n⛔ Stopped by user")

        self.print_final_summary()


def run_demo():
    """Run demonstration with simulated time progression"""
    print(f"\n{'*'*70}")
    print(f"* BINARY OPTIONS BOT v2 - DEMO MODE")
    print(f"* Simulating multiple periods with varying conditions")
    print(f"{'*'*70}")

    bot = BinaryOptionsBot(
        call_file="/home/ubuntu/013_2025_polymarket/15M_CALL_nonoise.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_PUT_nonoise.json",
        imbalance_ratio=2.0
    )

    # Load base books for simulation
    call_base = bot.load_order_book(bot.call_file)
    put_base = bot.load_order_book(bot.put_file)

    if not call_base or not put_base:
        print("Failed to load base books")
        return

    # Simulate across multiple periods
    base_time = datetime(2026, 1, 15, 10, 0, 0)

    scenarios = [
        # (minute_offset, name, call_bid_mult, call_ask_mult, put_bid_mult, put_ask_mult, price_shift)
        # Period 1: 10:00-10:15
        (0, "Period start - neutral", 1.0, 1.0, 1.0, 1.0, 0),
        (3, "CALL bullish signal", 3.0, 0.8, 0.8, 1.2, 0),
        (6, "Holding", 2.5, 0.9, 0.9, 1.1, 0.02),
        (9, "CALL hits 0.99!", 2.0, 1.0, 1.0, 1.0, 0.50),  # Price spike to 0.99
        (12, "Neutral again", 1.2, 1.0, 1.0, 1.2, 0),
        (14, "Last 10s warning", 1.0, 1.0, 1.0, 1.0, 0),
        (14.85, "Force close", 1.0, 1.0, 2.5, 0.8, 0),  # Last 10s

        # Period 2: 10:15-10:30
        (15, "New period start", 1.0, 1.0, 1.0, 1.0, 0),
        (17, "PUT bullish signal", 0.8, 1.2, 3.0, 0.7, 0),
        (20, "PUT holding", 0.9, 1.1, 2.8, 0.8, 0.01),
        (23, "PUT sell signal", 1.2, 0.9, 0.6, 2.5, 0),
        (26, "Price too high - no buy", 1.0, 1.0, 1.0, 1.0, 0.47),  # Ask at 0.98
        (29, "Period ending", 1.0, 1.0, 1.0, 1.0, 0),

        # Period 3: 10:30-10:45
        (30, "Period 3 start", 1.0, 1.0, 1.0, 1.0, 0),
        (32, "CALL signal - normal price", 2.8, 0.7, 1.0, 1.0, 0),
        (35, "Both positions", 2.5, 0.8, 2.5, 0.8, 0),
        (38, "Market reversal", 0.6, 2.2, 0.6, 2.2, -0.02),
        (41, "Both exit on signal", 0.5, 2.5, 0.5, 2.5, 0),
        (44, "Near period end", 1.0, 1.0, 1.0, 1.0, 0),
    ]

    for offset_min, name, cb, ca, pb, pa, shift in scenarios:
        # Calculate simulated time
        sim_time = base_time + timedelta(minutes=offset_min)

        # Create simulated books
        call_book = simulate_book(call_base, cb, ca, shift)
        put_book = simulate_book(put_base, pb, pa, -shift)

        # Temporarily override the load function
        original_load = bot.load_order_book
        bot.load_order_book = lambda f: call_book if 'CALL' in f else put_book

        print(f"\n[SIM: {name}]")
        bot.run_iteration(current_time=sim_time)

        bot.load_order_book = original_load
        time.sleep(0.3)

    bot.print_final_summary()


def simulate_book(base_book: Dict, bid_mult: float, ask_mult: float, price_shift: float) -> Dict:
    """Create simulated order book"""
    book = json.loads(json.dumps(base_book))

    # Modify top levels
    for level in book['complete_book']['bids'][-4:]:
        level['size'] = str(float(level['size']) * bid_mult)

    for level in book['complete_book']['asks'][:4]:
        level['size'] = str(float(level['size']) * ask_mult)

    # Price shift
    if price_shift != 0:
        for level in book['complete_book']['bids']:
            new_price = max(0.01, min(0.99, float(level['price']) + price_shift))
            level['price'] = f"{new_price:.2f}"
        for level in book['complete_book']['asks']:
            new_price = max(0.01, min(0.99, float(level['price']) + price_shift))
            level['price'] = f"{new_price:.2f}"

        book['best_bid']['price'] = max(0.01, min(0.99, book['best_bid']['price'] + price_shift))
        book['best_ask']['price'] = max(0.01, min(0.99, book['best_ask']['price'] + price_shift))

    return book


if __name__ == "__main__":
    import sys

    bot = BinaryOptionsBot(
        call_file="/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_PUT.json"
    )
    bot.run_continuous(interval=1.0)
