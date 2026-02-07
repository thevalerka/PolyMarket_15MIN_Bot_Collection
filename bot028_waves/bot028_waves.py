#!/usr/bin/env python3
"""
bot028_waves.py — Wave Analysis Trading Bot for Polymarket 15M Binary Options

The order book as two opposing ocean waves:
- BID wave pushes price UP (buying pressure)
- ASK wave pushes price DOWN (selling pressure)
- When one wave overwhelms the other, price moves

We use the CALL book as primary reference:
  - CALL bids = buy pressure (wave pushing price UP toward 1.00)
  - CALL asks = sell pressure (wave pushing price DOWN toward 0.00)
  - PUT book is the complement/mirror — used for cross-validation

The books are read every 0.1s, 30-snapshot rolling window (3s span).

Signals:
  1. BUY BACKWASH    — quasi-spread forming on ask side, volume building on bid side
  2. BUY BIGGER WAVE — big bid volume, average asks, price already moving up
  3. HOLD VOLATILITY  — big volumes close to price BOTH sides, thin away
  4. SELL COUNTERWAVE — price moving but losing side building bigger wall
  5. SELL EXHAUSTION   — post-move, volumes dropping & flattening on both sides
  6. DISCOVERED        — emergent patterns logged for review
"""

import json
import os
import sys
import time
import copy
import signal as sig_module
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/home/ubuntu/013_2025_polymarket")
BOT_DIR = BASE_DIR / "bot028_waves"
DATA_DIR = BASE_DIR  # where the JSON order books live

CALL_FILE = DATA_DIR / "15M_CALL_nonoise.json"
PUT_FILE = DATA_DIR / "15M_PUT_nonoise.json"

SIGNAL_LOG = BOT_DIR / "wave_signals.json"
STATE_LOG = BOT_DIR / "wave_state.json"
DEBUG_LOG = BOT_DIR / "wave_debug.log"

SNAPSHOT_INTERVAL = 0.1    # seconds between snapshots
WINDOW_SIZE = 30           # 30 snapshots = 3 seconds rolling window
PRICE_TICK = 0.01          # each level = $0.01

# ── Signal detection thresholds (tunable) ──
QUASI_SPREAD_RATIO = 0.25         # level volume < 25% of neighbor avg = quasi-empty
QUASI_SPREAD_DEPTH = 5            # check first N levels for quasi-spread
BIG_VOLUME_MULTIPLIER = 2.0       # volume > 2x avg of that depth band = "big"
WAVE_IMBALANCE_RATIO = 1.8        # one side > 1.8x the other = imbalance
EXHAUSTION_DROP_PCT = 0.40        # volume dropped 40% from window peak = exhaustion
FLATNESS_CV_THRESHOLD = 0.35      # coefficient of variation < 0.35 = "flat/even"
MOVEMENT_THRESHOLD_TICKS = 2      # price moved >= 2 ticks across window = "moving"
COUNTERWAVE_BUILD_PCT = 0.30      # losing side grew 30%+ while price moved against it
VOLATILITY_WALL_RATIO = 2.5       # near-price vol > 2.5x far-price vol = walls
MIN_SNAPSHOTS_FOR_SIGNAL = 10     # need at least 10 snapshots before generating signals
SIGNAL_COOLDOWN_S = 1.5           # min seconds between same signal type

# ─────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────

def log_debug(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

def safe_read_json(path: Path) -> Optional[dict]:
    """Read JSON, handle partial writes from other processes."""
    try:
        with open(path, "r") as f:
            raw = f.read()
        if not raw.strip():
            return None
        return json.loads(raw)
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return None

def atomic_write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        log_debug(f"ERROR writing {path}: {e}")

def now_ms() -> int:
    return int(time.time() * 1000)

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ─────────────────────────────────────────────────────────────────────
# ORDER BOOK PARSING — UNIFIED WAVE VIEW
# ─────────────────────────────────────────────────────────────────────

class BookLevel:
    __slots__ = ("price", "size")
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size

class UnifiedBook:
    """
    Unified wave view from the CALL order book.

    Convention (from CALL perspective):
      - bids: sorted DESCENDING from best_bid (closest to price -> away)
              These are the BUY WAVE — pushing price UP
      - asks: sorted ASCENDING from best_ask (closest to price -> away)
              These are the SELL WAVE — pushing price DOWN

    The midpoint is (best_bid + best_ask) / 2.

    IMPORTANT: Since CALL + PUT = 1.00, the CALL bid at price X mirrors
    the PUT ask at price (1-X). We read BOTH books and merge them into
    a single unified view. The CALL book has the natural bids, and the
    PUT book's asks (flipped) provide additional ask-side depth for the
    CALL view, and vice versa.
    """

    def __init__(self):
        self.timestamp_ms: int = 0
        self.bids: List[BookLevel] = []     # descending from best bid
        self.asks: List[BookLevel] = []     # ascending from best ask
        self.best_bid: float = 0.0
        self.best_ask: float = 1.0
        self.midpoint: float = 0.5
        self.spread: float = 1.0
        self.spread_ticks: int = 100
        self.total_bid_volume: float = 0.0
        self.total_ask_volume: float = 0.0

    @staticmethod
    def from_call_json(data: dict) -> Optional["UnifiedBook"]:
        """Build unified book from CALL JSON structure."""
        if not data or "complete_book" not in data:
            return None

        book = UnifiedBook()

        try:
            book.timestamp_ms = int(data.get("timestamp", 0))
        except (ValueError, TypeError):
            book.timestamp_ms = now_ms()

        cb = data["complete_book"]

        # Parse CALL bids -> bid wave (sorted descending by price, best first)
        raw_bids = []
        for lvl in cb.get("bids", []):
            p = float(lvl["price"])
            s = float(lvl["size"])
            if s > 0:
                raw_bids.append(BookLevel(p, s))
        raw_bids.sort(key=lambda x: x.price, reverse=True)
        book.bids = raw_bids

        # Parse CALL asks -> ask wave (sorted ascending by price, best first)
        raw_asks = []
        for lvl in cb.get("asks", []):
            p = float(lvl["price"])
            s = float(lvl["size"])
            if s > 0:
                raw_asks.append(BookLevel(p, s))
        raw_asks.sort(key=lambda x: x.price)
        book.asks = raw_asks

        # Best bid/ask
        if data.get("best_bid") and data["best_bid"].get("price"):
            book.best_bid = float(data["best_bid"]["price"])
        elif raw_bids:
            book.best_bid = raw_bids[0].price
        else:
            book.best_bid = 0.0

        if data.get("best_ask") and data["best_ask"].get("price"):
            book.best_ask = float(data["best_ask"]["price"])
        elif raw_asks:
            book.best_ask = raw_asks[0].price
        else:
            book.best_ask = 1.0

        book.spread = round(book.best_ask - book.best_bid, 4)
        book.spread_ticks = max(0, round(book.spread / PRICE_TICK))
        book.midpoint = round((book.best_bid + book.best_ask) / 2, 4)
        book.total_bid_volume = sum(l.size for l in raw_bids)
        book.total_ask_volume = sum(l.size for l in raw_asks)

        return book

    # ── Volume extraction by level count ──

    def bid_volume_levels(self, n: int) -> Tuple[float, List[BookLevel]]:
        """Volume of first N bid levels (closest to price)."""
        levels = self.bids[:n]
        return sum(l.size for l in levels), levels

    def ask_volume_levels(self, n: int) -> Tuple[float, List[BookLevel]]:
        """Volume of first N ask levels (closest to price)."""
        levels = self.asks[:n]
        return sum(l.size for l in levels), levels

    # ── Quasi-spread detection ──

    def _quasi_spread(self, levels: List[BookLevel], depth: int) -> dict:
        """
        Check if the first `depth` levels have quasi-empty volumes
        compared to the levels behind them (depth..depth+10).
        """
        if len(levels) < depth + 3:
            return {"detected": False, "ratio": 1.0, "thin_levels": 0,
                    "front_avg": 0, "behind_avg": 0}

        front = levels[:depth]
        behind = levels[depth:depth + 10]

        front_avg = sum(l.size for l in front) / max(len(front), 1)
        behind_avg = sum(l.size for l in behind) / max(len(behind), 1)

        if behind_avg == 0:
            return {"detected": False, "ratio": 1.0, "thin_levels": 0,
                    "front_avg": round(front_avg, 2), "behind_avg": 0}

        ratio = front_avg / behind_avg
        thin_count = sum(1 for l in front if l.size < behind_avg * QUASI_SPREAD_RATIO)

        return {
            "detected": ratio < QUASI_SPREAD_RATIO * 2 or thin_count >= depth // 2,
            "ratio": round(ratio, 4),
            "thin_levels": thin_count,
            "front_avg": round(front_avg, 2),
            "behind_avg": round(behind_avg, 2)
        }

    def bid_quasi_spread(self, depth: int = QUASI_SPREAD_DEPTH) -> dict:
        return self._quasi_spread(self.bids, depth)

    def ask_quasi_spread(self, depth: int = QUASI_SPREAD_DEPTH) -> dict:
        return self._quasi_spread(self.asks, depth)

    def depth_profile(self) -> dict:
        """Full depth analysis: first 3, 5, 10, 20 levels + spread info."""
        profile = {}
        for n in [3, 5, 10, 20]:
            bv, bl = self.bid_volume_levels(n)
            av, al = self.ask_volume_levels(n)
            profile[f"L{n}"] = {
                "bid_vol": round(bv, 2),
                "ask_vol": round(av, 2),
                "bid_levels_actual": len(bl),
                "ask_levels_actual": len(al),
                "imbalance": round(bv / max(av, 0.01), 4),
            }

        bq = self.bid_quasi_spread()
        aq = self.ask_quasi_spread()

        profile["spread"] = {
            "ticks": self.spread_ticks,
            "dollars": self.spread,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "midpoint": self.midpoint,
            "spread_side": "ASK" if self.best_ask > (1.0 - 0.01) else (
                "BID" if self.best_bid < 0.01 else "BOTH"
            ),
            "bid_quasi": bq,
            "ask_quasi": aq,
        }
        profile["totals"] = {
            "bid_volume": round(self.total_bid_volume, 2),
            "ask_volume": round(self.total_ask_volume, 2),
            "ratio": round(self.total_bid_volume / max(self.total_ask_volume, 0.01), 4),
        }
        return profile


# ─────────────────────────────────────────────────────────────────────
# SNAPSHOT — compressed book state for rolling window
# ─────────────────────────────────────────────────────────────────────

class WaveSnapshot:

    def __init__(self, book: UnifiedBook):
        self.ts_ms = now_ms()
        self.midpoint = book.midpoint
        self.best_bid = book.best_bid
        self.best_ask = book.best_ask
        self.spread = book.spread
        self.spread_ticks = book.spread_ticks

        # Volume at depth bands
        self.bid_v3, _ = book.bid_volume_levels(3)
        self.bid_v5, _ = book.bid_volume_levels(5)
        self.bid_v10, _ = book.bid_volume_levels(10)
        self.bid_v20, _ = book.bid_volume_levels(20)

        self.ask_v3, _ = book.ask_volume_levels(3)
        self.ask_v5, _ = book.ask_volume_levels(5)
        self.ask_v10, _ = book.ask_volume_levels(10)
        self.ask_v20, _ = book.ask_volume_levels(20)

        self.total_bid = book.total_bid_volume
        self.total_ask = book.total_ask_volume

        # Quasi-spread info
        self.bid_quasi = book.bid_quasi_spread()
        self.ask_quasi = book.ask_quasi_spread()

        # Near-price detail (first 10 each side) for deeper analysis
        self.bid_near = [(l.price, l.size) for l in book.bids[:10]]
        self.ask_near = [(l.price, l.size) for l in book.asks[:10]]

    def to_dict(self) -> dict:
        return {
            "ts_ms": self.ts_ms,
            "mid": self.midpoint,
            "bb": self.best_bid,
            "ba": self.best_ask,
            "spread": self.spread,
            "bid": {"v3": round(self.bid_v3, 1), "v5": round(self.bid_v5, 1),
                    "v10": round(self.bid_v10, 1), "v20": round(self.bid_v20, 1),
                    "total": round(self.total_bid, 1)},
            "ask": {"v3": round(self.ask_v3, 1), "v5": round(self.ask_v5, 1),
                    "v10": round(self.ask_v10, 1), "v20": round(self.ask_v20, 1),
                    "total": round(self.total_ask, 1)},
            "bid_quasi": self.bid_quasi["detected"],
            "ask_quasi": self.ask_quasi["detected"],
        }


# ─────────────────────────────────────────────────────────────────────
# WAVE ANALYZER — Signal detection engine
# ─────────────────────────────────────────────────────────────────────

class WaveAnalyzer:

    def __init__(self):
        self.window: deque[WaveSnapshot] = deque(maxlen=WINDOW_SIZE)
        self.last_signal_time: Dict[str, float] = {}
        self.signal_count = 0
        self.all_signals_log: List[dict] = []

        # Load existing signals
        if SIGNAL_LOG.exists():
            try:
                self.all_signals_log = json.loads(SIGNAL_LOG.read_text())
                self.signal_count = len(self.all_signals_log)
            except Exception:
                self.all_signals_log = []

    def add_snapshot(self, snap: WaveSnapshot):
        self.window.append(snap)

    def can_emit(self, signal_type: str) -> bool:
        last = self.last_signal_time.get(signal_type, 0)
        return (time.time() - last) >= SIGNAL_COOLDOWN_S

    def emit_signal(self, signal_type: str, action: str, confidence: float,
                    details: dict, snap: WaveSnapshot):
        if not self.can_emit(signal_type):
            return

        self.last_signal_time[signal_type] = time.time()
        self.signal_count += 1

        record = {
            "id": self.signal_count,
            "timestamp": now_iso(),
            "ts_ms": now_ms(),
            "signal": signal_type,
            "action": action,
            "confidence": round(confidence, 3),
            "midpoint": snap.midpoint,
            "best_bid": snap.best_bid,
            "best_ask": snap.best_ask,
            "spread": snap.spread,
            "bid_v5": round(snap.bid_v5, 1),
            "ask_v5": round(snap.ask_v5, 1),
            "bid_v10": round(snap.bid_v10, 1),
            "ask_v10": round(snap.ask_v10, 1),
            "bid_total": round(snap.total_bid, 1),
            "ask_total": round(snap.total_ask, 1),
            "details": details,
        }

        self.all_signals_log.append(record)
        atomic_write_json(SIGNAL_LOG, self.all_signals_log)

        emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "HOLD": "\U0001f7e1"}.get(action, "\u26aa")
        print(f"\n{emoji} SIGNAL #{self.signal_count}: {signal_type} -> {action} "
              f"(conf={confidence:.1%})")
        print(f"   mid={snap.midpoint:.4f}  bid={snap.best_bid:.3f}  "
              f"ask={snap.best_ask:.3f}  spread={snap.spread:.4f}")
        print(f"   bid_v5={snap.bid_v5:.0f}  ask_v5={snap.ask_v5:.0f}  "
              f"bid_v10={snap.bid_v10:.0f}  ask_v10={snap.ask_v10:.0f}")
        print(f"   >> {details.get('reason', '')}")
        log_debug(f"SIGNAL: {json.dumps(record)}")

    # ── Window-derived metrics ──

    def _price_movement(self) -> Tuple[float, int]:
        """(delta, ticks_moved). Positive = UP."""
        if len(self.window) < 2:
            return 0.0, 0
        delta = self.window[-1].midpoint - self.window[0].midpoint
        return delta, round(delta / PRICE_TICK)

    def _volume_trend(self, side: str, band: str = "v10") -> Tuple[float, float, float]:
        """(early_avg, late_avg, change_pct) across first vs last third of window."""
        if len(self.window) < 6:
            return 0, 0, 0

        third = max(len(self.window) // 3, 1)
        attr = f"{side}_{band}" if band != "total" else f"total_{side}"

        early = [getattr(s, attr, 0) for i, s in enumerate(self.window) if i < third]
        late = [getattr(s, attr, 0) for i, s in enumerate(self.window) if i >= len(self.window) - third]

        ea = sum(early) / max(len(early), 1)
        la = sum(late) / max(len(late), 1)
        pct = (la - ea) / ea if ea > 0 else 0
        return ea, la, pct

    def _volume_peak_and_current(self, side: str, band: str = "v10") -> Tuple[float, float]:
        attr = f"{side}_{band}" if band != "total" else f"total_{side}"
        peak = max((getattr(s, attr, 0) for s in self.window), default=0)
        current = getattr(self.window[-1], attr, 0)
        return peak, current

    def _volume_flatness(self) -> float:
        """Coefficient of variation of total volume across window."""
        if len(self.window) < 5:
            return 1.0
        vals = [s.total_bid + s.total_ask for s in self.window]
        mean = sum(vals) / len(vals)
        if mean == 0:
            return 0
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return (variance ** 0.5) / mean

    def _imbalance_ratio(self, band: str = "v10") -> float:
        """Current bid/ask volume ratio. >1 = bid heavy."""
        snap = self.window[-1]
        bv = getattr(snap, f"bid_{band}", 0)
        av = getattr(snap, f"ask_{band}", 0)
        return bv / av if av > 0 else 999.0

    # ── Signal detectors ──

    def detect_signals(self):
        if len(self.window) < MIN_SNAPSHOTS_FOR_SIGNAL:
            return

        snap = self.window[-1]
        price_delta, price_ticks = self._price_movement()

        self._detect_backwash(snap, price_delta, price_ticks)
        self._detect_bigger_wave(snap, price_delta, price_ticks)
        self._detect_volatility_hold(snap, price_delta, price_ticks)
        self._detect_counterwave(snap, price_delta, price_ticks)
        self._detect_exhaustion(snap, price_delta, price_ticks)
        self._detect_emergent(snap, price_delta, price_ticks)

    def _detect_backwash(self, snap, price_delta, price_ticks):
        """
        Signal 1: BACKWASH
        Quasi-spread forming on one side (thinning), volume building on other.
        The retreating side is about to be overwhelmed.
        """
        bq = snap.bid_quasi
        aq = snap.ask_quasi

        # Ask side thinning + bids building -> BUY
        if aq["detected"] and not bq["detected"]:
            _, _, bid_trend = self._volume_trend("bid", "v10")
            if bid_trend > 0.05:
                confidence = min(0.95, 0.5 + bid_trend + (1 - aq.get("ratio", 1)))
                self.emit_signal("BACKWASH", "BUY", confidence, {
                    "reason": "Ask wave retreating (quasi-spread) + bid wave building",
                    "ask_quasi_ratio": aq.get("ratio"),
                    "ask_thin_levels": aq.get("thin_levels"),
                    "bid_growth_pct": round(bid_trend, 3),
                }, snap)

        # Bid side thinning + asks building -> SELL
        if bq["detected"] and not aq["detected"]:
            _, _, ask_trend = self._volume_trend("ask", "v10")
            if ask_trend > 0.05:
                confidence = min(0.95, 0.5 + ask_trend + (1 - bq.get("ratio", 1)))
                self.emit_signal("BACKWASH", "SELL", confidence, {
                    "reason": "Bid wave retreating (quasi-spread) + ask wave building",
                    "bid_quasi_ratio": bq.get("ratio"),
                    "bid_thin_levels": bq.get("thin_levels"),
                    "ask_growth_pct": round(ask_trend, 3),
                }, snap)

    def _detect_bigger_wave(self, snap, price_delta, price_ticks):
        """
        Signal 2: BIGGER WAVE
        Big volume on one side + average on other + price already moving that way.
        Momentum confirmation.
        """
        if abs(price_ticks) < MOVEMENT_THRESHOLD_TICKS:
            return

        imb5 = self._imbalance_ratio("v5")
        imb10 = self._imbalance_ratio("v10")

        # Price UP + bids dominating
        if price_ticks >= MOVEMENT_THRESHOLD_TICKS and imb10 >= WAVE_IMBALANCE_RATIO:
            conf = min(0.95, 0.4 + 0.1 * abs(price_ticks) + 0.1 * (imb10 - 1))
            self.emit_signal("BIGGER_WAVE", "BUY", conf, {
                "reason": f"Price up {price_ticks}t, bid wave {imb10:.1f}x stronger",
                "price_ticks": price_ticks,
                "imbalance_L5": round(imb5, 3),
                "imbalance_L10": round(imb10, 3),
            }, snap)

        # Price DOWN + asks dominating
        if price_ticks <= -MOVEMENT_THRESHOLD_TICKS and imb10 <= (1 / WAVE_IMBALANCE_RATIO):
            inv = 1 / max(imb10, 0.001)
            conf = min(0.95, 0.4 + 0.1 * abs(price_ticks) + 0.1 * (inv - 1))
            self.emit_signal("BIGGER_WAVE", "SELL", conf, {
                "reason": f"Price down {abs(price_ticks)}t, ask wave {inv:.1f}x stronger",
                "price_ticks": price_ticks,
                "imbalance_L5": round(imb5, 3),
                "imbalance_L10": round(imb10, 3),
            }, snap)

    def _detect_volatility_hold(self, snap, price_delta, price_ticks):
        """
        Signal 3: HOLD — VOLATILITY AHEAD
        Big walls on both sides near price, thin away. Tension building.
        """
        bid_near_avg = snap.bid_v5 / 5 if snap.bid_v5 > 0 else 0
        ask_near_avg = snap.ask_v5 / 5 if snap.ask_v5 > 0 else 0

        bid_far = max(snap.bid_v20 - snap.bid_v10, 0)
        ask_far = max(snap.ask_v20 - snap.ask_v10, 0)
        bid_far_avg = bid_far / 10
        ask_far_avg = ask_far / 10

        if bid_far_avg > 0 and ask_far_avg > 0 and bid_near_avg > 0 and ask_near_avg > 0:
            bid_wall = bid_near_avg / bid_far_avg
            ask_wall = ask_near_avg / ask_far_avg

            if (bid_wall >= VOLATILITY_WALL_RATIO and
                ask_wall >= VOLATILITY_WALL_RATIO and
                abs(price_ticks) <= 1):
                conf = min(0.9, 0.3 + 0.1 * min(bid_wall, ask_wall))
                self.emit_signal("VOLATILITY_HOLD", "HOLD", conf, {
                    "reason": f"Dual walls near price: bid {bid_wall:.1f}x, ask {ask_wall:.1f}x vs far levels",
                    "bid_wall_ratio": round(bid_wall, 2),
                    "ask_wall_ratio": round(ask_wall, 2),
                    "bid_near_avg": round(bid_near_avg, 1),
                    "ask_near_avg": round(ask_near_avg, 1),
                }, snap)

    def _detect_counterwave(self, snap, price_delta, price_ticks):
        """
        Signal 4: COUNTERWAVE
        Price moving one way but the losing side is BUILDING volume.
        Reversal incoming.
        """
        if abs(price_ticks) < MOVEMENT_THRESHOLD_TICKS:
            return

        if price_ticks > 0:
            # Price UP -> asks should be weakening. If asks GROWING = counterwave
            _, _, ask_growth = self._volume_trend("ask", "v10")
            if ask_growth >= COUNTERWAVE_BUILD_PCT:
                conf = min(0.9, 0.4 + ask_growth)
                self.emit_signal("COUNTERWAVE", "SELL", conf, {
                    "reason": f"Price up {price_ticks}t but ask wave growing {ask_growth:.0%}",
                    "price_ticks": price_ticks,
                    "losing_side_growth": round(ask_growth, 3),
                    "side": "asks_building_against_uptrend",
                }, snap)
        else:
            # Price DOWN -> bids should be weakening. If bids GROWING = counterwave
            _, _, bid_growth = self._volume_trend("bid", "v10")
            if bid_growth >= COUNTERWAVE_BUILD_PCT:
                conf = min(0.9, 0.4 + bid_growth)
                self.emit_signal("COUNTERWAVE", "BUY", conf, {
                    "reason": f"Price down {abs(price_ticks)}t but bid wave growing {bid_growth:.0%}",
                    "price_ticks": price_ticks,
                    "losing_side_growth": round(bid_growth, 3),
                    "side": "bids_building_against_downtrend",
                }, snap)

    def _detect_exhaustion(self, snap, price_delta, price_ticks):
        """
        Signal 5: EXHAUSTION
        After a big move: volumes dropping on BOTH sides, distribution flattening.
        The wave has crashed.
        """
        if abs(price_ticks) < MOVEMENT_THRESHOLD_TICKS:
            return

        bid_peak, bid_curr = self._volume_peak_and_current("bid", "v10")
        ask_peak, ask_curr = self._volume_peak_and_current("ask", "v10")

        bid_drop = (bid_peak - bid_curr) / max(bid_peak, 1)
        ask_drop = (ask_peak - ask_curr) / max(ask_peak, 1)

        if bid_drop >= EXHAUSTION_DROP_PCT and ask_drop >= EXHAUSTION_DROP_PCT:
            flatness = self._volume_flatness()
            conf = min(0.9, 0.3 + bid_drop * 0.3 + ask_drop * 0.3)
            self.emit_signal("EXHAUSTION", "SELL", conf, {
                "reason": f"Post-move collapse: bids -{bid_drop:.0%}, asks -{ask_drop:.0%}, CV={flatness:.2f}",
                "price_ticks": price_ticks,
                "bid_drop_pct": round(bid_drop, 3),
                "ask_drop_pct": round(ask_drop, 3),
                "volume_cv": round(flatness, 3),
            }, snap)

    def _detect_emergent(self, snap, price_delta, price_ticks):
        """
        Signal 6: EMERGENT / DISCOVERED
        Unusual patterns that don't fit the above categories.
        """
        # Pattern A: Extreme imbalance with no movement = coiled spring
        imb = self._imbalance_ratio("v10")
        if abs(price_ticks) <= 1 and (imb > 3.0 or imb < 0.33):
            dominant = "bids" if imb > 3.0 else "asks"
            direction = "BUY" if dominant == "bids" else "SELL"
            self.emit_signal("EMERGENT_COILED_SPRING", direction, 0.4, {
                "reason": f"Extreme {dominant} pressure ({imb:.1f}x) with no movement — spring loaded",
                "imbalance_L10": round(imb, 3),
                "price_ticks": price_ticks,
            }, snap)

        # Pattern B: Spread blowout = liquidity vacuum
        if len(self.window) >= 10:
            early_sp = [s.spread for s in list(self.window)[:5]]
            late_sp = [s.spread for s in list(self.window)[-5:]]
            ea = sum(early_sp) / len(early_sp)
            la = sum(late_sp) / len(late_sp)
            if ea > 0 and la / ea > 2.0:
                self.emit_signal("EMERGENT_SPREAD_BLOW", "SELL", 0.5, {
                    "reason": f"Spread blowout: {ea:.4f} -> {la:.4f} ({la/ea:.1f}x)",
                    "early_spread": round(ea, 4),
                    "late_spread": round(la, 4),
                }, snap)

        # Pattern C: Volume surge on one side while other collapses
        if len(self.window) >= 10:
            _, _, bid_trend = self._volume_trend("bid", "v10")
            _, _, ask_trend = self._volume_trend("ask", "v10")
            # One surging > 50% while other dropping > 30%
            if bid_trend > 0.50 and ask_trend < -0.30:
                self.emit_signal("EMERGENT_TSUNAMI", "BUY", 0.6, {
                    "reason": f"Bid tsunami: bids +{bid_trend:.0%}, asks {ask_trend:.0%}",
                    "bid_trend": round(bid_trend, 3),
                    "ask_trend": round(ask_trend, 3),
                }, snap)
            elif ask_trend > 0.50 and bid_trend < -0.30:
                self.emit_signal("EMERGENT_TSUNAMI", "SELL", 0.6, {
                    "reason": f"Ask tsunami: asks +{ask_trend:.0%}, bids {bid_trend:.0%}",
                    "bid_trend": round(bid_trend, 3),
                    "ask_trend": round(ask_trend, 3),
                }, snap)


# ─────────────────────────────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────────────────────────────

class WaveBot:

    def __init__(self):
        self.analyzer = WaveAnalyzer()
        self.running = True
        self.cycle_count = 0
        self.last_book_ts = None

        BOT_DIR.mkdir(parents=True, exist_ok=True)

        sig_module.signal(sig_module.SIGINT, self._shutdown)
        sig_module.signal(sig_module.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        print("\n\n  Wave bot shutting down...")
        self.running = False

    def read_book(self) -> Optional[UnifiedBook]:
        data = safe_read_json(CALL_FILE)
        if data is None:
            return None
        return UnifiedBook.from_call_json(data)

    def print_depth_profile(self, book: UnifiedBook):
        profile = book.depth_profile()
        sp = profile["spread"]

        print("\n" + "=" * 74)
        print("  WAVE DEPTH PROFILE")
        print("=" * 74)
        print(f"  Midpoint: {book.midpoint:.4f}  |  "
              f"Best Bid: {book.best_bid:.3f}  |  Best Ask: {book.best_ask:.3f}")
        print(f"  Spread: {sp['ticks']} ticks ({sp['dollars']:.4f}$)  "
              f"|  Spread forming: {sp['spread_side']} side")

        bq = sp["bid_quasi"]
        aq = sp["ask_quasi"]
        if bq["detected"]:
            print(f"  >> BID quasi-spread: front/back={bq['ratio']:.2f}, "
                  f"thin={bq['thin_levels']} levels "
                  f"(front_avg={bq['front_avg']:.0f} vs behind_avg={bq['behind_avg']:.0f})")
        if aq["detected"]:
            print(f"  >> ASK quasi-spread: front/back={aq['ratio']:.2f}, "
                  f"thin={aq['thin_levels']} levels "
                  f"(front_avg={aq['front_avg']:.0f} vs behind_avg={aq['behind_avg']:.0f})")

        print()
        hdr = f"  {'Depth':<8} {'Bid Vol':>10} {'Ask Vol':>10} {'Imbalance':>12} {'B.Lvls':>7} {'A.Lvls':>7}"
        print(hdr)
        print("  " + "-" * 60)
        for key in ["L3", "L5", "L10", "L20"]:
            d = profile[key]
            imb = d["imbalance"]
            tag = ""
            if imb > WAVE_IMBALANCE_RATIO:
                tag = " BID>>"
            elif imb < 1 / WAVE_IMBALANCE_RATIO:
                tag = " <<ASK"
            print(f"  {key:<8} {d['bid_vol']:>10.1f} {d['ask_vol']:>10.1f} "
                  f"{imb:>8.2f}x{tag:<4} {d['bid_levels_actual']:>7} {d['ask_levels_actual']:>7}")

        t = profile["totals"]
        print(f"  {'TOTAL':<8} {t['bid_volume']:>10.1f} {t['ask_volume']:>10.1f} "
              f"{t['ratio']:>8.2f}x")
        print("=" * 74)

    def print_wave_bar(self, snap: WaveSnapshot):
        """Compact live status line."""
        w = self.analyzer.window
        if len(w) >= 2:
            delta = snap.midpoint - w[0].midpoint
            ticks = round(delta / PRICE_TICK)
            trend = f"{'+'if ticks>=0 else ''}{ticks}t" if ticks != 0 else " 0t"
        else:
            trend = " --"

        imb = snap.bid_v10 / max(snap.ask_v10, 0.01)
        bar_len = 20
        bid_pct = imb / (imb + 1)
        bid_bar = min(int(bid_pct * bar_len), bar_len)
        ask_bar = bar_len - bid_bar

        bq = "Q" if snap.bid_quasi["detected"] else "."
        aq = "Q" if snap.ask_quasi["detected"] else "."

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-4]

        sys.stdout.write(
            f"\r  [{ts}] mid={snap.midpoint:.4f} {trend:>4} "
            f"| {bq}" + "\u2588" * bid_bar + "\u2591" * ask_bar + f"{aq} "
            f"| v10: {snap.bid_v10:>7.0f} vs {snap.ask_v10:>7.0f} "
            f"| {imb:>5.2f}x "
            f"| spr={snap.spread_ticks:>2}t "
            f"| w={len(w):>2}/{WINDOW_SIZE}"
        )
        sys.stdout.flush()

    def run(self):
        print()
        print("  " + "~" * 60)
        print("  WAVE ANALYSIS BOT - Polymarket 15M Binary Options")
        print(f"  Snapshot: {SNAPSHOT_INTERVAL}s | Window: {WINDOW_SIZE} ({WINDOW_SIZE * SNAPSHOT_INTERVAL}s)")
        print(f"  Signals -> {SIGNAL_LOG}")
        print(f"  State   -> {STATE_LOG}")
        print(f"  Source  -> {CALL_FILE}")
        print("  " + "~" * 60)
        print()

        # Wait for first valid book
        print("  Waiting for order book data...")
        while self.running:
            book = self.read_book()
            if book and (len(book.bids) > 0 or len(book.asks) > 0):
                self.print_depth_profile(book)
                break
            time.sleep(0.2)

        if not self.running:
            return

        print("\n  Live wave monitoring started (Ctrl+C to stop)\n")

        while self.running:
            t0 = time.monotonic()
            self.cycle_count += 1

            # Read
            book = self.read_book()
            if book is None:
                time.sleep(SNAPSHOT_INTERVAL)
                continue

            # Skip if book timestamp hasn't changed (no new data)
            book_ts = book.timestamp_ms
            if book_ts == self.last_book_ts:
                time.sleep(SNAPSHOT_INTERVAL)
                continue
            self.last_book_ts = book_ts

            # Snapshot + analyze
            snap = WaveSnapshot(book)
            self.analyzer.add_snapshot(snap)
            self.analyzer.detect_signals()

            # Display
            self.print_wave_bar(snap)

            # Periodic full profile every 10s
            if self.cycle_count % 100 == 0:
                print()
                self.print_depth_profile(book)
                print()

            # Save state every 3s
            if self.cycle_count % 30 == 0:
                self._save_state(snap)

            # Sleep remainder
            elapsed = time.monotonic() - t0
            time.sleep(max(0, SNAPSHOT_INTERVAL - elapsed))

        # Final save
        if self.analyzer.window:
            self._save_state(self.analyzer.window[-1])
        print(f"\n\n  Stopped. Total signals emitted: {self.analyzer.signal_count}")
        print(f"  Signal log: {SIGNAL_LOG}\n")

    def _save_state(self, snap: Optional[WaveSnapshot]):
        if snap is None:
            return
        state = {
            "updated_at": now_iso(),
            "cycle_count": self.cycle_count,
            "window_fill": len(self.analyzer.window),
            "total_signals": self.analyzer.signal_count,
            "current": snap.to_dict(),
            "window_midpoints": [round(s.midpoint, 4) for s in self.analyzer.window],
        }
        atomic_write_json(STATE_LOG, state)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = WaveBot()
    bot.run()
