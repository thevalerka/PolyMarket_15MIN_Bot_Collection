#!/usr/bin/env python3
"""
Advanced Market Maker Engine for Polymarket Binary Options
==========================================================

Implements all advanced MM techniques:
1. Asymmetric Quote Skewing (momentum-based)
2. Inventory-Based Skewing
3. Quote Fading on Volatility Spikes
4. Layered/Tiered Quoting
5. Lead-Lag Arbitrage Detection
6. Greeks-Aware Position Management
7. Order Flow Toxicity Detection
8. Time-of-Period Awareness
9. Correlation-Based Hedging
10. Regime Detection

Author: Claude for Valerio
"""

import json
import time
import math
import statistics
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Deque
from collections import deque
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class TradePeriodPhase(Enum):
    EARLY = "early"           # 0-5 min: Low gamma, price discovery
    MID = "mid"               # 5-10 min: Building conviction
    LATE = "late"             # 10-14 min: High gamma, dangerous
    FINAL = "final"           # 14-15 min: Extreme gamma, don't quote


class OrderFlowToxicity(Enum):
    LOW = "low"               # Retail/noise - safe to quote tight
    MEDIUM = "medium"         # Mixed signals
    HIGH = "high"             # Informed flow - widen or pull
    EXTREME = "extreme"       # Run away


class QuoteAction(Enum):
    QUOTE_BOTH = "quote_both"
    QUOTE_BID_ONLY = "quote_bid_only"
    QUOTE_ASK_ONLY = "quote_ask_only"
    WIDEN_SPREADS = "widen_spreads"
    PULL_QUOTES = "pull_quotes"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriceData:
    """Current price data from a source."""
    source: str
    price: float
    timestamp: int
    direction: Optional[str] = None
    direction_15s: Optional[str] = None
    direction_30s: Optional[str] = None


@dataclass
class OptionsData:
    """Current options market data."""
    asset_type: str  # 'CALL' or 'PUT'
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    timestamp: int
    
    @property
    def implied_prob(self) -> float:
        """Implied probability from mid price."""
        return self.mid


@dataclass
class Position:
    """Current position in an asset."""
    asset_type: str  # 'CALL' or 'PUT'
    size: float  # Positive = long, negative = short
    avg_entry: float
    unrealized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        return self.size < 0


@dataclass
class Quote:
    """A quote to place in the market."""
    asset_type: str
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    layer: int = 0  # For tiered quoting


@dataclass
class MarketState:
    """Complete market state snapshot."""
    # Price sources
    bybit: Optional[PriceData] = None
    chainlink: Optional[PriceData] = None
    coinbase: Optional[PriceData] = None
    
    # Options
    call: Optional[OptionsData] = None
    put: Optional[OptionsData] = None
    
    # Strike info
    strike_price: Optional[float] = None
    strike_timestamp: Optional[int] = None
    
    # Derived
    regime: MarketRegime = MarketRegime.UNKNOWN
    phase: TradePeriodPhase = TradePeriodPhase.EARLY
    toxicity: OrderFlowToxicity = OrderFlowToxicity.LOW
    
    # Timing
    time_to_expiry_seconds: float = 900  # 15 minutes
    timestamp: int = 0


@dataclass
class MMConfig:
    """Market maker configuration."""
    # Base spreads
    base_spread: float = 0.02  # 2 cents
    min_spread: float = 0.01
    max_spread: float = 0.10
    
    # Position limits
    max_position_size: float = 1000.0
    max_inventory_imbalance: float = 500.0
    
    # Skewing
    inventory_skew_factor: float = 0.001  # Per unit of inventory
    momentum_skew_factor: float = 0.005
    
    # Volatility
    vol_spike_threshold: float = 0.002  # 0.2% move in short window
    vol_pullback_seconds: float = 5.0
    
    # Time decay
    gamma_danger_threshold: float = 120  # seconds to expiry
    
    # Lead-lag
    lead_lag_threshold_ms: float = 100  # ms advantage needed
    
    # Tiered quoting
    tier_sizes: List[float] = field(default_factory=lambda: [50, 100, 200, 500])
    tier_spreads: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.03, 0.05])


# =============================================================================
# ANALYSIS MODULES
# =============================================================================

class LeadLagAnalyzer:
    """
    Technique #5: Lead-Lag Arbitrage Detection
    Determines which price source leads and provides early signals.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: Dict[str, Deque[Tuple[int, float]]] = {
            'bybit': deque(maxlen=window_size),
            'chainlink': deque(maxlen=window_size),
            'coinbase': deque(maxlen=window_size),
        }
        self.leadership_scores: Dict[str, int] = {
            'bybit': 0, 'chainlink': 0, 'coinbase': 0
        }
        self.last_move_leader: Optional[str] = None
        self.move_threshold: float = 5.0  # $5 minimum move
        
    def update(self, source: str, timestamp: int, price: float):
        """Record a price update."""
        self.price_history[source].append((timestamp, price))
        
    def detect_leader(self) -> Tuple[Optional[str], float]:
        """
        Detect which source is currently leading.
        Returns (leader_source, confidence_score)
        """
        # Need enough history
        min_samples = 10
        for source, history in self.price_history.items():
            if len(history) < min_samples:
                return None, 0.0
        
        # Check for recent significant moves
        moves = {}
        for source, history in self.price_history.items():
            if len(history) >= 2:
                recent_prices = [p for _, p in list(history)[-10:]]
                if len(recent_prices) >= 2:
                    move = recent_prices[-1] - recent_prices[0]
                    if abs(move) >= self.move_threshold:
                        # Find when the move started
                        first_ts = list(history)[-10][0]
                        moves[source] = (first_ts, move)
        
        if not moves:
            return None, 0.0
        
        # Leader is the one who moved first
        sorted_moves = sorted(moves.items(), key=lambda x: x[1][0])
        leader = sorted_moves[0][0]
        
        # Confidence based on time gap to next mover
        if len(sorted_moves) > 1:
            time_gap = sorted_moves[1][1][0] - sorted_moves[0][1][0]
            confidence = min(1.0, time_gap / 500)  # 500ms = max confidence
        else:
            confidence = 1.0
        
        self.leadership_scores[leader] += 1
        self.last_move_leader = leader
        
        return leader, confidence
    
    def get_predicted_direction(self, state: MarketState) -> Optional[str]:
        """
        Use the leading source to predict where lagging sources will go.
        """
        leader, confidence = self.detect_leader()
        
        if leader and confidence > 0.5:
            leader_data = getattr(state, leader)
            if leader_data and leader_data.direction:
                return leader_data.direction
        
        # Fallback to Bybit's direction signals
        if state.bybit and state.bybit.direction:
            return state.bybit.direction
        
        return None


class RegimeDetector:
    """
    Technique #10: Regime Detection
    Identifies current market regime for strategy adaptation.
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.price_history: Deque[Tuple[int, float]] = deque(maxlen=window_size)
        self.return_history: Deque[float] = deque(maxlen=window_size)
        
    def update(self, timestamp: int, price: float):
        """Add a price observation."""
        if self.price_history:
            last_price = self.price_history[-1][1]
            if last_price > 0:
                ret = (price - last_price) / last_price
                self.return_history.append(ret)
        
        self.price_history.append((timestamp, price))
    
    def detect_regime(self) -> MarketRegime:
        """Detect the current market regime."""
        if len(self.return_history) < 10:
            return MarketRegime.UNKNOWN
        
        returns = list(self.return_history)
        
        # Calculate metrics
        mean_return = statistics.mean(returns)
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Count direction changes (mean reversion indicator)
        direction_changes = 0
        for i in range(1, len(returns)):
            if (returns[i] > 0) != (returns[i-1] > 0):
                direction_changes += 1
        
        mean_reversion_ratio = direction_changes / len(returns) if returns else 0
        
        # Determine regime
        vol_threshold_high = 0.001  # 0.1%
        vol_threshold_low = 0.0002  # 0.02%
        trend_threshold = 0.0001  # 0.01%
        mean_reversion_threshold = 0.6
        
        if volatility > vol_threshold_high:
            return MarketRegime.HIGH_VOLATILITY
        
        if volatility < vol_threshold_low:
            return MarketRegime.LOW_VOLATILITY
        
        if mean_reversion_ratio > mean_reversion_threshold:
            return MarketRegime.MEAN_REVERTING
        
        if mean_return > trend_threshold:
            return MarketRegime.TRENDING_UP
        elif mean_return < -trend_threshold:
            return MarketRegime.TRENDING_DOWN
        
        return MarketRegime.MEAN_REVERTING


class VolatilityMonitor:
    """
    Technique #3: Quote Fading on Volatility Spikes
    Detects sudden volatility spikes for quote management.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 60):
        self.short_window = short_window
        self.long_window = long_window
        self.price_history: Deque[Tuple[int, float]] = deque(maxlen=long_window)
        self.spike_detected_at: Optional[int] = None
        self.spike_cooldown_ms: int = 5000  # 5 seconds cooldown
        
    def update(self, timestamp: int, price: float):
        """Add price observation."""
        self.price_history.append((timestamp, price))
    
    def get_volatility(self, window: int) -> float:
        """Calculate volatility over specified window."""
        if len(self.price_history) < window:
            return 0.0
        
        prices = [p for _, p in list(self.price_history)[-window:]]
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                   for i in range(1, len(prices)) if prices[i-1] > 0]
        
        if not returns:
            return 0.0
        
        return statistics.stdev(returns) if len(returns) > 1 else abs(returns[0])
    
    def detect_spike(self, threshold: float = 2.0) -> bool:
        """
        Detect if current short-term vol is spiking vs long-term.
        Returns True if vol spike detected.
        """
        short_vol = self.get_volatility(self.short_window)
        long_vol = self.get_volatility(self.long_window)
        
        if long_vol > 0 and short_vol > threshold * long_vol:
            now = int(time.time() * 1000)
            self.spike_detected_at = now
            return True
        
        return False
    
    def is_in_cooldown(self) -> bool:
        """Check if we're still in post-spike cooldown."""
        if self.spike_detected_at is None:
            return False
        
        now = int(time.time() * 1000)
        return (now - self.spike_detected_at) < self.spike_cooldown_ms
    
    def should_pull_quotes(self, threshold: float = 2.0) -> bool:
        """Determine if quotes should be pulled due to volatility."""
        return self.detect_spike(threshold) or self.is_in_cooldown()


class OrderFlowAnalyzer:
    """
    Technique #7: Order Flow Toxicity Detection
    Analyzes order flow patterns to detect informed trading.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.trades: Deque[dict] = deque(maxlen=window_size)
        
    def record_trade(self, side: str, size: float, aggressor: str, timestamp: int):
        """
        Record a trade execution.
        side: 'buy' or 'sell'
        aggressor: 'taker' or 'maker'
        """
        self.trades.append({
            'side': side,
            'size': size,
            'aggressor': aggressor,
            'timestamp': timestamp,
        })
    
    def calculate_toxicity(self) -> OrderFlowToxicity:
        """
        Calculate order flow toxicity based on recent trades.
        """
        if len(self.trades) < 5:
            return OrderFlowToxicity.LOW
        
        recent = list(self.trades)[-20:]
        
        # Calculate buy/sell imbalance
        buy_volume = sum(t['size'] for t in recent if t['side'] == 'buy')
        sell_volume = sum(t['size'] for t in recent if t['side'] == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return OrderFlowToxicity.LOW
        
        imbalance = abs(buy_volume - sell_volume) / total_volume
        
        # Check for large aggressive orders
        avg_size = total_volume / len(recent)
        large_orders = [t for t in recent if t['size'] > 2 * avg_size]
        large_order_ratio = len(large_orders) / len(recent)
        
        # Calculate toxicity score
        toxicity_score = imbalance * 0.5 + large_order_ratio * 0.5
        
        if toxicity_score > 0.7:
            return OrderFlowToxicity.EXTREME
        elif toxicity_score > 0.5:
            return OrderFlowToxicity.HIGH
        elif toxicity_score > 0.3:
            return OrderFlowToxicity.MEDIUM
        else:
            return OrderFlowToxicity.LOW
    
    def get_dominant_side(self) -> Optional[str]:
        """Get the dominant order flow side."""
        if len(self.trades) < 5:
            return None
        
        recent = list(self.trades)[-20:]
        buy_volume = sum(t['size'] for t in recent if t['side'] == 'buy')
        sell_volume = sum(t['size'] for t in recent if t['side'] == 'sell')
        
        if buy_volume > sell_volume * 1.2:
            return 'buy'
        elif sell_volume > buy_volume * 1.2:
            return 'sell'
        return None


class GreeksCalculator:
    """
    Technique #6: Greeks-Aware Position Management
    Calculates option Greeks for binary options.
    """
    
    @staticmethod
    def binary_delta(price: float, strike: float, time_to_expiry: float, 
                     volatility: float) -> float:
        """
        Calculate delta for a binary option.
        For binary options, delta represents the probability density at strike.
        """
        if time_to_expiry <= 0 or volatility <= 0:
            # At expiry, delta is infinite at strike, 0 elsewhere
            return 1.0 if abs(price - strike) < 0.01 * strike else 0.0
        
        # Simplified binary delta approximation
        # Delta = d(prob)/d(price) ≈ PDF at strike
        sigma_sqrt_t = volatility * math.sqrt(time_to_expiry / 31536000)  # Annualized
        
        if sigma_sqrt_t <= 0:
            return 0.0
        
        d = (price - strike) / (strike * sigma_sqrt_t)
        
        # Normal PDF
        pdf = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
        delta = pdf / (strike * sigma_sqrt_t)
        
        return min(delta, 10.0)  # Cap at reasonable value
    
    @staticmethod
    def binary_gamma(price: float, strike: float, time_to_expiry: float,
                     volatility: float) -> float:
        """
        Calculate gamma for a binary option.
        Gamma = d(delta)/d(price)
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return float('inf') if abs(price - strike) < 0.01 * strike else 0.0
        
        # Delta at slightly different prices
        epsilon = price * 0.0001  # 0.01%
        delta_up = GreeksCalculator.binary_delta(price + epsilon, strike, 
                                                  time_to_expiry, volatility)
        delta_down = GreeksCalculator.binary_delta(price - epsilon, strike,
                                                    time_to_expiry, volatility)
        
        gamma = (delta_up - delta_down) / (2 * epsilon)
        return abs(gamma)
    
    @staticmethod
    def time_decay_factor(time_to_expiry: float) -> float:
        """
        Get spread multiplier based on time to expiry.
        As expiry approaches, gamma increases, so spreads should widen.
        """
        if time_to_expiry <= 0:
            return 10.0  # Maximum widening
        
        # Exponential increase as time decreases
        # At 15 min: 1.0x, at 5 min: 1.5x, at 1 min: 3x, at 10s: 5x
        if time_to_expiry > 600:  # > 10 min
            return 1.0
        elif time_to_expiry > 300:  # 5-10 min
            return 1.0 + (600 - time_to_expiry) / 600
        elif time_to_expiry > 60:  # 1-5 min
            return 1.5 + (300 - time_to_expiry) / 160
        elif time_to_expiry > 10:  # 10s - 1 min
            return 3.0 + (60 - time_to_expiry) / 25
        else:  # < 10s
            return 5.0 + (10 - time_to_expiry)


class PeriodPhaseManager:
    """
    Technique #8: Time-of-Period Awareness
    Manages strategy based on where we are in the 15-minute period.
    """
    
    @staticmethod
    def get_phase(time_to_expiry_seconds: float) -> TradePeriodPhase:
        """Determine current phase of the trading period."""
        if time_to_expiry_seconds > 600:  # > 10 min
            return TradePeriodPhase.EARLY
        elif time_to_expiry_seconds > 300:  # 5-10 min
            return TradePeriodPhase.MID
        elif time_to_expiry_seconds > 60:  # 1-5 min
            return TradePeriodPhase.LATE
        else:  # < 1 min
            return TradePeriodPhase.FINAL
    
    @staticmethod
    def get_phase_config(phase: TradePeriodPhase) -> dict:
        """Get configuration adjustments for current phase."""
        configs = {
            TradePeriodPhase.EARLY: {
                'spread_multiplier': 1.0,
                'max_position_multiplier': 1.0,
                'quote_both_sides': True,
                'aggressive_quoting': True,
            },
            TradePeriodPhase.MID: {
                'spread_multiplier': 1.2,
                'max_position_multiplier': 0.8,
                'quote_both_sides': True,
                'aggressive_quoting': True,
            },
            TradePeriodPhase.LATE: {
                'spread_multiplier': 1.5,
                'max_position_multiplier': 0.5,
                'quote_both_sides': True,
                'aggressive_quoting': False,
            },
            TradePeriodPhase.FINAL: {
                'spread_multiplier': 3.0,
                'max_position_multiplier': 0.0,  # No new positions
                'quote_both_sides': False,
                'aggressive_quoting': False,
            },
        }
        return configs.get(phase, configs[TradePeriodPhase.EARLY])


# =============================================================================
# MAIN MARKET MAKER ENGINE
# =============================================================================

class AdvancedMarketMaker:
    """
    Main market maker engine integrating all techniques.
    """
    
    def __init__(self, config: Optional[MMConfig] = None):
        self.config = config or MMConfig()
        
        # Analysis modules
        self.lead_lag = LeadLagAnalyzer()
        self.regime_detector = RegimeDetector()
        self.vol_monitor = VolatilityMonitor()
        self.flow_analyzer = OrderFlowAnalyzer()
        
        # State
        self.positions: Dict[str, Position] = {}
        self.last_quotes: Dict[str, List[Quote]] = {'CALL': [], 'PUT': []}
        self.last_state: Optional[MarketState] = None
        
        # Performance tracking
        self.quotes_generated = 0
        self.quotes_pulled = 0
        
    def update_market_state(self, state: MarketState):
        """Update all analyzers with new market state."""
        # Update lead-lag analyzer
        if state.bybit:
            self.lead_lag.update('bybit', state.bybit.timestamp, state.bybit.price)
        if state.chainlink:
            self.lead_lag.update('chainlink', state.chainlink.timestamp, state.chainlink.price)
        if state.coinbase:
            self.lead_lag.update('coinbase', state.coinbase.timestamp, state.coinbase.price)
        
        # Update regime detector with primary price source
        primary_price = state.bybit or state.chainlink or state.coinbase
        if primary_price:
            self.regime_detector.update(primary_price.timestamp, primary_price.price)
            self.vol_monitor.update(primary_price.timestamp, primary_price.price)
        
        # Update derived state
        state.regime = self.regime_detector.detect_regime()
        state.phase = PeriodPhaseManager.get_phase(state.time_to_expiry_seconds)
        state.toxicity = self.flow_analyzer.calculate_toxicity()
        
        self.last_state = state
    
    def calculate_fair_value(self, state: MarketState, asset_type: str) -> float:
        """
        Calculate fair value for CALL or PUT.
        
        Strategy: Use MARKET MID as the base fair value, then apply adjustments.
        The market is generally efficient, so we trust its pricing and only
        make small adjustments based on our edge (momentum, inventory, etc.)
        
        We do NOT try to calculate theoretical probability from BTC price alone
        because that requires accurate volatility estimates we don't have.
        """
        # PRIMARY: Use current market mid price
        if asset_type == 'CALL' and state.call:
            return state.call.mid
        elif asset_type == 'PUT' and state.put:
            return state.put.mid
        
        # FALLBACK: If no market data, estimate from price vs strike
        if state.chainlink and state.strike_price:
            price = state.chainlink.price
            strike = state.strike_price
            
            # Simple linear approximation near the strike
            # This is just a rough fallback, not meant to be precise
            distance_pct = (price - strike) / strike
            
            # Clamp probability to reasonable range
            if asset_type == 'CALL':
                # CALL value increases when price > strike
                prob = 0.5 + distance_pct * 10  # 10x leverage for small moves
                return max(0.05, min(0.95, prob))
            else:  # PUT
                # PUT value increases when price < strike
                prob = 0.5 - distance_pct * 10
                return max(0.05, min(0.95, prob))
        
        # Ultimate fallback
        return 0.5
    
    def _estimate_implied_vol(self, state: MarketState) -> float:
        """Estimate implied volatility from market data."""
        # Use realized vol from our monitor as proxy
        realized_vol = self.vol_monitor.get_volatility(30)
        
        # Annualize (assuming ~1 second samples)
        annualized = realized_vol * math.sqrt(31536000)
        
        # Clamp to reasonable range
        return max(0.3, min(2.0, annualized)) if annualized > 0 else 0.5
    
    def calculate_inventory_skew(self, asset_type: str) -> float:
        """
        Technique #2: Inventory-Based Skewing
        Returns price adjustment based on current inventory.
        Positive = raise prices (reduce buying), Negative = lower prices.
        """
        position = self.positions.get(asset_type)
        if not position:
            return 0.0
        
        # Skew proportional to position size
        skew = position.size * self.config.inventory_skew_factor
        
        return skew
    
    def calculate_momentum_skew(self, state: MarketState) -> float:
        """
        Technique #1: Asymmetric Quote Skewing
        Returns price adjustment based on detected momentum.
        """
        # Use lead-lag analyzer for direction
        predicted_dir = self.lead_lag.get_predicted_direction(state)
        
        if predicted_dir == 'UP':
            return self.config.momentum_skew_factor
        elif predicted_dir == 'DOWN':
            return -self.config.momentum_skew_factor
        
        # Also check Bybit's short-term direction
        if state.bybit:
            if state.bybit.direction_15s == 'UP':
                return self.config.momentum_skew_factor * 0.5
            elif state.bybit.direction_15s == 'DOWN':
                return -self.config.momentum_skew_factor * 0.5
        
        return 0.0
    
    def calculate_spread(self, state: MarketState, asset_type: str) -> float:
        """
        Calculate appropriate spread based on all factors.
        """
        base_spread = self.config.base_spread
        
        # Time decay factor (Technique #6 & #8)
        time_factor = GreeksCalculator.time_decay_factor(state.time_to_expiry_seconds)
        
        # Regime factor (Technique #10)
        regime_factors = {
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.MEAN_REVERTING: 1.0,
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.HIGH_VOLATILITY: 2.0,
            MarketRegime.UNKNOWN: 1.5,
        }
        regime_factor = regime_factors.get(state.regime, 1.0)
        
        # Toxicity factor (Technique #7)
        toxicity_factors = {
            OrderFlowToxicity.LOW: 1.0,
            OrderFlowToxicity.MEDIUM: 1.3,
            OrderFlowToxicity.HIGH: 1.8,
            OrderFlowToxicity.EXTREME: 3.0,
        }
        toxicity_factor = toxicity_factors.get(state.toxicity, 1.0)
        
        # Volatility spike factor (Technique #3)
        vol_factor = 2.0 if self.vol_monitor.should_pull_quotes() else 1.0
        
        # Phase factor (Technique #8)
        phase_config = PeriodPhaseManager.get_phase_config(state.phase)
        phase_factor = phase_config['spread_multiplier']
        
        # Combine all factors
        final_spread = base_spread * time_factor * regime_factor * toxicity_factor * vol_factor * phase_factor
        
        # Clamp to configured limits
        return max(self.config.min_spread, min(self.config.max_spread, final_spread))
    
    def should_quote(self, state: MarketState, asset_type: str, side: str) -> bool:
        """
        Determine if we should quote this side based on all signals.
        """
        # Check volatility (Technique #3)
        if self.vol_monitor.should_pull_quotes(threshold=3.0):
            logger.warning("Pulling quotes due to volatility spike")
            return False
        
        # Check phase (Technique #8)
        phase_config = PeriodPhaseManager.get_phase_config(state.phase)
        if state.phase == TradePeriodPhase.FINAL:
            # Only quote to reduce position, not to add
            position = self.positions.get(asset_type)
            if position:
                if side == 'bid' and position.is_long:
                    return False  # Don't buy more if already long
                if side == 'ask' and position.is_short:
                    return False  # Don't sell more if already short
            else:
                return False  # No position, don't start one in final phase
        
        # Check toxicity (Technique #7)
        if state.toxicity == OrderFlowToxicity.EXTREME:
            logger.warning("Pulling quotes due to extreme order flow toxicity")
            return False
        
        # Check momentum for asymmetric quoting (Technique #1)
        predicted_dir = self.lead_lag.get_predicted_direction(state)
        
        if predicted_dir == 'UP':
            # Price going up: be careful about selling CALLs or buying PUTs
            if asset_type == 'CALL' and side == 'ask':
                if state.toxicity in [OrderFlowToxicity.HIGH, OrderFlowToxicity.EXTREME]:
                    return False
            if asset_type == 'PUT' and side == 'bid':
                if state.toxicity in [OrderFlowToxicity.HIGH, OrderFlowToxicity.EXTREME]:
                    return False
        
        elif predicted_dir == 'DOWN':
            # Price going down: be careful about buying CALLs or selling PUTs
            if asset_type == 'CALL' and side == 'bid':
                if state.toxicity in [OrderFlowToxicity.HIGH, OrderFlowToxicity.EXTREME]:
                    return False
            if asset_type == 'PUT' and side == 'ask':
                if state.toxicity in [OrderFlowToxicity.HIGH, OrderFlowToxicity.EXTREME]:
                    return False
        
        return True
    
    def generate_tiered_quotes(self, state: MarketState, asset_type: str) -> List[Quote]:
        """
        Technique #4: Layered/Tiered Quoting
        Generate multiple quote layers with increasing size and spread.
        """
        quotes = []
        
        fair_value = self.calculate_fair_value(state, asset_type)
        inventory_skew = self.calculate_inventory_skew(asset_type)
        momentum_skew = self.calculate_momentum_skew(state)
        base_spread = self.calculate_spread(state, asset_type)
        
        # Apply skews to mid price
        adjusted_mid = fair_value + inventory_skew + momentum_skew
        
        # Generate tiered quotes
        for layer, (size, spread_mult) in enumerate(zip(
            self.config.tier_sizes, 
            self.config.tier_spreads
        )):
            layer_spread = base_spread * (1 + spread_mult * layer)
            half_spread = layer_spread / 2
            
            # Bid quote
            if self.should_quote(state, asset_type, 'bid'):
                bid_price = adjusted_mid - half_spread
                # Additional momentum adjustment for bids
                if momentum_skew > 0:  # Upward momentum
                    bid_price -= momentum_skew * 0.5  # More conservative bid
                
                bid_price = max(0.01, min(0.99, bid_price))
                
                quotes.append(Quote(
                    asset_type=asset_type,
                    side='bid',
                    price=round(bid_price, 2),
                    size=size,
                    layer=layer,
                ))
            
            # Ask quote
            if self.should_quote(state, asset_type, 'ask'):
                ask_price = adjusted_mid + half_spread
                # Additional momentum adjustment for asks
                if momentum_skew < 0:  # Downward momentum
                    ask_price -= momentum_skew * 0.5  # More conservative ask (higher)
                
                ask_price = max(0.01, min(0.99, ask_price))
                
                quotes.append(Quote(
                    asset_type=asset_type,
                    side='ask',
                    price=round(ask_price, 2),
                    size=size,
                    layer=layer,
                ))
        
        self.quotes_generated += len(quotes)
        return quotes
    
    def evaluate_hedge(self, state: MarketState) -> Optional[dict]:
        """
        Technique #9: Correlation-Based Hedging
        Evaluate if hedging with the opposite option is better than taking a loss.
        """
        call_pos = self.positions.get('CALL')
        put_pos = self.positions.get('PUT')
        
        recommendations = []
        
        # Check if CALL position needs hedging
        if call_pos and call_pos.unrealized_pnl < -0.05 * abs(call_pos.size):
            # CALL is losing money
            if state.put and state.call:
                # Cost to hedge with PUT
                hedge_size = abs(call_pos.size)
                hedge_cost = state.put.best_ask * hedge_size
                
                # Cost to close CALL at loss
                close_cost = abs(call_pos.unrealized_pnl)
                
                if hedge_cost < close_cost * 0.8:  # Hedge is significantly cheaper
                    recommendations.append({
                        'action': 'hedge',
                        'asset': 'PUT',
                        'side': 'buy',
                        'size': hedge_size,
                        'price': state.put.best_ask,
                        'reason': f'Hedge CALL loss: ${close_cost:.2f} vs hedge cost ${hedge_cost:.2f}'
                    })
        
        # Check if PUT position needs hedging
        if put_pos and put_pos.unrealized_pnl < -0.05 * abs(put_pos.size):
            if state.call and state.put:
                hedge_size = abs(put_pos.size)
                hedge_cost = state.call.best_ask * hedge_size
                close_cost = abs(put_pos.unrealized_pnl)
                
                if hedge_cost < close_cost * 0.8:
                    recommendations.append({
                        'action': 'hedge',
                        'asset': 'CALL',
                        'side': 'buy',
                        'size': hedge_size,
                        'price': state.call.best_ask,
                        'reason': f'Hedge PUT loss: ${close_cost:.2f} vs hedge cost ${hedge_cost:.2f}'
                    })
        
        return recommendations if recommendations else None
    
    def get_quote_action(self, state: MarketState) -> QuoteAction:
        """
        Determine overall quote action based on all signals.
        """
        # Volatility check (Technique #3)
        if self.vol_monitor.should_pull_quotes(threshold=3.0):
            return QuoteAction.PULL_QUOTES
        
        # Phase check (Technique #8)
        if state.phase == TradePeriodPhase.FINAL:
            return QuoteAction.PULL_QUOTES
        
        # Toxicity check (Technique #7)
        if state.toxicity == OrderFlowToxicity.EXTREME:
            return QuoteAction.PULL_QUOTES
        if state.toxicity == OrderFlowToxicity.HIGH:
            return QuoteAction.WIDEN_SPREADS
        
        # Regime check (Technique #10)
        if state.regime == MarketRegime.HIGH_VOLATILITY:
            return QuoteAction.WIDEN_SPREADS
        
        # Momentum/direction check for asymmetric quoting (Technique #1)
        predicted_dir = self.lead_lag.get_predicted_direction(state)
        
        if predicted_dir == 'UP' and state.toxicity == OrderFlowToxicity.MEDIUM:
            return QuoteAction.QUOTE_BID_ONLY  # Don't sell into rally
        elif predicted_dir == 'DOWN' and state.toxicity == OrderFlowToxicity.MEDIUM:
            return QuoteAction.QUOTE_ASK_ONLY  # Don't buy into selloff
        
        return QuoteAction.QUOTE_BOTH
    
    def generate_all_quotes(self, state: MarketState) -> Dict[str, List[Quote]]:
        """
        Main entry point: Generate all quotes based on current state.
        Returns dict with 'CALL' and 'PUT' quote lists.
        """
        self.update_market_state(state)
        
        action = self.get_quote_action(state)
        
        result = {'CALL': [], 'PUT': []}
        
        if action == QuoteAction.PULL_QUOTES:
            self.quotes_pulled += 1
            logger.info(f"Pulling all quotes: regime={state.regime.value}, "
                       f"phase={state.phase.value}, toxicity={state.toxicity.value}")
            return result
        
        # Generate quotes for both assets
        for asset_type in ['CALL', 'PUT']:
            quotes = self.generate_tiered_quotes(state, asset_type)
            
            # Filter based on action
            if action == QuoteAction.QUOTE_BID_ONLY:
                quotes = [q for q in quotes if q.side == 'bid']
            elif action == QuoteAction.QUOTE_ASK_ONLY:
                quotes = [q for q in quotes if q.side == 'ask']
            elif action == QuoteAction.WIDEN_SPREADS:
                # Widen by adjusting prices
                for q in quotes:
                    if q.side == 'bid':
                        q.price = max(0.01, q.price - 0.01)
                    else:
                        q.price = min(0.99, q.price + 0.01)
            
            result[asset_type] = quotes
        
        self.last_quotes = result
        return result
    
    def get_status_report(self, state: MarketState) -> dict:
        """Generate a status report of current MM state."""
        return {
            'regime': state.regime.value,
            'phase': state.phase.value,
            'toxicity': state.toxicity.value,
            'time_to_expiry': state.time_to_expiry_seconds,
            'vol_spike': self.vol_monitor.should_pull_quotes(),
            'leadership_scores': dict(self.lead_lag.leadership_scores),
            'predicted_direction': self.lead_lag.get_predicted_direction(state),
            'positions': {k: {'size': v.size, 'pnl': v.unrealized_pnl} 
                         for k, v in self.positions.items()},
            'quotes_generated': self.quotes_generated,
            'quotes_pulled': self.quotes_pulled,
        }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_market_state_from_files(
    bybit_path: str,
    chainlink_path: str,
    coinbase_path: str,
    call_path: str,
    put_path: str,
    period_start_timestamp: int
) -> MarketState:
    """
    Create MarketState from JSON files.
    """
    state = MarketState()
    
    # Load Bybit
    try:
        with open(bybit_path) as f:
            data = json.load(f)
            state.bybit = PriceData(
                source='bybit',
                price=data['price'],
                timestamp=data['timestamp'],
                direction=data.get('direction'),
                direction_15s=data.get('direction_15s'),
                direction_30s=data.get('direction_30s'),
            )
    except:
        pass
    
    # Load Chainlink
    try:
        with open(chainlink_path) as f:
            data = json.load(f)
            state.chainlink = PriceData(
                source='chainlink',
                price=data['price'],
                timestamp=data['timestamp'],
            )
            state.strike_price = data.get('strike')
            state.strike_timestamp = data.get('strike_timestamp')
    except:
        pass
    
    # Load Coinbase
    try:
        with open(coinbase_path) as f:
            data = json.load(f)
            state.coinbase = PriceData(
                source='coinbase',
                price=data['price'],
                timestamp=data['timestamp'],
            )
    except:
        pass
    
    # Load CALL options
    try:
        with open(call_path) as f:
            data = json.load(f)
            state.call = OptionsData(
                asset_type='CALL',
                best_bid=data['best_bid'],
                best_ask=data['best_ask'],
                mid=(data['best_bid'] + data['best_ask']) / 2,
                spread=data['spread'],
                timestamp=data['timestamp'],
            )
    except:
        pass
    
    # Load PUT options
    try:
        with open(put_path) as f:
            data = json.load(f)
            state.put = OptionsData(
                asset_type='PUT',
                best_bid=data['best_bid'],
                best_ask=data['best_ask'],
                mid=(data['best_bid'] + data['best_ask']) / 2,
                spread=data['spread'],
                timestamp=data['timestamp'],
            )
    except:
        pass
    
    # Calculate time to expiry
    now = int(time.time() * 1000)
    period_end = period_start_timestamp + (15 * 60 * 1000)  # 15 minutes
    state.time_to_expiry_seconds = max(0, (period_end - now) / 1000)
    state.timestamp = now
    
    return state
