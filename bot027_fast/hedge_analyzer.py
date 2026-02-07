#!/usr/bin/env python3
"""
Correlation-Based Hedging Module
================================

Technique #9: Correlation-Based Hedging

This module provides sophisticated hedging logic for binary options positions.
Instead of closing a losing position at a loss, it evaluates whether hedging
with the opposite option is more cost-effective.

Key Concepts:
- For binary options, CALL + PUT should sum to ~1.0 (minus spreads)
- If you're long CALL and price drops, instead of selling CALL at loss:
  * Consider buying PUT to create a synthetic hedge
  * If PUT is cheap enough, your max loss is capped

Example:
- Long 100 CALL @ 0.50 entry, current bid is 0.35 (loss: $15)
- PUT ask is 0.60
- Buying 100 PUT @ 0.60 costs $60
- Now you have both CALL and PUT:
  * If price goes UP: CALL pays $100, PUT pays $0, net = $100 - $50 - $60 = -$10
  * If price goes DOWN: CALL pays $0, PUT pays $100, net = $0 - $50 + $40 = -$10
- Max loss is capped at $10 vs potential $50 loss on CALL alone
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class HedgeAction(Enum):
    HOLD = "hold"                    # Do nothing
    CLOSE_POSITION = "close"         # Close at current market
    HEDGE_WITH_OPPOSITE = "hedge"    # Buy opposite option
    PARTIAL_HEDGE = "partial_hedge"  # Hedge part of position


@dataclass
class Position:
    """Represents a position in a binary option."""
    asset_type: str  # 'CALL' or 'PUT'
    size: float      # Positive = long, negative = short
    entry_price: float
    
    @property
    def is_long(self) -> bool:
        return self.size > 0
    
    @property
    def cost_basis(self) -> float:
        """Total cost to enter position."""
        return abs(self.size) * self.entry_price


@dataclass
class MarketPrices:
    """Current market prices for CALL and PUT."""
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    
    @property
    def call_mid(self) -> float:
        return (self.call_bid + self.call_ask) / 2
    
    @property
    def put_mid(self) -> float:
        return (self.put_bid + self.put_ask) / 2
    
    @property
    def market_sum(self) -> float:
        """CALL + PUT should be close to 1.0 in efficient market."""
        return self.call_mid + self.put_mid


@dataclass
class HedgeRecommendation:
    """Recommendation from hedge analysis."""
    action: HedgeAction
    asset: Optional[str] = None     # Which asset to trade
    side: Optional[str] = None      # 'buy' or 'sell'
    size: Optional[float] = None
    price: Optional[float] = None
    
    # Analysis
    current_pnl: float = 0.0
    pnl_if_close: float = 0.0
    pnl_if_hedge: float = 0.0
    max_loss_if_hedge: float = 0.0
    reasoning: str = ""


class HedgeAnalyzer:
    """
    Analyzes positions and provides hedging recommendations.
    """
    
    def __init__(
        self,
        min_edge_for_hedge: float = 0.1,    # Hedge must be 10% cheaper than closing
        max_hedge_cost_ratio: float = 0.7,   # Don't hedge if cost > 70% of loss
        slippage_estimate: float = 0.005,    # Estimated slippage per trade
    ):
        self.min_edge_for_hedge = min_edge_for_hedge
        self.max_hedge_cost_ratio = max_hedge_cost_ratio
        self.slippage_estimate = slippage_estimate
    
    def calculate_unrealized_pnl(self, position: Position, prices: MarketPrices) -> float:
        """Calculate current unrealized P&L."""
        if position.asset_type == 'CALL':
            exit_price = prices.call_bid if position.is_long else prices.call_ask
        else:
            exit_price = prices.put_bid if position.is_long else prices.put_ask
        
        if position.is_long:
            return (exit_price - position.entry_price) * abs(position.size)
        else:
            return (position.entry_price - exit_price) * abs(position.size)
    
    def calculate_close_pnl(self, position: Position, prices: MarketPrices) -> float:
        """Calculate P&L if we close position now (including slippage)."""
        pnl = self.calculate_unrealized_pnl(position, prices)
        slippage_cost = abs(position.size) * self.slippage_estimate
        return pnl - slippage_cost
    
    def calculate_hedge_outcomes(
        self, 
        position: Position, 
        prices: MarketPrices
    ) -> Tuple[float, float, float]:
        """
        Calculate outcomes if we hedge with opposite option.
        
        Returns: (hedge_cost, pnl_if_win, pnl_if_lose)
        
        For a CALL position:
          - "win" means price ends ABOVE strike (CALL pays 1.0)
          - "lose" means price ends BELOW strike (PUT pays 1.0)
        """
        size = abs(position.size)
        
        if position.asset_type == 'CALL' and position.is_long:
            # Long CALL, hedge by buying PUT
            hedge_cost = size * prices.put_ask * (1 + self.slippage_estimate)
            
            # If price ends ABOVE strike: CALL = 1.0, PUT = 0
            pnl_win = (1.0 - position.entry_price) * size - hedge_cost
            
            # If price ends BELOW strike: CALL = 0, PUT = 1.0
            pnl_lose = (0 - position.entry_price) * size + (1.0 * size - hedge_cost)
            
        elif position.asset_type == 'CALL' and not position.is_long:
            # Short CALL, hedge by buying CALL (reduces short)
            # Or buy PUT to profit if price drops
            hedge_cost = size * prices.put_ask * (1 + self.slippage_estimate)
            
            pnl_win = (position.entry_price - 1.0) * size - hedge_cost
            pnl_lose = position.entry_price * size + (1.0 * size - hedge_cost)
            
        elif position.asset_type == 'PUT' and position.is_long:
            # Long PUT, hedge by buying CALL
            hedge_cost = size * prices.call_ask * (1 + self.slippage_estimate)
            
            # If price ends ABOVE strike: PUT = 0, CALL = 1.0
            pnl_win = (0 - position.entry_price) * size + (1.0 * size - hedge_cost)
            
            # If price ends BELOW strike: PUT = 1.0, CALL = 0
            pnl_lose = (1.0 - position.entry_price) * size - hedge_cost
            
        else:  # Short PUT
            hedge_cost = size * prices.call_ask * (1 + self.slippage_estimate)
            
            pnl_win = position.entry_price * size + (1.0 * size - hedge_cost)
            pnl_lose = (position.entry_price - 1.0) * size - hedge_cost
        
        return hedge_cost, pnl_win, pnl_lose
    
    def analyze(self, position: Position, prices: MarketPrices) -> HedgeRecommendation:
        """
        Analyze position and provide hedge recommendation.
        """
        # Calculate current state
        unrealized_pnl = self.calculate_unrealized_pnl(position, prices)
        close_pnl = self.calculate_close_pnl(position, prices)
        
        # If not losing money, no need to hedge
        if unrealized_pnl >= 0:
            return HedgeRecommendation(
                action=HedgeAction.HOLD,
                current_pnl=unrealized_pnl,
                pnl_if_close=close_pnl,
                reasoning="Position is profitable, no hedge needed"
            )
        
        # Calculate hedge outcomes
        hedge_cost, pnl_win, pnl_lose = self.calculate_hedge_outcomes(position, prices)
        
        # Max loss if hedged
        max_loss_hedged = min(pnl_win, pnl_lose)
        
        # Compare: close now vs hedge
        # If we close, we lock in close_pnl (a loss)
        # If we hedge, max loss is max_loss_hedged
        
        loss_if_close = abs(close_pnl)
        loss_if_hedge = abs(max_loss_hedged)
        
        # Determine opposite asset
        opposite_asset = 'PUT' if position.asset_type == 'CALL' else 'CALL'
        hedge_price = prices.put_ask if opposite_asset == 'PUT' else prices.call_ask
        
        # Decision logic
        if loss_if_hedge < loss_if_close * (1 - self.min_edge_for_hedge):
            # Hedging is significantly better
            if hedge_cost < loss_if_close * self.max_hedge_cost_ratio:
                return HedgeRecommendation(
                    action=HedgeAction.HEDGE_WITH_OPPOSITE,
                    asset=opposite_asset,
                    side='buy',
                    size=abs(position.size),
                    price=hedge_price,
                    current_pnl=unrealized_pnl,
                    pnl_if_close=close_pnl,
                    pnl_if_hedge=max_loss_hedged,
                    max_loss_if_hedge=loss_if_hedge,
                    reasoning=f"Hedge caps loss at ${loss_if_hedge:.2f} vs ${loss_if_close:.2f} if close"
                )
            else:
                # Hedge cost too high relative to potential savings
                return HedgeRecommendation(
                    action=HedgeAction.CLOSE_POSITION,
                    asset=position.asset_type,
                    side='sell' if position.is_long else 'buy',
                    size=abs(position.size),
                    price=prices.call_bid if position.asset_type == 'CALL' else prices.put_bid,
                    current_pnl=unrealized_pnl,
                    pnl_if_close=close_pnl,
                    reasoning=f"Hedge cost (${hedge_cost:.2f}) too high, better to close"
                )
        
        # Check if partial hedge makes sense
        optimal_hedge_size = self._find_optimal_hedge_size(position, prices, loss_if_close)
        
        if optimal_hedge_size and optimal_hedge_size < abs(position.size) * 0.9:
            partial_cost = optimal_hedge_size * hedge_price * (1 + self.slippage_estimate)
            return HedgeRecommendation(
                action=HedgeAction.PARTIAL_HEDGE,
                asset=opposite_asset,
                side='buy',
                size=optimal_hedge_size,
                price=hedge_price,
                current_pnl=unrealized_pnl,
                pnl_if_close=close_pnl,
                reasoning=f"Partial hedge of {optimal_hedge_size:.0f} units optimal"
            )
        
        # Default: hold and wait for better opportunity
        return HedgeRecommendation(
            action=HedgeAction.HOLD,
            current_pnl=unrealized_pnl,
            pnl_if_close=close_pnl,
            pnl_if_hedge=max_loss_hedged,
            reasoning="Neither hedging nor closing is clearly better, hold"
        )
    
    def _find_optimal_hedge_size(
        self, 
        position: Position, 
        prices: MarketPrices,
        loss_if_close: float
    ) -> Optional[float]:
        """
        Find optimal partial hedge size that minimizes max loss.
        Uses simple grid search.
        """
        size = abs(position.size)
        best_size = None
        best_max_loss = loss_if_close
        
        for ratio in [0.25, 0.5, 0.75]:
            hedge_size = size * ratio
            
            # Create partial position for analysis
            opposite = 'PUT' if position.asset_type == 'CALL' else 'CALL'
            hedge_price = prices.put_ask if opposite == 'PUT' else prices.call_ask
            hedge_cost = hedge_size * hedge_price * (1 + self.slippage_estimate)
            
            # Calculate outcomes with partial hedge
            # This is simplified - full analysis would account for partial coverage
            max_loss = max(
                abs((1.0 - position.entry_price) * (size - hedge_size) - hedge_cost),
                abs((position.entry_price) * size - (1.0 * hedge_size - hedge_cost))
            )
            
            if max_loss < best_max_loss:
                best_max_loss = max_loss
                best_size = hedge_size
        
        return best_size


def demo():
    """Demo of hedge analyzer."""
    print("="*60)
    print("   Hedge Analyzer Demo")
    print("="*60)
    
    analyzer = HedgeAnalyzer()
    
    # Scenario 1: Long CALL losing money
    print("\n--- Scenario 1: Long CALL Underwater ---")
    position = Position(asset_type='CALL', size=100, entry_price=0.55)
    prices = MarketPrices(call_bid=0.40, call_ask=0.42, put_bid=0.57, put_ask=0.59)
    
    rec = analyzer.analyze(position, prices)
    print(f"Position: Long 100 CALL @ 0.55")
    print(f"Market: CALL {prices.call_bid}/{prices.call_ask}, PUT {prices.put_bid}/{prices.put_ask}")
    print(f"Current P&L: ${rec.current_pnl:.2f}")
    print(f"P&L if close: ${rec.pnl_if_close:.2f}")
    print(f"Recommendation: {rec.action.value}")
    print(f"Reasoning: {rec.reasoning}")
    
    if rec.action in [HedgeAction.HEDGE_WITH_OPPOSITE, HedgeAction.PARTIAL_HEDGE]:
        print(f"→ {rec.side.upper()} {rec.size:.0f} {rec.asset} @ {rec.price:.4f}")
    
    # Scenario 2: Long PUT profitable
    print("\n--- Scenario 2: Long PUT Profitable ---")
    position = Position(asset_type='PUT', size=100, entry_price=0.45)
    prices = MarketPrices(call_bid=0.35, call_ask=0.37, put_bid=0.62, put_ask=0.64)
    
    rec = analyzer.analyze(position, prices)
    print(f"Position: Long 100 PUT @ 0.45")
    print(f"Market: CALL {prices.call_bid}/{prices.call_ask}, PUT {prices.put_bid}/{prices.put_ask}")
    print(f"Current P&L: ${rec.current_pnl:.2f}")
    print(f"Recommendation: {rec.action.value}")
    print(f"Reasoning: {rec.reasoning}")
    
    # Scenario 3: Deep loss - hedge makes sense
    print("\n--- Scenario 3: Deep Loss - Hedge Analysis ---")
    position = Position(asset_type='CALL', size=100, entry_price=0.60)
    prices = MarketPrices(call_bid=0.25, call_ask=0.27, put_bid=0.72, put_ask=0.74)
    
    rec = analyzer.analyze(position, prices)
    print(f"Position: Long 100 CALL @ 0.60")
    print(f"Market: CALL {prices.call_bid}/{prices.call_ask}, PUT {prices.put_bid}/{prices.put_ask}")
    print(f"Current P&L: ${rec.current_pnl:.2f}")
    print(f"P&L if close: ${rec.pnl_if_close:.2f}")
    print(f"Max loss if hedge: ${rec.max_loss_if_hedge:.2f}" if rec.max_loss_if_hedge else "")
    print(f"Recommendation: {rec.action.value}")
    print(f"Reasoning: {rec.reasoning}")


if __name__ == '__main__':
    demo()
