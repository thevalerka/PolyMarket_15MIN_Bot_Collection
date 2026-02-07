"""
Simple API wrapper for binary options sensitivity prediction
Easy integration with trading bots
"""

from sensitivity_predictor import SensitivityPredictor
from typing import Dict, Optional, List
import json


class SensitivityAPI:
    """
    Simplified API for bot integration
    """

    def __init__(self, data_file: str):
        """Initialize the API with data file"""
        self.predictor = SensitivityPredictor(data_file)
        print(f"[SensitivityAPI] Ready - loaded {len(self.predictor.bins_data)} bins")

    def get_sensitivity(self,
                       btc_price: float,
                       strike_price: float,
                       time_to_expiry_seconds: int,
                       volatility_percent: float) -> Dict:
        """
        Get sensitivity for a specific option

        Args:
            btc_price: Current BTC price (e.g., 100000)
            strike_price: Option strike price (e.g., 100050)
            time_to_expiry_seconds: Time until expiry in seconds (e.g., 300)
            volatility_percent: Implied volatility in % (e.g., 50)

        Returns:
            Dict with sensitivities and metadata
        """
        distance = abs(strike_price - btc_price)

        pred = self.predictor.predict(
            distance=distance,
            time_to_expiry=time_to_expiry_seconds,
            volatility=volatility_percent,
            method='ensemble'
        )

        return {
            'input': {
                'btc_price': btc_price,
                'strike_price': strike_price,
                'distance': distance,
                'time_to_expiry': time_to_expiry_seconds,
                'volatility': volatility_percent
            },
            'sensitivity': {
                'put': round(pred['put_sensitivity'], 6),
                'call': round(pred['call_sensitivity'], 6)
            },
            'confidence': round(pred.get('confidence', 0), 3)
        }

    def estimate_price_change(self,
                             btc_price: float,
                             strike_price: float,
                             time_to_expiry_seconds: int,
                             volatility_percent: float,
                             expected_btc_move: float) -> Dict:
        """
        Estimate how option prices will change given expected BTC move

        Args:
            btc_price: Current BTC price
            strike_price: Option strike price
            time_to_expiry_seconds: Time until expiry
            volatility_percent: Implied volatility
            expected_btc_move: Expected BTC price change (positive or negative)

        Returns:
            Dict with estimated price changes
        """
        sensitivity = self.get_sensitivity(
            btc_price, strike_price, time_to_expiry_seconds, volatility_percent
        )

        call_change = sensitivity['sensitivity']['call'] * expected_btc_move
        put_change = sensitivity['sensitivity']['put'] * expected_btc_move

        return {
            'input': sensitivity['input'],
            'expected_btc_move': expected_btc_move,
            'estimated_changes': {
                'call': round(call_change, 6),
                'put': round(put_change, 6)
            },
            'confidence': sensitivity['confidence']
        }

    def get_expected_return(self,
                           btc_price: float,
                           strike_price: float,
                           time_to_expiry_seconds: int,
                           volatility_percent: float,
                           expected_btc_move: float,
                           option_premium: float) -> Dict:
        """
        Calculate expected return percentage for a trade

        Args:
            btc_price: Current BTC price
            strike_price: Option strike price
            time_to_expiry_seconds: Time until expiry
            volatility_percent: Implied volatility
            expected_btc_move: Expected BTC price change
            option_premium: Current option price/premium

        Returns:
            Dict with expected returns and trading signals
        """
        change = self.estimate_price_change(
            btc_price, strike_price, time_to_expiry_seconds,
            volatility_percent, expected_btc_move
        )

        call_return_pct = (change['estimated_changes']['call'] / option_premium * 100) if option_premium > 0 else 0
        put_return_pct = (change['estimated_changes']['put'] / option_premium * 100) if option_premium > 0 else 0

        # Generate signals
        def get_signal(return_pct):
            if return_pct > 10:
                return "STRONG_BUY"
            elif return_pct > 5:
                return "BUY"
            elif return_pct > -5:
                return "HOLD"
            elif return_pct > -10:
                return "SELL"
            else:
                return "STRONG_SELL"

        return {
            'input': change['input'],
            'option_premium': option_premium,
            'expected_returns': {
                'call_percent': round(call_return_pct, 2),
                'put_percent': round(put_return_pct, 2)
            },
            'signals': {
                'call': get_signal(call_return_pct),
                'put': get_signal(put_return_pct)
            },
            'confidence': change['confidence']
        }

    def calculate_breakeven(self,
                           btc_price: float,
                           strike_price: float,
                           time_to_expiry_seconds: int,
                           volatility_percent: float,
                           option_premium: float) -> Dict:
        """
        Calculate breakeven BTC price move needed to recover premium

        Args:
            btc_price: Current BTC price
            strike_price: Option strike price
            time_to_expiry_seconds: Time until expiry
            volatility_percent: Implied volatility
            option_premium: Premium paid for option

        Returns:
            Dict with breakeven analysis
        """
        sensitivity = self.get_sensitivity(
            btc_price, strike_price, time_to_expiry_seconds, volatility_percent
        )

        call_sens = sensitivity['sensitivity']['call']
        put_sens = abs(sensitivity['sensitivity']['put'])

        call_breakeven = option_premium / call_sens if call_sens > 0 else float('inf')
        put_breakeven = option_premium / put_sens if put_sens > 0 else float('inf')

        return {
            'input': sensitivity['input'],
            'option_premium': option_premium,
            'breakeven_moves': {
                'call': round(call_breakeven, 2),  # BTC needs to move up by this much
                'put': round(put_breakeven, 2)      # BTC needs to move down by this much
            },
            'sensitivity': sensitivity['sensitivity'],
            'confidence': sensitivity['confidence']
        }

    def get_best_strikes(self,
                        btc_price: float,
                        time_to_expiry_seconds: int,
                        volatility_percent: float,
                        n_strikes: int = 5,
                        distance_range: tuple = (5, 100)) -> List[Dict]:
        """
        Find strikes with highest sensitivity (maximum leverage)

        Args:
            btc_price: Current BTC price
            time_to_expiry_seconds: Time until expiry
            volatility_percent: Implied volatility
            n_strikes: Number of top strikes to return
            distance_range: (min_distance, max_distance) to search

        Returns:
            List of top strikes sorted by sensitivity
        """
        import numpy as np

        distances = np.linspace(distance_range[0], distance_range[1], 20)
        results = []

        for dist in distances:
            pred = self.predictor.predict(dist, time_to_expiry_seconds, volatility_percent)

            results.append({
                'strike_price': round(btc_price + dist, 2),
                'distance': round(dist, 2),
                'call_sensitivity': round(pred['call_sensitivity'], 6),
                'put_sensitivity': round(pred['put_sensitivity'], 6),
                'abs_sensitivity': round(abs(pred['call_sensitivity']), 6),
                'confidence': round(pred.get('confidence', 0), 3)
            })

        # Sort by absolute sensitivity
        results.sort(key=lambda x: x['abs_sensitivity'], reverse=True)

        return results[:n_strikes]

    def monitor_position(self,
                        btc_price: float,
                        strike_price: float,
                        time_to_expiry_seconds: int,
                        volatility_percent: float,
                        entry_premium: float,
                        current_premium: float) -> Dict:
        """
        Monitor an existing position and provide guidance

        Args:
            btc_price: Current BTC price
            strike_price: Option strike price
            time_to_expiry_seconds: Time remaining
            volatility_percent: Current implied volatility
            entry_premium: Premium paid at entry
            current_premium: Current option price

        Returns:
            Dict with position analysis and recommendations
        """
        sensitivity = self.get_sensitivity(
            btc_price, strike_price, time_to_expiry_seconds, volatility_percent
        )

        current_pnl = current_premium - entry_premium
        pnl_percent = (current_pnl / entry_premium * 100) if entry_premium > 0 else 0

        # Time urgency
        if time_to_expiry_seconds < 60:
            urgency = "CRITICAL"
        elif time_to_expiry_seconds < 180:
            urgency = "HIGH"
        elif time_to_expiry_seconds < 300:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # Position recommendation
        if pnl_percent > 20:
            recommendation = "TAKE_PROFIT"
        elif pnl_percent < -30:
            recommendation = "CUT_LOSS"
        elif time_to_expiry_seconds < 60 and pnl_percent < 5:
            recommendation = "EXIT_NOW"
        else:
            recommendation = "HOLD"

        return {
            'input': sensitivity['input'],
            'position_status': {
                'entry_premium': entry_premium,
                'current_premium': current_premium,
                'pnl': round(current_pnl, 4),
                'pnl_percent': round(pnl_percent, 2)
            },
            'current_sensitivity': sensitivity['sensitivity'],
            'time_urgency': urgency,
            'recommendation': recommendation,
            'confidence': sensitivity['confidence']
        }


def example_usage():
    """Example usage for bot integration"""

    # Initialize API
    api = SensitivityAPI('/mnt/user-data/uploads/sensitivity_transformed.json')

    print("\n" + "="*80)
    print("EXAMPLE 1: Get Sensitivity")
    print("="*80)

    result = api.get_sensitivity(
        btc_price=100000,
        strike_price=100050,
        time_to_expiry_seconds=300,
        volatility_percent=50
    )
    print(json.dumps(result, indent=2))

    print("\n" + "="*80)
    print("EXAMPLE 2: Estimate Price Change")
    print("="*80)

    result = api.estimate_price_change(
        btc_price=100000,
        strike_price=100050,
        time_to_expiry_seconds=300,
        volatility_percent=50,
        expected_btc_move=150  # Expecting BTC to rise $150
    )
    print(json.dumps(result, indent=2))

    print("\n" + "="*80)
    print("EXAMPLE 3: Get Expected Return")
    print("="*80)

    result = api.get_expected_return(
        btc_price=100000,
        strike_price=100050,
        time_to_expiry_seconds=300,
        volatility_percent=50,
        expected_btc_move=150,
        option_premium=0.45
    )
    print(json.dumps(result, indent=2))

    print("\n" + "="*80)
    print("EXAMPLE 4: Calculate Breakeven")
    print("="*80)

    result = api.calculate_breakeven(
        btc_price=100000,
        strike_price=100050,
        time_to_expiry_seconds=300,
        volatility_percent=50,
        option_premium=0.40
    )
    print(json.dumps(result, indent=2))

    print("\n" + "="*80)
    print("EXAMPLE 5: Find Best Strikes")
    print("="*80)

    result = api.get_best_strikes(
        btc_price=100000,
        time_to_expiry_seconds=300,
        volatility_percent=50,
        n_strikes=3
    )
    print(json.dumps(result, indent=2))

    print("\n" + "="*80)
    print("EXAMPLE 6: Monitor Position")
    print("="*80)

    result = api.monitor_position(
        btc_price=100000,
        strike_price=100040,
        time_to_expiry_seconds=120,
        volatility_percent=55,
        entry_premium=0.40,
        current_premium=0.52
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    example_usage()
