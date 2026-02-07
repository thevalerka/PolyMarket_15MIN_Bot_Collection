import numpy as np
import pandas as pd
import requests
from typing import Dict, Tuple, Optional
from datetime import datetime

class MarketRegimeDetector:
    """
    Detect market regime (TRENDING/RANGING) using 1 hour of 1-minute candles.

    Usage:
        detector = MarketRegimeDetector()
        result = detector.analyze_from_bybit('BTCUSDT')
        current_regime = result['regime']
        trend_score = result['trend_score']
    """

    def __init__(
        self,
        lookback: int = 60,  # 60 minutes = 1 hour
        adx_period: int = 14,
        trending_threshold: float = 2.0,
        ranging_threshold: float = -2.0,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the regime detector.

        Args:
            lookback: Number of periods to analyze (default 60 = 1 hour of 1-min candles)
            adx_period: Period for ADX calculation (default 14)
            trending_threshold: Score above which market is TRENDING (default 2.0)
            ranging_threshold: Score below which market is RANGING (default -2.0)
            weights: Dict with keys 'adx', 'slope', 'efficiency'
        """
        self.lookback = lookback
        self.adx_period = adx_period
        self.trending_threshold = trending_threshold
        self.ranging_threshold = ranging_threshold

        self.weights = weights or {
            'adx': 0.4,
            'slope': 0.3,
            'efficiency': 0.3
        }

        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def fetch_bybit_data(
        self,
        symbol: str = 'BTCUSDT',
        limit: int = 200  # Fetch extra for calculations
    ) -> pd.DataFrame:
        """
        Fetch 1-minute candle data from Bybit API.

        Args:
            symbol: Trading pair (default 'BTCUSDT')
            limit: Number of candles to fetch (max 200)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close
        """
        url = "https://api.bybit.com/v5/market/mark-price-kline"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': '1',  # 1 minute
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['retCode'] != 0:
                raise ValueError(f"Bybit API error: {data['retMsg']}")

            # Parse the data
            candles = data['result']['list']

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close'])

            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)

            # Bybit returns newest first, so reverse
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch Bybit data: {e}")

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """Calculate Average True Range."""
        if period is None:
            period = self.adx_period

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """Calculate Average Directional Index."""
        if period is None:
            period = self.adx_period

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_normalized_slope(
        self,
        close: pd.Series,
        atr: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """Calculate linear regression slope normalized by ATR."""
        if period is None:
            period = self.lookback

        slopes = []
        for i in range(len(close)):
            if i < period:
                slopes.append(np.nan)
            else:
                y = close.iloc[i-period:i].values
                x = np.arange(period)
                slope = np.polyfit(x, y, 1)[0]
                normalized_slope = abs(slope) / atr.iloc[i] if atr.iloc[i] > 0 else 0
                slopes.append(normalized_slope)

        return pd.Series(slopes, index=close.index)

    def _calculate_efficiency_ratio(
        self,
        close: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """Calculate efficiency ratio (price change / path traveled)."""
        if period is None:
            period = self.lookback

        price_change = abs(close - close.shift(period))
        path_traveled = abs(close.diff()).rolling(period).sum()
        efficiency = price_change / path_traveled

        efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)

        return efficiency

    def _calculate_scores(
        self,
        adx: pd.Series,
        norm_slope: pd.Series,
        efficiency: pd.Series
    ) -> pd.Series:
        """Calculate composite trend score."""
        adx_score = (adx - 20) / 5
        slope_score = norm_slope * 10
        efficiency_score = (efficiency - 0.3) / 0.2 * 10

        trend_score = (
            self.weights['adx'] * adx_score +
            self.weights['slope'] * slope_score +
            self.weights['efficiency'] * efficiency_score
        )

        return trend_score

    def analyze(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Analyze market regime for the given dataframe.

        Args:
            df: DataFrame with columns 'close', 'high', 'low'

        Returns:
            Dictionary containing regime indicators
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']

        # Calculate ATR
        atr = self._calculate_atr(high, low, close)

        # Calculate indicators
        adx = self._calculate_adx(high, low, close)
        norm_slope = self._calculate_normalized_slope(close, atr)
        efficiency = self._calculate_efficiency_ratio(close)

        # Calculate composite score
        trend_score = self._calculate_scores(adx, norm_slope, efficiency)

        # Classify regime
        regime = pd.Series('NEUTRAL', index=df.index)
        regime[trend_score > self.trending_threshold] = 'TRENDING'
        regime[trend_score < self.ranging_threshold] = 'RANGING'

        return {
            'regime': regime,
            'trend_score': trend_score,
            'adx': adx,
            'norm_slope': norm_slope,
            'efficiency': efficiency,
            'atr': atr
        }

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """
        Get current market regime and detailed metrics.

        Returns:
            Tuple of (regime_string, trend_score, metrics_dict)
        """
        result = self.analyze(df)

        metrics = {
            'adx': result['adx'].iloc[-1],
            'norm_slope': result['norm_slope'].iloc[-1],
            'efficiency': result['efficiency'].iloc[-1],
            'atr': result['atr'].iloc[-1]
        }

        return result['regime'].iloc[-1], result['trend_score'].iloc[-1], metrics

    def analyze_from_bybit(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Fetch data from Bybit and analyze current regime.

        Args:
            symbol: Trading pair to analyze

        Returns:
            Dict with regime, score, and metrics
        """
        df = self.fetch_bybit_data(symbol)
        regime, score, metrics = self.get_current_regime(df)

        return {
            'symbol': symbol,
            'regime': regime,
            'trend_score': score,
            'timestamp': df['timestamp'].iloc[-1],
            'current_price': df['close'].iloc[-1],
            'metrics': metrics
        }

    def is_trending(self, df: pd.DataFrame) -> bool:
        """Quick check if market is currently trending."""
        regime, _, _ = self.get_current_regime(df)
        return regime == 'TRENDING'

    def is_ranging(self, df: pd.DataFrame) -> bool:
        """Quick check if market is currently ranging."""
        regime, _, _ = self.get_current_regime(df)
        return regime == 'RANGING'
