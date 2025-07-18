"""
Enhanced technical indicators for ML feature engineering
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate various technical indicators for trading"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

        try:
            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD, Signal line, and Histogram"""
        if len(prices) < slow_period + signal_period:
            return 0.0, 0.0, 0.0

        try:
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

            # MACD line
            macd_line = ema_fast - ema_slow

            # Signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # MACD histogram
            macd_histogram = macd_line - signal_line

            # Return latest values
            return (
                float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
                float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
                float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0.0,
            )

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate Bollinger Bands
        Returns: (upper_band, middle_band, lower_band, bb_position, bb_squeeze)
        """
        if len(prices) < period:
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            return current_price, current_price, current_price, 0.5, 0.0

        try:
            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=period).mean()

            # Calculate standard deviation
            std_dev = prices.rolling(window=period).std()

            # Calculate bands
            upper_band = middle_band + (std_dev * num_std)
            lower_band = middle_band - (std_dev * num_std)

            # Get latest values
            current_price = float(prices.iloc[-1])
            upper = float(upper_band.iloc[-1])
            middle = float(middle_band.iloc[-1])
            lower = float(lower_band.iloc[-1])

            # Calculate position within bands (0 = lower band, 1 = upper band)
            band_width = upper - lower
            if band_width > 0:
                bb_position = (current_price - lower) / band_width
                bb_position = max(0.0, min(1.0, bb_position))  # Clamp to [0, 1]
            else:
                bb_position = 0.5

            # Calculate squeeze (low volatility)
            # Normalize by middle band to make it relative
            bb_squeeze = band_width / middle if middle > 0 else 0.0

            return upper, middle, lower, bb_position, bb_squeeze

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            return current_price, current_price, current_price, 0.5, 0.0

    @staticmethod
    def calculate_atr(
        high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14
    ) -> float:
        """Calculate Average True Range"""
        if len(close_prices) < period + 1:
            return 0.0

        try:
            # Calculate True Range
            high_low = high_prices - low_prices
            high_close = abs(high_prices - close_prices.shift(1))
            low_close = abs(low_prices - close_prices.shift(1))

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Calculate ATR
            atr = true_range.rolling(window=period).mean()

            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0

    @staticmethod
    def calculate_stochastic(
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator (%K and %D)"""
        if len(close_prices) < k_period + d_period:
            return 50.0, 50.0

        try:
            # Calculate %K
            lowest_low = low_prices.rolling(window=k_period).min()
            highest_high = high_prices.rolling(window=k_period).max()

            k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))

            # Calculate %D (SMA of %K)
            d_percent = k_percent.rolling(window=d_period).mean()

            return (
                float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else 50.0,
                float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else 50.0,
            )

        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return 50.0, 50.0

    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 10) -> float:
        """Calculate momentum indicator"""
        if len(prices) < period + 1:
            return 0.0

        try:
            momentum = (prices.iloc[-1] / prices.iloc[-period - 1] - 1) * 100
            return float(momentum) if not pd.isna(momentum) else 0.0
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0

    @staticmethod
    def calculate_vwap(
        prices: pd.Series, volumes: pd.Series, period: Optional[int] = None
    ) -> float:
        """Calculate Volume Weighted Average Price"""
        if len(prices) != len(volumes) or len(prices) == 0:
            return float(prices.iloc[-1]) if len(prices) > 0 else 0.0

        try:
            if period:
                # Use rolling window
                price_volume = prices * volumes
                vwap = (
                    price_volume.rolling(window=period).sum() / volumes.rolling(window=period).sum()
                )
            else:
                # Use all available data
                vwap = (prices * volumes).sum() / volumes.sum()

            if isinstance(vwap, pd.Series):
                return (
                    float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else float(prices.iloc[-1])
                )
            else:
                return float(vwap) if not pd.isna(vwap) else float(prices.iloc[-1])

        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return float(prices.iloc[-1]) if len(prices) > 0 else 0.0

    @staticmethod
    def calculate_obv(prices: pd.Series, volumes: pd.Series) -> float:
        """Calculate On Balance Volume"""
        if len(prices) != len(volumes) or len(prices) < 2:
            return 0.0

        try:
            # Calculate price direction
            price_diff = prices.diff()

            # Calculate OBV
            obv = volumes.copy()
            obv[price_diff < 0] *= -1
            obv[price_diff == 0] = 0

            return float(obv.sum())

        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return 0.0

    @staticmethod
    def calculate_support_resistance(
        prices: pd.Series, window: int = 20, num_levels: int = 3
    ) -> Tuple[List[float], List[float], float]:
        """
        Calculate support and resistance levels
        Returns: (support_levels, resistance_levels, strength)
        """
        if len(prices) < window:
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            return [current_price], [current_price], 0.0

        try:
            # Find local minima and maxima
            rolling_min = prices.rolling(window=window, center=True).min()
            rolling_max = prices.rolling(window=window, center=True).max()

            # Identify turning points
            support_levels = []
            resistance_levels = []

            for i in range(window, len(prices) - window):
                if prices.iloc[i] == rolling_min.iloc[i]:
                    support_levels.append(float(prices.iloc[i]))
                elif prices.iloc[i] == rolling_max.iloc[i]:
                    resistance_levels.append(float(prices.iloc[i]))

            # Get unique levels and sort
            support_levels = sorted(list(set(support_levels)))[-num_levels:]
            resistance_levels = sorted(list(set(resistance_levels)))[:num_levels]

            # Calculate strength based on how many times price bounced off levels
            current_price = float(prices.iloc[-1])
            strength = 0.0

            # Check distance to nearest support/resistance
            all_levels = support_levels + resistance_levels
            if all_levels:
                distances = [abs(current_price - level) / current_price for level in all_levels]
                min_distance = min(distances)
                strength = 1.0 - min(min_distance * 10, 1.0)  # Normalize to [0, 1]

            return support_levels, resistance_levels, strength

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            return [current_price], [current_price], 0.0

    @staticmethod
    def calculate_mean_reversion_signal(
        prices: pd.Series, lookback: int = 20, num_std: float = 2.0
    ) -> float:
        """
        Calculate mean reversion signal
        Returns: Signal strength (-1 to 1, negative = oversold, positive = overbought)
        """
        if len(prices) < lookback:
            return 0.0

        try:
            # Calculate rolling mean and std
            rolling_mean = prices.rolling(window=lookback).mean()
            rolling_std = prices.rolling(window=lookback).std()

            # Calculate z-score
            current_price = float(prices.iloc[-1])
            mean = float(rolling_mean.iloc[-1])
            std = float(rolling_std.iloc[-1])

            if std > 0:
                z_score = (current_price - mean) / std
                # Normalize to [-1, 1] range
                signal = np.tanh(z_score / num_std)
                return float(signal)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating mean reversion signal: {e}")
            return 0.0
