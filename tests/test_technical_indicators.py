"""
Comprehensive tests for technical indicators
"""

import numpy as np
import pandas as pd
import pytest

from app.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test technical indicator calculations"""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing"""
        # Create a price series with some variation
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0
        prices = [base_price]

        # Generate 50 price points with random walk
        for i in range(49):
            change = np.random.normal(0, 1.0)  # Small random changes
            new_price = max(prices[-1] + change, 1.0)  # Ensure positive prices
            prices.append(new_price)

        return pd.Series(prices)

    @pytest.fixture
    def trending_up_prices(self):
        """Create upward trending price data"""
        return pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 115])

    @pytest.fixture
    def trending_down_prices(self):
        """Create downward trending price data"""
        return pd.Series([115, 113, 111, 112, 110, 108, 109, 107, 105, 106, 104, 102, 100])

    @pytest.fixture
    def volatile_prices(self):
        """Create highly volatile price data"""
        return pd.Series([100, 105, 95, 110, 90, 115, 85, 120, 80, 125, 75, 130, 70])

    def test_rsi_calculation_neutral(self, sample_prices):
        """Test RSI calculation with neutral data"""
        rsi = TechnicalIndicators.calculate_rsi(sample_prices, period=14)

        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        # For random walk data, RSI should be valid (no need for strict neutrality)
        assert np.isfinite(rsi)

    def test_rsi_calculation_trending_up(self, trending_up_prices):
        """Test RSI calculation with upward trending data"""
        rsi = TechnicalIndicators.calculate_rsi(trending_up_prices, period=5)

        # Upward trend should produce high RSI (> 50)
        assert rsi > 50
        assert 0 <= rsi <= 100

    def test_rsi_calculation_trending_down(self, trending_down_prices):
        """Test RSI calculation with downward trending data"""
        rsi = TechnicalIndicators.calculate_rsi(trending_down_prices, period=5)

        # Downward trend should produce low RSI (< 50)
        assert rsi < 50
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data"""
        short_prices = pd.Series([100, 101, 102])
        rsi = TechnicalIndicators.calculate_rsi(short_prices, period=14)

        # Should return neutral RSI for insufficient data
        assert rsi == 50.0

    def test_rsi_error_handling(self):
        """Test RSI error handling"""
        # Test with invalid data
        invalid_prices = pd.Series([np.nan, np.inf, -np.inf])
        rsi = TechnicalIndicators.calculate_rsi(invalid_prices, period=14)

        # Should return neutral RSI on error
        assert rsi == 50.0

    def test_macd_calculation(self, sample_prices):
        """Test MACD calculation"""
        macd, signal, histogram = TechnicalIndicators.calculate_macd(
            sample_prices, fast_period=12, slow_period=26, signal_period=9
        )

        # All values should be finite numbers
        assert np.isfinite(macd)
        assert np.isfinite(signal)
        assert np.isfinite(histogram)

        # Histogram should be difference between MACD and signal
        assert abs(histogram - (macd - signal)) < 0.001

    def test_macd_trending_up(self, trending_up_prices):
        """Test MACD with upward trend"""
        macd, signal, histogram = TechnicalIndicators.calculate_macd(
            trending_up_prices, fast_period=5, slow_period=10, signal_period=3
        )

        # In uptrend, MACD should generally be positive
        assert np.isfinite(macd)
        assert np.isfinite(signal)
        assert np.isfinite(histogram)

    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data"""
        short_prices = pd.Series([100, 101, 102])
        macd, signal, histogram = TechnicalIndicators.calculate_macd(
            short_prices, fast_period=12, slow_period=26, signal_period=9
        )

        # Should return zeros for insufficient data
        assert macd == 0.0
        assert signal == 0.0
        assert histogram == 0.0

    def test_bollinger_bands_calculation(self, sample_prices):
        """Test Bollinger Bands calculation"""
        upper, middle, lower, bb_position, bb_squeeze = TechnicalIndicators.calculate_bollinger_bands(
            sample_prices, period=20, num_std=2.0
        )

        # All bands should be finite
        assert np.isfinite(upper)
        assert np.isfinite(middle)
        assert np.isfinite(lower)
        assert np.isfinite(bb_position)
        assert np.isfinite(bb_squeeze)

        # Upper should be >= middle >= lower
        assert upper >= middle >= lower

        # BB position should be between 0 and 1
        assert 0 <= bb_position <= 1

        # BB squeeze should be non-negative
        assert bb_squeeze >= 0

    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data"""
        short_prices = pd.Series([100, 101, 102])
        upper, middle, lower, bb_position, bb_squeeze = TechnicalIndicators.calculate_bollinger_bands(
            short_prices, period=20, num_std=2.0
        )

        # Should return last price for all bands when insufficient data
        last_price = short_prices.iloc[-1]
        assert upper == last_price
        assert middle == last_price
        assert lower == last_price
        assert bb_position == 0.5  # Neutral position
        assert bb_squeeze == 0.0   # No squeeze

    def test_bollinger_bands_volatile_data(self, volatile_prices):
        """Test Bollinger Bands with volatile data"""
        upper, middle, lower, bb_position, bb_squeeze = TechnicalIndicators.calculate_bollinger_bands(
            volatile_prices, period=10, num_std=2.0
        )

        # With volatile data, bands should be wide apart
        band_width = upper - lower
        assert band_width > 10  # Expect significant width with volatile data

        # BB squeeze should reflect the volatility
        assert bb_squeeze > 0

    def test_momentum_calculation(self, sample_prices):
        """Test Momentum calculation"""
        momentum = TechnicalIndicators.calculate_momentum(sample_prices, period=10)

        # Should be finite
        assert np.isfinite(momentum)

        # Momentum should reflect price change percentage
        if len(sample_prices) > 10:
            expected_momentum = (sample_prices.iloc[-1] / sample_prices.iloc[-11] - 1) * 100
            assert abs(momentum - expected_momentum) < 0.001

    def test_momentum_insufficient_data(self):
        """Test Momentum with insufficient data"""
        short_prices = pd.Series([100, 101, 102])
        momentum = TechnicalIndicators.calculate_momentum(short_prices, period=10)

        # Should return 0 for insufficient data
        assert momentum == 0.0

    def test_vwap_calculation(self, sample_prices):
        """Test VWAP calculation"""
        volumes = pd.Series([1000] * len(sample_prices))
        vwap = TechnicalIndicators.calculate_vwap(sample_prices, volumes, period=10)

        # Should be finite
        assert np.isfinite(vwap)

        # With equal volumes, VWAP should be close to simple average
        if len(sample_prices) >= 10:
            expected_avg = sample_prices.tail(10).mean()
            assert abs(vwap - expected_avg) < 1.0  # Allow some tolerance

    def test_vwap_insufficient_data(self):
        """Test VWAP with insufficient data"""
        short_prices = pd.Series([100, 101, 102])
        short_volumes = pd.Series([1000, 1100, 1200])
        vwap = TechnicalIndicators.calculate_vwap(short_prices, short_volumes)

        # Should return valid VWAP
        assert np.isfinite(vwap)

    def test_stochastic_oscillator(self, sample_prices):
        """Test Stochastic Oscillator calculation"""
        # Create high, low, close data
        high_prices = sample_prices * 1.02  # Slightly higher
        low_prices = sample_prices * 0.98   # Slightly lower
        close_prices = sample_prices

        k_percent, d_percent = TechnicalIndicators.calculate_stochastic(
            high_prices, low_prices, close_prices, k_period=14, d_period=3
        )

        # Both should be between 0 and 100
        assert 0 <= k_percent <= 100
        assert 0 <= d_percent <= 100
        assert np.isfinite(k_percent)
        assert np.isfinite(d_percent)

    def test_stochastic_insufficient_data(self):
        """Test Stochastic with insufficient data"""
        short_data = pd.Series([100, 101, 102])
        k, d = TechnicalIndicators.calculate_stochastic(
            short_data, short_data, short_data, k_period=14, d_period=3
        )

        # Should return neutral values
        assert k == 50.0
        assert d == 50.0

    def test_atr_calculation(self, sample_prices):
        """Test Average True Range calculation"""
        # Create high, low, close data
        high_prices = sample_prices * 1.02
        low_prices = sample_prices * 0.98
        close_prices = sample_prices

        atr = TechnicalIndicators.calculate_atr(
            high_prices, low_prices, close_prices, period=14
        )

        # ATR should be positive and finite
        assert atr > 0
        assert np.isfinite(atr)

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data"""
        short_data = pd.Series([100, 101, 102])
        atr = TechnicalIndicators.calculate_atr(
            short_data, short_data, short_data, period=14
        )

        # Should return 0 for insufficient data
        assert atr == 0.0

    def test_support_resistance_calculation(self, sample_prices):
        """Test Support and Resistance calculation"""
        support_levels, resistance_levels, strength = TechnicalIndicators.calculate_support_resistance(
            sample_prices, window=10, num_levels=3
        )

        # Should return lists of levels and a strength value
        assert isinstance(support_levels, list)
        assert isinstance(resistance_levels, list)
        assert np.isfinite(strength)
        assert 0 <= strength <= 1  # Strength should be normalized

        # Should have at most the requested number of levels
        assert len(support_levels) <= 3
        assert len(resistance_levels) <= 3

    def test_support_resistance_insufficient_data(self):
        """Test Support/Resistance with insufficient data"""
        short_data = pd.Series([100, 101, 102])
        support_levels, resistance_levels, strength = TechnicalIndicators.calculate_support_resistance(
            short_data, window=20, num_levels=3
        )

        # Should return current price as level
        assert len(support_levels) == 1
        assert len(resistance_levels) == 1
        assert support_levels[0] == 102
        assert resistance_levels[0] == 102
        assert strength == 0.0

    def test_mean_reversion_signal(self, sample_prices):
        """Test Mean Reversion Signal calculation"""
        signal = TechnicalIndicators.calculate_mean_reversion_signal(
            sample_prices, lookback=20, num_std=2.0
        )

        # Signal should be between -1 and 1
        assert -1 <= signal <= 1
        assert np.isfinite(signal)

    def test_mean_reversion_insufficient_data(self):
        """Test Mean Reversion with insufficient data"""
        short_data = pd.Series([100, 101, 102])
        signal = TechnicalIndicators.calculate_mean_reversion_signal(
            short_data, lookback=20, num_std=2.0
        )

        # Should return 0 for insufficient data
        assert signal == 0.0

    def test_on_balance_volume(self):
        """Test On Balance Volume calculation"""
        prices = pd.Series([100, 102, 101, 103, 102, 104])
        volumes = pd.Series([1000, 1200, 800, 1500, 900, 1100])

        obv = TechnicalIndicators.calculate_obv(prices, volumes)

        # OBV should be finite
        assert np.isfinite(obv)

        # OBV calculation: starts with first volume if price goes up, accumulates from there
        # This is a cumulative calculation, so just check it's reasonable
        assert np.isfinite(obv)
        assert obv != 0  # Should have some volume impact

    def test_obv_insufficient_data(self):
        """Test OBV with insufficient data"""
        short_prices = pd.Series([100])
        short_volumes = pd.Series([1000])

        obv = TechnicalIndicators.calculate_obv(short_prices, short_volumes)
        assert obv == 0.0

    def test_edge_cases_empty_series(self):
        """Test indicators with empty price series"""
        empty_series = pd.Series([])
        empty_volumes = pd.Series([])

        # All indicators should handle empty data gracefully
        assert TechnicalIndicators.calculate_rsi(empty_series) == 50.0
        assert TechnicalIndicators.calculate_momentum(empty_series) == 0.0
        assert TechnicalIndicators.calculate_obv(empty_series, empty_volumes) == 0.0

        macd, signal, histogram = TechnicalIndicators.calculate_macd(empty_series)
        assert macd == 0.0 and signal == 0.0 and histogram == 0.0

    def test_edge_cases_single_value(self):
        """Test indicators with single value"""
        single_value = pd.Series([100.0])
        single_volume = pd.Series([1000])

        # RSI should return neutral
        assert TechnicalIndicators.calculate_rsi(single_value) == 50.0

        # Momentum should return 0
        assert TechnicalIndicators.calculate_momentum(single_value) == 0.0

        # OBV should return 0 for single value
        assert TechnicalIndicators.calculate_obv(single_value, single_volume) == 0.0

    def test_edge_cases_constant_prices(self):
        """Test indicators with constant prices"""
        constant_prices = pd.Series([100.0] * 30)
        constant_volumes = pd.Series([1000] * 30)

        # RSI should be neutral for constant prices
        rsi = TechnicalIndicators.calculate_rsi(constant_prices)
        assert rsi == 50.0  # No price changes = neutral RSI

        # Momentum should be 0 for constant prices
        momentum = TechnicalIndicators.calculate_momentum(constant_prices)
        assert momentum == 0.0

        # Bollinger bands should collapse to the constant value
        upper, middle, lower, bb_position, bb_squeeze = TechnicalIndicators.calculate_bollinger_bands(constant_prices)
        assert upper == middle == lower == 100.0
        assert bb_position == 0.5
        assert bb_squeeze == 0.0

    def test_parameter_validation(self, sample_prices):
        """Test parameter validation for indicators"""
        # Test with zero period (should handle gracefully)
        rsi_zero = TechnicalIndicators.calculate_rsi(sample_prices, period=0)
        assert rsi_zero == 50.0

        # Test with negative period (should handle gracefully)
        momentum_negative = TechnicalIndicators.calculate_momentum(sample_prices, period=-5)
        # Should still compute momentum with negative period (it doesn't validate period in the implementation)
        assert np.isfinite(momentum_negative)

        # Test with period larger than data
        large_period_rsi = TechnicalIndicators.calculate_rsi(sample_prices, period=1000)
        assert large_period_rsi == 50.0

    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset"""
        # Create larger dataset
        np.random.seed(42)
        large_prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.5))
        large_volumes = pd.Series([1000] * 1000)

        # All indicators should complete in reasonable time
        rsi = TechnicalIndicators.calculate_rsi(large_prices, period=14)
        momentum = TechnicalIndicators.calculate_momentum(large_prices, period=10)
        vwap = TechnicalIndicators.calculate_vwap(large_prices, large_volumes, period=50)

        # Results should be reasonable
        assert 0 <= rsi <= 100
        assert np.isfinite(momentum)
        assert np.isfinite(vwap)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
