"""
Comprehensive tests for market indicators and technical analysis components
"""

from collections import deque
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.market_indicators import (
    MarketFeatures,
    MarketIndicatorEngine,
    MarketMicrostructure,
    MarketRegimeDetector,
    TechnicalIndicators,
    VolatilityAnalyzer,
)


class TestTechnicalIndicators:
    """Test technical analysis calculations"""

    def test_rsi_calculation(self):
        """Test RSI calculation with known values"""
        # Simple test case with known RSI result
        prices = [
            44,
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.85,
            46.08,
            45.89,
            46.03,
            46.83,
            47.69,
            46.49,
            46.26,
            47.09,
            47.37,
            47.20,
            47.72,
            48.90,
        ]

        rsi = TechnicalIndicators.rsi(prices, period=14)

        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        # For this ascending price series, RSI should be > 50
        assert rsi > 50

    def test_rsi_edge_cases(self):
        """Test RSI edge cases"""
        # Insufficient data
        short_prices = [100, 101, 102]
        rsi = TechnicalIndicators.rsi(short_prices, period=14)
        assert rsi == 50.0  # Default when insufficient data

        # All prices same (no change)
        flat_prices = [100] * 20
        rsi = TechnicalIndicators.rsi(flat_prices, period=14)
        assert rsi == 50.0  # Should handle zero variance

        # All gains (no losses)
        rising_prices = list(range(100, 120))
        rsi = TechnicalIndicators.rsi(rising_prices, period=14)
        assert rsi == 100.0  # All gains should give RSI of 100

    def test_macd_calculation(self):
        """Test MACD calculation"""
        # Generate price series
        prices = [i + np.sin(i / 10) * 5 for i in range(100, 150)]

        macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)

        # Basic validation
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

        # Histogram should be difference between MACD and signal
        assert abs(histogram - (macd_line - signal_line)) < 1e-10

    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data"""
        short_prices = [100, 101, 102]
        macd_line, signal_line, histogram = TechnicalIndicators.macd(short_prices)

        assert macd_line == 0.0
        assert signal_line == 0.0
        assert histogram == 0.0

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        # Generate price series with some volatility
        np.random.seed(42)
        base_price = 100
        prices = [base_price + np.random.normal(0, 2) for _ in range(30)]

        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, period=20, std_dev=2.0)

        # Basic validations
        assert upper > middle > lower
        assert isinstance(upper, float)
        assert isinstance(middle, float)
        assert isinstance(lower, float)

        # Middle should be approximately the mean of recent prices
        recent_mean = np.mean(prices[-20:])
        assert abs(middle - recent_mean) < 1e-10

    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data"""
        short_prices = [100, 101]
        upper, middle, lower = TechnicalIndicators.bollinger_bands(short_prices, period=20)

        # Should return current price for all bands
        assert upper == middle == lower == 101

    def test_ema_calculation(self):
        """Test EMA calculation"""
        prices = np.array([100, 102, 101, 103, 105, 104, 106])
        ema = TechnicalIndicators._ema(prices, period=5)

        assert isinstance(ema, float)
        assert ema > 0
        # EMA should be closer to recent prices
        assert abs(ema - prices[-1]) < abs(np.mean(prices) - prices[-1])


class TestVolatilityAnalyzer:
    """Test volatility analysis functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create volatility analyzer instance"""
        return VolatilityAnalyzer()

    def test_price_addition(self, analyzer):
        """Test adding prices to history"""
        now = datetime.utcnow()
        analyzer.add_price(100.0, now)
        analyzer.add_price(102.0, now + timedelta(minutes=1))

        assert len(analyzer.price_history) == 2
        assert analyzer.price_history[0] == (now, 100.0)
        assert analyzer.price_history[1] == (now + timedelta(minutes=1), 102.0)

    def test_price_history_limit(self, analyzer):
        """Test price history length limitation"""
        now = datetime.utcnow()

        # Add more than max length
        for i in range(1500):  # More than maxlen=1440
            analyzer.add_price(100.0 + i, now + timedelta(minutes=i))

        assert len(analyzer.price_history) == 1440  # Should be capped

    def test_realized_volatility_calculation(self, analyzer):
        """Test realized volatility calculation"""
        now = datetime.utcnow()

        # Add price series with known volatility
        base_price = 100
        for i in range(60):
            # Add some realistic price movement
            price_change = np.random.normal(0, 0.5)  # 0.5% std dev
            price = base_price * (1 + price_change / 100)
            analyzer.add_price(price, now - timedelta(minutes=60 - i))
            base_price = price

        vol = analyzer.calculate_realized_volatility(30)

        assert vol >= 0  # Volatility should be non-negative
        assert vol < 2.0  # Should be reasonable for our test data

    def test_realized_volatility_insufficient_data(self, analyzer):
        """Test volatility calculation with insufficient data"""
        vol = analyzer.calculate_realized_volatility(60)
        assert vol == 0.0

        # Add only one price
        analyzer.add_price(100.0, datetime.utcnow())
        vol = analyzer.calculate_realized_volatility(60)
        assert vol == 0.0

    def test_volatility_rank_calculation(self, analyzer):
        """Test volatility rank calculation"""
        now = datetime.utcnow()

        # Create historical volatility pattern
        # Add 30 days of data (simplified)
        for day in range(30):
            for minute in range(0, 390, 10):  # Every 10 minutes during trading day
                timestamp = now - timedelta(days=30 - day, minutes=390 - minute)
                # Create varying volatility - higher in middle of period
                vol_multiplier = 1 + 0.5 * np.sin(day / 30 * np.pi)
                price_change = np.random.normal(0, 0.3 * vol_multiplier)
                price = 100 * (1 + price_change / 100)
                analyzer.add_price(price, timestamp)

        # Test with current moderate volatility
        rank = analyzer.calculate_volatility_rank(0.15)  # 15% annualized vol

        assert 0 <= rank <= 100
        assert isinstance(rank, float)

    def test_volatility_rank_insufficient_history(self, analyzer):
        """Test volatility rank with insufficient historical data"""
        rank = analyzer.calculate_volatility_rank(0.15)
        assert rank == 50.0  # Default when insufficient data


class TestMarketMicrostructure:
    """Test market microstructure analysis"""

    @pytest.fixture
    def microstructure(self):
        """Create market microstructure analyzer"""
        return MarketMicrostructure()

    def test_market_data_addition(self, microstructure):
        """Test adding market data"""
        now = datetime.utcnow()
        microstructure.add_market_data(bid=99.5, ask=100.5, volume=1000, timestamp=now)

        assert len(microstructure.bid_ask_history) == 1
        assert len(microstructure.volume_history) == 1

        data = microstructure.bid_ask_history[0]
        assert data["spread"] == 1.0  # 100.5 - 99.5
        assert data["relative_spread"] == 0.01  # 1.0 / 100.0
        assert data["mid_price"] == 100.0

    def test_avg_bid_ask_spread(self, microstructure):
        """Test average bid-ask spread calculation"""
        now = datetime.utcnow()

        # Add several data points
        spreads = [0.5, 0.75, 1.0, 0.25]
        for i, spread in enumerate(spreads):
            bid = 100 - spread / 2
            ask = 100 + spread / 2
            timestamp = now - timedelta(minutes=30 - i * 5)
            microstructure.add_market_data(bid, ask, 1000, timestamp)

        avg_spread = microstructure.get_avg_bid_ask_spread(30)
        expected_avg = np.mean([s / 100 for s in spreads])  # Relative spreads

        assert abs(avg_spread - expected_avg) < 1e-10

    def test_avg_bid_ask_spread_no_data(self, microstructure):
        """Test spread calculation with no data"""
        avg_spread = microstructure.get_avg_bid_ask_spread(30)
        assert avg_spread == 0.0

    def test_volume_profile(self, microstructure):
        """Test volume profile calculation"""
        now = datetime.utcnow()

        # Add volume data with increasing trend
        volumes = [800, 1000, 1200, 1500, 1000]
        for i, volume in enumerate(volumes):
            timestamp = now - timedelta(minutes=60 - i * 10)
            microstructure.add_market_data(99.5, 100.5, volume, timestamp)

        profile = microstructure.get_volume_profile(60)

        # Current volume (average of last few) vs total average
        current_vol = np.mean(volumes[-2:])  # Last 2 volumes
        avg_vol = np.mean(volumes)
        expected_profile = current_vol / avg_vol

        assert abs(profile - expected_profile) < 1e-10

    def test_volume_profile_no_data(self, microstructure):
        """Test volume profile with no data"""
        profile = microstructure.get_volume_profile(60)
        assert profile == 0.0


class TestMarketRegimeDetector:
    """Test market regime detection"""

    @pytest.fixture
    def detector(self):
        """Create regime detector"""
        return MarketRegimeDetector()

    def test_vix_data_addition(self, detector):
        """Test adding VIX data"""
        now = datetime.utcnow()
        detector.add_vix_data(20.5, now)

        assert len(detector.vix_history) == 1
        assert detector.vix_history[0]["vix"] == 20.5
        assert detector.vix_history[0]["timestamp"] == now

    def test_vix_regime_classification(self, detector):
        """Test VIX regime classification"""
        now = datetime.utcnow()

        # Test different VIX levels
        test_cases = [
            (12.0, "LOW_VOL"),
            (18.0, "NORMAL_VOL"),
            (28.0, "HIGH_VOL"),
            (40.0, "EXTREME_VOL"),
        ]

        for vix_level, expected_regime in test_cases:
            detector.add_vix_data(vix_level, now)
            regime = detector.get_vix_regime()
            assert regime == expected_regime

    def test_vix_regime_no_data(self, detector):
        """Test regime detection with no data"""
        regime = detector.get_vix_regime()
        assert regime == "UNKNOWN"

    def test_vix_percentile(self, detector):
        """Test VIX percentile calculation"""
        now = datetime.utcnow()

        # Add historical VIX data
        vix_values = [15, 18, 22, 16, 25, 19, 21, 14, 28, 17]
        for i, vix in enumerate(vix_values):
            timestamp = now - timedelta(days=len(vix_values) - i)
            detector.add_vix_data(vix, timestamp)

        # Test current VIX at 20 (should be around 60th percentile)
        detector.add_vix_data(20, now)
        percentile = detector.get_vix_percentile()

        # 20 is higher than 6 out of 10 values = 60th percentile
        expected_percentile = 60.0
        assert abs(percentile - expected_percentile) < 5.0  # Allow some tolerance

    def test_vix_percentile_insufficient_data(self, detector):
        """Test percentile with insufficient data"""
        percentile = detector.get_vix_percentile()
        assert percentile == 50.0


class TestMarketIndicatorEngine:
    """Test the main market indicator engine"""

    @pytest.fixture
    def engine(self):
        """Create market indicator engine"""
        return MarketIndicatorEngine()

    def test_market_data_update(self, engine):
        """Test market data update"""
        timestamp = datetime.utcnow()

        engine.update_market_data(
            price=4200.0, bid=4199.5, ask=4200.5, volume=1000, atm_iv=0.20, timestamp=timestamp
        )

        # Check that data was stored in different timeframes
        assert len(engine.price_data["1m"]) == 1
        assert len(engine.option_data) == 1

        # Verify data content
        price_data = engine.price_data["1m"][0]
        assert price_data == (timestamp, 4200.0)

        option_data = engine.option_data[0]
        assert option_data["price"] == 4200.0
        assert option_data["atm_iv"] == 0.20

    def test_feature_calculation(self, engine):
        """Test comprehensive feature calculation"""
        # Add some market data history
        now = datetime.utcnow()
        base_price = 4200.0

        for i in range(60):  # 60 minutes of data
            timestamp = now - timedelta(minutes=60 - i)
            price = base_price + np.random.normal(0, 5)  # Some price movement

            engine.update_market_data(
                price=price,
                bid=price - 0.25,
                ask=price + 0.25,
                volume=1000 + np.random.randint(-200, 200),
                atm_iv=0.20 + np.random.normal(0, 0.02),
                timestamp=timestamp,
            )
            base_price = price

        # Calculate features
        features = engine.calculate_all_features(
            current_price=base_price, implied_move=25.0, vix_level=20.0
        )

        # Validate feature structure
        assert isinstance(features, MarketFeatures)
        assert isinstance(features.timestamp, datetime)

        # Validate volatility features
        assert features.realized_vol_15m >= 0
        assert features.realized_vol_30m >= 0
        assert features.realized_vol_60m >= 0
        assert features.atm_iv > 0

        # Validate technical indicators
        assert 0 <= features.rsi_15m <= 100
        assert 0 <= features.rsi_30m <= 100
        assert -1 <= features.bb_position <= 2  # Can be outside bands

        # Validate time features
        assert 0 <= features.time_of_day <= 24
        assert 0 <= features.day_of_week <= 6
        assert features.time_to_expiry >= 0

    def test_feature_calculation_insufficient_data(self, engine):
        """Test feature calculation with minimal data"""
        features = engine.calculate_all_features(
            current_price=4200.0, implied_move=25.0, vix_level=20.0
        )

        # Should return valid features even with no history
        assert isinstance(features, MarketFeatures)

        # Many features should be at default values
        assert features.realized_vol_15m == 0.0
        assert features.rsi_15m == 50.0  # Default RSI
        assert features.bb_position == 0.5  # Default BB position

    def test_momentum_calculation(self, engine):
        """Test price momentum calculation"""
        prices = [100, 102, 101, 105, 103]

        # Test different periods
        momentum_5 = engine._calculate_momentum(prices, 5)
        momentum_3 = engine._calculate_momentum(prices, 3)

        # 5-period: (103 - 100) / 100 = 0.03
        assert abs(momentum_5 - 0.03) < 1e-10

        # 3-period: (103 - 101) / 101 ≈ 0.0198
        expected_momentum_3 = (103 - 101) / 101
        assert abs(momentum_3 - expected_momentum_3) < 1e-10

    def test_momentum_insufficient_data(self, engine):
        """Test momentum with insufficient data"""
        momentum = engine._calculate_momentum([100], 5)
        assert momentum == 0.0  # Should handle single price

        momentum = engine._calculate_momentum([], 5)
        assert momentum == 0.0  # Should handle empty list

    def test_time_to_expiry_calculation(self, engine):
        """Test time to expiry calculation"""
        with patch("app.market_indicators.datetime") as mock_datetime:
            # Mock current time as 2PM ET (6PM UTC)
            mock_now = datetime(2024, 1, 15, 18, 0, 0)  # 6PM UTC = 2PM ET
            mock_datetime.utcnow.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            time_to_expiry = engine._calculate_time_to_expiry()

            # Should be 3 hours until 4PM ET (21:00 UTC)
            assert abs(time_to_expiry - 3.0) < 0.1

    def test_trade_result_tracking(self, engine):
        """Test trade result tracking"""
        engine.add_trade_result(150.0, {"strike": 4200, "type": "strangle"})
        engine.add_trade_result(-75.0, {"strike": 4175, "type": "strangle"})

        assert len(engine.trade_history) == 2

        # Test recent performance calculations
        win_rate = engine._calculate_recent_win_rate()
        assert win_rate == 0.5  # 1 win out of 2 trades

        profit_factor = engine._calculate_recent_profit_factor()
        assert profit_factor == 2.0  # 150 profit / 75 loss


class TestMarketFeaturesDataclass:
    """Test MarketFeatures dataclass functionality"""

    def test_market_features_creation(self):
        """Test creating MarketFeatures instance"""
        timestamp = datetime.utcnow()

        features = MarketFeatures(
            realized_vol_15m=0.15,
            realized_vol_30m=0.18,
            realized_vol_60m=0.20,
            realized_vol_2h=0.22,
            realized_vol_daily=0.25,
            atm_iv=0.23,
            iv_rank=65.0,
            iv_percentile=70.0,
            iv_skew=0.05,
            iv_term_structure=0.02,
            rsi_15m=55.0,
            rsi_30m=58.0,
            macd_signal=0.15,
            macd_histogram=0.05,
            bb_position=0.6,
            bb_squeeze=0.02,
            price_momentum_15m=0.01,
            price_momentum_30m=0.015,
            price_momentum_60m=0.02,
            support_resistance_strength=0.3,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.005,
            option_volume_ratio=1.2,
            put_call_ratio=0.9,
            gamma_exposure=1500.0,
            vix_level=22.0,
            vix_term_structure=0.03,
            market_correlation=0.75,
            volume_profile=1.1,
            time_of_day=14.5,
            day_of_week=2.0,
            time_to_expiry=3.5,
            days_since_last_trade=1.0,
            win_rate_recent=0.28,
            profit_factor_recent=1.8,
            sharpe_ratio_recent=1.2,
            timestamp=timestamp,
        )

        assert features.realized_vol_15m == 0.15
        assert features.atm_iv == 0.23
        assert features.rsi_15m == 55.0
        assert features.vix_level == 22.0
        assert features.timestamp == timestamp

    def test_market_features_validation(self):
        """Test that features have reasonable values"""
        features = MarketFeatures(
            realized_vol_15m=0.15,
            realized_vol_30m=0.18,
            realized_vol_60m=0.20,
            realized_vol_2h=0.22,
            realized_vol_daily=0.25,
            atm_iv=0.23,
            iv_rank=65.0,
            iv_percentile=70.0,
            iv_skew=0.05,
            iv_term_structure=0.02,
            rsi_15m=55.0,
            rsi_30m=58.0,
            macd_signal=0.15,
            macd_histogram=0.05,
            bb_position=0.6,
            bb_squeeze=0.02,
            price_momentum_15m=0.01,
            price_momentum_30m=0.015,
            price_momentum_60m=0.02,
            support_resistance_strength=0.3,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.005,
            option_volume_ratio=1.2,
            put_call_ratio=0.9,
            gamma_exposure=1500.0,
            vix_level=22.0,
            vix_term_structure=0.03,
            market_correlation=0.75,
            volume_profile=1.1,
            time_of_day=14.5,
            day_of_week=2.0,
            time_to_expiry=3.5,
            days_since_last_trade=1.0,
            win_rate_recent=0.28,
            profit_factor_recent=1.8,
            sharpe_ratio_recent=1.2,
            timestamp=datetime.utcnow(),
        )

        # Validate RSI ranges
        assert 0 <= features.rsi_15m <= 100
        assert 0 <= features.rsi_30m <= 100

        # Validate BB position
        assert -1 <= features.bb_position <= 2  # Can be outside bands

        # Validate time features
        assert 0 <= features.time_of_day <= 24
        assert 0 <= features.day_of_week <= 6

        # Validate volatility features are non-negative
        assert features.realized_vol_15m >= 0
        assert features.atm_iv >= 0


# Integration tests for the complete indicator engine
class TestIndicatorEngineIntegration:
    """Integration tests for the complete indicator engine workflow"""

    @pytest.fixture
    def engine(self):
        """Create engine for integration tests"""
        return MarketIndicatorEngine()

    def test_complete_market_session_simulation(self, engine):
        """Test complete market session data processing"""
        # Simulate a trading day with realistic data
        start_time = datetime(2024, 1, 15, 14, 30)  # 9:30 AM ET
        base_price = 4200.0
        base_iv = 0.20

        for minute in range(390):  # Full trading day
            timestamp = start_time + timedelta(minutes=minute)

            # Simulate realistic price movement
            price_change = np.random.normal(0, 0.1)  # 0.1% std dev per minute
            base_price *= 1 + price_change / 100

            # Simulate IV changes
            iv_change = np.random.normal(0, 0.001)  # Small IV changes
            base_iv = max(0.05, min(0.50, base_iv + iv_change))

            # Add to engine
            engine.update_market_data(
                price=base_price,
                bid=base_price - 0.25,
                ask=base_price + 0.25,
                volume=1000 + np.random.randint(-300, 300),
                atm_iv=base_iv,
                timestamp=timestamp,
            )

        # Calculate final features
        features = engine.calculate_all_features(
            current_price=base_price,
            implied_move=base_price * base_iv * np.sqrt(1 / 365),  # 1-day implied move
            vix_level=20.0,
        )

        # Validate that all timeframes have data
        assert len(engine.price_data["1m"]) > 0
        assert len(engine.price_data["5m"]) > 0
        assert len(engine.price_data["15m"]) > 0
        assert len(engine.price_data["30m"]) > 0
        assert len(engine.price_data["60m"]) > 0

        # Validate feature quality
        assert features.realized_vol_15m > 0
        assert features.realized_vol_30m > 0
        assert features.realized_vol_60m > 0
        assert 20 <= features.rsi_30m <= 80  # Should be in reasonable range
        assert 0.0 <= features.bb_position <= 1.0  # Should be in valid range (can touch bands)

    def test_feature_consistency_across_timeframes(self, engine):
        """Test that features are consistent across different timeframes"""
        # Add consistent price trend
        base_price = 4200.0
        start_time = datetime.utcnow()

        # Create upward trend
        for i in range(120):  # 2 hours of data
            timestamp = start_time - timedelta(minutes=120 - i)
            price = base_price + i * 0.5  # Steady upward trend

            engine.update_market_data(
                price=price,
                bid=price - 0.25,
                ask=price + 0.25,
                volume=1000,
                atm_iv=0.20,
                timestamp=timestamp,
            )

        features = engine.calculate_all_features(
            current_price=base_price + 60, implied_move=25.0, vix_level=20.0  # Final price
        )

        # In an uptrend, shorter-term volatility might be higher
        # But all momentum indicators should be positive
        assert features.price_momentum_15m > 0
        assert features.price_momentum_30m > 0
        assert features.price_momentum_60m > 0

        # RSI should reflect upward momentum
        assert features.rsi_15m > 50
        assert features.rsi_30m > 50
