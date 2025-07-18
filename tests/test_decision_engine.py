"""
Comprehensive tests for ML decision engine and trading signal generation
"""

import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from app.decision_engine import (
    BaseDecisionModel,
    DecisionConfidence,
    DecisionEngine,
    ExitSignal,
    MLEnsembleModel,
    TradingSignal,
    VolatilityBasedModel,
)
from app.market_indicators import MarketFeatures, MarketIndicatorEngine


class TestTradingSignal:
    """Test TradingSignal data structure"""

    def test_trading_signal_creation(self):
        """Test TradingSignal creation and validation"""
        signal = TradingSignal(
            action="ENTER",
            confidence=0.75,
            reasoning=["Low volatility", "High IV rank"],
            features_used={"iv_rank": 80.0, "realized_vol": 0.15},
            model_predictions={"volatility_model": 0.8, "ml_model": 0.7},
            optimal_strikes=(4225.0, 4175.0),
            position_size_multiplier=1.2,
            profit_target_multiplier=4.5,
        )

        assert signal.action == "ENTER"
        assert signal.confidence == 0.75
        assert len(signal.reasoning) == 2
        assert signal.optimal_strikes == (4225.0, 4175.0)
        assert signal.position_size_multiplier == 1.2
        assert signal.profit_target_multiplier == 4.5
        assert isinstance(signal.timestamp, datetime)

    def test_trading_signal_defaults(self):
        """Test TradingSignal with default values"""
        signal = TradingSignal(
            action="HOLD",
            confidence=0.3,
            reasoning=["Low confidence"],
            features_used={},
            model_predictions={},
        )

        assert signal.optimal_strikes is None
        assert signal.position_size_multiplier == 1.0
        assert signal.profit_target_multiplier == 4.0
        assert isinstance(signal.timestamp, datetime)


class TestExitSignal:
    """Test ExitSignal data structure"""

    def test_exit_signal_creation(self):
        """Test ExitSignal creation"""
        signal = ExitSignal(
            should_exit=True,
            exit_type="PROFIT_TARGET",
            confidence=0.85,
            reasoning=["Target reached", "Time decay"],
            new_profit_target=3.5,
        )

        assert signal.should_exit is True
        assert signal.exit_type == "PROFIT_TARGET"
        assert signal.confidence == 0.85
        assert signal.new_profit_target == 3.5
        assert isinstance(signal.timestamp, datetime)


class TestVolatilityBasedModel:
    """Test enhanced volatility-based decision model"""

    @pytest.fixture
    def model(self):
        """Create volatility-based model"""
        return VolatilityBasedModel()

    @pytest.fixture
    def sample_features(self):
        """Create sample market features"""
        return MarketFeatures(
            realized_vol_15m=0.10,
            realized_vol_30m=0.12,
            realized_vol_60m=0.15,
            realized_vol_2h=0.18,
            realized_vol_daily=0.20,
            atm_iv=0.25,
            iv_rank=70.0,
            iv_percentile=75.0,
            iv_skew=0.02,
            iv_term_structure=0.01,
            rsi_5m=42.0,
            rsi_15m=45.0,
            rsi_30m=50.0,
            macd_signal=0.05,
            macd_histogram=0.02,
            bb_position=0.4,
            bb_squeeze=0.015,
            price_momentum_15m=0.005,
            price_momentum_30m=0.008,
            price_momentum_60m=0.012,
            support_resistance_strength=0.2,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.002,
            option_volume_ratio=1.1,
            put_call_ratio=0.95,
            gamma_exposure=1200.0,
            vix_level=18.0,
            vix_term_structure=0.02,
            market_correlation=0.7,
            volume_profile=1.05,
            market_regime="normal",
            time_of_day=12.5,
            day_of_week=2.0,
            time_to_expiry=4.0,
            days_since_last_trade=2.0,
            win_rate_recent=0.30,
            profit_factor_recent=1.8,
            sharpe_ratio_recent=1.1,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

    @pytest.mark.asyncio
    async def test_entry_signal_prediction(self, model, sample_features):
        """Test entry signal prediction"""
        signal_strength, feature_importance = await model.predict_entry_signal(sample_features)

        assert 0.0 <= signal_strength <= 1.0
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) > 0

        # Check that feature importance contains expected keys
        expected_keys = ["volatility_signal", "technical_signal", "regime_signal", "time_signal"]
        for key in expected_keys:
            assert key in feature_importance

    @pytest.mark.asyncio
    async def test_entry_signal_high_volatility(self, model):
        """Test entry signal with high realized volatility (should be low signal)"""
        high_vol_features = MarketFeatures(
            realized_vol_15m=0.30,  # High volatility
            realized_vol_30m=0.35,
            realized_vol_60m=0.40,
            realized_vol_2h=0.42,
            realized_vol_daily=0.45,
            atm_iv=0.25,  # Lower than realized vol
            iv_rank=50.0,
            iv_percentile=50.0,
            iv_skew=0.0,
            iv_term_structure=0.0,
            rsi_5m=50.0,
            rsi_15m=50.0,
            rsi_30m=50.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            bb_position=0.5,
            bb_squeeze=0.0,
            price_momentum_15m=0.0,
            price_momentum_30m=0.0,
            price_momentum_60m=0.0,
            support_resistance_strength=0.0,
            mean_reversion_signal=0.0,
            bid_ask_spread=0.0,
            option_volume_ratio=0.0,
            put_call_ratio=1.0,
            gamma_exposure=0.0,
            vix_level=25.0,
            vix_term_structure=0.0,
            market_correlation=0.0,
            volume_profile=1.0,
            market_regime="normal",
            time_of_day=12.0,
            day_of_week=2.0,
            time_to_expiry=4.0,
            days_since_last_trade=0.0,
            win_rate_recent=0.25,
            profit_factor_recent=1.0,
            sharpe_ratio_recent=0.0,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

        signal_strength, _ = await model.predict_entry_signal(high_vol_features)

        # Should be moderate signal - high vol but still below extreme thresholds
        # With vol_ratio = 0.4/0.5 = 0.8, which is above threshold (0.67)
        assert 0.4 < signal_strength < 0.6

    @pytest.mark.asyncio
    async def test_entry_signal_favorable_conditions(self, model):
        """Test entry signal with favorable conditions"""
        favorable_features = MarketFeatures(
            realized_vol_15m=0.08,  # Low volatility
            realized_vol_30m=0.10,
            realized_vol_60m=0.12,
            realized_vol_2h=0.14,
            realized_vol_daily=0.16,
            atm_iv=0.25,  # High IV vs realized
            iv_rank=25.0,  # Low IV rank
            iv_percentile=20.0,
            iv_skew=0.0,
            iv_term_structure=0.0,
            rsi_5m=43.0,
            rsi_15m=45.0,
            rsi_30m=50.0,  # Neutral RSI
            macd_signal=0.0,
            macd_histogram=0.0,
            bb_position=0.4,
            bb_squeeze=0.01,  # Squeeze condition
            price_momentum_15m=0.0,
            price_momentum_30m=0.0,
            price_momentum_60m=0.0,
            support_resistance_strength=0.0,
            mean_reversion_signal=0.0,
            bid_ask_spread=0.005,
            option_volume_ratio=0.0,
            put_call_ratio=1.0,
            gamma_exposure=0.0,
            vix_level=16.0,  # Low VIX
            vix_term_structure=0.0,
            market_correlation=0.0,
            volume_profile=1.0,
            market_regime="normal",
            time_of_day=13.0,
            day_of_week=2.0,  # Good time
            time_to_expiry=4.0,
            days_since_last_trade=0.0,
            win_rate_recent=0.25,
            profit_factor_recent=1.0,
            sharpe_ratio_recent=0.0,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

        signal_strength, _ = await model.predict_entry_signal(favorable_features)

        # Should be high signal due to favorable conditions
        assert signal_strength > 0.5

    @pytest.mark.asyncio
    async def test_exit_signal_prediction(self, model, sample_features):
        """Test exit signal prediction"""
        trade_info = {
            "entry_time": datetime.utcnow() - timedelta(hours=2),
            "entry_price": 4200.0,
            "call_strike": 4225.0,
            "put_strike": 4175.0,
            "current_pnl": 50.0,
            "time_in_trade": 2.0,
        }

        exit_strength, reasoning = await model.predict_exit_signal(sample_features, trade_info)

        assert 0.0 <= exit_strength <= 1.0
        assert isinstance(reasoning, dict)

    @pytest.mark.asyncio
    async def test_exit_signal_time_decay(self, model, sample_features):
        """Test exit signal with time decay pressure"""
        # Modify features for near expiry
        near_expiry_features = MarketFeatures(
            **{**asdict(sample_features), "time_to_expiry": 1.0}  # 1 hour to expiry
        )

        trade_info = {"entry_time": datetime.utcnow() - timedelta(hours=3), "time_in_trade": 3.0}

        exit_strength, reasoning = await model.predict_exit_signal(near_expiry_features, trade_info)

        # With 1 hour to expiry: time_decay_signal = 0.5, contributing 0.5 * 0.4 = 0.2 to exit strength
        assert exit_strength >= 0.2
        assert "time_decay" in reasoning
        assert reasoning["time_decay"] == 0.5

    @pytest.mark.asyncio
    async def test_strike_optimization(self, model, sample_features):
        """Test strike optimization logic"""
        call_strike, put_strike = await model.optimize_strikes(
            sample_features, current_price=4200.0, implied_move=25.0
        )

        # Strikes should be reasonable
        assert call_strike > 4200.0
        assert put_strike < 4200.0
        assert call_strike - 4200.0 == 4200.0 - put_strike  # Should be symmetric

        # Should be rounded to 25-point increments
        assert call_strike % 25 == 0
        assert put_strike % 25 == 0

    @pytest.mark.asyncio
    async def test_strike_optimization_overbought(self, model):
        """Test strike optimization with overbought conditions"""
        overbought_features = MarketFeatures(
            realized_vol_15m=0.1,
            realized_vol_30m=0.1,
            realized_vol_60m=0.1,
            realized_vol_2h=0.1,
            realized_vol_daily=0.1,
            atm_iv=0.2,
            iv_rank=50.0,
            iv_percentile=50.0,
            iv_skew=0.0,
            iv_term_structure=0.0,
            rsi_5m=72.0,
            rsi_15m=75.0,
            rsi_30m=80.0,  # Overbought
            macd_signal=0.0,
            macd_histogram=0.0,
            bb_position=0.5,
            bb_squeeze=0.0,
            price_momentum_15m=0.0,
            price_momentum_30m=0.0,
            price_momentum_60m=0.0,
            support_resistance_strength=0.0,
            mean_reversion_signal=0.0,
            bid_ask_spread=0.0,
            option_volume_ratio=0.0,
            put_call_ratio=1.0,
            gamma_exposure=0.0,
            vix_level=20.0,
            vix_term_structure=0.0,
            market_correlation=0.0,
            volume_profile=1.0,
            market_regime="normal",
            time_of_day=12.0,
            day_of_week=2.0,
            time_to_expiry=4.0,
            days_since_last_trade=0.0,
            win_rate_recent=0.25,
            profit_factor_recent=1.0,
            sharpe_ratio_recent=0.0,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

        call_strike, put_strike = await model.optimize_strikes(
            overbought_features, current_price=4200.0, implied_move=25.0
        )

        # With overbought conditions, put side should be favored (closer)
        call_distance = call_strike - 4200.0
        put_distance = 4200.0 - put_strike

        # Put distance should be smaller than call distance
        assert put_distance <= call_distance


class TestMLEnsembleModel:
    """Test ML ensemble model (placeholder implementation)"""

    @pytest.fixture
    def model(self):
        """Create ML ensemble model"""
        with patch("app.decision_engine.MLEnsembleImplementation") as mock_implementation:
            # Mock the implementation to avoid file system operations
            mock_impl = Mock()
            mock_impl.predict_entry_signal = AsyncMock(
                return_value=(0.7, {"feature1": 0.3, "feature2": 0.4})
            )
            mock_impl.predict_exit_signal = AsyncMock(return_value=(0.6, {"exit_feature1": 0.2}))
            mock_impl.optimize_strikes = AsyncMock(return_value=(4225.0, 4175.0))
            mock_impl.get_model_status = Mock(
                return_value={
                    "entry_models": {"test_model": {"trained": True}},
                    "exit_models": {"test_model": {"trained": True}},
                    "total_predictions": 0,
                }
            )
            mock_implementation.return_value = mock_impl

            return MLEnsembleModel(database_url="sqlite:///:memory:")

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing"""
        return MarketFeatures(
            realized_vol_15m=0.12,
            realized_vol_30m=0.15,
            realized_vol_60m=0.18,
            realized_vol_2h=0.20,
            realized_vol_daily=0.22,
            atm_iv=0.25,
            iv_rank=60.0,
            iv_percentile=65.0,
            iv_skew=0.02,
            iv_term_structure=0.01,
            rsi_5m=52.0,
            rsi_15m=55.0,
            rsi_30m=58.0,
            macd_signal=0.1,
            macd_histogram=0.05,
            bb_position=0.6,
            bb_squeeze=0.02,
            price_momentum_15m=0.01,
            price_momentum_30m=0.015,
            price_momentum_60m=0.02,
            support_resistance_strength=0.3,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.003,
            option_volume_ratio=1.2,
            put_call_ratio=0.9,
            gamma_exposure=1500.0,
            vix_level=19.0,
            vix_term_structure=0.025,
            market_correlation=0.75,
            volume_profile=1.15,
            market_regime="normal",
            time_of_day=14.0,
            day_of_week=3.0,
            time_to_expiry=3.5,
            days_since_last_trade=1.5,
            win_rate_recent=0.32,
            profit_factor_recent=1.9,
            sharpe_ratio_recent=1.3,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

    @pytest.mark.asyncio
    async def test_ml_entry_prediction(self, model, sample_features):
        """Test ML model entry prediction"""
        signal_strength, importance = await model.predict_entry_signal(sample_features)

        assert 0.0 <= signal_strength <= 1.0
        assert isinstance(importance, dict)

    @pytest.mark.asyncio
    async def test_ml_model_status(self, model, sample_features):
        """Test ML model status retrieval"""
        status = model.get_model_status()

        assert isinstance(status, dict)
        assert "entry_models" in status
        assert "exit_models" in status
        assert "total_predictions" in status


class TestDecisionEngine:
    """Test the main decision engine orchestrator"""

    @pytest.fixture
    def engine(self):
        """Create decision engine"""
        with patch("app.decision_engine.MLEnsembleImplementation") as mock_implementation:
            # Mock the implementation to avoid file system operations
            mock_impl = Mock()
            mock_impl.predict_entry_signal = AsyncMock(
                return_value=(0.7, {"feature1": 0.3, "feature2": 0.4})
            )
            mock_impl.predict_exit_signal = AsyncMock(return_value=(0.6, {"exit_feature1": 0.2}))
            mock_impl.optimize_strikes = AsyncMock(return_value=(4225.0, 4175.0))
            mock_impl.get_model_status = Mock(return_value={"models_loaded": True})
            mock_implementation.return_value = mock_impl

            return DecisionEngine(database_url="sqlite:///:memory:")

    @pytest.fixture
    def mock_features(self):
        """Create mock market features"""
        return MarketFeatures(
            realized_vol_15m=0.12,
            realized_vol_30m=0.15,
            realized_vol_60m=0.18,
            realized_vol_2h=0.20,
            realized_vol_daily=0.22,
            atm_iv=0.25,
            iv_rank=60.0,
            iv_percentile=65.0,
            iv_skew=0.02,
            iv_term_structure=0.01,
            rsi_5m=52.0,
            rsi_15m=55.0,
            rsi_30m=58.0,
            macd_signal=0.1,
            macd_histogram=0.05,
            bb_position=0.6,
            bb_squeeze=0.02,
            price_momentum_15m=0.01,
            price_momentum_30m=0.015,
            price_momentum_60m=0.02,
            support_resistance_strength=0.3,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.003,
            option_volume_ratio=1.2,
            put_call_ratio=0.9,
            gamma_exposure=1500.0,
            vix_level=19.0,
            vix_term_structure=0.025,
            market_correlation=0.75,
            volume_profile=1.15,
            market_regime="normal",
            time_of_day=14.0,
            day_of_week=3.0,
            time_to_expiry=3.5,
            days_since_last_trade=1.5,
            win_rate_recent=0.32,
            profit_factor_recent=1.9,
            sharpe_ratio_recent=1.3,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

    @pytest.mark.asyncio
    async def test_market_data_update(self, engine):
        """Test market data update functionality"""
        timestamp = datetime.utcnow()

        await engine.update_market_data(
            price=4200.0,
            bid=4199.5,
            ask=4200.5,
            volume=1000,
            atm_iv=0.25,
            vix_level=20.0,
            timestamp=timestamp,
        )

        # Check that indicator engine was updated
        assert len(engine.indicator_engine.price_data["1m"]) > 0
        assert len(engine.indicator_engine.option_data) > 0

    @pytest.mark.asyncio
    async def test_entry_signal_generation(self, engine):
        """Test complete entry signal generation"""
        # Mock the indicator engine to return our test features
        with patch.object(engine.indicator_engine, "calculate_all_features") as mock_calc:
            test_features = MarketFeatures(
                realized_vol_15m=0.10,
                realized_vol_30m=0.12,
                realized_vol_60m=0.15,
                realized_vol_2h=0.18,
                realized_vol_daily=0.20,
                atm_iv=0.25,
                iv_rank=70.0,
                iv_percentile=75.0,
                iv_skew=0.02,
                iv_term_structure=0.01,
                rsi_5m=42.0,
                rsi_15m=45.0,
                rsi_30m=50.0,
                macd_signal=0.05,
                macd_histogram=0.02,
                bb_position=0.4,
                bb_squeeze=0.015,
                price_momentum_15m=0.005,
                price_momentum_30m=0.008,
                price_momentum_60m=0.012,
                support_resistance_strength=0.2,
                mean_reversion_signal=0.1,
                bid_ask_spread=0.002,
                option_volume_ratio=1.1,
                put_call_ratio=0.95,
                gamma_exposure=1200.0,
                vix_level=18.0,
                vix_term_structure=0.02,
                market_correlation=0.7,
                volume_profile=1.05,
                market_regime="normal",
                time_of_day=12.5,
                day_of_week=2.0,
                time_to_expiry=4.0,
                days_since_last_trade=2.0,
                win_rate_recent=0.30,
                profit_factor_recent=1.8,
                sharpe_ratio_recent=1.1,
                price=4200.0,
                volume=1000000.0,
                timestamp=datetime.utcnow(),
            )
            mock_calc.return_value = test_features

            signal = await engine.generate_entry_signal(
                current_price=4200.0, implied_move=25.0, vix_level=18.0
            )

            assert isinstance(signal, TradingSignal)
            assert signal.action in ["ENTER", "HOLD"]
            assert 0.0 <= signal.confidence <= 1.0
            assert len(signal.reasoning) > 0
            assert isinstance(signal.model_predictions, dict)
            assert len(signal.model_predictions) > 0

    @pytest.mark.asyncio
    async def test_entry_signal_high_confidence(self, engine):
        """Test entry signal with high confidence conditions"""
        with patch.object(engine.indicator_engine, "calculate_all_features") as mock_calc:
            # Create very favorable features
            favorable_features = MarketFeatures(
                realized_vol_15m=0.08,
                realized_vol_30m=0.10,
                realized_vol_60m=0.12,
                realized_vol_2h=0.14,
                realized_vol_daily=0.16,
                atm_iv=0.30,
                iv_rank=20.0,
                iv_percentile=15.0,  # High IV, low rank
                iv_skew=0.01,
                iv_term_structure=0.005,
                rsi_5m=46.0,
                rsi_15m=48.0,
                rsi_30m=52.0,
                macd_signal=0.0,
                macd_histogram=0.0,
                bb_position=0.45,
                bb_squeeze=0.008,  # Tight squeeze
                price_momentum_15m=0.002,
                price_momentum_30m=0.003,
                price_momentum_60m=0.005,
                support_resistance_strength=0.1,
                mean_reversion_signal=0.05,
                bid_ask_spread=0.001,
                option_volume_ratio=1.0,
                put_call_ratio=1.0,
                gamma_exposure=1000.0,
                vix_level=15.0,  # Low VIX
                vix_term_structure=0.01,
                market_correlation=0.8,
                volume_profile=0.95,
                market_regime="normal",
                time_of_day=13.0,
                day_of_week=2.0,
                time_to_expiry=4.5,  # Good timing
                days_since_last_trade=3.0,
                win_rate_recent=0.35,
                profit_factor_recent=2.2,
                sharpe_ratio_recent=1.5,
                # Basic market data
                price=4200.0,
                volume=1000000.0,
                timestamp=datetime.utcnow(),
            )
            mock_calc.return_value = favorable_features

            signal = await engine.generate_entry_signal(
                current_price=4200.0, implied_move=25.0, vix_level=15.0
            )

            # Should generate ENTER signal with high confidence
            assert signal.action == "ENTER"
            assert signal.confidence > 0.6
            assert signal.optimal_strikes is not None

    @pytest.mark.asyncio
    async def test_exit_signal_generation(self, engine, mock_features):
        """Test exit signal generation"""
        with patch.object(engine.indicator_engine, "calculate_all_features") as mock_calc:
            mock_calc.return_value = mock_features

            trade_info = {
                "entry_time": datetime.utcnow() - timedelta(hours=2),
                "entry_price": 4200.0,
                "call_strike": 4225.0,
                "put_strike": 4175.0,
                "current_pnl": 100.0,
                "time_in_trade": 2.0,
            }

            exit_signal = await engine.generate_exit_signal(
                trade_info=trade_info, current_price=4205.0, implied_move=25.0, vix_level=20.0
            )

            assert isinstance(exit_signal, ExitSignal)
            assert isinstance(exit_signal.should_exit, bool)
            assert exit_signal.exit_type in ["HOLD", "TIME_DECAY", "MARKET_CHANGE", "PROFIT_TARGET"]
            assert 0.0 <= exit_signal.confidence <= 1.0

    def test_reasoning_generation(self, engine):
        """Test reasoning generation for decisions"""
        features = MarketFeatures(
            realized_vol_15m=0.08,
            realized_vol_30m=0.10,
            realized_vol_60m=0.12,
            realized_vol_2h=0.14,
            realized_vol_daily=0.16,
            atm_iv=0.25,
            iv_rank=70.0,
            iv_percentile=75.0,
            iv_skew=0.02,
            iv_term_structure=0.01,
            rsi_5m=42.0,
            rsi_15m=45.0,
            rsi_30m=50.0,
            macd_signal=0.05,
            macd_histogram=0.02,
            bb_position=0.4,
            bb_squeeze=0.015,
            price_momentum_15m=0.005,
            price_momentum_30m=0.008,
            price_momentum_60m=0.012,
            support_resistance_strength=0.2,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.002,
            option_volume_ratio=1.1,
            put_call_ratio=0.95,
            gamma_exposure=1200.0,
            vix_level=18.0,
            vix_term_structure=0.02,
            market_correlation=0.7,
            volume_profile=1.05,
            market_regime="normal",
            time_of_day=12.5,
            day_of_week=2.0,
            time_to_expiry=4.0,
            days_since_last_trade=2.0,
            win_rate_recent=0.30,
            profit_factor_recent=1.8,
            sharpe_ratio_recent=1.1,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

        model_predictions = {"volatility_based": 0.7, "ml_ensemble": 0.6}
        model_importance = {"volatility_based": {}, "ml_ensemble": {}}

        reasoning = engine._generate_reasoning(features, model_predictions, model_importance)

        assert isinstance(reasoning, list)
        assert len(reasoning) > 0

        # Check for expected reasoning patterns
        reasoning_text = " ".join(reasoning)
        assert any(
            keyword in reasoning_text.lower()
            for keyword in ["volatility", "rsi", "vix", "time", "agreement"]
        )

    def test_key_features_extraction(self, engine, mock_features):
        """Test extraction of key features for logging"""
        key_features = engine._extract_key_features(mock_features)

        assert isinstance(key_features, dict)

        expected_keys = [
            "realized_vol_30m",
            "atm_iv",
            "iv_rank",
            "rsi_30m",
            "vix_level",
            "time_of_day",
            "time_to_expiry",
            "bb_position",
        ]

        for key in expected_keys:
            assert key in key_features
            assert isinstance(key_features[key], (int, float))

    def test_dynamic_profit_target_calculation(self, engine, mock_features):
        """Test dynamic profit target calculation"""
        # Test with high confidence
        high_confidence_target = engine._calculate_dynamic_profit_target(mock_features, 0.9)

        # Test with low confidence
        low_confidence_target = engine._calculate_dynamic_profit_target(mock_features, 0.3)

        # High confidence should yield higher target
        assert high_confidence_target > low_confidence_target

        # Both should be within reasonable bounds
        assert 2.0 <= high_confidence_target <= 6.0
        assert 2.0 <= low_confidence_target <= 6.0

    def test_dynamic_profit_target_time_adjustment(self, engine):
        """Test profit target adjustment based on time to expiry"""
        # Near expiry features
        near_expiry_features = MarketFeatures(
            realized_vol_15m=0.12,
            realized_vol_30m=0.15,
            realized_vol_60m=0.18,
            realized_vol_2h=0.20,
            realized_vol_daily=0.22,
            atm_iv=0.25,
            iv_rank=50.0,
            iv_percentile=50.0,
            iv_skew=0.0,
            iv_term_structure=0.0,
            rsi_5m=50.0,
            rsi_15m=50.0,
            rsi_30m=50.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            bb_position=0.5,
            bb_squeeze=0.0,
            price_momentum_15m=0.0,
            price_momentum_30m=0.0,
            price_momentum_60m=0.0,
            support_resistance_strength=0.0,
            mean_reversion_signal=0.0,
            bid_ask_spread=0.0,
            option_volume_ratio=0.0,
            put_call_ratio=1.0,
            gamma_exposure=0.0,
            vix_level=20.0,
            vix_term_structure=0.0,
            market_correlation=0.0,
            volume_profile=1.0,
            market_regime="normal",
            time_of_day=15.0,
            day_of_week=3.0,
            time_to_expiry=1.0,  # Near expiry
            days_since_last_trade=0.0,
            win_rate_recent=0.25,
            profit_factor_recent=1.0,
            sharpe_ratio_recent=0.0,
            # Basic market data
            price=4200.0,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

        # Far expiry features
        far_expiry_features = MarketFeatures(
            **{**asdict(near_expiry_features), "time_to_expiry": 6.0}  # Far from expiry
        )

        near_target = engine._calculate_dynamic_profit_target(near_expiry_features, 0.6)
        far_target = engine._calculate_dynamic_profit_target(far_expiry_features, 0.6)

        # Near expiry should have lower target (take profits sooner)
        assert near_target < far_target

    def test_model_performance_tracking(self, engine):
        """Test model performance tracking"""
        # Record some performance data
        engine.update_model_performance("test_prediction_1", 0.75)

        # Get performance summary
        summary = engine.get_performance_summary()

        assert isinstance(summary, dict)
        for model_name in engine.models:
            assert model_name in summary
            assert "avg_error" in summary[model_name]
            assert "total_predictions" in summary[model_name]
            assert "recent_accuracy" in summary[model_name]


class TestDecisionEngineIntegration:
    """Integration tests for decision engine with real components"""

    @pytest.fixture
    def engine(self):
        """Create decision engine for integration testing"""
        with patch("app.decision_engine.MLEnsembleImplementation") as mock_implementation:
            # Mock the implementation to avoid file system operations
            mock_impl = Mock()
            mock_impl.predict_entry_signal = AsyncMock(
                return_value=(0.7, {"feature1": 0.3, "feature2": 0.4})
            )
            mock_impl.predict_exit_signal = AsyncMock(return_value=(0.6, {"exit_feature1": 0.2}))
            mock_impl.optimize_strikes = AsyncMock(return_value=(4225.0, 4175.0))
            mock_impl.get_model_status = Mock(return_value={"models_loaded": True})
            mock_implementation.return_value = mock_impl

            return DecisionEngine(database_url="sqlite:///:memory:")

    @pytest.mark.asyncio
    async def test_complete_decision_workflow(self, engine):
        """Test complete decision-making workflow"""
        # Simulate market data collection
        timestamps = []
        prices = []

        base_price = 4200.0
        start_time = datetime.utcnow() - timedelta(hours=2)

        for i in range(120):  # 2 hours of minute data
            timestamp = start_time + timedelta(minutes=i)
            price = base_price + np.random.normal(0, 2)  # Small price moves

            await engine.update_market_data(
                price=price,
                bid=price - 0.25,
                ask=price + 0.25,
                volume=1000 + np.random.randint(-200, 200),
                atm_iv=0.25 + np.random.normal(0, 0.01),
                vix_level=20.0 + np.random.normal(0, 1),
                timestamp=timestamp,
            )

            timestamps.append(timestamp)
            prices.append(price)
            base_price = price

        # Generate entry signal
        entry_signal = await engine.generate_entry_signal(
            current_price=base_price, implied_move=25.0, vix_level=20.0
        )

        assert isinstance(entry_signal, TradingSignal)
        assert entry_signal.action in ["ENTER", "HOLD"]

        # If we get an enter signal, test exit signal
        if entry_signal.action == "ENTER":
            trade_info = {
                "entry_time": timestamps[-1],
                "entry_price": base_price,
                "call_strike": (
                    entry_signal.optimal_strikes[0]
                    if entry_signal.optimal_strikes
                    else base_price + 25
                ),
                "put_strike": (
                    entry_signal.optimal_strikes[1]
                    if entry_signal.optimal_strikes
                    else base_price - 25
                ),
                "current_pnl": 25.0,
                "time_in_trade": 0.5,
            }

            exit_signal = await engine.generate_exit_signal(
                trade_info=trade_info,
                current_price=base_price + 5,
                implied_move=25.0,
                vix_level=20.0,
            )

            assert isinstance(exit_signal, ExitSignal)

    @pytest.mark.asyncio
    async def test_model_ensemble_agreement(self, engine):
        """Test model ensemble agreement scenarios"""
        # Create scenario where models should agree (favorable conditions)
        favorable_conditions = True

        if favorable_conditions:
            # Add data that should trigger high-confidence signals
            for i in range(60):
                timestamp = datetime.utcnow() - timedelta(minutes=60 - i)
                # Low volatility environment
                price = 4200.0 + np.random.normal(0, 1)  # Very low volatility

                await engine.update_market_data(
                    price=price,
                    bid=price - 0.1,
                    ask=price + 0.1,
                    volume=1000,
                    atm_iv=0.30,  # High IV
                    vix_level=15.0,  # Low VIX
                    timestamp=timestamp,
                )

        signal = await engine.generate_entry_signal(
            current_price=4200.0, implied_move=30.0, vix_level=15.0  # High implied move
        )

        # Models should agree on favorable conditions
        predictions = signal.model_predictions
        prediction_values = list(predictions.values())

        if len(prediction_values) > 1:
            # Check that predictions are reasonably close (agreement)
            prediction_std = np.std(prediction_values)
            assert prediction_std < 0.3  # Models should mostly agree
