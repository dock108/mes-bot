"""
Comprehensive tests for enhanced ML-powered trading strategy integration
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine

from app.decision_engine import DecisionEngine, ExitSignal, TradingSignal
from app.enhanced_strategy import EnhancedLottoGridStrategy
from app.feature_pipeline import FeatureCollector
from app.ib_client import IBClient
from app.ml_training import ModelTrainer
from app.models import Base, DecisionHistory, Trade
from app.risk_manager import RiskManager
from app.strategy import LottoGridStrategy


class TestEnhancedLottoGridStrategy:
    """Test enhanced trading strategy with ML integration"""

    @pytest.fixture
    def database_url(self):
        """Create in-memory database for testing"""
        return "sqlite:///:memory:"

    @pytest.fixture
    def mock_ib_client(self):
        """Mock IB client for testing"""
        client = Mock(spec=IBClient)
        client.get_current_price = AsyncMock(return_value=4200.0)
        client.get_atm_straddle_price = AsyncMock(return_value=(15.0, 12.0, 27.0))
        client.place_strangle = AsyncMock(
            return_value={
                "call_strike": 4225.0,
                "put_strike": 4175.0,
                "call_price": 15.0,
                "put_price": 12.0,
                "total_premium": 27.0,
                "call_trades": [Mock(order=Mock(orderId=12345))],
                "put_trades": [Mock(order=Mock(orderId=12346))],
            }
        )
        client.is_market_hours = Mock(return_value=True)
        client.get_today_expiry_string = Mock(return_value=datetime.now().strftime("%Y%m%d"))
        client.get_mes_contract = AsyncMock(return_value=Mock())
        client.get_account_values = AsyncMock(return_value={"NetLiquidation": 10000.0})
        client.get_mes_option_contract = AsyncMock(return_value=Mock())
        client.close_position_at_market = AsyncMock(return_value=Mock())
        client.cancel_order = AsyncMock()
        return client

    @pytest.fixture
    def mock_risk_manager(self):
        """Mock risk manager for testing"""
        risk_mgr = Mock(spec=RiskManager)
        risk_mgr.can_open_new_trade = Mock(return_value=(True, "Trade approved"))
        risk_mgr.set_daily_start_equity = Mock()
        risk_mgr.should_halt_trading = Mock(return_value=(False, "No halt required"))
        risk_mgr.update_daily_summary = Mock()
        risk_mgr.get_risk_metrics_summary = Mock(return_value={})
        return risk_mgr

    @pytest.fixture
    def enhanced_strategy(self, mock_ib_client, mock_risk_manager, database_url):
        """Create enhanced strategy instance"""
        # Create database tables
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        # Mock ML implementation to avoid file system operations
        with patch("app.decision_engine.MLEnsembleImplementation") as mock_ml_impl:
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
            mock_ml_impl.return_value = mock_impl

            strategy = EnhancedLottoGridStrategy(mock_ib_client, mock_risk_manager, database_url)

            # Mock some internal components for testing
            strategy.feature_collector = Mock(spec=FeatureCollector)
            strategy.model_trainer = Mock(spec=ModelTrainer)

            return strategy

    @pytest.mark.asyncio
    async def test_enhanced_initialization(self, enhanced_strategy, mock_ib_client):
        """Test enhanced daily initialization"""
        # Mock decision engine initialization
        enhanced_strategy._initialize_decision_engine = AsyncMock()
        enhanced_strategy.model_scheduler.check_and_retrain_models = AsyncMock()
        enhanced_strategy._collect_market_features = AsyncMock()

        result = await enhanced_strategy.initialize_daily_session()

        assert result is True
        enhanced_strategy._initialize_decision_engine.assert_called_once()
        enhanced_strategy.model_scheduler.check_and_retrain_models.assert_called_once()
        enhanced_strategy._collect_market_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_initialization_ml_failure_fallback(
        self, enhanced_strategy, mock_ib_client
    ):
        """Test fallback to basic strategy when ML initialization fails"""
        enhanced_strategy.fallback_to_basic = True

        # Mock ML initialization failure
        enhanced_strategy._initialize_decision_engine = AsyncMock(
            side_effect=Exception("ML init failed")
        )

        with patch.object(LottoGridStrategy, "initialize_daily_session", return_value=True):
            result = await enhanced_strategy.initialize_daily_session()

        assert result is True
        assert enhanced_strategy.ml_enabled is False  # Should disable ML

    @pytest.mark.asyncio
    async def test_should_place_trade_enhanced_ml_enabled(self, enhanced_strategy):
        """Test enhanced trade decision with ML enabled"""
        enhanced_strategy.ml_enabled = True

        # Mock basic strategy decision
        with patch.object(
            enhanced_strategy, "should_place_trade", return_value=(True, "Basic conditions met")
        ):
            # Mock ML signal generation
            ml_signal = TradingSignal(
                action="ENTER",
                confidence=0.75,
                reasoning=["High IV rank", "Low realized volatility"],
                features_used={"iv_rank": 80.0, "realized_vol": 0.12},
                model_predictions={"volatility_model": 0.8, "ml_model": 0.7},
                optimal_strikes=(4225.0, 4175.0),
                position_size_multiplier=1.2,
                profit_target_multiplier=4.5,
            )

            enhanced_strategy._collect_market_features = AsyncMock()
            enhanced_strategy.decision_engine.generate_entry_signal = AsyncMock(
                return_value=ml_signal
            )
            enhanced_strategy._record_decision = AsyncMock()
            enhanced_strategy._get_current_vix = AsyncMock(return_value=20.0)

            enhanced_strategy.underlying_price = 4200.0
            enhanced_strategy.implied_move = 27.0

            should_trade, reason, signal = await enhanced_strategy.should_place_trade_enhanced()

            assert should_trade is True
            assert "ML+Basic agreement" in reason or "High ML confidence" in reason
            assert signal == ml_signal
            enhanced_strategy._record_decision.assert_called_once_with(ml_signal)

    @pytest.mark.asyncio
    async def test_should_place_trade_enhanced_ml_disabled(self, enhanced_strategy):
        """Test enhanced trade decision with ML disabled"""
        enhanced_strategy.ml_enabled = False

        with patch.object(
            enhanced_strategy, "should_place_trade", return_value=(True, "Basic conditions met")
        ):
            should_trade, reason, signal = await enhanced_strategy.should_place_trade_enhanced()

            assert should_trade is True
            assert reason == "Basic conditions met"
            assert signal is None

    @pytest.mark.asyncio
    async def test_should_place_trade_enhanced_ml_error_fallback(self, enhanced_strategy):
        """Test fallback when ML signal generation fails"""
        enhanced_strategy.ml_enabled = True
        enhanced_strategy.fallback_to_basic = True

        with patch.object(
            enhanced_strategy, "should_place_trade", return_value=(True, "Basic conditions met")
        ):
            # Mock ML error
            enhanced_strategy._collect_market_features = AsyncMock(
                side_effect=Exception("ML error")
            )

            should_trade, reason, signal = await enhanced_strategy.should_place_trade_enhanced()

            assert should_trade is True
            assert "ML fallback" in reason
            assert signal is None

    def test_combine_signals_high_ml_confidence(self, enhanced_strategy):
        """Test signal combination with high ML confidence"""
        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.85,  # High confidence
            reasoning=["Strong signals"],
            features_used={},
            model_predictions={},
        )

        should_trade, reason = enhanced_strategy._combine_signals(
            basic_should_trade=False,  # Basic disagrees
            basic_reason="High volatility",
            ml_signal=ml_signal,
        )

        # High ML confidence should override basic signal
        assert should_trade is True
        assert "High ML confidence" in reason

    def test_combine_signals_moderate_ml_confidence_agreement(self, enhanced_strategy):
        """Test signal combination with moderate ML confidence and agreement"""
        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.6,  # Moderate confidence
            reasoning=["Moderate signals"],
            features_used={},
            model_predictions={},
        )

        should_trade, reason = enhanced_strategy._combine_signals(
            basic_should_trade=True,  # Basic agrees
            basic_reason="Low volatility",
            ml_signal=ml_signal,
        )

        # Should trade when both agree
        assert should_trade is True
        assert "ML+Basic agreement" in reason

    def test_combine_signals_moderate_ml_confidence_disagreement(self, enhanced_strategy):
        """Test signal combination with moderate ML confidence and disagreement"""
        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.6,  # Moderate confidence
            reasoning=["Moderate signals"],
            features_used={},
            model_predictions={},
        )

        should_trade, reason = enhanced_strategy._combine_signals(
            basic_should_trade=False,  # Basic disagrees
            basic_reason="High volatility",
            ml_signal=ml_signal,
        )

        # Should not trade when they disagree
        assert should_trade is False
        assert "No consensus" in reason

    def test_combine_signals_low_ml_confidence(self, enhanced_strategy):
        """Test signal combination with low ML confidence"""
        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.2,  # Low confidence
            reasoning=["Weak signals"],
            features_used={},
            model_predictions={},
        )

        should_trade, reason = enhanced_strategy._combine_signals(
            basic_should_trade=True, basic_reason="Low volatility", ml_signal=ml_signal
        )

        # Should defer to basic strategy
        assert should_trade is True
        assert "Low ML confidence" in reason and "basic" in reason

    @pytest.mark.asyncio
    async def test_execute_trading_cycle_enhanced_with_ml_strikes(
        self, enhanced_strategy, mock_ib_client
    ):
        """Test enhanced trading cycle using ML-optimized strikes"""
        enhanced_strategy.ml_enabled = True

        # Mock successful ML signal with optimal strikes
        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.8,
            reasoning=["Favorable conditions"],
            features_used={},
            model_predictions={},
            optimal_strikes=(4230.0, 4170.0),
            position_size_multiplier=1.3,
            profit_target_multiplier=4.8,
        )

        enhanced_strategy.should_place_trade_enhanced = AsyncMock(
            return_value=(True, "Good conditions", ml_signal)
        )
        enhanced_strategy.place_enhanced_strangle_trade = AsyncMock(return_value={"success": True})
        enhanced_strategy.update_price_history = Mock()

        enhanced_strategy.underlying_price = 4200.0

        result = await enhanced_strategy.execute_trading_cycle_enhanced()

        assert result is True
        enhanced_strategy.place_enhanced_strangle_trade.assert_called_once()

        # Check that ML-optimized strikes were used
        call_args = enhanced_strategy.place_enhanced_strangle_trade.call_args
        assert call_args.kwargs["call_strike"] == 4230.0
        assert call_args.kwargs["put_strike"] == 4170.0
        assert call_args.kwargs["position_multiplier"] == 1.3
        assert call_args.kwargs["profit_target_multiplier"] == 4.8

    @pytest.mark.asyncio
    async def test_execute_trading_cycle_enhanced_fallback_strikes(
        self, enhanced_strategy, mock_ib_client
    ):
        """Test enhanced trading cycle falling back to basic strike calculation"""
        enhanced_strategy.ml_enabled = True

        # Mock ML signal without optimal strikes
        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.6,
            reasoning=["Moderate conditions"],
            features_used={},
            model_predictions={},
            optimal_strikes=None,  # No ML strikes
            position_size_multiplier=1.0,
            profit_target_multiplier=4.0,
        )

        enhanced_strategy.should_place_trade_enhanced = AsyncMock(
            return_value=(True, "Good conditions", ml_signal)
        )
        enhanced_strategy.calculate_strike_levels = Mock(return_value=[(4225.0, 4175.0)])
        enhanced_strategy.place_enhanced_strangle_trade = AsyncMock(return_value={"success": True})
        enhanced_strategy.update_price_history = Mock()

        enhanced_strategy.underlying_price = 4200.0

        result = await enhanced_strategy.execute_trading_cycle_enhanced()

        assert result is True

        # Check that basic strikes were used
        call_args = enhanced_strategy.place_enhanced_strangle_trade.call_args
        assert call_args.kwargs["call_strike"] == 4225.0
        assert call_args.kwargs["put_strike"] == 4175.0

    @pytest.mark.asyncio
    async def test_execute_trading_cycle_enhanced_ml_failure_fallback(
        self, enhanced_strategy, mock_ib_client
    ):
        """Test fallback to basic trading cycle when ML fails"""
        enhanced_strategy.fallback_to_basic = True

        # Mock ML failure
        enhanced_strategy.should_place_trade_enhanced = AsyncMock(side_effect=Exception("ML error"))

        with patch.object(
            LottoGridStrategy, "execute_trading_cycle", return_value=True
        ) as mock_basic:
            result = await enhanced_strategy.execute_trading_cycle_enhanced()

            assert result is True
            mock_basic.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_enhanced_strangle_trade(
        self, enhanced_strategy, mock_ib_client, mock_risk_manager
    ):
        """Test placing enhanced strangle with ML parameters"""
        enhanced_strategy.underlying_price = 4200.0
        enhanced_strategy.implied_move = 27.0

        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.75,
            reasoning=["Good signals"],
            features_used={},
            model_predictions={},
        )

        enhanced_strategy._record_enhanced_trade = AsyncMock(return_value=Mock(id=123))

        result = await enhanced_strategy.place_enhanced_strangle_trade(
            call_strike=4225.0,
            put_strike=4175.0,
            position_multiplier=1.2,
            profit_target_multiplier=4.5,
            ml_signal=ml_signal,
        )

        assert result is not None
        assert "trade_record" in result
        assert "strangle_result" in result
        assert "ml_enhanced" in result
        assert result["ml_enhanced"] is True

        # Verify enhanced parameters were applied
        mock_ib_client.place_strangle_legacy.assert_called_once()
        call_args = mock_ib_client.place_strangle_legacy.call_args
        # Max premium should be adjusted by position multiplier
        # Original max_premium_per_strangle * 1.2

    @pytest.mark.asyncio
    async def test_place_enhanced_strangle_trade_risk_check_failure(
        self, enhanced_strategy, mock_ib_client, mock_risk_manager
    ):
        """Test enhanced strangle placement with risk check failure"""
        enhanced_strategy.underlying_price = 4200.0

        # Mock risk check failure
        mock_risk_manager.can_open_new_trade.return_value = (False, "Insufficient margin")

        result = await enhanced_strategy.place_enhanced_strangle_trade(
            call_strike=4225.0,
            put_strike=4175.0,
            position_multiplier=1.0,
            profit_target_multiplier=4.0,
        )

        assert result is None
        mock_ib_client.place_strangle.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_open_positions_enhanced_with_ml_exit(self, enhanced_strategy):
        """Test enhanced position updates with ML exit signals"""
        enhanced_strategy.ml_enabled = True

        # Mock open trade in database
        session = enhanced_strategy.session_maker()
        try:
            trade = Trade(
                id=1,
                date=datetime.utcnow().date(),
                entry_time=datetime.utcnow() - timedelta(hours=1),
                underlying_symbol="MES",
                underlying_price_at_entry=4200.0,
                implied_move=27.0,
                call_strike=4225.0,
                put_strike=4175.0,
                call_premium=15.0,
                put_premium=12.0,
                total_premium=27.0,
                status="OPEN",
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
        finally:
            session.close()

        # Mock methods
        enhanced_strategy._update_trade_pnl = AsyncMock()
        enhanced_strategy._check_ml_exit_signal = AsyncMock()

        await enhanced_strategy.update_open_positions_enhanced()

        enhanced_strategy._update_trade_pnl.assert_called_once()
        enhanced_strategy._check_ml_exit_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_ml_exit_signal(self, enhanced_strategy):
        """Test ML exit signal checking"""
        enhanced_strategy.ml_enabled = True
        enhanced_strategy.underlying_price = 4205.0
        enhanced_strategy.implied_move = 27.0
        enhanced_strategy._get_current_vix = AsyncMock(return_value=22.0)
        enhanced_strategy._log_exit_signal = AsyncMock()

        # Mock strong exit signal
        exit_signal = ExitSignal(
            should_exit=True,
            exit_type="PROFIT_TARGET",
            confidence=0.8,
            reasoning=["Target reached", "Time decay"],
        )

        enhanced_strategy.decision_engine.generate_exit_signal = AsyncMock(return_value=exit_signal)

        # Create mock trade
        trade = Mock()
        trade.id = 123
        trade.entry_time = datetime.utcnow() - timedelta(hours=2)
        trade.underlying_price_at_entry = 4200.0
        trade.call_strike = 4225.0
        trade.put_strike = 4175.0
        trade.unrealized_pnl = 100.0

        session = Mock()

        await enhanced_strategy._check_ml_exit_signal(trade, session)

        # Should log the exit signal for strong signals
        enhanced_strategy._log_exit_signal.assert_called_once_with(123, exit_signal)

    @pytest.mark.asyncio
    async def test_record_decision(self, enhanced_strategy):
        """Test decision recording in database"""
        enhanced_strategy.underlying_price = 4200.0
        enhanced_strategy.implied_move = 27.0

        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.75,
            reasoning=["Good conditions"],
            features_used={"iv_rank": 75.0},
            model_predictions={"vol_model": 0.8},
            optimal_strikes=(4225.0, 4175.0),
            position_size_multiplier=1.2,
            profit_target_multiplier=4.5,
        )

        await enhanced_strategy._record_decision(ml_signal)

        # Verify decision was recorded
        session = enhanced_strategy.session_maker()
        try:
            decision = session.query(DecisionHistory).first()
            assert decision is not None
            assert decision.action == "ENTER"
            assert decision.confidence == 0.75
            assert decision.underlying_price == 4200.0
            assert decision.suggested_call_strike == 4225.0
            assert decision.suggested_put_strike == 4175.0
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_record_enhanced_trade(self, enhanced_strategy, mock_ib_client):
        """Test recording enhanced trade with ML metadata"""
        strangle_result = {
            "call_strike": 4225.0,
            "put_strike": 4175.0,
            "call_price": 15.0,
            "put_price": 12.0,
            "total_premium": 27.0,
            "call_trades": [Mock(order=Mock(orderId=12345))],
            "put_trades": [Mock(order=Mock(orderId=12346))],
        }

        ml_signal = TradingSignal(
            action="ENTER",
            confidence=0.8,
            reasoning=["Excellent conditions"],
            features_used={},
            model_predictions={},
        )

        enhanced_strategy.underlying_price = 4200.0
        enhanced_strategy.implied_move = 27.0
        enhanced_strategy._link_decision_to_trade = AsyncMock()

        trade = await enhanced_strategy._record_enhanced_trade(
            strangle_result=strangle_result,
            position_multiplier=1.2,
            profit_target_multiplier=4.5,
            ml_signal=ml_signal,
        )

        assert trade is not None
        assert trade.underlying_price_at_entry == 4200.0
        assert trade.call_strike == 4225.0
        assert trade.put_strike == 4175.0
        assert trade.total_premium == 27.0 * 1.2  # Adjusted by multiplier

        enhanced_strategy._link_decision_to_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_decision_engine(self, enhanced_strategy, mock_ib_client):
        """Test decision engine initialization"""
        enhanced_strategy.decision_engine.update_market_data = AsyncMock()
        enhanced_strategy._get_current_vix = AsyncMock(return_value=20.0)

        await enhanced_strategy._initialize_decision_engine()

        enhanced_strategy.decision_engine.update_market_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_market_features(self, enhanced_strategy):
        """Test market feature collection"""
        enhanced_strategy.last_feature_collection = None
        enhanced_strategy.underlying_price = 4200.0
        enhanced_strategy.implied_move = 27.0
        enhanced_strategy.feature_collector.collect_market_data = AsyncMock(return_value=True)
        enhanced_strategy.feature_collector.calculate_and_store_features = AsyncMock(
            return_value=123
        )
        enhanced_strategy._get_current_vix = AsyncMock(return_value=20.0)
        enhanced_strategy.ib_client.get_today_expiry_string = Mock(
            return_value=datetime.now().strftime("%Y%m%d")
        )
        enhanced_strategy.ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        enhanced_strategy._round_to_strike = Mock(return_value=4200.0)

        await enhanced_strategy._collect_market_features()

        enhanced_strategy.feature_collector.collect_market_data.assert_called_once()
        enhanced_strategy.feature_collector.calculate_and_store_features.assert_called_once()
        assert enhanced_strategy.last_feature_collection is not None

    @pytest.mark.asyncio
    async def test_collect_market_features_rate_limiting(self, enhanced_strategy):
        """Test that feature collection is rate limited"""
        # Set recent collection time
        enhanced_strategy.last_feature_collection = datetime.utcnow() - timedelta(minutes=2)
        enhanced_strategy.feature_collector.collect_market_data = AsyncMock()

        await enhanced_strategy._collect_market_features()

        # Should not collect due to rate limiting
        enhanced_strategy.feature_collector.collect_market_data.assert_not_called()

    def test_get_enhanced_strategy_status(self, enhanced_strategy):
        """Test enhanced strategy status reporting"""
        enhanced_strategy.ml_enabled = True
        enhanced_strategy.ml_confidence_threshold = 0.4
        enhanced_strategy.last_feature_collection = datetime.utcnow()
        enhanced_strategy.decision_engine.get_performance_summary = Mock(return_value={})

        with patch.object(enhanced_strategy, "get_strategy_status", return_value={}):
            status = enhanced_strategy.get_enhanced_strategy_status()

            assert "ml_enabled" in status
            assert "ml_confidence_threshold" in status
            assert "last_feature_collection" in status
            assert "decision_engine_performance" in status
            assert status["ml_enabled"] is True

    @pytest.mark.asyncio
    async def test_perform_ml_maintenance(self, enhanced_strategy):
        """Test ML maintenance functionality"""
        enhanced_strategy.model_scheduler.check_and_retrain_models = AsyncMock()
        enhanced_strategy._update_model_weights = AsyncMock()
        enhanced_strategy._cleanup_old_data = AsyncMock()

        await enhanced_strategy.perform_ml_maintenance()

        enhanced_strategy.model_scheduler.check_and_retrain_models.assert_called_once()
        enhanced_strategy._update_model_weights.assert_called_once()
        enhanced_strategy._cleanup_old_data.assert_called_once()
        assert enhanced_strategy.last_ml_training is not None

    @pytest.mark.asyncio
    async def test_update_model_weights(self, enhanced_strategy):
        """Test model weight updates based on performance"""
        # Mock performance summary
        enhanced_strategy.decision_engine.get_performance_summary = Mock(
            return_value={
                "volatility_based": {"recent_accuracy": 0.7},
                "ml_ensemble": {"recent_accuracy": 0.4},
            }
        )

        enhanced_strategy.decision_engine.model_weights = {
            "volatility_based": 0.6,
            "ml_ensemble": 0.4,
        }

        await enhanced_strategy._update_model_weights()

        # Volatility model should get higher weight (good performance)
        # ML ensemble should get lower weight (poor performance)
        assert enhanced_strategy.decision_engine.model_weights["volatility_based"] > 0.6
        assert enhanced_strategy.decision_engine.model_weights["ml_ensemble"] < 0.4

        # Weights should sum to approximately 1
        total_weight = sum(enhanced_strategy.decision_engine.model_weights.values())
        assert abs(total_weight - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, enhanced_strategy):
        """Test cleanup of old ML data"""
        # Add some old data to cleanup
        session = enhanced_strategy.session_maker()
        try:
            # Add old decision (should be deleted)
            old_decision = DecisionHistory(
                timestamp=datetime.utcnow() - timedelta(days=400),
                action="ENTER",
                confidence=0.5,
                underlying_price=4200.0,
                implied_move=27.0,
            )
            session.add(old_decision)
            session.commit()
        finally:
            session.close()

        await enhanced_strategy._cleanup_old_data()

        # Verify old data was removed
        session = enhanced_strategy.session_maker()
        try:
            remaining_decisions = session.query(DecisionHistory).count()
            assert remaining_decisions == 0  # Should be cleaned up
        finally:
            session.close()


class TestEnhancedStrategyIntegration:
    """Integration tests for enhanced strategy with real components"""

    @pytest.fixture
    def database_url(self):
        """Create in-memory database for testing"""
        return "sqlite:///:memory:"

    @pytest.fixture
    def integration_strategy(self, database_url):
        """Create strategy for integration testing"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        mock_ib_client = Mock(spec=IBClient)
        mock_ib_client.get_current_price = AsyncMock(return_value=4200.0)
        mock_ib_client.get_atm_straddle_price = AsyncMock(return_value=(15.0, 12.0, 27.0))
        mock_ib_client.is_market_hours = Mock(return_value=True)
        mock_ib_client.get_today_expiry_string = Mock(
            return_value=datetime.now().strftime("%Y%m%d")
        )
        mock_ib_client.get_mes_contract = AsyncMock(return_value=Mock())
        mock_ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 10000.0})

        mock_risk_manager = Mock(spec=RiskManager)
        mock_risk_manager.can_open_new_trade = Mock(return_value=(True, "Approved"))
        mock_risk_manager.set_daily_start_equity = Mock()

        # Mock ML implementation to avoid file system operations
        with patch("app.decision_engine.MLEnsembleImplementation") as mock_ml_impl:
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
            mock_ml_impl.return_value = mock_impl

            return EnhancedLottoGridStrategy(mock_ib_client, mock_risk_manager, database_url)

    @pytest.mark.asyncio
    async def test_complete_enhanced_workflow(self, integration_strategy):
        """Test complete enhanced trading workflow"""
        # Initialize strategy
        with (
            patch.object(integration_strategy, "_initialize_decision_engine"),
            patch.object(integration_strategy.model_scheduler, "check_and_retrain_models"),
            patch.object(integration_strategy, "_collect_market_features"),
        ):
            init_result = await integration_strategy.initialize_daily_session()
            assert init_result is True

        # Test feature collection
        await integration_strategy._collect_market_features()

        # Test decision generation
        integration_strategy.underlying_price = 4200.0
        integration_strategy.implied_move = 27.0

        with patch.object(
            integration_strategy.decision_engine, "generate_entry_signal"
        ) as mock_signal:
            mock_signal.return_value = TradingSignal(
                action="ENTER",
                confidence=0.7,
                reasoning=["Good conditions"],
                features_used={},
                model_predictions={},
                optimal_strikes=(4225.0, 4175.0),
            )

            should_trade, reason, signal = await integration_strategy.should_place_trade_enhanced()
            assert should_trade is True
            assert signal is not None

    @pytest.mark.asyncio
    async def test_error_resilience(self, integration_strategy):
        """Test strategy resilience to various errors"""
        integration_strategy.fallback_to_basic = True

        # Test ML initialization error
        with (
            patch.object(
                integration_strategy,
                "_initialize_decision_engine",
                side_effect=Exception("ML error"),
            ),
            patch.object(LottoGridStrategy, "initialize_daily_session", return_value=True),
        ):
            result = await integration_strategy.initialize_daily_session()
            assert result is True
            assert integration_strategy.ml_enabled is False

        # Test decision engine error
        with patch.object(
            integration_strategy, "should_place_trade", return_value=(True, "Basic OK")
        ):
            should_trade, reason, signal = await integration_strategy.should_place_trade_enhanced()
            assert should_trade is True
            assert "ML fallback" in reason or reason == "Basic OK"

    def test_configuration_handling(self, integration_strategy):
        """Test configuration parameter handling"""
        assert hasattr(integration_strategy, "ml_enabled")
        assert hasattr(integration_strategy, "ml_confidence_threshold")
        assert hasattr(integration_strategy, "fallback_to_basic")

        # Test default values
        assert integration_strategy.ml_enabled is True
        assert integration_strategy.ml_confidence_threshold == 0.4
        assert integration_strategy.fallback_to_basic is True

    @pytest.mark.asyncio
    async def test_performance_tracking(self, integration_strategy):
        """Test performance tracking functionality"""
        # Set required market data for decision recording
        integration_strategy.underlying_price = 4200.0
        integration_strategy.implied_move = 27.0

        # Record some decisions
        signal1 = TradingSignal(
            action="ENTER", confidence=0.8, reasoning=[], features_used={}, model_predictions={}
        )
        signal2 = TradingSignal(
            action="HOLD", confidence=0.3, reasoning=[], features_used={}, model_predictions={}
        )

        await integration_strategy._record_decision(signal1)
        await integration_strategy._record_decision(signal2)

        # Check that decisions were recorded
        session = integration_strategy.session_maker()
        try:
            decision_count = session.query(DecisionHistory).count()
            assert decision_count == 2
        finally:
            session.close()

        # Test status reporting
        status = integration_strategy.get_enhanced_strategy_status()
        assert "ml_decisions_count" in status or "decision_engine_performance" in status
