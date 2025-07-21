"""
Tests for synthetic data generator
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from app.market_indicators import MarketFeatures
from app.synthetic_data_generator import SyntheticDataGenerator


@pytest.mark.integration
@pytest.mark.db
class TestSyntheticDataGenerator:
    """Test synthetic data generation functionality"""

    @pytest.fixture
    def mock_session_maker(self):
        """Create mock session maker"""
        session_maker = Mock()
        session = Mock()
        session_maker.return_value = session
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        session.add = Mock()
        session.add_all = Mock()
        session.flush = Mock()
        session.query = Mock()
        return session_maker

    @pytest.fixture
    def generator(self, mock_session_maker):
        """Create generator instance with mocked dependencies"""
        with patch(
            "app.synthetic_data_generator.get_session_maker", return_value=mock_session_maker
        ):
            return SyntheticDataGenerator("test_db_url")

    def test_init(self):
        """Test generator initialization"""
        with patch("app.synthetic_data_generator.get_session_maker") as mock_get_session:
            generator = SyntheticDataGenerator("test_db_url")

            assert generator.database_url == "test_db_url"
            assert "low_vol" in generator.market_regimes
            assert "favorable" in generator.outcome_probabilities
            mock_get_session.assert_called_once_with("test_db_url")

    def test_generate_market_features_normal_regime(self, generator):
        """Test market feature generation for normal regime"""
        timestamp = datetime.utcnow()
        features = generator.generate_market_features(timestamp, regime="normal", base_price=4500.0)

        # Check basic properties
        assert isinstance(features, MarketFeatures)
        assert features.timestamp == timestamp
        assert features.market_regime == "normal"
        assert 4400 <= features.price <= 4600  # Price should be near base

        # Check volatility measures are correlated
        assert 0.1 <= features.realized_vol_15m <= 0.5
        assert 0.1 <= features.realized_vol_30m <= 0.5
        assert abs(features.realized_vol_30m - features.realized_vol_15m) < 0.2

        # Check RSI bounds
        assert 5 <= features.rsi_5m <= 95
        assert 5 <= features.rsi_15m <= 95
        assert 5 <= features.rsi_30m <= 95

        # Check time features
        assert 0 <= features.time_of_day <= 24
        assert 0 <= features.day_of_week <= 6

    def test_generate_market_features_crisis_regime(self, generator):
        """Test market feature generation for crisis regime"""
        timestamp = datetime.utcnow()
        features = generator.generate_market_features(timestamp, regime="crisis", base_price=4000.0)

        assert features.market_regime == "crisis"
        # Crisis should have higher volatility
        assert features.realized_vol_daily > 0.2
        # VIX should be in crisis range
        assert 35 <= features.vix_level <= 60

    def test_classify_market_condition_favorable(self, generator):
        """Test market condition classification - favorable"""
        features = MarketFeatures(
            realized_vol_15m=0.15,
            realized_vol_30m=0.16,
            realized_vol_60m=0.17,
            realized_vol_2h=0.18,
            realized_vol_daily=0.20,
            atm_iv=0.25,  # High IV relative to realized vol
            iv_rank=70,  # High IV rank
            iv_percentile=75,
            iv_skew=0.0,
            iv_term_structure=0.0,
            rsi_5m=50,
            rsi_15m=50,
            rsi_30m=50,  # Neutral RSI
            macd_signal=0.0,
            macd_histogram=0.0,
            bb_position=0.5,
            bb_squeeze=0.0,
            price_momentum_15m=0.0,
            price_momentum_30m=0.0,
            price_momentum_60m=0.0,
            support_resistance_strength=0.5,
            mean_reversion_signal=0.0,
            bid_ask_spread=0.01,
            option_volume_ratio=1.0,
            put_call_ratio=1.0,
            gamma_exposure=0.0,
            vix_level=18,  # Low VIX
            vix_term_structure=0.0,
            market_correlation=0.5,
            volume_profile=1.0,
            market_regime="normal",
            time_of_day=12.0,  # Good trading hours
            day_of_week=2.0,
            time_to_expiry=4.0,
            days_since_last_trade=1.0,
            win_rate_recent=0.6,
            profit_factor_recent=1.5,
            sharpe_ratio_recent=1.2,
            price=4500.0,
            volume=300000,
            timestamp=datetime.utcnow(),
        )

        condition = generator.classify_market_condition(features)
        assert condition == "favorable"

    def test_classify_market_condition_unfavorable(self, generator):
        """Test market condition classification - unfavorable"""
        features = MarketFeatures(
            realized_vol_15m=0.35,
            realized_vol_30m=0.36,
            realized_vol_60m=0.37,
            realized_vol_2h=0.38,
            realized_vol_daily=0.40,
            atm_iv=0.30,  # Low IV relative to realized vol
            iv_rank=20,  # Low IV rank
            iv_percentile=25,
            iv_skew=0.0,
            iv_term_structure=0.0,
            rsi_5m=85,
            rsi_15m=85,
            rsi_30m=85,  # Overbought RSI
            macd_signal=0.0,
            macd_histogram=0.0,
            bb_position=0.9,
            bb_squeeze=0.0,
            price_momentum_15m=0.0,
            price_momentum_30m=0.0,
            price_momentum_60m=0.0,
            support_resistance_strength=0.5,
            mean_reversion_signal=0.0,
            bid_ask_spread=0.03,
            option_volume_ratio=1.0,
            put_call_ratio=1.5,
            gamma_exposure=-0.1,
            vix_level=35,  # High VIX
            vix_term_structure=0.0,
            market_correlation=0.9,
            volume_profile=0.5,
            market_regime="high_vol",
            time_of_day=8.0,  # Poor trading hours
            day_of_week=1.0,
            time_to_expiry=1.0,
            days_since_last_trade=0.5,
            win_rate_recent=0.3,
            profit_factor_recent=0.8,
            sharpe_ratio_recent=-0.5,
            price=4200.0,
            volume=100000,
            timestamp=datetime.utcnow(),
        )

        condition = generator.classify_market_condition(features)
        assert condition == "unfavorable"

    def test_generate_trade_outcome_winning(self, generator):
        """Test trade outcome generation - winning trade"""
        # Mock random to ensure winning trade
        with patch("numpy.random.random", return_value=0.1):  # Below win rate threshold
            features = Mock()
            is_winner, pnl_multiplier = generator.generate_trade_outcome("favorable", features)

            assert is_winner is True
            assert 2.7 <= pnl_multiplier <= 3.7  # avg_profit ± 0.5

    def test_generate_trade_outcome_losing(self, generator):
        """Test trade outcome generation - losing trade"""
        # Mock random to ensure losing trade
        with patch("numpy.random.random", return_value=0.9):  # Above win rate threshold
            features = Mock()
            is_winner, pnl_multiplier = generator.generate_trade_outcome("unfavorable", features)

            assert is_winner is False
            assert -2.8 <= pnl_multiplier <= -2.2  # avg_loss ± 0.3

    # @patch("app.synthetic_data_generator.logger")
    # def test_create_synthetic_dataset_success(self, mock_logger, generator, mock_session_maker):
    #     """Test successful synthetic dataset creation"""
    #     session = mock_session_maker()

    #     # Mock MarketFeaturesModel to track created instances
    #     with patch("app.synthetic_data_generator.MarketFeaturesModel") as mock_mf_model:
    #         with patch("app.synthetic_data_generator.DecisionHistory") as mock_decision:
    #             with patch("app.synthetic_data_generator.Trade") as mock_trade:
    #                 # Setup mocks
    #                 mock_feature_instance = Mock()
    #                 mock_feature_instance.id = 1
    #                 mock_mf_model.return_value = mock_feature_instance

    #                 mock_decision_instance = Mock()
    #                 mock_decision.return_value = mock_decision_instance

    #                 mock_trade_instance = Mock()
    #                 mock_trade_instance.id = 1
    #                 mock_trade.return_value = mock_trade_instance

    #                 # Run dataset creation
    #                 result = generator.create_synthetic_dataset(num_records=10)

    #                 assert result is True
    #                 # Use call_count for more reliable assertions across Python versions
    #                 assert session.add.call_count > 0
    #                 assert session.add_all.call_count > 0
    #                 assert session.commit.call_count > 0
    #                 assert session.rollback.call_count == 0
    #                 assert mock_logger.info.call_count > 0

    @patch("app.synthetic_data_generator.logger")
    def test_create_synthetic_dataset_filters_weekends(
        self, mock_logger, generator, mock_session_maker
    ):
        """Test that weekend timestamps are filtered out"""
        session = mock_session_maker()

        # Track timestamps
        created_timestamps = []

        def track_timestamp(*args, **kwargs):
            if "timestamp" in kwargs:
                created_timestamps.append(kwargs["timestamp"])
            return Mock(id=1)

        with patch("app.synthetic_data_generator.MarketFeaturesModel", side_effect=track_timestamp):
            with patch("app.synthetic_data_generator.DecisionHistory", return_value=Mock()):
                with patch("app.synthetic_data_generator.Trade", return_value=Mock(id=1)):
                    # Force some weekend timestamps
                    with patch("numpy.random.randint") as mock_randint:
                        # Generate timestamps that include weekends
                        mock_randint.side_effect = [
                            0,  # Monday
                            86400,  # Tuesday
                            172800,  # Wednesday
                            259200,  # Thursday
                            345600,  # Friday
                            432000,  # Saturday (should be skipped)
                            518400,  # Sunday (should be skipped)
                            604800,  # Next Monday
                            691200,  # Next Tuesday
                            777600,  # Next Wednesday
                        ]

                        generator.create_synthetic_dataset(num_records=10)

                        # Check no weekend timestamps were used
                        for ts in created_timestamps:
                            assert ts.weekday() < 5  # 0-4 are weekdays

    @patch("app.synthetic_data_generator.logger")
    def test_create_synthetic_dataset_exception_handling(
        self, mock_logger, generator, mock_session_maker
    ):
        """Test exception handling during dataset creation"""
        session = mock_session_maker()

        # Make session.add raise an exception
        session.add.side_effect = Exception("Database error")

        result = generator.create_synthetic_dataset(num_records=10)

        assert result == False
        assert session.rollback.called
        assert session.close.called
        mock_logger.error.assert_called()

    def test_validate_generated_data_with_data(self, generator, mock_session_maker):
        """Test data validation with existing data"""
        session = mock_session_maker()

        # Mock query results
        mock_features = [Mock(iv_rank=50, vix_level=20, market_regime="normal") for _ in range(10)]
        mock_trades = [
            Mock(realized_pnl=100),
            Mock(realized_pnl=150),
            Mock(realized_pnl=-50),
            Mock(realized_pnl=-75),
            Mock(realized_pnl=200),
        ]

        # Setup query mocks
        query_mock = Mock()
        session.query.return_value = query_mock
        query_mock.count.side_effect = [10, 8, 5]  # features, decisions, trades
        query_mock.all.side_effect = [mock_trades, mock_features]

        result = generator.validate_generated_data()

        assert result["data_counts"]["features"] == 10
        assert result["data_counts"]["decisions"] == 8
        assert result["data_counts"]["trades"] == 5
        assert result["trade_metrics"]["win_rate"] == 0.6  # 3 wins out of 5
        assert result["trade_metrics"]["avg_profit"] == 150.0  # (100+150+200)/3
        assert result["trade_metrics"]["avg_loss"] == -62.5  # (-50-75)/2
        assert result["trade_metrics"]["total_pnl"] == 325.0
        assert result["market_characteristics"]["avg_iv_rank"] == 50.0
        assert result["market_characteristics"]["avg_vix"] == 20.0

    def test_validate_generated_data_empty_database(self, generator, mock_session_maker):
        """Test data validation with empty database"""
        session = mock_session_maker()

        # Mock empty query results
        query_mock = Mock()
        session.query.return_value = query_mock
        query_mock.count.return_value = 0
        query_mock.all.return_value = []

        result = generator.validate_generated_data()

        assert result["data_counts"]["features"] == 0
        assert result["data_counts"]["decisions"] == 0
        assert result["data_counts"]["trades"] == 0
        assert result["trade_metrics"]["win_rate"] == 0
        assert result["trade_metrics"]["avg_profit"] == 0
        assert result["trade_metrics"]["avg_loss"] == 0

    @patch("app.synthetic_data_generator.logger")
    def test_validate_generated_data_exception(self, mock_logger, generator, mock_session_maker):
        """Test validation error handling"""
        session = mock_session_maker()
        session.query.side_effect = Exception("Query error")

        result = generator.validate_generated_data()

        assert result == {}
        mock_logger.error.assert_called()
        assert session.close.called

    def test_main_function(self):
        """Test main function execution"""
        with patch("app.synthetic_data_generator.SyntheticDataGenerator") as mock_generator_class:
            with patch("app.synthetic_data_generator.config") as mock_config:
                with patch("builtins.print") as mock_print:
                    # Setup mocks
                    mock_config.database.url = "test_url"
                    mock_generator = Mock()
                    mock_generator_class.return_value = mock_generator
                    mock_generator.create_synthetic_dataset.return_value = True
                    mock_generator.validate_generated_data.return_value = {
                        "data_counts": {"features": 100, "decisions": 80, "trades": 50},
                        "trade_metrics": {
                            "win_rate": 0.6,
                            "avg_profit": 100,
                            "avg_loss": -50,
                            "total_pnl": 2500,
                        },
                        "market_characteristics": {
                            "avg_iv_rank": 50,
                            "avg_vix": 20,
                            "regime_distribution": {"normal": 80, "high_vol": 20},
                        },
                    }

                    # Import and run main
                    from app.synthetic_data_generator import main

                    main()

                    # Verify execution
                    mock_generator.create_synthetic_dataset.assert_called_once_with(
                        num_records=1200
                    )
                    mock_generator.validate_generated_data.assert_called_once()
                    assert mock_print.call_count > 5  # Should print multiple results

    def test_main_function_failure(self):
        """Test main function when generation fails"""
        with patch("app.synthetic_data_generator.SyntheticDataGenerator") as mock_generator_class:
            with patch("app.synthetic_data_generator.config") as mock_config:
                with patch("builtins.print") as mock_print:
                    # Setup mocks for failure
                    mock_config.database.url = "test_url"
                    mock_generator = Mock()
                    mock_generator_class.return_value = mock_generator
                    mock_generator.create_synthetic_dataset.return_value = False

                    # Import and run main
                    from app.synthetic_data_generator import main

                    main()

                    # Should not call validate when creation fails
                    mock_generator.validate_generated_data.assert_not_called()
                    # Should print failure message
                    mock_print.assert_called_with("Failed to generate synthetic data")
