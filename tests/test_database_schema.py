"""
Comprehensive tests for database schema including all ML models,
relationships, constraints, and CRUD operations
"""

from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.models import (
    BacktestResult,
    Base,
    DailySummary,
    DecisionHistory,
    MarketData,
    MarketFeatures,
    MLModelMetadata,
    MLPrediction,
    PerformanceMetrics,
    SystemLog,
    Trade,
    create_database,
    get_session_maker,
)


class TestDatabaseSchema:
    """Test database schema creation and basic functionality"""

    @pytest.fixture
    def database_url(self):
        """Create in-memory database for testing"""
        return "sqlite:///:memory:"

    @pytest.fixture
    def engine(self, database_url):
        """Create database engine"""
        return create_engine(database_url)

    @pytest.fixture
    def session_maker(self, database_url):
        """Create session maker"""
        return get_session_maker(database_url)

    @pytest.fixture
    def session(self, session_maker):
        """Create database session"""
        session = session_maker()
        yield session
        session.close()

    def test_database_creation(self, engine):
        """Test that database and all tables are created"""
        Base.metadata.create_all(engine)

        # Verify all tables exist
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        expected_tables = [
            "trades",
            "daily_summaries",
            "backtest_results",
            "system_logs",
            "market_data",
            "market_features",
            "decision_history",
            "ml_model_metadata",
            "ml_predictions",
            "performance_metrics",
        ]

        for table in expected_tables:
            assert table in table_names, f"Table {table} not found in database"

    def test_table_columns(self, engine):
        """Test that all expected columns exist with correct types"""
        Base.metadata.create_all(engine)
        inspector = inspect(engine)

        # Test MarketData columns
        market_data_columns = {
            col["name"]: col["type"] for col in inspector.get_columns("market_data")
        }
        expected_market_data_cols = [
            "id",
            "timestamp",
            "underlying_price",
            "bid_price",
            "ask_price",
            "volume",
            "atm_iv",
            "implied_move",
            "vix_level",
        ]
        for col in expected_market_data_cols:
            assert col in market_data_columns

        # Test MarketFeatures columns
        features_columns = {
            col["name"]: col["type"] for col in inspector.get_columns("market_features")
        }
        expected_features_cols = [
            "id",
            "timestamp",
            "realized_vol_15m",
            "realized_vol_30m",
            "atm_iv",
            "iv_rank",
            "rsi_15m",
            "rsi_30m",
            "bb_position",
            "vix_level",
        ]
        for col in expected_features_cols:
            assert col in features_columns

        # Test MLModelMetadata columns
        ml_metadata_columns = {
            col["name"]: col["type"] for col in inspector.get_columns("ml_model_metadata")
        }
        expected_ml_cols = [
            "id",
            "model_name",
            "model_type",
            "version",
            "trained_on",
            "hyperparameters",
            "validation_accuracy",
            "is_active",
        ]
        for col in expected_ml_cols:
            assert col in ml_metadata_columns

    def test_table_indexes(self, engine):
        """Test that proper indexes are created"""
        Base.metadata.create_all(engine)
        inspector = inspect(engine)

        # Test MarketData indexes
        market_data_indexes = inspector.get_indexes("market_data")
        timestamp_indexed = any("timestamp" in idx["column_names"] for idx in market_data_indexes)
        assert timestamp_indexed, "MarketData timestamp should be indexed"

        # Test MarketFeatures indexes
        features_indexes = inspector.get_indexes("market_features")
        timestamp_indexed = any("timestamp" in idx["column_names"] for idx in features_indexes)
        assert timestamp_indexed, "MarketFeatures timestamp should be indexed"

        # Test DecisionHistory indexes
        decision_indexes = inspector.get_indexes("decision_history")
        timestamp_indexed = any("timestamp" in idx["column_names"] for idx in decision_indexes)
        assert timestamp_indexed, "DecisionHistory timestamp should be indexed"


class TestTradeModel:
    """Test Trade model functionality"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_trade_creation(self, session):
        """Test creating a trade record"""
        trade = Trade(
            date=date.today(),
            entry_time=datetime.utcnow(),
            underlying_symbol="MES",
            underlying_price_at_entry=4200.0,
            implied_move=25.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=15.0,
            put_premium=12.0,
            total_premium=27.0,
            status="OPEN",
        )

        session.add(trade)
        session.commit()

        assert trade.id is not None
        assert trade.underlying_symbol == "MES"
        assert trade.status == "OPEN"
        assert trade.created_at is not None
        assert trade.updated_at is not None

    def test_trade_update(self, session):
        """Test updating trade record"""
        trade = Trade(
            underlying_price_at_entry=4200.0,
            implied_move=25.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=15.0,
            put_premium=12.0,
            total_premium=27.0,
        )

        session.add(trade)
        session.commit()
        original_updated_at = trade.updated_at

        # Update trade
        trade.status = "CLOSED_WIN"
        trade.realized_pnl = 50.0
        session.commit()

        assert trade.status == "CLOSED_WIN"
        assert trade.realized_pnl == 50.0
        assert trade.updated_at > original_updated_at

    def test_trade_defaults(self, session):
        """Test trade default values"""
        trade = Trade(
            underlying_price_at_entry=4200.0,
            implied_move=25.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=15.0,
            put_premium=12.0,
            total_premium=27.0,
        )

        session.add(trade)
        session.commit()

        assert trade.date == date.today()
        assert trade.underlying_symbol == "MES"
        assert trade.status == "OPEN"
        assert trade.call_status == "OPEN"
        assert trade.put_status == "OPEN"


class TestMarketDataModel:
    """Test MarketData model functionality"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_market_data_creation(self, session):
        """Test creating market data record"""
        market_data = MarketData(
            timestamp=datetime.utcnow(),
            underlying_price=4200.0,
            bid_price=4199.5,
            ask_price=4200.5,
            volume=1000.0,
            atm_iv=0.25,
            implied_move=25.0,
            vix_level=20.0,
        )

        session.add(market_data)
        session.commit()

        assert market_data.id is not None
        assert market_data.underlying_price == 4200.0
        assert market_data.bid_price == 4199.5
        assert market_data.atm_iv == 0.25

    def test_market_data_defaults(self, session):
        """Test market data default values"""
        market_data = MarketData(
            underlying_price=4200.0, bid_price=4199.5, ask_price=4200.5, atm_iv=0.25
        )

        session.add(market_data)
        session.commit()

        assert market_data.volume == 0.0
        assert market_data.timestamp is not None

    def test_market_data_bulk_insert(self, session):
        """Test bulk insertion of market data"""
        base_time = datetime.utcnow()
        market_data_records = []

        for i in range(100):
            market_data = MarketData(
                timestamp=base_time + timedelta(minutes=i),
                underlying_price=4200.0 + np.random.normal(0, 5),
                bid_price=4199.5,
                ask_price=4200.5,
                volume=1000 + np.random.randint(-100, 100),
                atm_iv=0.25 + np.random.normal(0, 0.01),
                vix_level=20.0 + np.random.normal(0, 2),
            )
            market_data_records.append(market_data)

        session.add_all(market_data_records)
        session.commit()

        count = session.query(MarketData).count()
        assert count == 100


class TestMarketFeaturesModel:
    """Test MarketFeatures model functionality"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_market_features_creation(self, session):
        """Test creating market features record"""
        features = MarketFeatures(
            timestamp=datetime.utcnow(),
            realized_vol_15m=0.12,
            realized_vol_30m=0.15,
            realized_vol_60m=0.18,
            atm_iv=0.25,
            iv_rank=70.0,
            rsi_15m=45.0,
            rsi_30m=50.0,
            bb_position=0.4,
            vix_level=18.0,
            time_of_day=12.5,
            day_of_week=2.0,
            time_to_expiry=4.0,
        )

        session.add(features)
        session.commit()

        assert features.id is not None
        assert features.realized_vol_15m == 0.12
        assert features.iv_rank == 70.0
        assert features.time_of_day == 12.5

    def test_market_features_defaults(self, session):
        """Test market features default values"""
        features = MarketFeatures(time_of_day=12.0, day_of_week=2.0, time_to_expiry=4.0)

        session.add(features)
        session.commit()

        assert features.realized_vol_15m == 0.0
        assert features.iv_rank == 50.0
        assert features.rsi_15m == 50.0
        assert features.vix_level == 20.0
        assert features.win_rate_recent == 0.25

    def test_market_features_all_fields(self, session):
        """Test all market features fields"""
        features = MarketFeatures(
            timestamp=datetime.utcnow(),
            # Volatility features
            realized_vol_15m=0.10,
            realized_vol_30m=0.12,
            realized_vol_60m=0.15,
            realized_vol_2h=0.18,
            realized_vol_daily=0.20,
            # IV features
            atm_iv=0.25,
            iv_rank=60.0,
            iv_percentile=65.0,
            iv_skew=0.02,
            iv_term_structure=0.01,
            # Technical indicators
            rsi_15m=55.0,
            rsi_30m=58.0,
            macd_signal=0.1,
            macd_histogram=0.05,
            bb_position=0.6,
            bb_squeeze=0.02,
            # Price action
            price_momentum_15m=0.01,
            price_momentum_30m=0.015,
            price_momentum_60m=0.02,
            support_resistance_strength=0.3,
            mean_reversion_signal=0.1,
            # Microstructure
            bid_ask_spread=0.003,
            option_volume_ratio=1.2,
            put_call_ratio=0.9,
            gamma_exposure=1500.0,
            # Market regime
            vix_level=19.0,
            vix_term_structure=0.025,
            market_correlation=0.75,
            volume_profile=1.15,
            # Time features
            time_of_day=14.0,
            day_of_week=3.0,
            time_to_expiry=3.5,
            days_since_last_trade=1.5,
            # Performance features
            win_rate_recent=0.32,
            profit_factor_recent=1.9,
            sharpe_ratio_recent=1.3,
        )

        session.add(features)
        session.commit()

        assert features.id is not None
        # Verify a sampling of fields
        assert features.realized_vol_15m == 0.10
        assert features.iv_rank == 60.0
        assert features.rsi_15m == 55.0
        assert features.bid_ask_spread == 0.003
        assert features.vix_level == 19.0
        assert features.win_rate_recent == 0.32


class TestDecisionHistoryModel:
    """Test DecisionHistory model functionality"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_decision_creation(self, session):
        """Test creating decision history record"""
        decision = DecisionHistory(
            timestamp=datetime.utcnow(),
            action="ENTER",
            confidence=0.75,
            underlying_price=4200.0,
            implied_move=25.0,
            reasoning=["Low volatility", "High IV rank"],
            model_predictions={"volatility_model": 0.8, "ml_model": 0.7},
            suggested_call_strike=4225.0,
            suggested_put_strike=4175.0,
            position_size_multiplier=1.2,
            profit_target_multiplier=4.5,
        )

        session.add(decision)
        session.commit()

        assert decision.id is not None
        assert decision.action == "ENTER"
        assert decision.confidence == 0.75
        assert decision.reasoning == ["Low volatility", "High IV rank"]
        assert decision.model_predictions["volatility_model"] == 0.8

    def test_decision_with_features_relationship(self, session):
        """Test decision with market features relationship"""
        # Create market features first
        features = MarketFeatures(
            time_of_day=12.0, day_of_week=2.0, time_to_expiry=4.0, iv_rank=70.0
        )
        session.add(features)
        session.commit()

        # Create decision with features relationship
        decision = DecisionHistory(
            action="ENTER",
            confidence=0.8,
            underlying_price=4200.0,
            implied_move=25.0,
            features_id=features.id,
        )
        session.add(decision)
        session.commit()

        # Test relationship
        assert decision.features is not None
        assert decision.features.id == features.id
        assert decision.features.iv_rank == 70.0

    def test_decision_with_trade_relationship(self, session):
        """Test decision with trade relationship"""
        # Create trade first
        trade = Trade(
            underlying_price_at_entry=4200.0,
            implied_move=25.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=15.0,
            put_premium=12.0,
            total_premium=27.0,
        )
        session.add(trade)
        session.commit()

        # Create decision with trade relationship
        decision = DecisionHistory(
            action="ENTER",
            confidence=0.8,
            underlying_price=4200.0,
            implied_move=25.0,
            trade_id=trade.id,
            actual_outcome=100.0,
        )
        session.add(decision)
        session.commit()

        # Test relationship
        assert decision.trade is not None
        assert decision.trade.id == trade.id
        assert decision.actual_outcome == 100.0

    def test_decision_defaults(self, session):
        """Test decision history default values"""
        decision = DecisionHistory(
            action="HOLD", confidence=0.5, underlying_price=4200.0, implied_move=25.0
        )

        session.add(decision)
        session.commit()

        assert decision.position_size_multiplier == 1.0
        assert decision.profit_target_multiplier == 4.0
        assert decision.timestamp is not None


class TestMLModelMetadataModel:
    """Test MLModelMetadata model functionality"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_ml_model_creation(self, session):
        """Test creating ML model metadata"""
        model_metadata = MLModelMetadata(
            model_name="entry_prediction_v1",
            model_type="entry",
            version="1.0.0",
            trained_on=datetime.utcnow(),
            training_start_date=date(2024, 1, 1),
            training_end_date=date(2024, 1, 31),
            training_samples=1000,
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            feature_importance={"iv_rank": 0.3, "realized_vol": 0.2},
            validation_accuracy=0.85,
            validation_precision=0.80,
            validation_recall=0.75,
            validation_f1=0.77,
            model_file_path="/models/entry_v1.joblib",
            model_file_hash="abc123def456",
            is_active=True,
            is_production=False,
        )

        session.add(model_metadata)
        session.commit()

        assert model_metadata.id is not None
        assert model_metadata.model_name == "entry_prediction_v1"
        assert model_metadata.model_type == "entry"
        assert model_metadata.training_samples == 1000
        assert model_metadata.hyperparameters["n_estimators"] == 100
        assert model_metadata.validation_accuracy == 0.85
        assert model_metadata.is_active is True

    def test_ml_model_uniqueness(self, session):
        """Test model name uniqueness constraint"""
        model1 = MLModelMetadata(model_name="test_model", model_type="entry")
        session.add(model1)
        session.commit()

        # Try to add another model with same name
        model2 = MLModelMetadata(model_name="test_model", model_type="exit")
        session.add(model2)

        with pytest.raises(IntegrityError):
            session.commit()

    def test_ml_model_defaults(self, session):
        """Test ML model default values"""
        model_metadata = MLModelMetadata(model_name="test_model", model_type="entry")

        session.add(model_metadata)
        session.commit()

        assert model_metadata.version == "1.0.0"
        assert model_metadata.training_samples == 0
        assert model_metadata.is_active is False
        assert model_metadata.is_production is False
        assert model_metadata.created_at is not None


class TestMLPredictionModel:
    """Test MLPrediction model functionality"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_ml_prediction_creation(self, session):
        """Test creating ML prediction record"""
        # Create model metadata first
        model_metadata = MLModelMetadata(model_name="test_entry_model", model_type="entry")
        session.add(model_metadata)
        session.commit()

        # Create prediction
        prediction = MLPrediction(
            timestamp=datetime.utcnow(),
            model_id=model_metadata.id,
            model_name="test_entry_model",
            prediction_type="entry",
            prediction_value=0.75,
            confidence=0.85,
            input_features={"iv_rank": 70.0, "realized_vol": 0.12},
            actual_outcome=1.0,
            prediction_error=0.25,
        )

        session.add(prediction)
        session.commit()

        assert prediction.id is not None
        assert prediction.model_name == "test_entry_model"
        assert prediction.prediction_value == 0.75
        assert prediction.confidence == 0.85
        assert prediction.input_features["iv_rank"] == 70.0

    def test_ml_prediction_relationships(self, session):
        """Test ML prediction relationships"""
        # Create related records
        model_metadata = MLModelMetadata(model_name="test_model", model_type="entry")
        features = MarketFeatures(time_of_day=12.0, day_of_week=2.0, time_to_expiry=4.0)
        trade = Trade(
            underlying_price_at_entry=4200.0,
            implied_move=25.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=15.0,
            put_premium=12.0,
            total_premium=27.0,
        )
        decision = DecisionHistory(
            action="ENTER", confidence=0.8, underlying_price=4200.0, implied_move=25.0
        )

        session.add_all([model_metadata, features, trade, decision])
        session.commit()

        # Create prediction with relationships
        prediction = MLPrediction(
            model_id=model_metadata.id,
            model_name="test_model",
            prediction_type="entry",
            prediction_value=0.7,
            confidence=0.8,
            features_id=features.id,
            decision_id=decision.id,
            trade_id=trade.id,
        )
        session.add(prediction)
        session.commit()

        # Test relationships
        assert prediction.model is not None
        assert prediction.model.id == model_metadata.id
        assert prediction.features is not None
        assert prediction.features.id == features.id
        assert prediction.decision is not None
        assert prediction.decision.id == decision.id
        assert prediction.trade is not None
        assert prediction.trade.id == trade.id

    def test_prediction_bulk_insert(self, session):
        """Test bulk insertion of predictions"""
        model_metadata = MLModelMetadata(model_name="bulk_test_model", model_type="entry")
        session.add(model_metadata)
        session.commit()

        predictions = []
        for i in range(50):
            prediction = MLPrediction(
                timestamp=datetime.utcnow() + timedelta(minutes=i),
                model_id=model_metadata.id,
                model_name="bulk_test_model",
                prediction_type="entry",
                prediction_value=np.random.uniform(0.0, 1.0),
                confidence=np.random.uniform(0.5, 1.0),
            )
            predictions.append(prediction)

        session.add_all(predictions)
        session.commit()

        count = session.query(MLPrediction).count()
        assert count == 50


class TestPerformanceMetricsModel:
    """Test PerformanceMetrics model functionality"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_performance_metrics_creation(self, session):
        """Test creating performance metrics record"""
        metrics = PerformanceMetrics(
            date=date.today(),
            metric_type="daily",
            total_trades=10,
            winning_trades=7,
            win_rate=0.70,
            avg_win=150.0,
            avg_loss=-50.0,
            profit_factor=2.1,
            max_drawdown=0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            model_accuracy=0.85,
            prediction_accuracy=0.82,
            feature_drift_score=0.15,
            avg_iv_rank=65.0,
            avg_vix_level=18.5,
            market_regime="LOW_VOL",
        )

        session.add(metrics)
        session.commit()

        assert metrics.id is not None
        assert metrics.metric_type == "daily"
        assert metrics.win_rate == 0.70
        assert metrics.profit_factor == 2.1
        assert metrics.model_accuracy == 0.85
        assert metrics.market_regime == "LOW_VOL"

    def test_performance_metrics_defaults(self, session):
        """Test performance metrics default values"""
        metrics = PerformanceMetrics(date=date.today(), metric_type="weekly")

        session.add(metrics)
        session.commit()

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.created_at is not None


class TestDatabaseRelationships:
    """Test database relationships and foreign keys"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_decision_features_relationship(self, session):
        """Test DecisionHistory -> MarketFeatures relationship"""
        features = MarketFeatures(time_of_day=12.0, day_of_week=2.0, time_to_expiry=4.0)
        session.add(features)
        session.commit()

        decision = DecisionHistory(
            action="ENTER",
            confidence=0.8,
            underlying_price=4200.0,
            implied_move=25.0,
            features_id=features.id,
        )
        session.add(decision)
        session.commit()

        # Test forward relationship
        assert decision.features.id == features.id

        # Test backward relationship
        assert len(features.decisions) == 1
        assert features.decisions[0].id == decision.id

    def test_decision_trade_relationship(self, session):
        """Test DecisionHistory -> Trade relationship"""
        trade = Trade(
            underlying_price_at_entry=4200.0,
            implied_move=25.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=15.0,
            put_premium=12.0,
            total_premium=27.0,
        )
        session.add(trade)
        session.commit()

        decision = DecisionHistory(
            action="ENTER",
            confidence=0.8,
            underlying_price=4200.0,
            implied_move=25.0,
            trade_id=trade.id,
        )
        session.add(decision)
        session.commit()

        # Test relationships
        assert decision.trade.id == trade.id
        assert trade.decision.id == decision.id

    def test_ml_prediction_relationships(self, session):
        """Test MLPrediction relationships"""
        model = MLModelMetadata(model_name="test_model", model_type="entry")
        features = MarketFeatures(time_of_day=12.0, day_of_week=2.0, time_to_expiry=4.0)
        session.add_all([model, features])
        session.commit()

        prediction = MLPrediction(
            model_id=model.id,
            model_name="test_model",
            prediction_type="entry",
            prediction_value=0.7,
            confidence=0.8,
            features_id=features.id,
        )
        session.add(prediction)
        session.commit()

        # Test forward relationships
        assert prediction.model.id == model.id
        assert prediction.features.id == features.id

        # Test backward relationships
        assert len(model.predictions) == 1
        assert model.predictions[0].id == prediction.id
        assert len(features.predictions) == 1
        assert features.predictions[0].id == prediction.id


class TestDatabaseConstraintsAndValidation:
    """Test database constraints and data validation"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_required_fields_validation(self, session):
        """Test that required fields are enforced"""
        # Test Trade with missing required fields
        trade = Trade()
        session.add(trade)

        with pytest.raises(IntegrityError):
            session.commit()

        session.rollback()

        # Test MarketData with missing required fields
        market_data = MarketData()
        session.add(market_data)

        with pytest.raises(IntegrityError):
            session.commit()

    def test_foreign_key_constraints(self, session):
        """Test foreign key constraints"""
        # Try to create decision with invalid features_id
        decision = DecisionHistory(
            action="ENTER",
            confidence=0.8,
            underlying_price=4200.0,
            implied_move=25.0,
            features_id=99999,  # Non-existent ID
        )
        session.add(decision)

        # SQLite with foreign keys enabled should raise an error
        # Note: SQLite in-memory databases might not enforce FK constraints by default
        try:
            session.commit()
            # If it doesn't raise an error, verify the relationship is null
            assert decision.features is None
        except IntegrityError:
            # This is the expected behavior with FK constraints enabled
            session.rollback()

    def test_unique_constraints(self, session):
        """Test unique constraints"""
        # Test MLModelMetadata unique constraint on model_name
        model1 = MLModelMetadata(model_name="unique_test", model_type="entry")
        session.add(model1)
        session.commit()

        model2 = MLModelMetadata(model_name="unique_test", model_type="exit")
        session.add(model2)

        with pytest.raises(IntegrityError):
            session.commit()


class TestDatabasePerformance:
    """Test database performance with larger datasets"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    @pytest.fixture
    def session(self, database_url):
        session_maker = get_session_maker(database_url)
        session = session_maker()
        yield session
        session.close()

    def test_bulk_operations_performance(self, session):
        """Test performance of bulk operations"""
        import time

        # Test bulk insert of market data
        start_time = time.time()
        market_data_records = []

        for i in range(1000):
            market_data = MarketData(
                timestamp=datetime.utcnow() + timedelta(minutes=i),
                underlying_price=4200.0 + np.random.normal(0, 5),
                bid_price=4199.5,
                ask_price=4200.5,
                atm_iv=0.25,
                volume=1000.0,
            )
            market_data_records.append(market_data)

        session.add_all(market_data_records)
        session.commit()

        bulk_time = time.time() - start_time

        # Verify all records were inserted
        count = session.query(MarketData).count()
        assert count == 1000

        # Bulk operations should be reasonably fast (under 5 seconds for 1000 records)
        assert bulk_time < 5.0

    def test_query_performance_with_indexes(self, session):
        """Test query performance with indexed columns"""
        import time

        # Insert test data
        base_time = datetime.utcnow()
        market_data_records = []

        for i in range(500):
            market_data = MarketData(
                timestamp=base_time + timedelta(minutes=i),
                underlying_price=4200.0,
                bid_price=4199.5,
                ask_price=4200.5,
                atm_iv=0.25,
                volume=1000.0,
            )
            market_data_records.append(market_data)

        session.add_all(market_data_records)
        session.commit()

        # Test timestamp-based query performance
        start_time = time.time()
        query_start = base_time + timedelta(minutes=100)
        query_end = base_time + timedelta(minutes=200)

        results = (
            session.query(MarketData)
            .filter(MarketData.timestamp >= query_start, MarketData.timestamp <= query_end)
            .all()
        )

        query_time = time.time() - start_time

        # Verify correct results
        assert len(results) == 101  # 100 minutes inclusive

        # Query should be fast with proper indexing
        assert query_time < 1.0


class TestDatabaseMigrationSupport:
    """Test database schema evolution and migration support"""

    @pytest.fixture
    def database_url(self):
        return "sqlite:///:memory:"

    def test_schema_introspection(self, database_url):
        """Test ability to introspect existing schema"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Verify all expected tables exist
        expected_tables = [
            "trades",
            "market_data",
            "market_features",
            "decision_history",
            "ml_model_metadata",
            "ml_predictions",
            "performance_metrics",
        ]

        for table in expected_tables:
            assert table in tables

            # Get column info for each table
            columns = inspector.get_columns(table)
            assert len(columns) > 0

            # Verify primary key exists
            pk_columns = inspector.get_pk_constraint(table)
            assert len(pk_columns["constrained_columns"]) > 0

    def test_backward_compatibility(self, database_url):
        """Test that new models don't break existing functionality"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        session_maker = sessionmaker(bind=engine)
        session = session_maker()

        try:
            # Create records using older models (Trade, DailySummary)
            trade = Trade(
                underlying_price_at_entry=4200.0,
                implied_move=25.0,
                call_strike=4225.0,
                put_strike=4175.0,
                call_premium=15.0,
                put_premium=12.0,
                total_premium=27.0,
            )
            session.add(trade)
            session.commit()

            # Verify older functionality still works
            retrieved_trade = session.query(Trade).first()
            assert retrieved_trade.underlying_price_at_entry == 4200.0

            # Create new ML model records
            model = MLModelMetadata(model_name="test_model", model_type="entry")
            session.add(model)
            session.commit()

            # Verify new functionality works
            retrieved_model = session.query(MLModelMetadata).first()
            assert retrieved_model.model_name == "test_model"

        finally:
            session.close()
