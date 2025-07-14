"""
Comprehensive tests for ML training framework including model training,
hyperparameter optimization, and model lifecycle management
"""

import asyncio
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from app.feature_pipeline import FeatureEngineer
from app.ml_training import (
    BaseMLModel,
    EntryPredictionModel,
    ExitPredictionModel,
    ModelPerformance,
    ModelScheduler,
    ModelTrainer,
    StrikeOptimizationModel,
    TrainingConfig,
)
from app.models import Base, MLModelMetadata, get_session_maker


class TestTrainingConfig:
    """Test training configuration data structure"""

    def test_training_config_creation(self):
        """Test TrainingConfig creation and validation"""
        config = TrainingConfig(
            model_type="entry",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            train_start_date=date(2024, 1, 1),
            train_end_date=date(2024, 1, 31),
            validation_split=0.2,
            cross_validation_folds=5,
            feature_selection=True,
            scale_features=True,
        )

        assert config.model_type == "entry"
        assert config.algorithm == "random_forest"
        assert config.hyperparameters["n_estimators"] == 100
        assert config.validation_split == 0.2
        assert config.cross_validation_folds == 5
        assert config.feature_selection is True
        assert config.scale_features is True


class TestModelPerformance:
    """Test model performance metrics structure"""

    def test_model_performance_creation(self):
        """Test ModelPerformance creation"""
        performance = ModelPerformance(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            mse=0.15,
            r2_score=0.65,
            cross_val_mean=0.82,
            cross_val_std=0.03,
            feature_importance={"feature1": 0.3, "feature2": 0.7},
        )

        assert performance.accuracy == 0.85
        assert performance.precision == 0.80
        assert performance.recall == 0.75
        assert performance.f1_score == 0.77
        assert performance.mse == 0.15
        assert performance.r2_score == 0.65
        assert performance.cross_val_mean == 0.82
        assert performance.cross_val_std == 0.03
        assert performance.feature_importance["feature1"] == 0.3


class TestEntryPredictionModel:
    """Test entry prediction model functionality"""

    @pytest.fixture
    def training_config(self):
        """Create training configuration for entry model"""
        return TrainingConfig(
            model_type="entry",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 5, "random_state": 42},
            train_start_date=date(2024, 1, 1),
            train_end_date=date(2024, 1, 31),
            validation_split=0.2,
            cross_validation_folds=3,
        )

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100

        # Create realistic features
        features = pd.DataFrame(
            {
                "realized_vol_30m": np.random.uniform(0.1, 0.3, n_samples),
                "atm_iv": np.random.uniform(0.15, 0.35, n_samples),
                "iv_rank": np.random.uniform(0, 100, n_samples),
                "rsi_30m": np.random.uniform(20, 80, n_samples),
                "vix_level": np.random.uniform(12, 35, n_samples),
                "time_of_day": np.random.uniform(9, 16, n_samples),
                "bb_position": np.random.uniform(0, 1, n_samples),
                "price_momentum_30m": np.random.normal(0, 0.01, n_samples),
            }
        )

        # Create target based on features (synthetic relationship)
        # Higher IV rank and lower realized vol should lead to profitable trades
        target_probs = (
            features["iv_rank"] / 100 * 0.3
            + (0.3 - features["realized_vol_30m"]) / 0.2 * 0.3
            + np.random.uniform(0, 0.4, n_samples)
        )
        targets = (target_probs > 0.5).astype(int)

        return features, targets

    def test_entry_model_creation(self, training_config):
        """Test entry model creation"""
        model = EntryPredictionModel(training_config)

        assert model.config.model_type == "entry"
        assert model.config.algorithm == "random_forest"
        assert not model.is_trained
        assert model.model is None

    def test_entry_model_create_model(self, training_config):
        """Test creating the underlying ML model"""
        model = EntryPredictionModel(training_config)
        ml_model = model.create_model()

        # Should return a RandomForestClassifier
        assert hasattr(ml_model, "fit")
        assert hasattr(ml_model, "predict")
        assert hasattr(ml_model, "predict_proba")

    def test_entry_model_prepare_targets(self, training_config):
        """Test target preparation"""
        model = EntryPredictionModel(training_config)

        # Test with target_profitable column
        df_with_target = pd.DataFrame(
            {"target_profitable": [1, 0, 1, 0, 1], "actual_outcome": [100, -50, 75, -25, 120]}
        )

        targets = model.prepare_targets(df_with_target)
        expected = np.array([1, 0, 1, 0, 1])
        np.testing.assert_array_equal(targets, expected)

        # Test without target_profitable column (use actual_outcome)
        df_without_target = pd.DataFrame({"actual_outcome": [100, -50, 75, -25, 120]})

        targets = model.prepare_targets(df_without_target)
        expected = np.array([1, 0, 1, 0, 1])  # Positive outcomes = 1
        np.testing.assert_array_equal(targets, expected)

    def test_entry_model_training(self, training_config, sample_training_data):
        """Test complete model training"""
        model = EntryPredictionModel(training_config)
        X, y = sample_training_data

        performance = model.train(X, y)

        assert model.is_trained
        assert isinstance(performance, ModelPerformance)
        assert 0 <= performance.accuracy <= 1
        assert 0 <= performance.precision <= 1
        assert 0 <= performance.recall <= 1
        assert 0 <= performance.f1_score <= 1
        assert performance.cross_val_mean is not None
        assert performance.cross_val_std is not None

    def test_entry_model_prediction(self, training_config, sample_training_data):
        """Test model prediction"""
        model = EntryPredictionModel(training_config)
        X, y = sample_training_data

        # Train model
        model.train(X, y)

        # Test prediction
        predictions = model.predict(X.head(10))

        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

        # Test probability prediction
        probabilities = model.predict_proba(X.head(10))

        assert probabilities is not None
        assert probabilities.shape == (10, 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_entry_model_save_load(self, training_config, sample_training_data):
        """Test model saving and loading"""
        model = EntryPredictionModel(training_config)
        X, y = sample_training_data

        # Train model
        model.train(X, y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = f.name

        try:
            file_hash = model.save_model(temp_path)
            assert isinstance(file_hash, str)
            assert len(file_hash) == 64  # SHA256 hash length
            assert os.path.exists(temp_path)

            # Load model
            loaded_model = EntryPredictionModel.load_model(temp_path)

            assert loaded_model.is_trained
            assert loaded_model.feature_names == model.feature_names

            # Test that loaded model makes same predictions
            original_preds = model.predict(X.head(5))
            loaded_preds = loaded_model.predict(X.head(5))
            np.testing.assert_array_equal(original_preds, loaded_preds)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_entry_model_feature_importance(self, training_config, sample_training_data):
        """Test feature importance extraction"""
        model = EntryPredictionModel(training_config)
        X, y = sample_training_data

        model.train(X, y)
        importance = model._get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)

        # All features should have non-negative importance
        for feature, imp in importance.items():
            assert imp >= 0
            assert feature in X.columns

        # Importance should sum to approximately 1 for tree-based models
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.1


class TestExitPredictionModel:
    """Test exit prediction model functionality"""

    @pytest.fixture
    def exit_config(self):
        """Create configuration for exit model"""
        return TrainingConfig(
            model_type="exit",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 5, "random_state": 42},
            train_start_date=date(2024, 1, 1),
            train_end_date=date(2024, 1, 31),
        )

    @pytest.fixture
    def exit_training_data(self):
        """Create sample exit training data"""
        np.random.seed(42)
        n_samples = 100

        features = pd.DataFrame(
            {
                "time_to_expiry": np.random.uniform(0.5, 6, n_samples),
                "realized_vol_15m": np.random.uniform(0.1, 0.4, n_samples),
                "atm_iv": np.random.uniform(0.15, 0.35, n_samples),
                "rsi_15m": np.random.uniform(10, 90, n_samples),
                "current_pnl_pct": np.random.uniform(-0.5, 2.0, n_samples),
                "time_in_trade": np.random.uniform(0.5, 5, n_samples),
            }
        )

        # Create exit targets based on time to expiry and P&L
        # Should exit if close to expiry or high profit
        exit_signals = (
            (features["time_to_expiry"] < 1.5) | (features["current_pnl_pct"] > 1.5)
        ).astype(int)

        return features, exit_signals

    def test_exit_model_creation(self, exit_config):
        """Test exit model creation"""
        model = ExitPredictionModel(exit_config)

        assert model.config.model_type == "exit"
        assert not model.is_trained

    def test_exit_model_training(self, exit_config, exit_training_data):
        """Test exit model training"""
        model = ExitPredictionModel(exit_config)
        X, y = exit_training_data

        performance = model.train(X, y)

        assert model.is_trained
        assert isinstance(performance, ModelPerformance)
        assert 0 <= performance.accuracy <= 1


class TestStrikeOptimizationModel:
    """Test strike optimization model functionality"""

    @pytest.fixture
    def strike_config(self):
        """Create configuration for strike model"""
        return TrainingConfig(
            model_type="strike_selection",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 5, "random_state": 42},
            train_start_date=date(2024, 1, 1),
            train_end_date=date(2024, 1, 31),
        )

    @pytest.fixture
    def strike_training_data(self):
        """Create sample strike training data"""
        np.random.seed(42)
        n_samples = 100

        features = pd.DataFrame(
            {
                "implied_move": np.random.uniform(15, 35, n_samples),
                "iv_rank": np.random.uniform(0, 100, n_samples),
                "realized_vol_30m": np.random.uniform(0.1, 0.3, n_samples),
                "rsi_30m": np.random.uniform(20, 80, n_samples),
                "vix_level": np.random.uniform(12, 35, n_samples),
                "time_to_expiry": np.random.uniform(2, 6, n_samples),
            }
        )

        # Create profit targets (continuous variable)
        # Higher IV rank and optimal time to expiry should yield better profits
        profits = (
            features["iv_rank"] / 100 * 100
            + features["time_to_expiry"] * 10
            + np.random.normal(0, 25, n_samples)
        )

        return features, profits

    def test_strike_model_creation(self, strike_config):
        """Test strike model creation"""
        model = StrikeOptimizationModel(strike_config)

        assert model.config.model_type == "strike_selection"
        assert not model.is_trained

    def test_strike_model_training(self, strike_config, strike_training_data):
        """Test strike model training"""
        model = StrikeOptimizationModel(strike_config)
        X, y = strike_training_data

        performance = model.train(X, y)

        assert model.is_trained
        assert isinstance(performance, ModelPerformance)
        assert performance.accuracy == 0.0  # N/A for regression
        assert performance.mse is not None
        assert performance.r2_score is not None

    def test_strike_model_regression_prediction(self, strike_config, strike_training_data):
        """Test strike model continuous predictions"""
        model = StrikeOptimizationModel(strike_config)
        X, y = strike_training_data

        model.train(X, y)
        predictions = model.predict(X.head(10))

        assert len(predictions) == 10
        assert all(isinstance(pred, (int, float)) for pred in predictions)

        # Predictions should be reasonable profit values
        assert all(-200 <= pred <= 500 for pred in predictions)


class TestModelTrainer:
    """Test main model trainer functionality"""

    @pytest.fixture
    def database_url(self):
        """Create in-memory database for testing"""
        return "sqlite:///:memory:"

    @pytest.fixture
    def trainer(self, database_url):
        """Create model trainer with test database"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        return ModelTrainer(database_url)

    @pytest.fixture
    def mock_feature_engineer(self):
        """Create mock feature engineer with sample data"""
        mock_engineer = Mock(spec=FeatureEngineer)

        # Create sample training dataset
        np.random.seed(42)
        n_samples = 150  # Increased to meet minimum requirement

        sample_data = pd.DataFrame(
            {
                "realized_vol_30m": np.random.uniform(0.1, 0.3, n_samples),
                "atm_iv": np.random.uniform(0.15, 0.35, n_samples),
                "iv_rank": np.random.uniform(0, 100, n_samples),
                "rsi_30m": np.random.uniform(20, 80, n_samples),
                "vix_level": np.random.uniform(12, 35, n_samples),
                "time_of_day": np.random.uniform(9, 16, n_samples),
                "bb_position": np.random.uniform(0, 1, n_samples),
                "price_momentum_30m": np.random.normal(0, 0.01, n_samples),
                "actual_outcome": np.random.normal(0, 100, n_samples),
            }
        )

        feature_cols = [col for col in sample_data.columns if col != "actual_outcome"]

        mock_engineer.prepare_ml_dataset.return_value = (sample_data, feature_cols)
        return mock_engineer

    @pytest.mark.asyncio
    async def test_train_entry_model_success(self, trainer, mock_feature_engineer):
        """Test successful entry model training"""
        # Mock the feature engineer
        trainer.feature_engineer = mock_feature_engineer

        # Mock data quality check
        trainer.data_quality_monitor.check_data_quality = Mock(
            return_value={"completeness": 0.9, "consistency": 0.85, "freshness": 0.8}
        )

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        model = await trainer.train_entry_model(start_date, end_date, "random_forest")

        assert model is not None
        assert isinstance(model, EntryPredictionModel)
        assert model.is_trained

    @pytest.mark.asyncio
    async def test_train_entry_model_insufficient_data_quality(
        self, trainer, mock_feature_engineer
    ):
        """Test entry model training with poor data quality"""
        trainer.feature_engineer = mock_feature_engineer

        # Mock poor data quality
        trainer.data_quality_monitor.check_data_quality = Mock(
            return_value={
                "completeness": 0.5,  # Poor completeness
                "consistency": 0.6,
                "freshness": 0.7,
            }
        )

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        model = await trainer.train_entry_model(start_date, end_date, "random_forest")

        assert model is None  # Should return None due to poor data quality

    @pytest.mark.asyncio
    async def test_train_entry_model_insufficient_samples(self, trainer):
        """Test entry model training with insufficient samples"""
        # Mock feature engineer to return empty dataset
        mock_engineer = Mock(spec=FeatureEngineer)
        mock_engineer.prepare_ml_dataset.return_value = (pd.DataFrame(), [])
        trainer.feature_engineer = mock_engineer

        trainer.data_quality_monitor.check_data_quality = Mock(
            return_value={"completeness": 0.9, "consistency": 0.85, "freshness": 0.8}
        )

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        model = await trainer.train_entry_model(start_date, end_date, "random_forest")

        assert model is None  # Should return None due to insufficient data

    @pytest.mark.asyncio
    async def test_train_exit_model(self, trainer, mock_feature_engineer):
        """Test exit model training"""
        trainer.feature_engineer = mock_feature_engineer

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        model = await trainer.train_exit_model(start_date, end_date, "random_forest")

        assert model is not None
        assert isinstance(model, ExitPredictionModel)
        assert model.is_trained

    @pytest.mark.asyncio
    async def test_train_strike_model(self, trainer, mock_feature_engineer):
        """Test strike optimization model training"""
        trainer.feature_engineer = mock_feature_engineer

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        model = await trainer.train_strike_model(start_date, end_date, "random_forest")

        assert model is not None
        assert isinstance(model, StrikeOptimizationModel)
        assert model.is_trained

    @pytest.mark.asyncio
    async def test_hyperparameter_optimization(self, trainer, mock_feature_engineer):
        """Test hyperparameter optimization"""
        trainer.feature_engineer = mock_feature_engineer

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        best_params = await trainer.hyperparameter_optimization(
            "entry", "random_forest", start_date, end_date
        )

        assert isinstance(best_params, dict)
        # Should contain RandomForest parameters
        expected_params = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
        for param in expected_params:
            assert param in best_params

    def test_get_default_hyperparameters(self, trainer):
        """Test default hyperparameter retrieval"""
        # Test classification parameters
        rf_class_params = trainer._get_default_hyperparameters("random_forest", "classification")
        assert "n_estimators" in rf_class_params
        assert "random_state" in rf_class_params

        # Test regression parameters
        rf_reg_params = trainer._get_default_hyperparameters("random_forest", "regression")
        assert "n_estimators" in rf_reg_params
        assert "random_state" in rf_reg_params

        # Test unknown algorithm
        unknown_params = trainer._get_default_hyperparameters("unknown_algo", "classification")
        assert unknown_params == {}

    @pytest.mark.asyncio
    async def test_save_model_metadata(self, trainer):
        """Test saving model metadata to database"""
        config = TrainingConfig(
            model_type="entry",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 100},
            train_start_date=date(2024, 1, 1),
            train_end_date=date(2024, 1, 31),
        )

        performance = ModelPerformance(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            feature_importance={"feature1": 0.5, "feature2": 0.5},
        )

        await trainer._save_model_metadata(
            model_name="test_entry_model",
            model_type="entry",
            config=config,
            performance=performance,
            model_path="/path/to/model.joblib",
            file_hash="abc123",
            training_samples=100,
        )

        # Verify metadata was saved
        session = trainer.session_maker()
        try:
            metadata = (
                session.query(MLModelMetadata)
                .filter(MLModelMetadata.model_name == "test_entry_model")
                .first()
            )

            assert metadata is not None
            assert metadata.model_type == "entry"
            assert metadata.training_samples == 100
            assert metadata.validation_accuracy == 0.85
            assert metadata.is_active is True
        finally:
            session.close()


class TestModelScheduler:
    """Test automated model retraining scheduler"""

    @pytest.fixture
    def database_url(self):
        """Create in-memory database for testing"""
        return "sqlite:///:memory:"

    @pytest.fixture
    def mock_trainer(self, database_url):
        """Create mock trainer for scheduler testing"""
        mock_trainer = Mock(spec=ModelTrainer)
        mock_trainer.session_maker = get_session_maker(database_url)

        # Create database tables
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        return mock_trainer

    @pytest.fixture
    def scheduler(self, mock_trainer):
        """Create model scheduler with mock trainer"""
        return ModelScheduler(mock_trainer)

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler.trainer is not None
        assert "entry" in scheduler.retraining_schedule
        assert "exit" in scheduler.retraining_schedule
        assert "strike_selection" in scheduler.retraining_schedule

        # Check default intervals
        assert scheduler.retraining_schedule["entry"].days == 7
        assert scheduler.retraining_schedule["exit"].days == 14
        assert scheduler.retraining_schedule["strike_selection"].days == 30

    @pytest.mark.asyncio
    async def test_check_and_retrain_no_models(self, scheduler):
        """Test retraining check when no models exist"""
        # Mock training methods
        scheduler.trainer.train_entry_model = AsyncMock(return_value=Mock())
        scheduler.trainer.train_exit_model = AsyncMock(return_value=Mock())
        scheduler.trainer.train_strike_model = AsyncMock(return_value=Mock())

        await scheduler.check_and_retrain_models()

        # Should attempt to train all model types since none exist
        scheduler.trainer.train_entry_model.assert_called_once()
        scheduler.trainer.train_exit_model.assert_called_once()
        scheduler.trainer.train_strike_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_retrain_stale_models(self, scheduler, mock_trainer):
        """Test retraining of stale models"""
        # Add old model metadata to database
        session = mock_trainer.session_maker()
        try:
            old_model = MLModelMetadata(
                model_name="old_entry_model",
                model_type="entry",
                trained_on=datetime.utcnow() - timedelta(days=10),  # 10 days old
                is_active=True,
            )
            session.add(old_model)
            session.commit()
        finally:
            session.close()

        # Mock training methods
        scheduler.trainer.train_entry_model = AsyncMock(return_value=Mock())
        scheduler.trainer.train_exit_model = AsyncMock(return_value=Mock())
        scheduler.trainer.train_strike_model = AsyncMock(return_value=Mock())

        await scheduler.check_and_retrain_models()

        # Should retrain entry model (>7 days old) but not others
        scheduler.trainer.train_entry_model.assert_called_once()
        scheduler.trainer.train_exit_model.assert_called_once()  # No models exist
        scheduler.trainer.train_strike_model.assert_called_once()  # No models exist

    @pytest.mark.asyncio
    async def test_check_and_retrain_recent_models(self, scheduler, mock_trainer):
        """Test that recent models are not retrained"""
        # Add recent model metadata to database
        session = mock_trainer.session_maker()
        try:
            for model_type in ["entry", "exit", "strike_selection"]:
                recent_model = MLModelMetadata(
                    model_name=f"recent_{model_type}_model",
                    model_type=model_type,
                    trained_on=datetime.utcnow() - timedelta(hours=1),  # 1 hour old
                    is_active=True,
                )
                session.add(recent_model)
            session.commit()
        finally:
            session.close()

        # Mock training methods
        scheduler.trainer.train_entry_model = AsyncMock()
        scheduler.trainer.train_exit_model = AsyncMock()
        scheduler.trainer.train_strike_model = AsyncMock()

        await scheduler.check_and_retrain_models()

        # Should not retrain any models since they're all recent
        scheduler.trainer.train_entry_model.assert_not_called()
        scheduler.trainer.train_exit_model.assert_not_called()
        scheduler.trainer.train_strike_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrain_specific_model(self, scheduler):
        """Test retraining specific model types"""
        # Mock training methods
        scheduler.trainer.train_entry_model = AsyncMock(return_value=Mock())
        scheduler.trainer.train_exit_model = AsyncMock(return_value=Mock())
        scheduler.trainer.train_strike_model = AsyncMock(return_value=Mock())

        # Test retraining entry model
        await scheduler._retrain_model("entry")
        scheduler.trainer.train_entry_model.assert_called_once()

        # Test retraining exit model
        await scheduler._retrain_model("exit")
        scheduler.trainer.train_exit_model.assert_called_once()

        # Test retraining strike model
        await scheduler._retrain_model("strike_selection")
        scheduler.trainer.train_strike_model.assert_called_once()


class TestMLTrainingIntegration:
    """Integration tests for complete ML training pipeline"""

    @pytest.fixture
    def database_url(self):
        """Create in-memory database for testing"""
        return "sqlite:///:memory:"

    @pytest.fixture
    def full_trainer(self, database_url):
        """Create trainer with real database"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        return ModelTrainer(database_url)

    def test_model_directory_creation(self, full_trainer):
        """Test that model directory is created"""
        assert full_trainer.model_dir.exists()
        assert full_trainer.model_dir.is_dir()

    @pytest.mark.asyncio
    async def test_complete_training_workflow(self, full_trainer):
        """Test complete training workflow with mocked data"""
        # Mock feature engineer to provide training data
        np.random.seed(42)
        n_samples = 100

        training_data = pd.DataFrame(
            {
                "realized_vol_30m": np.random.uniform(0.1, 0.3, n_samples),
                "atm_iv": np.random.uniform(0.15, 0.35, n_samples),
                "iv_rank": np.random.uniform(0, 100, n_samples),
                "rsi_30m": np.random.uniform(20, 80, n_samples),
                "vix_level": np.random.uniform(12, 35, n_samples),
                "time_of_day": np.random.uniform(9, 16, n_samples),
                "bb_position": np.random.uniform(0, 1, n_samples),
                "price_momentum_30m": np.random.normal(0, 0.01, n_samples),
                "actual_outcome": np.random.normal(0, 100, n_samples),
            }
        )

        feature_cols = [col for col in training_data.columns if col != "actual_outcome"]

        full_trainer.feature_engineer.prepare_ml_dataset = Mock(
            return_value=(training_data, feature_cols)
        )

        # Mock data quality check
        full_trainer.data_quality_monitor.check_data_quality = Mock(
            return_value={"completeness": 0.9, "consistency": 0.85, "freshness": 0.8}
        )

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        # Train entry model
        entry_model = await full_trainer.train_entry_model(start_date, end_date)

        assert entry_model is not None
        assert entry_model.is_trained

        # Verify metadata was saved
        session = full_trainer.session_maker()
        try:
            metadata = (
                session.query(MLModelMetadata).filter(MLModelMetadata.model_type == "entry").first()
            )

            assert metadata is not None
            assert metadata.is_active is True
            assert metadata.training_samples == n_samples
        finally:
            session.close()

    def test_algorithm_support(self, full_trainer):
        """Test that trainer supports different algorithms"""
        algorithms = ["random_forest", "gradient_boosting", "logistic_regression", "neural_network"]

        for algorithm in algorithms:
            params = full_trainer._get_default_hyperparameters(algorithm, "classification")
            assert isinstance(params, dict)

            # Each algorithm should have some default parameters
            if algorithm in ["random_forest", "gradient_boosting"]:
                assert "n_estimators" in params
            elif algorithm == "logistic_regression":
                assert "random_state" in params
            elif algorithm == "neural_network":
                assert "hidden_layer_sizes" in params

    def test_error_handling_in_training(self, full_trainer):
        """Test error handling during training"""
        # Mock feature engineer to raise exception
        full_trainer.feature_engineer.prepare_ml_dataset = Mock(
            side_effect=Exception("Data processing error")
        )

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        # Should handle error gracefully and return None
        result = asyncio.run(full_trainer.train_entry_model(start_date, end_date))
        assert result is None

    @pytest.mark.asyncio
    async def test_model_versioning(self, full_trainer):
        """Test model versioning and replacement"""
        # Create minimal training data
        training_data = pd.DataFrame(
            {
                "feature1": np.random.random(150),  # Increased to meet minimum requirement
                "feature2": np.random.random(150),
                "actual_outcome": np.random.normal(0, 100, 150),
            }
        )

        full_trainer.feature_engineer.prepare_ml_dataset = Mock(
            return_value=(training_data, ["feature1", "feature2"])
        )

        full_trainer.data_quality_monitor.check_data_quality = Mock(
            return_value={"completeness": 0.9, "consistency": 0.85, "freshness": 0.8}
        )

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        # Train first model
        model1 = await full_trainer.train_entry_model(start_date, end_date)
        assert model1 is not None

        # Train second model (should replace first)
        model2 = await full_trainer.train_entry_model(start_date, end_date)
        assert model2 is not None

        # Check that only one active model exists
        session = full_trainer.session_maker()
        try:
            active_models = (
                session.query(MLModelMetadata)
                .filter(MLModelMetadata.model_type == "entry", MLModelMetadata.is_active == True)
                .count()
            )

            assert active_models == 1
        finally:
            session.close()
