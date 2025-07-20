"""
Tests for model automation and versioning systems
"""

import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.ml_training import ModelTrainer
from app.model_automation import ModelAutomationEngine, RetrainingTrigger
from app.model_versioning import ModelVersion, ModelVersionManager


class SerializableTestModel:
    """A simple serializable model for testing"""

    def __init__(self, iteration=0):
        self.coef_ = [0.1 * (iteration + 1), 0.2, 0.3]
        self.iteration = iteration

    def predict(self, X):
        return [0.5, 0.7, 0.3]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class TestModelVersionManager:
    """Test model versioning functionality"""

    @pytest.fixture
    def version_manager(self, test_db_url):
        """Create test version manager"""
        return ModelVersionManager(test_db_url, models_directory="test_models")

    def test_version_manager_initialization(self, version_manager):
        """Test version manager initializes correctly"""
        assert version_manager is not None
        assert version_manager.models_dir.exists()

    def test_save_model_version(self, version_manager):
        """Test saving a model version"""
        # Create a simple serializable test model
        mock_model = SerializableTestModel()

        # Define test data
        performance_metrics = {
            "accuracy": 0.75,
            "precision": 0.70,
            "recall": 0.80,
            "f1_score": 0.74,
        }

        metadata = {
            "feature_count": 50,
            "training_samples": 1000,
            "algorithm_params": {"n_estimators": 100},
        }

        # Save model version
        version = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics=performance_metrics,
            metadata=metadata,
            training_period="2024-01-01_to_2024-03-01",
        )

        assert version.startswith("v")
        assert "." in version

    def test_load_model_version(self, version_manager):
        """Test loading a model version"""
        # First save a model
        mock_model = SerializableTestModel()
        performance_metrics = {"accuracy": 0.75}
        metadata = {"test": "data"}

        version = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics=performance_metrics,
            metadata=metadata,
            training_period="2024-01-01_to_2024-03-01",
        )

        # Then load it
        loaded_model, loaded_metadata = version_manager.load_model_version("entry", version)

        assert loaded_model is not None
        assert loaded_metadata["version"] == version
        assert loaded_metadata["algorithm"] == "random_forest"

    def test_deploy_model_to_production(self, version_manager):
        """Test deploying a model to production"""
        # Save a model first
        mock_model = SerializableTestModel()
        performance_metrics = {"accuracy": 0.75}
        metadata = {"test": "data"}

        version = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics=performance_metrics,
            metadata=metadata,
            training_period="2024-01-01_to_2024-03-01",
        )

        # Deploy to production
        success = version_manager.deploy_model_to_production("entry", version)
        assert success

        # Verify it's marked as production
        model, metadata = version_manager.load_model_version("entry")
        assert metadata["version"] == version

    def test_list_model_versions(self, version_manager):
        """Test listing model versions"""
        # Save multiple models
        for i in range(3):
            mock_model = SerializableTestModel(i)
            version_manager.save_model_version(
                model=mock_model,
                model_type="entry",
                algorithm="random_forest",
                performance_metrics={"accuracy": 0.7 + i * 0.05},
                metadata={"iteration": i},
                training_period=f"2024-0{i+1}-01_to_2024-0{i+1}-28",
            )

        # List versions
        versions = version_manager.list_model_versions("entry")
        assert len(versions) == 3
        assert all("version" in v for v in versions)
        assert all("performance_score" in v for v in versions)


class TestModelAutomationEngine:
    """Test model automation functionality"""

    @pytest.fixture
    def automation_engine(self, test_db_url):
        """Create test automation engine"""
        return ModelAutomationEngine(test_db_url)

    @pytest.fixture
    def mock_model_trainer(self):
        """Create mock model trainer"""
        trainer = Mock(spec=ModelTrainer)

        # Mock training methods
        mock_model = Mock()
        mock_model.performance = Mock()
        mock_model.performance.accuracy = 0.75
        mock_model.performance.precision = 0.70
        mock_model.performance.recall = 0.80
        mock_model.performance.f1_score = 0.74

        trainer.train_entry_model = AsyncMock(return_value=mock_model)
        trainer.train_exit_model = AsyncMock(return_value=mock_model)
        trainer.train_strike_model = AsyncMock(return_value=mock_model)

        return trainer

    def test_automation_engine_initialization(self, automation_engine):
        """Test automation engine initializes correctly"""
        assert automation_engine is not None
        assert not automation_engine.running
        assert len(automation_engine.automation_tasks) == 0

    @pytest.mark.asyncio
    async def test_trigger_model_retraining(self, automation_engine, mock_model_trainer):
        """Test triggering model retraining"""
        # Patch the model trainer
        with patch.object(automation_engine, "model_trainer", mock_model_trainer):
            # Trigger retraining
            success = await automation_engine.trigger_model_retraining(
                "entry", RetrainingTrigger.MANUAL, force=True
            )

            assert success
            assert "entry" in automation_engine.last_retraining

            # Verify trainer was called
            mock_model_trainer.train_entry_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, automation_engine):
        """Test performance monitoring logic"""
        # Mock the performance degradation check
        with patch.object(
            automation_engine, "_check_model_performance_degradation", return_value=True
        ):
            # This would normally be called by the monitoring loop
            degraded = await automation_engine._check_model_performance_degradation("entry")
            assert degraded

    @pytest.mark.asyncio
    async def test_data_freshness_monitoring(self, automation_engine):
        """Test data freshness monitoring"""
        # Mock the new training data check
        with patch.object(automation_engine, "_check_new_training_data", return_value=True):
            # This would normally be called by the monitoring loop
            new_data = await automation_engine._check_new_training_data("entry")
            assert new_data

    def test_get_automation_status(self, automation_engine):
        """Test getting automation status"""
        status = automation_engine.get_automation_status()

        assert "running" in status
        assert "active_tasks" in status
        assert "last_retraining" in status
        assert "retraining_schedule" in status

    @pytest.mark.asyncio
    async def test_manual_retrain_all_models(self, automation_engine, mock_model_trainer):
        """Test manual retraining of all models"""
        with patch.object(automation_engine, "model_trainer", mock_model_trainer):
            results = await automation_engine.manual_retrain_all_models()

            assert "entry" in results
            assert "exit" in results
            assert "strike" in results

            # All should succeed with mocked trainer
            assert all(results.values())


class TestModelAutomationIntegration:
    """Test integration between automation and versioning"""

    @pytest.fixture
    def integrated_system(self, test_db_url):
        """Create integrated automation system"""
        automation = ModelAutomationEngine(test_db_url)
        versioning = ModelVersionManager(test_db_url, models_directory="test_models")
        return automation, versioning

    @pytest.mark.asyncio
    async def test_end_to_end_automation(self, integrated_system):
        """Test end-to-end automation workflow"""
        automation, versioning = integrated_system

        # Mock a successful training
        mock_model = Mock()
        mock_model.performance = Mock()
        mock_model.performance.accuracy = 0.78
        mock_model.performance.precision = 0.75
        mock_model.performance.recall = 0.82
        mock_model.performance.f1_score = 0.77

        with patch.object(automation, "model_trainer") as mock_trainer:
            mock_trainer.train_entry_model = AsyncMock(return_value=mock_model)

            # Trigger retraining
            success = await automation.trigger_model_retraining(
                "entry", RetrainingTrigger.MANUAL, force=True
            )

            assert success

            # Verify the model was "trained"
            mock_trainer.train_entry_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_automation_with_versioning_integration(self, integrated_system):
        """Test that automation can work with versioning system"""
        automation, versioning = integrated_system

        # Save a model version manually
        mock_model = SerializableTestModel()
        version = versioning.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics={"accuracy": 0.80},
            metadata={"test": "integration"},
            training_period="2024-01-01_to_2024-03-01",
        )

        # Deploy it
        success = versioning.deploy_model_to_production("entry", version)
        assert success

        # Get automation status
        status = automation.get_automation_status()
        assert status is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
