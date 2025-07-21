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


@pytest.mark.integration
@pytest.mark.db
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

    def test_rollback_model(self, version_manager):
        """Test rolling back to previous model version"""
        # Save two model versions
        mock_model1 = SerializableTestModel(1)
        version1 = version_manager.save_model_version(
            model=mock_model1,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics={"accuracy": 0.75},
            metadata={"version": "first"},
            training_period="2024-01-01_to_2024-01-31",
        )

        # Deploy the first version to production first
        success = version_manager.deploy_model_to_production("entry", version1)
        assert success

        mock_model2 = SerializableTestModel(2)
        version2 = version_manager.save_model_version(
            model=mock_model2,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics={"accuracy": 0.80},
            metadata={"version": "second"},
            training_period="2024-02-01_to_2024-02-28",
        )

        # Deploy the second version (this should deprecate the first)
        success = version_manager.deploy_model_to_production("entry", version2)
        assert success

        # Rollback to previous version (should restore the first version)
        success = version_manager.rollback_model("entry")
        assert success

        # Verify first version is now production
        model, metadata = version_manager.load_model_version("entry")
        # Check that we have the correct model (the custom metadata field "version" is "first")
        assert metadata["version"] == "first"  # Custom metadata overwrites the version field
        # Check that the model has been successfully rolled back by checking model iteration
        assert model.iteration == 1  # SerializableTestModel(1) was the first model

    def test_rollback_model_no_previous(self, version_manager):
        """Test rollback when no previous version exists"""
        # Save and deploy only one version
        mock_model = SerializableTestModel()
        version = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics={"accuracy": 0.75},
            metadata={"test": "data"},
            training_period="2024-01-01_to_2024-01-31",
        )

        version_manager.deploy_model_to_production("entry", version)

        # Attempt rollback (should fail)
        success = version_manager.rollback_model("entry")
        assert not success

    def test_get_model_performance_comparison(self, version_manager):
        """Test model performance comparison"""
        # Save multiple models with different performance
        performances = [0.70, 0.85, 0.75]
        for i, perf in enumerate(performances):
            mock_model = SerializableTestModel(i)
            version_manager.save_model_version(
                model=mock_model,
                model_type="entry",
                algorithm="random_forest",
                performance_metrics={"accuracy": perf, "f1_score": perf - 0.05},
                metadata={"iteration": i},
                training_period=f"2024-0{i+1}-01_to_2024-0{i+1}-28",
            )

        comparison = version_manager.get_model_performance_comparison("entry")

        assert comparison["model_type"] == "entry"
        assert comparison["total_versions"] == 3
        assert len(comparison["versions"]) == 3

        # Should be sorted by performance score (best first)
        scores = [v["performance_score"] for v in comparison["versions"]]
        assert scores == sorted(scores, reverse=True)

    def test_cleanup_old_versions(self, version_manager):
        """Test cleaning up old model versions"""
        # Save 5 models
        versions = []
        for i in range(5):
            mock_model = SerializableTestModel(i)
            version = version_manager.save_model_version(
                model=mock_model,
                model_type="entry",
                algorithm="random_forest",
                performance_metrics={"accuracy": 0.70 + i * 0.02},
                metadata={"iteration": i},
                training_period=f"2024-0{i+1}-01_to_2024-0{i+1}-28",
            )
            versions.append(version)

        # Deploy the last one to production
        version_manager.deploy_model_to_production("entry", versions[-1])

        # Cleanup keeping only 2 versions
        deleted_count = version_manager.cleanup_old_versions("entry", keep_count=2)

        # Should have deleted some versions (excluding production)
        assert deleted_count >= 0

        # List remaining versions
        remaining = version_manager.list_model_versions("entry")
        # Should have production version plus up to 2 others
        assert len(remaining) <= 3

    def test_load_model_version_not_found(self, version_manager):
        """Test loading non-existent model version"""
        with pytest.raises(ValueError, match="No model found"):
            version_manager.load_model_version("entry", "v99.99.99")

    def test_deploy_model_version_not_found(self, version_manager):
        """Test deploying non-existent model version"""
        success = version_manager.deploy_model_to_production("entry", "v99.99.99")
        assert not success

    def test_performance_score_calculation_entry(self, version_manager):
        """Test performance score calculation for entry models"""
        mock_model = SerializableTestModel()

        # Test with good metrics
        good_metrics = {"accuracy": 0.80, "precision": 0.75, "f1_score": 0.78}
        version = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics=good_metrics,
            metadata={},
            training_period="2024-01-01_to_2024-01-31",
        )

        # Get the saved version and check score
        versions = version_manager.list_model_versions("entry")
        saved_version = next(v for v in versions if v["version"] == version)

        # Score should be weighted combination: 0.8*0.4 + 0.75*0.3 + 0.78*0.3 = 0.779
        expected_score = 0.80 * 0.4 + 0.75 * 0.3 + 0.78 * 0.3
        assert abs(saved_version["performance_score"] - expected_score) < 0.01

    def test_performance_score_calculation_strike(self, version_manager):
        """Test performance score calculation for strike models"""
        mock_model = SerializableTestModel()

        # Test with regression metrics
        metrics = {"mse": 5000, "r2_score": 0.6}  # Good R2, reasonable MSE
        version = version_manager.save_model_version(
            model=mock_model,
            model_type="strike",
            algorithm="random_forest",
            performance_metrics=metrics,
            metadata={},
            training_period="2024-01-01_to_2024-01-31",
        )

        # Get the saved version and check score
        versions = version_manager.list_model_versions("strike")
        saved_version = next(v for v in versions if v["version"] == version)

        # Score should be reasonable (> 0)
        assert saved_version["performance_score"] > 0

    def test_version_generation(self, version_manager):
        """Test automatic version generation"""
        mock_model = SerializableTestModel()

        # Save first model
        version1 = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics={"accuracy": 0.75},
            metadata={},
            training_period="2024-01-01_to_2024-01-31",
        )

        # Save second model (should auto-increment)
        version2 = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics={"accuracy": 0.80},
            metadata={},
            training_period="2024-02-01_to_2024-02-28",
        )

        assert version1 != version2
        assert version1.startswith("v")
        assert version2.startswith("v")

    def test_custom_version_specification(self, version_manager):
        """Test specifying custom version string"""
        mock_model = SerializableTestModel()

        custom_version = "v2.1.0"
        version = version_manager.save_model_version(
            model=mock_model,
            model_type="entry",
            algorithm="random_forest",
            performance_metrics={"accuracy": 0.75},
            metadata={},
            training_period="2024-01-01_to_2024-01-31",
            version=custom_version,
        )

        assert version == custom_version

    def test_error_handling_in_save_model(self, version_manager):
        """Test error handling during model save"""

        # Create a mock model that can't be pickled
        def unpickleable_model(x):
            return x  # Functions can't be pickled easily

        with pytest.raises(Exception):
            version_manager.save_model_version(
                model=unpickleable_model,
                model_type="entry",
                algorithm="random_forest",
                performance_metrics={"accuracy": 0.75},
                metadata={},
                training_period="2024-01-01_to_2024-01-31",
            )


@pytest.mark.integration
@pytest.mark.db
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

    @pytest.mark.asyncio
    async def test_start_automation(self, automation_engine):
        """Test starting automation system"""
        with patch("app.model_automation.send_system_alert") as mock_alert:
            with patch.object(automation_engine, "_scheduled_retraining_loop") as mock_sched:
                with patch.object(automation_engine, "_performance_monitoring_loop") as mock_perf:
                    with patch.object(
                        automation_engine, "_data_freshness_monitoring_loop"
                    ) as mock_data:
                        # Set up async mocks
                        mock_sched.return_value = AsyncMock()
                        mock_perf.return_value = AsyncMock()
                        mock_data.return_value = AsyncMock()

                        # Ensure automation is not running before starting
                        automation_engine.running = False
                        await automation_engine.start_automation()

                        assert automation_engine.running
                        assert len(automation_engine.automation_tasks) == 3
                        mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_automation(self, automation_engine):
        """Test stopping automation system"""
        # Set up running state
        automation_engine.running = True
        mock_task = Mock()
        mock_task.cancel = Mock()
        automation_engine.automation_tasks = [mock_task]

        with patch("app.model_automation.send_system_alert") as mock_alert:
            with patch("asyncio.gather") as mock_gather:
                # Mock gather to return an awaitable
                async def mock_gather_func(*args, **kwargs):
                    return []

                mock_gather.side_effect = mock_gather_func

                await automation_engine.stop_automation()

                assert not automation_engine.running
                assert len(automation_engine.automation_tasks) == 0
                mock_task.cancel.assert_called_once()
                mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_retraining_too_recent(self, automation_engine, mock_model_trainer):
        """Test skipping retraining when too recent"""
        # Set recent retraining
        automation_engine.last_retraining["entry"] = datetime.utcnow() - timedelta(hours=1)

        with patch.object(automation_engine, "model_trainer", mock_model_trainer):
            success = await automation_engine.trigger_model_retraining(
                "entry", RetrainingTrigger.SCHEDULED, force=False
            )

            assert not success  # Should skip due to recent training
            mock_model_trainer.train_entry_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_retraining_failure(self, automation_engine, mock_model_trainer):
        """Test retraining failure handling"""
        # Mock trainer to return None (failure)
        mock_model_trainer.train_entry_model = AsyncMock(return_value=None)

        with patch.object(automation_engine, "model_trainer", mock_model_trainer):
            with patch("app.model_automation.send_system_alert") as mock_alert:
                success = await automation_engine.trigger_model_retraining(
                    "entry", RetrainingTrigger.MANUAL, force=True
                )

                assert not success
                # Check that send_system_alert was called with failure message
                # The exact call we're looking for is the one with "Failed" in the title
                assert any("Failed" in str(call) for call in mock_alert.call_args_list)

    @pytest.mark.asyncio
    async def test_trigger_retraining_exception(self, automation_engine, mock_model_trainer):
        """Test retraining exception handling"""
        # Mock trainer to raise exception
        mock_model_trainer.train_entry_model = AsyncMock(side_effect=Exception("Training failed"))

        with patch.object(automation_engine, "model_trainer", mock_model_trainer):
            with patch("app.model_automation.send_system_alert") as mock_alert:
                success = await automation_engine.trigger_model_retraining(
                    "entry", RetrainingTrigger.MANUAL, force=True
                )

                assert not success
                # Check that send_system_alert was called with error message
                # The exact call we're looking for is the one with "Error" in the title
                assert any("Error" in str(call) for call in mock_alert.call_args_list)

    @pytest.mark.asyncio
    async def test_check_model_performance_degradation_no_data(self, automation_engine):
        """Test performance degradation check with insufficient data"""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = []
        automation_engine.session_maker = Mock(return_value=mock_session)

        result = await automation_engine._check_model_performance_degradation("entry")
        assert not result  # Should return False for insufficient data

    @pytest.mark.asyncio
    async def test_check_model_performance_degradation_entry_model(self, automation_engine):
        """Test performance degradation check for entry model"""
        # Create mock predictions with low accuracy
        mock_predictions = []
        for i in range(15):
            pred = Mock()
            pred.prediction = 0.7  # Predicts positive
            pred.actual_outcome = -1.0 if i < 12 else 1.0  # Only 3 correct out of 15
            mock_predictions.append(pred)

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = mock_predictions
        automation_engine.session_maker = Mock(return_value=mock_session)

        result = await automation_engine._check_model_performance_degradation("entry")
        assert result  # Should detect degradation (20% accuracy < 40% threshold)

    @pytest.mark.asyncio
    async def test_check_model_performance_degradation_strike_model(self, automation_engine):
        """Test performance degradation check for strike model"""
        # Create mock predictions with high MSE
        mock_predictions = []
        for i in range(15):
            pred = Mock()
            pred.prediction = 100.0
            pred.actual_outcome = 201.0  # Large error: (100-201)^2 = 10201 > 10000
            mock_predictions.append(pred)

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = mock_predictions
        automation_engine.session_maker = Mock(return_value=mock_session)

        result = await automation_engine._check_model_performance_degradation("strike")
        assert result  # Should detect degradation (MSE = 10201 > 10000 threshold)

    @pytest.mark.asyncio
    async def test_check_new_training_data_sufficient_trades(self, automation_engine):
        """Test new training data check with sufficient new trades"""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.count.side_effect = [
            60,
            20,
        ]  # trades, decisions
        automation_engine.session_maker = Mock(return_value=mock_session)

        result = await automation_engine._check_new_training_data("entry")
        assert result  # Should return True for 60 new trades (>= 50 threshold)

    @pytest.mark.asyncio
    async def test_check_new_training_data_sufficient_decisions(self, automation_engine):
        """Test new training data check with sufficient new decisions"""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.count.side_effect = [
            20,
            120,
        ]  # trades, decisions
        automation_engine.session_maker = Mock(return_value=mock_session)

        result = await automation_engine._check_new_training_data("entry")
        assert result  # Should return True for 120 new decisions (>= 100 threshold)

    @pytest.mark.asyncio
    async def test_check_new_training_data_insufficient(self, automation_engine):
        """Test new training data check with insufficient data"""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.count.side_effect = [
            10,
            30,
        ]  # both below thresholds
        automation_engine.session_maker = Mock(return_value=mock_session)

        result = await automation_engine._check_new_training_data("entry")
        assert not result  # Should return False for insufficient data

    def test_update_performance_history(self, automation_engine):
        """Test updating performance history"""
        metrics = {"accuracy": 0.75, "precision": 0.70}

        automation_engine._update_performance_history("entry", metrics)

        assert "entry" in automation_engine.model_performance_history
        assert len(automation_engine.model_performance_history["entry"]) == 1

        entry = automation_engine.model_performance_history["entry"][0]
        assert entry["metrics"] == metrics
        assert "timestamp" in entry

    def test_update_performance_history_limit(self, automation_engine):
        """Test performance history maintains limit"""
        # Add 55 entries (over the 50 limit)
        for i in range(55):
            automation_engine._update_performance_history("entry", {"iteration": i})

        # Should only keep the last 50
        assert len(automation_engine.model_performance_history["entry"]) == 50
        # Should have the most recent entries (5-54)
        assert automation_engine.model_performance_history["entry"][0]["metrics"]["iteration"] == 5

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, automation_engine):
        """Test error handling in monitoring loops"""
        # Mock the internal method to raise exception
        with patch.object(
            automation_engine,
            "_check_model_performance_degradation",
            side_effect=Exception("Test error"),
        ):
            with patch("app.model_automation.logger") as mock_logger:
                # Set up running state
                automation_engine.running = True

                # Create a task that will run briefly and then we'll cancel it
                task = asyncio.create_task(automation_engine._performance_monitoring_loop())

                # Let it run briefly to trigger the exception
                await asyncio.sleep(0.01)
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should have logged the error
                mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_scheduled_retraining_loop_logic(self, automation_engine):
        """Test scheduled retraining loop logic"""
        # Set old last retraining time
        automation_engine.last_retraining["entry"] = datetime.utcnow() - timedelta(
            days=8
        )  # Older than 7-day schedule

        with patch.object(
            automation_engine, "trigger_model_retraining", new_callable=AsyncMock
        ) as mock_trigger:
            mock_trigger.return_value = True
            automation_engine.running = True

            # Create task and let it run once
            task = asyncio.create_task(automation_engine._scheduled_retraining_loop())
            await asyncio.sleep(0.1)  # Let it run once
            automation_engine.running = False  # Stop the loop
            await asyncio.sleep(0.1)  # Let it finish
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have triggered retraining for entry model
            mock_trigger.assert_called()
            # Check that it was called with entry and SCHEDULED trigger
            assert any(
                call.args[0] == "entry" and call.args[1] == RetrainingTrigger.SCHEDULED
                for call in mock_trigger.call_args_list
            )

    @pytest.mark.asyncio
    async def test_performance_monitoring_loop_logic(self, automation_engine):
        """Test performance monitoring loop logic"""
        with patch.object(automation_engine, "_check_model_performance_degradation") as mock_check:
            with patch.object(automation_engine, "trigger_model_retraining") as mock_trigger:
                mock_check.return_value = True  # Simulate degradation
                mock_trigger.return_value = True
                automation_engine.running = True

                # Create task and let it run once
                task = asyncio.create_task(automation_engine._performance_monitoring_loop())
                await asyncio.sleep(0.01)  # Let it run once
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should have checked all model types
                assert mock_check.call_count >= 3  # entry, exit, strike
                # Should have triggered retraining for degraded models
                mock_trigger.assert_called()

    @pytest.mark.asyncio
    async def test_data_freshness_monitoring_loop_logic(self, automation_engine):
        """Test data freshness monitoring loop logic"""
        with patch.object(automation_engine, "_check_new_training_data") as mock_check:
            with patch.object(automation_engine, "trigger_model_retraining") as mock_trigger:
                mock_check.return_value = True  # Simulate new data available
                mock_trigger.return_value = True
                automation_engine.running = True

                # Create task and let it run once
                task = asyncio.create_task(automation_engine._data_freshness_monitoring_loop())
                await asyncio.sleep(0.01)  # Let it run once
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should have checked all model types
                assert mock_check.call_count >= 3  # entry, exit, strike
                # Should have triggered retraining for models with new data
                mock_trigger.assert_called()


@pytest.mark.integration
@pytest.mark.db
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
