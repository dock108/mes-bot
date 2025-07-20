"""
Automated Model Training and Management System

This module provides comprehensive automation for ML model lifecycle management,
including scheduled retraining, performance monitoring, and deployment automation.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.config import config
from app.ml_training import ModelTrainer
from app.models import (
    DecisionHistory,
    MarketFeatures,
    MLPrediction,
    PerformanceMetrics,
    Trade,
    get_session_maker,
)
from app.notification_service import (
    NotificationLevel,
    send_performance_alert,
    send_system_alert,
)

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Types of events that can trigger model retraining"""

    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_FRESHNESS = "data_freshness"
    MANUAL = "manual"
    MARKET_REGIME_CHANGE = "market_regime_change"


class ModelStatus(Enum):
    """Model lifecycle status"""

    TRAINING = "training"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    TESTING = "testing"


class ModelAutomationEngine:
    """
    Orchestrates automated model training, monitoring, and deployment
    """

    def __init__(self, database_url: str):
        self.session_maker = get_session_maker(database_url)
        self.model_trainer = ModelTrainer(database_url)
        self.running = False
        self.automation_tasks = []

        # Configuration
        self.retraining_schedule = {
            "entry": timedelta(days=7),  # Retrain entry models weekly
            "exit": timedelta(days=14),  # Retrain exit models bi-weekly
            "strike": timedelta(days=10),  # Retrain strike models every 10 days
        }

        self.performance_thresholds = {
            "entry": {"min_accuracy": 0.4, "min_f1": 0.35, "max_degradation": 0.1},
            "exit": {"min_accuracy": 0.45, "min_f1": 0.4, "max_degradation": 0.1},
            "strike": {"max_mse": 10000, "min_r2": -0.5, "max_degradation": 0.2},
        }

        self.last_retraining = {}
        self.model_performance_history = {}

        logger.info("Model automation engine initialized")

    async def start_automation(self):
        """Start the automated model management system"""
        if self.running:
            logger.warning("Model automation already running")
            return

        self.running = True
        logger.info("Starting model automation engine...")

        # Schedule periodic tasks
        self.automation_tasks = [
            asyncio.create_task(self._scheduled_retraining_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._data_freshness_monitoring_loop()),
        ]

        await send_system_alert(
            "Model Automation Started",
            "Automated model training and monitoring system is now active",
            NotificationLevel.INFO,
            context={"automation_tasks": len(self.automation_tasks)},
        )

    async def stop_automation(self):
        """Stop the automated model management system"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping model automation engine...")

        # Cancel all running tasks
        for task in self.automation_tasks:
            task.cancel()

        await asyncio.gather(*self.automation_tasks, return_exceptions=True)
        self.automation_tasks.clear()

        await send_system_alert(
            "Model Automation Stopped",
            "Automated model training and monitoring system has been stopped",
            NotificationLevel.INFO,
        )

    async def _scheduled_retraining_loop(self):
        """Main loop for scheduled model retraining"""
        while self.running:
            try:
                logger.debug("Checking scheduled retraining requirements...")

                for model_type, schedule_interval in self.retraining_schedule.items():
                    last_training = self.last_retraining.get(model_type)

                    if (
                        not last_training
                        or (datetime.utcnow() - last_training) >= schedule_interval
                    ):
                        logger.info(f"Scheduled retraining triggered for {model_type} model")
                        await self.trigger_model_retraining(model_type, RetrainingTrigger.SCHEDULED)

                # Check every hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduled retraining loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _performance_monitoring_loop(self):
        """Monitor model performance and trigger retraining if degraded"""
        while self.running:
            try:
                logger.debug("Monitoring model performance...")

                for model_type in ["entry", "exit", "strike"]:
                    performance_degraded = await self._check_model_performance_degradation(
                        model_type
                    )

                    if performance_degraded:
                        logger.warning(f"Performance degradation detected for {model_type} model")
                        await self.trigger_model_retraining(
                            model_type, RetrainingTrigger.PERFORMANCE_DEGRADATION
                        )

                # Check every 4 hours
                await asyncio.sleep(14400)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes before retrying

    async def _data_freshness_monitoring_loop(self):
        """Monitor data freshness and trigger retraining when sufficient new data available"""
        while self.running:
            try:
                logger.debug("Checking data freshness...")

                for model_type in ["entry", "exit", "strike"]:
                    new_data_available = await self._check_new_training_data(model_type)

                    if new_data_available:
                        logger.info(f"Sufficient new data available for {model_type} model")
                        await self.trigger_model_retraining(
                            model_type, RetrainingTrigger.DATA_FRESHNESS
                        )

                # Check every 6 hours
                await asyncio.sleep(21600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data freshness monitoring loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying

    async def trigger_model_retraining(
        self, model_type: str, trigger: RetrainingTrigger, force: bool = False
    ) -> bool:
        """
        Trigger retraining for a specific model type

        Args:
            model_type: Type of model to retrain ('entry', 'exit', 'strike')
            trigger: What triggered the retraining
            force: Force retraining even if recently trained

        Returns:
            True if retraining was successful, False otherwise
        """
        try:
            # Check if recently retrained (unless forced)
            if not force and model_type in self.last_retraining:
                time_since_last = datetime.utcnow() - self.last_retraining[model_type]
                if time_since_last < timedelta(hours=12):
                    logger.info(
                        f"Skipping {model_type} retraining - too recent ({time_since_last})"
                    )
                    return False

            logger.info(f"Starting {model_type} model retraining (trigger: {trigger.value})")

            # Use last 60 days of data for training
            end_date = date.today()
            start_date = end_date - timedelta(days=60)

            # Send notification about retraining start
            await send_system_alert(
                f"Model Retraining Started",
                f"Starting automated retraining of {model_type} model",
                NotificationLevel.INFO,
                context={
                    "model_type": model_type,
                    "trigger": trigger.value,
                    "training_period": f"{start_date} to {end_date}",
                },
            )

            # Perform retraining based on model type
            success = False
            performance_metrics = {}

            if model_type == "entry":
                model = await self.model_trainer.train_entry_model(
                    start_date, end_date, algorithm="random_forest"
                )
                if model:
                    success = True
                    performance_metrics = {
                        "accuracy": model.performance.accuracy,
                        "precision": model.performance.precision,
                        "recall": model.performance.recall,
                        "f1_score": model.performance.f1_score,
                    }

            elif model_type == "exit":
                model = await self.model_trainer.train_exit_model(
                    start_date, end_date, algorithm="random_forest"
                )
                if model:
                    success = True
                    performance_metrics = {
                        "accuracy": model.performance.accuracy,
                        "cross_val_mean": model.performance.cross_val_mean,
                    }

            elif model_type == "strike":
                model = await self.model_trainer.train_strike_model(
                    start_date, end_date, algorithm="random_forest"
                )
                if model:
                    success = True
                    performance_metrics = {
                        "mse": model.performance.mse,
                        "r2_score": model.performance.r2_score,
                    }

            if success:
                self.last_retraining[model_type] = datetime.utcnow()
                self._update_performance_history(model_type, performance_metrics)

                # Send success notification
                await send_performance_alert(
                    f"{model_type.title()} Model Retrained Successfully",
                    f"Automated retraining of {model_type} model completed successfully",
                    performance_metrics,
                    NotificationLevel.INFO,
                )

                logger.info(f"{model_type} model retraining completed successfully")
                return True
            else:
                # Send failure notification
                await send_system_alert(
                    f"Model Retraining Failed",
                    f"Automated retraining of {model_type} model failed",
                    NotificationLevel.ERROR,
                    context={
                        "model_type": model_type,
                        "trigger": trigger.value,
                        "error": "Training returned no model",
                    },
                )

                logger.error(f"{model_type} model retraining failed")
                return False

        except Exception as e:
            logger.error(f"Error during {model_type} model retraining: {e}")

            # Send error notification
            await send_system_alert(
                f"Model Retraining Error",
                f"Error during automated retraining of {model_type} model: {str(e)}",
                NotificationLevel.ERROR,
                context={
                    "model_type": model_type,
                    "trigger": trigger.value,
                    "error": str(e),
                },
            )

            return False

    async def _check_model_performance_degradation(self, model_type: str) -> bool:
        """Check if model performance has degraded below thresholds"""
        try:
            session = self.session_maker()

            # Get recent predictions and their actual outcomes
            recent_cutoff = datetime.utcnow() - timedelta(days=7)

            predictions = (
                session.query(MLPrediction)
                .filter(
                    MLPrediction.model_name.like(f"{model_type}_%"),
                    MLPrediction.timestamp >= recent_cutoff,
                    MLPrediction.actual_outcome.isnot(None),
                )
                .all()
            )

            if len(predictions) < 10:  # Need minimum predictions to assess
                return False

            # Calculate current performance
            if model_type == "entry":
                # Calculate accuracy for entry models
                correct_predictions = sum(
                    1
                    for p in predictions
                    if (p.prediction > 0.5 and p.actual_outcome > 0)
                    or (p.prediction <= 0.5 and p.actual_outcome <= 0)
                )
                current_accuracy = correct_predictions / len(predictions)

                threshold = self.performance_thresholds[model_type]["min_accuracy"]
                return current_accuracy < threshold

            elif model_type == "exit":
                # Similar logic for exit models
                correct_predictions = sum(
                    1
                    for p in predictions
                    if (p.prediction > 0.5 and p.actual_outcome > 0)
                    or (p.prediction <= 0.5 and p.actual_outcome <= 0)
                )
                current_accuracy = correct_predictions / len(predictions)

                threshold = self.performance_thresholds[model_type]["min_accuracy"]
                return current_accuracy < threshold

            elif model_type == "strike":
                # Calculate MSE for strike models
                mse = sum((p.prediction - p.actual_outcome) ** 2 for p in predictions) / len(
                    predictions
                )

                threshold = self.performance_thresholds[model_type]["max_mse"]
                return mse > threshold

            return False

        except Exception as e:
            logger.error(f"Error checking {model_type} model performance: {e}")
            return False
        finally:
            session.close()

    async def _check_new_training_data(self, model_type: str) -> bool:
        """Check if sufficient new training data is available since last training"""
        try:
            session = self.session_maker()

            last_training_time = self.last_retraining.get(
                model_type, datetime.utcnow() - timedelta(days=30)
            )

            # Count new completed trades since last training
            new_trades = (
                session.query(Trade)
                .filter(
                    Trade.exit_time >= last_training_time,
                    Trade.status == "CLOSED",
                    Trade.realized_pnl.isnot(None),
                )
                .count()
            )

            # Count new decision records
            new_decisions = (
                session.query(DecisionHistory)
                .filter(
                    DecisionHistory.timestamp >= last_training_time,
                    DecisionHistory.actual_outcome.isnot(None),
                )
                .count()
            )

            # Trigger retraining if we have significant new data
            min_new_trades = 50  # Minimum new trades to trigger retraining
            min_new_decisions = 100  # Minimum new decisions to trigger retraining

            return new_trades >= min_new_trades or new_decisions >= min_new_decisions

        except Exception as e:
            logger.error(f"Error checking new training data for {model_type}: {e}")
            return False
        finally:
            session.close()

    def _update_performance_history(self, model_type: str, metrics: Dict):
        """Update performance history for tracking trends"""
        if model_type not in self.model_performance_history:
            self.model_performance_history[model_type] = []

        entry = {
            "timestamp": datetime.utcnow(),
            "metrics": metrics,
        }

        self.model_performance_history[model_type].append(entry)

        # Keep only last 50 entries
        if len(self.model_performance_history[model_type]) > 50:
            self.model_performance_history[model_type] = self.model_performance_history[model_type][
                -50:
            ]

    def get_automation_status(self) -> Dict:
        """Get current status of the automation system"""
        return {
            "running": self.running,
            "active_tasks": len(self.automation_tasks),
            "last_retraining": {
                model_type: timestamp.isoformat() if timestamp else None
                for model_type, timestamp in self.last_retraining.items()
            },
            "retraining_schedule": {
                model_type: str(interval)
                for model_type, interval in self.retraining_schedule.items()
            },
            "performance_history_entries": {
                model_type: len(history)
                for model_type, history in self.model_performance_history.items()
            },
        }

    async def manual_retrain_all_models(self) -> Dict[str, bool]:
        """Manually trigger retraining of all models"""
        logger.info("Manual retraining of all models triggered")

        results = {}
        for model_type in ["entry", "exit", "strike"]:
            results[model_type] = await self.trigger_model_retraining(
                model_type, RetrainingTrigger.MANUAL, force=True
            )

        await send_system_alert(
            "Manual Model Retraining Completed",
            f"Manual retraining completed. Results: {results}",
            NotificationLevel.INFO,
            context={"results": results},
        )

        return results
