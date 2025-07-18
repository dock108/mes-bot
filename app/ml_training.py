"""
Machine Learning Model Training Framework
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR

from app.config import config
from app.feature_pipeline import DataQualityMonitor, FeatureEngineer
from app.models import MLModelMetadata, get_session_maker

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML model training"""

    model_type: str  # 'entry' or 'exit'
    algorithm: str  # 'gradient_boosting', 'random_forest', 'neural_network'
    hyperparameters: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Container for model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    cross_val_mean: Optional[float] = None
    cross_val_std: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    model_type: str  # 'entry', 'exit', 'strike_selection'
    algorithm: str  # 'random_forest', 'gradient_boosting', 'neural_network', etc.
    hyperparameters: Dict[str, Any]
    train_start_date: date
    train_end_date: date
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    feature_selection: bool = True
    scale_features: bool = True


class BaseMLModel(ABC):
    """Abstract base class for ML models"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.performance = None
        self.feature_names = []
        self.is_trained = False

    @abstractmethod
    def create_model(self) -> Any:
        """Create the ML model instance"""
        pass

    @abstractmethod
    def prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variables from DataFrame"""
        pass

    def train(self, X: pd.DataFrame, y: np.ndarray) -> ModelPerformance:
        """Train the model"""
        try:
            # Store feature names
            self.feature_names = list(X.columns)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=y if self._is_classification() else None,
            )

            # Create preprocessing pipeline
            preprocessors = []

            if self.config.scale_features:
                scaler = RobustScaler()
                preprocessors.append(("scaler", scaler))

            # Create full pipeline
            if preprocessors:
                self.model = Pipeline(preprocessors + [("model", self.create_model())])
            else:
                self.model = self.create_model()

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate performance
            y_pred = self.model.predict(X_val)

            if self._is_classification():
                performance = ModelPerformance(
                    accuracy=accuracy_score(y_val, y_pred),
                    precision=precision_score(y_val, y_pred, average="weighted", zero_division=0),
                    recall=recall_score(y_val, y_pred, average="weighted", zero_division=0),
                    f1_score=f1_score(y_val, y_pred, average="weighted", zero_division=0),
                )
            else:
                performance = ModelPerformance(
                    accuracy=0.0,  # N/A for regression
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    mse=mean_squared_error(y_val, y_pred),
                    r2_score=r2_score(y_val, y_pred),
                )

            # Cross-validation
            cv_scores = cross_val_score(
                self.model,
                X_train,
                y_train,
                cv=self.config.cross_validation_folds,
                scoring="accuracy" if self._is_classification() else "r2",
            )
            performance.cross_val_mean = np.mean(cv_scores)
            performance.cross_val_std = np.std(cv_scores)

            # Feature importance
            performance.feature_importance = self._get_feature_importance()

            self.performance = performance
            self.is_trained = True

            logger.info(f"Model training completed. Performance: {performance}")
            return performance

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure feature order matches training
        X_ordered = X[self.feature_names]
        return self.model.predict(X_ordered)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make probability predictions (classification only)"""
        if not self.is_trained or not self._is_classification():
            return None

        X_ordered = X[self.feature_names]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_ordered)
        return None

    def save_model(self, filepath: str) -> str:
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": self.config,
            "performance": self.performance,
        }

        joblib.dump(model_data, filepath)

        # Calculate file hash
        with open(filepath, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        return file_hash

    @classmethod
    def load_model(cls, filepath: str) -> "BaseMLModel":
        """Load model from file"""
        model_data = joblib.load(filepath)

        # Create instance
        instance = cls(model_data["config"])
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.performance = model_data["performance"]
        instance.is_trained = True

        return instance

    def _is_classification(self) -> bool:
        """Check if this is a classification model"""
        return self.config.model_type in ["entry", "exit"]

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        try:
            model = self.model
            if hasattr(model, "named_steps"):
                model = model.named_steps["model"]

            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                return dict(zip(self.feature_names, importance))
            elif hasattr(model, "coef_"):
                # For linear models, use absolute coefficients
                importance = np.abs(model.coef_).flatten()
                return dict(zip(self.feature_names, importance))

            return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None


class EntryPredictionModel(BaseMLModel):
    """Model for predicting entry signals"""

    def create_model(self) -> Any:
        """Create entry prediction model"""
        algorithm = self.config.algorithm
        params = self.config.hyperparameters

        if algorithm == "random_forest":
            return RandomForestClassifier(**params)
        elif algorithm == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        elif algorithm == "logistic_regression":
            return LogisticRegression(**params)
        elif algorithm == "neural_network":
            return MLPClassifier(**params)
        elif algorithm == "svm":
            return SVC(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare binary classification targets"""
        if "target_profitable" in df.columns:
            return df["target_profitable"].values
        else:
            # Create binary target from actual outcome
            return (df["actual_outcome"] > 0).astype(int).values


class ExitPredictionModel(BaseMLModel):
    """Model for predicting exit signals"""

    def create_model(self) -> Any:
        """Create exit prediction model"""
        algorithm = self.config.algorithm
        params = self.config.hyperparameters

        if algorithm == "random_forest":
            return RandomForestClassifier(**params)
        elif algorithm == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        elif algorithm == "logistic_regression":
            return LogisticRegression(**params)
        elif algorithm == "neural_network":
            return MLPClassifier(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare exit timing targets"""
        if "target_should_exit" in df.columns:
            return df["target_should_exit"].values
        else:
            # Create target based on some logic
            return np.zeros(len(df))  # Placeholder


class StrikeOptimizationModel(BaseMLModel):
    """Model for optimizing strike selection"""

    def create_model(self) -> Any:
        """Create strike optimization model"""
        algorithm = self.config.algorithm
        params = self.config.hyperparameters

        if algorithm == "random_forest":
            return RandomForestRegressor(**params)
        elif algorithm == "gradient_boosting":
            return GradientBoostingClassifier(**params)  # Can be adapted for regression
        elif algorithm == "linear_regression":
            return LinearRegression(**params)
        elif algorithm == "neural_network":
            return MLPRegressor(**params)
        elif algorithm == "svm":
            return SVR(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare regression targets for profit optimization"""
        if "target_profit_magnitude" in df.columns:
            return df["target_profit_magnitude"].values
        else:
            return df["actual_outcome"].values


class ModelTrainer:
    """Main class for training and managing ML models"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.session_maker = get_session_maker(database_url)
        self.feature_engineer = FeatureEngineer(database_url)
        self.data_quality_monitor = DataQualityMonitor(database_url)

        # Model storage
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    async def train_entry_model(
        self, start_date: date, end_date: date, algorithm: str = "random_forest"
    ) -> Optional[EntryPredictionModel]:
        """Train entry prediction model"""
        try:
            logger.info(f"Training entry model from {start_date} to {end_date}")

            # Check data quality
            quality_metrics = self.data_quality_monitor.check_data_quality(start_date, end_date)
            if quality_metrics["completeness"] < 0.8:
                logger.warning(f"Data quality insufficient: {quality_metrics}")
                return None

            # Get training data
            df, feature_cols = self.feature_engineer.prepare_ml_dataset(
                start_date, end_date, target_type="entry"
            )

            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient training data: {len(df)} samples")
                return None

            # Prepare features and targets
            X = df[feature_cols]

            # Create model config
            config = TrainingConfig(
                model_type="entry",
                algorithm=algorithm,
                hyperparameters=self._get_default_hyperparameters(algorithm, "classification"),
                train_start_date=start_date,
                train_end_date=end_date,
            )

            # Create and train model
            model = EntryPredictionModel(config)
            y = model.prepare_targets(df)

            performance = model.train(X, y)

            # Save model
            model_filename = f"entry_model_{algorithm}_{end_date.strftime('%Y%m%d')}.joblib"
            model_path = self.model_dir / model_filename
            file_hash = model.save_model(str(model_path))

            # Save metadata to database
            await self._save_model_metadata(
                model_name=f"entry_{algorithm}",
                model_type="entry",
                config=config,
                performance=performance,
                model_path=str(model_path),
                file_hash=file_hash,
                training_samples=len(df),
            )

            logger.info(f"Entry model training completed. Accuracy: {performance.accuracy:.3f}")
            return model

        except Exception as e:
            logger.error(f"Error training entry model: {e}")
            return None

    async def train_exit_model(
        self, start_date: date, end_date: date, algorithm: str = "random_forest"
    ) -> Optional[ExitPredictionModel]:
        """Train exit prediction model"""
        try:
            logger.info(f"Training exit model from {start_date} to {end_date}")

            # Get training data
            df, feature_cols = self.feature_engineer.prepare_ml_dataset(
                start_date, end_date, target_type="exit"
            )

            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient exit training data: {len(df)} samples")
                return None

            # Prepare features and targets
            X = df[feature_cols]

            config = TrainingConfig(
                model_type="exit",
                algorithm=algorithm,
                hyperparameters=self._get_default_hyperparameters(algorithm, "classification"),
                train_start_date=start_date,
                train_end_date=end_date,
            )

            model = ExitPredictionModel(config)
            y = model.prepare_targets(df)

            performance = model.train(X, y)

            # Save model
            model_filename = f"exit_model_{algorithm}_{end_date.strftime('%Y%m%d')}.joblib"
            model_path = self.model_dir / model_filename
            file_hash = model.save_model(str(model_path))

            await self._save_model_metadata(
                model_name=f"exit_{algorithm}",
                model_type="exit",
                config=config,
                performance=performance,
                model_path=str(model_path),
                file_hash=file_hash,
                training_samples=len(df),
            )

            logger.info(f"Exit model training completed. Accuracy: {performance.accuracy:.3f}")
            return model

        except Exception as e:
            logger.error(f"Error training exit model: {e}")
            return None

    async def train_strike_model(
        self, start_date: date, end_date: date, algorithm: str = "random_forest"
    ) -> Optional[StrikeOptimizationModel]:
        """Train strike optimization model"""
        try:
            logger.info(f"Training strike optimization model from {start_date} to {end_date}")

            # Get training data
            df, feature_cols = self.feature_engineer.prepare_ml_dataset(
                start_date, end_date, target_type="entry"  # Use entry data with profit targets
            )

            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient strike training data: {len(df)} samples")
                return None

            X = df[feature_cols]

            config = TrainingConfig(
                model_type="strike_selection",
                algorithm=algorithm,
                hyperparameters=self._get_default_hyperparameters(algorithm, "regression"),
                train_start_date=start_date,
                train_end_date=end_date,
            )

            model = StrikeOptimizationModel(config)
            y = model.prepare_targets(df)

            performance = model.train(X, y)

            # Save model
            model_filename = f"strike_model_{algorithm}_{end_date.strftime('%Y%m%d')}.joblib"
            model_path = self.model_dir / model_filename
            file_hash = model.save_model(str(model_path))

            await self._save_model_metadata(
                model_name=f"strike_{algorithm}",
                model_type="strike_selection",
                config=config,
                performance=performance,
                model_path=str(model_path),
                file_hash=file_hash,
                training_samples=len(df),
            )

            logger.info(f"Strike model training completed. R2: {performance.r2_score:.3f}")
            return model

        except Exception as e:
            logger.error(f"Error training strike model: {e}")
            return None

    async def hyperparameter_optimization(
        self, model_type: str, algorithm: str, start_date: date, end_date: date
    ) -> Dict[str, Any]:
        """Perform hyperparameter optimization"""
        try:
            logger.info(f"Optimizing hyperparameters for {model_type} {algorithm}")

            # Get training data
            df, feature_cols = self.feature_engineer.prepare_ml_dataset(
                start_date, end_date, target_type=model_type
            )

            if df.empty:
                return {}

            X = df[feature_cols]

            # Prepare targets based on model type
            if model_type == "entry":
                y = (df["actual_outcome"] > 0).astype(int).values
            elif model_type == "exit":
                y = np.zeros(len(df))  # Placeholder
            else:
                y = df["actual_outcome"].values

            # Define parameter grids
            param_grids = {
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "gradient_boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0],
                },
                "neural_network": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "learning_rate_init": [0.001, 0.01, 0.1],
                    "alpha": [0.0001, 0.001, 0.01],
                },
            }

            if algorithm not in param_grids:
                logger.warning(f"No parameter grid defined for {algorithm}")
                return self._get_default_hyperparameters(
                    algorithm,
                    "classification" if model_type != "strike_selection" else "regression",
                )

            # Create model
            if model_type == "entry":
                base_model = EntryPredictionModel(
                    TrainingConfig(
                        model_type=model_type,
                        algorithm=algorithm,
                        hyperparameters={},
                        train_start_date=start_date,
                        train_end_date=end_date,
                    )
                ).create_model()
            elif model_type == "exit":
                base_model = ExitPredictionModel(
                    TrainingConfig(
                        model_type=model_type,
                        algorithm=algorithm,
                        hyperparameters={},
                        train_start_date=start_date,
                        train_end_date=end_date,
                    )
                ).create_model()
            else:
                base_model = StrikeOptimizationModel(
                    TrainingConfig(
                        model_type=model_type,
                        algorithm=algorithm,
                        hyperparameters={},
                        train_start_date=start_date,
                        train_end_date=end_date,
                    )
                ).create_model()

            # Perform grid search
            scoring = "accuracy" if model_type != "strike_selection" else "r2"

            grid_search = GridSearchCV(
                base_model, param_grids[algorithm], cv=3, scoring=scoring, n_jobs=-1, verbose=1
            )

            grid_search.fit(X, y)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best score: {grid_search.best_score_:.3f}")

            return grid_search.best_params_

        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return self._get_default_hyperparameters(
                algorithm, "classification" if model_type != "strike_selection" else "regression"
            )

    def _get_default_hyperparameters(self, algorithm: str, task_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for algorithm and task type"""
        defaults = {
            "random_forest": {
                "classification": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                },
                "regression": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                },
            },
            "gradient_boosting": {
                "classification": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42,
                },
                "regression": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42,
                },
            },
            "logistic_regression": {"classification": {"random_state": 42, "max_iter": 1000}},
            "linear_regression": {"regression": {}},
            "neural_network": {
                "classification": {
                    "hidden_layer_sizes": (100,),
                    "random_state": 42,
                    "max_iter": 500,
                },
                "regression": {"hidden_layer_sizes": (100,), "random_state": 42, "max_iter": 500},
            },
            "svm": {
                "classification": {"random_state": 42, "probability": True},
                "regression": {"random_state": 42},
            },
        }

        return defaults.get(algorithm, {}).get(task_type, {})

    async def _save_model_metadata(
        self,
        model_name: str,
        model_type: str,
        config: TrainingConfig,
        performance: ModelPerformance,
        model_path: str,
        file_hash: str,
        training_samples: int,
    ):
        """Save model metadata to database"""
        session = self.session_maker()
        try:
            # Check if model exists
            existing_model = (
                session.query(MLModelMetadata)
                .filter(MLModelMetadata.model_name == model_name)
                .first()
            )

            if existing_model:
                # Update existing model
                existing_model.version = "1.0.0"  # Could implement versioning
                existing_model.trained_on = datetime.utcnow()
                existing_model.training_start_date = config.train_start_date
                existing_model.training_end_date = config.train_end_date
                existing_model.training_samples = training_samples
                existing_model.hyperparameters = config.hyperparameters
                existing_model.feature_importance = performance.feature_importance
                existing_model.validation_accuracy = performance.accuracy
                existing_model.validation_precision = performance.precision
                existing_model.validation_recall = performance.recall
                existing_model.validation_f1 = performance.f1_score
                existing_model.model_file_path = model_path
                existing_model.model_file_hash = file_hash
                existing_model.is_active = True
                existing_model.updated_at = datetime.utcnow()
            else:
                # Create new model
                model_metadata = MLModelMetadata(
                    model_name=model_name,
                    model_type=model_type,
                    version="1.0.0",
                    trained_on=datetime.utcnow(),
                    training_start_date=config.train_start_date,
                    training_end_date=config.train_end_date,
                    training_samples=training_samples,
                    hyperparameters=config.hyperparameters,
                    feature_importance=performance.feature_importance,
                    validation_accuracy=performance.accuracy,
                    validation_precision=performance.precision,
                    validation_recall=performance.recall,
                    validation_f1=performance.f1_score,
                    model_file_path=model_path,
                    model_file_hash=file_hash,
                    is_active=True,
                )
                session.add(model_metadata)

            session.commit()
            logger.info(f"Saved metadata for model {model_name}")

        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
            session.rollback()
        finally:
            session.close()


class ModelScheduler:
    """Scheduler for automated model retraining"""

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.retraining_schedule = {
            "entry": timedelta(days=7),  # Retrain weekly
            "exit": timedelta(days=14),  # Retrain bi-weekly
            "strike_selection": timedelta(days=30),  # Retrain monthly
        }

    async def check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        session = self.trainer.session_maker()
        try:
            for model_type, interval in self.retraining_schedule.items():
                # Get latest model
                latest_model = (
                    session.query(MLModelMetadata)
                    .filter(
                        MLModelMetadata.model_type == model_type, MLModelMetadata.is_active == True
                    )
                    .order_by(MLModelMetadata.trained_on.desc())
                    .first()
                )

                should_retrain = False

                if not latest_model:
                    logger.info(f"No {model_type} model found, scheduling initial training")
                    should_retrain = True
                else:
                    # Check if model is stale
                    days_since_training = (datetime.utcnow() - latest_model.trained_on).days
                    if days_since_training >= interval.days:
                        logger.info(
                            f"{model_type} model is {days_since_training} days old, retraining"
                        )
                        should_retrain = True

                if should_retrain:
                    await self._retrain_model(model_type)

        except Exception as e:
            logger.error(f"Error in model retraining check: {e}")
        finally:
            session.close()

    async def _retrain_model(self, model_type: str):
        """Retrain a specific model type"""
        try:
            # Use last 60 days of data for training
            end_date = date.today()
            start_date = end_date - timedelta(days=60)

            if model_type == "entry":
                await self.trainer.train_entry_model(start_date, end_date)
            elif model_type == "exit":
                await self.trainer.train_exit_model(start_date, end_date)
            elif model_type == "strike_selection":
                await self.trainer.train_strike_model(start_date, end_date)

            logger.info(f"Completed retraining of {model_type} model")

        except Exception as e:
            logger.error(f"Error retraining {model_type} model: {e}")
