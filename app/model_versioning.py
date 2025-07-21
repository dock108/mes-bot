"""
Model Versioning and Deployment Management System

This module handles model versioning, deployment, rollback capabilities,
and A/B testing of different model versions.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import config
from app.notification_service import NotificationLevel, send_system_alert


class Base(DeclarativeBase):
    pass


logger = logging.getLogger(__name__)


class ModelVersion(Base):
    """Database model for tracking model versions"""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    model_type = Column(String(50), nullable=False)  # 'entry', 'exit', 'strike'
    version = Column(String(50), nullable=False)  # e.g., 'v1.2.3'
    algorithm = Column(String(50), nullable=False)  # 'random_forest', 'xgboost', etc.
    file_path = Column(String(500), nullable=False)  # Path to serialized model
    metadata_json = Column(Text)  # JSON metadata about the model
    performance_metrics = Column(Text)  # JSON performance metrics
    training_data_period = Column(String(100))  # e.g., '2024-01-01_to_2024-03-01'
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="active")  # 'active', 'deprecated', 'testing'
    deployment_date = Column(DateTime)
    is_production = Column(String(5), default="false")  # String boolean for compatibility
    performance_score = Column(Float)  # Overall performance score for ranking
    created_by = Column(String(100), default="automation")  # 'automation', 'manual', etc.


class ModelVersionManager:
    """
    Manages model versions, deployments, and rollbacks
    """

    def __init__(self, database_url: str, models_directory: str = "models"):
        self.database_url = database_url
        self.models_dir = Path(models_directory)
        self.models_dir.mkdir(exist_ok=True)

        # Create versioning database tables
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"Model version manager initialized with models directory: {self.models_dir}")

    def save_model_version(
        self,
        model: Any,
        model_type: str,
        algorithm: str,
        performance_metrics: Dict,
        metadata: Dict,
        training_period: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Save a new model version

        Args:
            model: The trained model object
            model_type: Type of model ('entry', 'exit', 'strike')
            algorithm: Algorithm used ('random_forest', 'xgboost', etc.)
            performance_metrics: Dictionary of performance metrics
            metadata: Additional metadata about the model
            training_period: String describing training data period
            version: Optional specific version string, auto-generated if None

        Returns:
            Version string of the saved model
        """
        session = self.SessionLocal()
        try:
            # Generate version if not provided
            if not version:
                version = self._generate_version(model_type, algorithm, session)

            # Create file path for the model
            model_filename = f"{model_type}_{algorithm}_{version}.pkl"
            model_path = self.models_dir / model_filename

            # Serialize and save the model
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Calculate performance score for ranking
            performance_score = self._calculate_performance_score(model_type, performance_metrics)

            # Create database record
            model_version = ModelVersion(
                model_type=model_type,
                version=version,
                algorithm=algorithm,
                file_path=str(model_path),
                metadata_json=json.dumps(metadata),
                performance_metrics=json.dumps(performance_metrics),
                training_data_period=training_period,
                performance_score=performance_score,
                status="active",
            )

            session.add(model_version)
            session.commit()

            logger.info(
                f"Saved {model_type} model version {version} with score {performance_score:.3f}"
            )
            return version

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving model version: {e}")
            raise
        finally:
            session.close()

    def load_model_version(
        self, model_type: str, version: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """
        Load a specific model version

        Args:
            model_type: Type of model to load
            version: Specific version to load, loads best production model if None

        Returns:
            Tuple of (model_object, metadata_dict)
        """
        session = self.SessionLocal()
        try:
            if version:
                # Load specific version
                model_record = (
                    session.query(ModelVersion)
                    .filter(
                        ModelVersion.model_type == model_type,
                        ModelVersion.version == version,
                    )
                    .first()
                )
            else:
                # Load best production model
                model_record = (
                    session.query(ModelVersion)
                    .filter(
                        ModelVersion.model_type == model_type,
                        ModelVersion.is_production == "true",
                        ModelVersion.status == "active",
                    )
                    .order_by(ModelVersion.performance_score.desc())
                    .first()
                )

            if not model_record:
                raise ValueError(f"No model found for {model_type} version {version}")

            # Load the model file
            with open(model_record.file_path, "rb") as f:
                model = pickle.load(f)

            # Parse metadata
            metadata = {
                "version": model_record.version,
                "algorithm": model_record.algorithm,
                "created_at": model_record.created_at,
                "performance_metrics": json.loads(model_record.performance_metrics),
                "training_period": model_record.training_data_period,
                "performance_score": model_record.performance_score,
                **json.loads(model_record.metadata_json),
            }

            logger.info(f"Loaded {model_type} model version {model_record.version}")
            return model, metadata

        except Exception as e:
            logger.error(f"Error loading model version: {e}")
            raise
        finally:
            session.close()

    def deploy_model_to_production(self, model_type: str, version: str) -> bool:
        """
        Deploy a specific model version to production

        Args:
            model_type: Type of model to deploy
            version: Version to deploy

        Returns:
            True if deployment successful, False otherwise
        """
        session = self.SessionLocal()
        try:
            # Find the model version to deploy
            new_model = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.version == version,
                )
                .first()
            )

            if not new_model:
                logger.error(f"Model version {version} not found for {model_type}")
                return False

            # Mark current production models as deprecated
            session.query(ModelVersion).filter(
                ModelVersion.model_type == model_type,
                ModelVersion.is_production == "true",
            ).update({"is_production": "false", "status": "deprecated"})

            # Deploy new model
            new_model.is_production = "true"
            new_model.deployment_date = datetime.utcnow()
            new_model.status = "active"

            session.commit()

            # Send deployment notification (fire and forget)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        send_system_alert(
                            f"Model Deployed to Production",
                            f"Successfully deployed {model_type} model version {version} to production",
                            NotificationLevel.INFO,
                            context={
                                "model_type": model_type,
                                "version": version,
                                "performance_score": new_model.performance_score,
                            },
                        )
                    )
            except RuntimeError:
                # No event loop running, skip notification
                pass

            logger.info(f"Deployed {model_type} model version {version} to production")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error deploying model to production: {e}")
            return False
        finally:
            session.close()

    def rollback_model(self, model_type: str) -> bool:
        """
        Rollback to the previous production model version

        Args:
            model_type: Type of model to rollback

        Returns:
            True if rollback successful, False otherwise
        """
        session = self.SessionLocal()
        try:
            # Get current production model
            current_model = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.is_production == "true",
                )
                .first()
            )

            # Get previous production model (most recent deprecated)
            previous_model = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.status == "deprecated",
                )
                .order_by(ModelVersion.deployment_date.desc())
                .first()
            )

            if not previous_model:
                logger.error(f"No previous model found for rollback of {model_type}")
                return False

            # Perform rollback
            if current_model:
                current_model.is_production = "false"
                current_model.status = "deprecated"

            previous_model.is_production = "true"
            previous_model.status = "active"
            previous_model.deployment_date = datetime.utcnow()

            session.commit()

            # Send rollback notification (fire and forget)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        send_system_alert(
                            f"Model Rollback Completed",
                            f"Successfully rolled back {model_type} model to version {previous_model.version}",
                            NotificationLevel.WARNING,
                            context={
                                "model_type": model_type,
                                "rollback_to_version": previous_model.version,
                                "previous_version": (
                                    current_model.version if current_model else "unknown"
                                ),
                            },
                        )
                    )
            except RuntimeError:
                # No event loop running, skip notification
                pass

            logger.info(f"Rolled back {model_type} model to version {previous_model.version}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error rolling back model: {e}")
            return False
        finally:
            session.close()

    def list_model_versions(self, model_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        List model versions with their metadata

        Args:
            model_type: Filter by model type, None for all types
            limit: Maximum number of versions to return

        Returns:
            List of model version dictionaries
        """
        session = self.SessionLocal()
        try:
            query = session.query(ModelVersion)

            if model_type:
                query = query.filter(ModelVersion.model_type == model_type)

            models = query.order_by(ModelVersion.created_at.desc()).limit(limit).all()

            result = []
            for model in models:
                result.append(
                    {
                        "id": model.id,
                        "model_type": model.model_type,
                        "version": model.version,
                        "algorithm": model.algorithm,
                        "performance_score": model.performance_score,
                        "created_at": model.created_at.isoformat(),
                        "deployment_date": (
                            model.deployment_date.isoformat() if model.deployment_date else None
                        ),
                        "status": model.status,
                        "is_production": model.is_production == "true",
                        "training_period": model.training_data_period,
                        "performance_metrics": json.loads(model.performance_metrics),
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error listing model versions: {e}")
            return []
        finally:
            session.close()

    def get_model_performance_comparison(self, model_type: str) -> Dict:
        """
        Compare performance of different model versions

        Args:
            model_type: Type of model to compare

        Returns:
            Dictionary with performance comparison data
        """
        session = self.SessionLocal()
        try:
            models = (
                session.query(ModelVersion)
                .filter(ModelVersion.model_type == model_type)
                .order_by(ModelVersion.performance_score.desc())
                .limit(10)
                .all()
            )

            comparison = {
                "model_type": model_type,
                "total_versions": len(models),
                "production_version": None,
                "best_version": None,
                "versions": [],
            }

            for model in models:
                version_data = {
                    "version": model.version,
                    "algorithm": model.algorithm,
                    "performance_score": model.performance_score,
                    "is_production": model.is_production == "true",
                    "created_at": model.created_at.isoformat(),
                    "performance_metrics": json.loads(model.performance_metrics),
                }

                comparison["versions"].append(version_data)

                if model.is_production == "true":
                    comparison["production_version"] = version_data

                if not comparison["best_version"]:
                    comparison["best_version"] = version_data

            return comparison

        except Exception as e:
            logger.error(f"Error getting model performance comparison: {e}")
            return {}
        finally:
            session.close()

    def cleanup_old_versions(self, model_type: str, keep_count: int = 10) -> int:
        """
        Clean up old model versions, keeping only the most recent ones

        Args:
            model_type: Type of model to clean up
            keep_count: Number of recent versions to keep

        Returns:
            Number of versions deleted
        """
        session = self.SessionLocal()
        try:
            # Get models to delete (excluding production models)
            models_to_delete = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.is_production == "false",
                )
                .order_by(ModelVersion.created_at.desc())
                .offset(keep_count)
                .all()
            )

            deleted_count = 0
            for model in models_to_delete:
                # Delete model file
                try:
                    Path(model.file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error deleting model file {model.file_path}: {e}")

                # Delete database record
                session.delete(model)
                deleted_count += 1

            session.commit()
            logger.info(f"Cleaned up {deleted_count} old {model_type} model versions")
            return deleted_count

        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old model versions: {e}")
            return 0
        finally:
            session.close()

    def _generate_version(self, model_type: str, algorithm: str, session: Session) -> str:
        """Generate a new version string for a model"""
        # Get the latest version for this model type and algorithm
        latest = (
            session.query(ModelVersion)
            .filter(
                ModelVersion.model_type == model_type,
                ModelVersion.algorithm == algorithm,
            )
            .order_by(ModelVersion.created_at.desc())
            .first()
        )

        if not latest:
            return "v1.0.0"

        # Parse current version and increment
        try:
            current_version = latest.version.replace("v", "")
            major, minor, patch = map(int, current_version.split("."))

            # Increment patch version
            patch += 1
            return f"v{major}.{minor}.{patch}"
        except Exception:
            # Fallback to timestamp-based version
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            return f"v{timestamp}"

    def _calculate_performance_score(self, model_type: str, metrics: Dict) -> float:
        """Calculate a single performance score for ranking models"""
        try:
            if model_type == "entry":
                # For entry models, combine accuracy, precision, and F1
                accuracy = metrics.get("accuracy", 0)
                precision = metrics.get("precision", 0)
                f1_score = metrics.get("f1_score", 0)
                return accuracy * 0.4 + precision * 0.3 + f1_score * 0.3

            elif model_type == "exit":
                # For exit models, focus on accuracy and cross-validation
                accuracy = metrics.get("accuracy", 0)
                cross_val = metrics.get("cross_val_mean", 0)
                return accuracy * 0.6 + cross_val * 0.4

            elif model_type == "strike":
                # For strike models, convert MSE and R2 to a score
                mse = metrics.get("mse", float("inf"))
                r2_score = metrics.get("r2_score", -1)

                # Normalize MSE (lower is better) and combine with R2 (higher is better)
                normalized_mse = max(0, 1 - (mse / 10000))  # Assume 10000 as max reasonable MSE
                normalized_r2 = max(0, (r2_score + 1) / 2)  # Convert R2 from [-1,1] to [0,1]

                return normalized_mse * 0.5 + normalized_r2 * 0.5

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
