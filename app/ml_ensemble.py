"""
Enhanced ML Ensemble Model Implementation for Decision Engine
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.config import config
from app.feature_pipeline import FeatureEngineer
from app.market_indicators import MarketFeatures
from app.ml_training import EntryPredictionModel, ExitPredictionModel, ModelConfig
from app.models import DecisionHistory, MLModelMetadata, Trade, get_session_maker

logger = logging.getLogger(__name__)


class MLEnsembleImplementation:
    """Production-ready ML ensemble model for trading decisions"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.session_maker = get_session_maker(database_url)

        # Model storage
        self.entry_models = {}
        self.exit_models = {}
        self.feature_engineer = FeatureEngineer(database_url)

        # Model paths
        self.model_dir = Path(config.data.cache_dir) / "ml_models"
        self.model_dir.mkdir(exist_ok=True)

        # Load trained models
        self._load_models()

        # Performance tracking
        self.prediction_history = []
        self.feature_importance_cache = {}

    def _load_models(self):
        """Load trained models from disk or train new ones"""
        try:
            # Load entry models
            self._load_entry_models()

            # Load exit models
            self._load_exit_models()

            logger.info(
                f"Loaded {len(self.entry_models)} entry models and {len(self.exit_models)} exit models"
            )

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Will train new models on first use")

    def _load_entry_models(self):
        """Load or train entry prediction models"""
        model_configs = [
            ModelConfig(
                model_type="entry",
                algorithm="gradient_boosting",
                hyperparameters={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "subsample": 0.8,
                },
            ),
            ModelConfig(
                model_type="entry",
                algorithm="random_forest",
                hyperparameters={"n_estimators": 100, "max_depth": 10, "min_samples_split": 10},
            ),
            ModelConfig(
                model_type="entry",
                algorithm="neural_network",
                hyperparameters={
                    "hidden_layer_sizes": (50, 30),
                    "learning_rate": "adaptive",
                    "max_iter": 500,
                },
            ),
        ]

        for config in model_configs:
            model_name = f"entry_{config.algorithm}"
            model_path = self.model_dir / f"{model_name}.joblib"

            if model_path.exists():
                # Load existing model
                try:
                    model = EntryPredictionModel.load_model(str(model_path))
                    self.entry_models[model_name] = model
                    logger.info(f"Loaded {model_name} from disk")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
            else:
                # Model doesn't exist, will be trained on first use
                self.entry_models[model_name] = EntryPredictionModel(config)

    def _load_exit_models(self):
        """Load or train exit prediction models"""
        model_configs = [
            ModelConfig(
                model_type="exit",
                algorithm="gradient_boosting",
                hyperparameters={"n_estimators": 50, "learning_rate": 0.15, "max_depth": 4},
            ),
            ModelConfig(
                model_type="exit",
                algorithm="logistic_regression",
                hyperparameters={"C": 1.0, "max_iter": 1000},
            ),
        ]

        for config in model_configs:
            model_name = f"exit_{config.algorithm}"
            model_path = self.model_dir / f"{model_name}.joblib"

            if model_path.exists():
                try:
                    model = ExitPredictionModel.load_model(str(model_path))
                    self.exit_models[model_name] = model
                    logger.info(f"Loaded {model_name} from disk")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
            else:
                self.exit_models[model_name] = ExitPredictionModel(config)

    async def predict_entry_signal(
        self, features: MarketFeatures
    ) -> Tuple[float, Dict[str, float]]:
        """Predict entry signal using ensemble of models"""
        try:
            # Convert MarketFeatures to feature vector
            feature_df = await self._prepare_features(features)

            if feature_df.empty:
                logger.warning("Empty feature dataframe, returning neutral signal")
                return 0.5, {}

            # Get predictions from all models
            predictions = []
            feature_importances = {}

            for model_name, model in self.entry_models.items():
                try:
                    # Ensure model is trained
                    if not model.is_trained:
                        await self._train_model_if_needed(model, "entry")

                    if model.is_trained:
                        # Get prediction
                        proba = model.predict_proba(feature_df)
                        if proba is not None and len(proba) > 0:
                            # Get probability of positive class (entry signal)
                            pred = proba[0][1] if len(proba[0]) > 1 else proba[0][0]
                            predictions.append(pred)

                            # Track feature importance
                            if model.performance and model.performance.feature_importance:
                                for feat, imp in model.performance.feature_importance.items():
                                    if feat not in feature_importances:
                                        feature_importances[feat] = []
                                    feature_importances[feat].append(imp)

                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    continue

            if not predictions:
                logger.warning("No model predictions available")
                return 0.5, {}

            # Ensemble prediction (weighted average)
            ensemble_prediction = np.mean(predictions)

            # Average feature importances
            avg_importances = {}
            for feat, imps in feature_importances.items():
                avg_importances[feat] = np.mean(imps)

            # Normalize importances
            total_imp = sum(avg_importances.values())
            if total_imp > 0:
                avg_importances = {k: v / total_imp for k, v in avg_importances.items()}

            # Log prediction
            self._log_prediction("entry", ensemble_prediction, predictions)

            return ensemble_prediction, avg_importances

        except Exception as e:
            logger.error(f"Error in predict_entry_signal: {e}")
            return 0.5, {}

    async def predict_exit_signal(
        self, features: MarketFeatures, trade_info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Predict exit signal using ensemble of models"""
        try:
            # Prepare features including trade-specific info
            feature_df = await self._prepare_features(features, trade_info)

            if feature_df.empty:
                return 0.0, {}

            predictions = []
            feature_importances = {}

            for model_name, model in self.exit_models.items():
                try:
                    if not model.is_trained:
                        await self._train_model_if_needed(model, "exit")

                    if model.is_trained:
                        proba = model.predict_proba(feature_df)
                        if proba is not None and len(proba) > 0:
                            pred = proba[0][1] if len(proba[0]) > 1 else proba[0][0]
                            predictions.append(pred)

                            if model.performance and model.performance.feature_importance:
                                for feat, imp in model.performance.feature_importance.items():
                                    if feat not in feature_importances:
                                        feature_importances[feat] = []
                                    feature_importances[feat].append(imp)

                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    continue

            if not predictions:
                return 0.0, {}

            ensemble_prediction = np.mean(predictions)

            # Average and normalize importances
            avg_importances = {}
            for feat, imps in feature_importances.items():
                avg_importances[feat] = np.mean(imps)

            total_imp = sum(avg_importances.values())
            if total_imp > 0:
                avg_importances = {k: v / total_imp for k, v in avg_importances.items()}

            self._log_prediction("exit", ensemble_prediction, predictions)

            return ensemble_prediction, avg_importances

        except Exception as e:
            logger.error(f"Error in predict_exit_signal: {e}")
            return 0.0, {}

    async def optimize_strikes(
        self, features: MarketFeatures, current_price: float, implied_move: float
    ) -> Tuple[float, float]:
        """Optimize strike selection using ML insights"""
        try:
            # Get entry prediction for confidence-based strike adjustment
            entry_signal, _ = await self.predict_entry_signal(features)

            # Base strikes on implied move
            base_call_strike = current_price + implied_move
            base_put_strike = current_price - implied_move

            # Adjust based on ML confidence
            if entry_signal > 0.7:  # High confidence
                # Tighter strikes for higher probability
                adjustment_factor = 0.9
            elif entry_signal > 0.5:  # Medium confidence
                adjustment_factor = 1.0
            else:  # Low confidence
                # Wider strikes for safety
                adjustment_factor = 1.1

            call_strike = self._round_to_strike(base_call_strike * adjustment_factor)
            put_strike = self._round_to_strike(base_put_strike * adjustment_factor)

            return call_strike, put_strike

        except Exception as e:
            logger.error(f"Error in optimize_strikes: {e}")
            # Fallback to standard strikes
            call_strike = self._round_to_strike(current_price + implied_move)
            put_strike = self._round_to_strike(current_price - implied_move)
            return call_strike, put_strike

    async def _prepare_features(
        self, market_features: MarketFeatures, trade_info: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Convert MarketFeatures to ML feature dataframe"""
        try:
            # Basic market features
            features = {
                # Price and volume
                "price": market_features.price,
                "volume": market_features.volume,
                "bid_ask_spread": market_features.bid_ask_spread,
                "volume_profile": market_features.volume_profile,
                # Volatility features
                "realized_vol_15m": market_features.realized_vol_15m,
                "realized_vol_30m": market_features.realized_vol_30m,
                "atm_iv": market_features.atm_iv,
                "iv_rank": market_features.iv_rank,
                "iv_percentile": market_features.iv_percentile,
                "put_call_ratio": market_features.put_call_ratio,
                # Technical indicators
                "rsi_5m": market_features.rsi_5m,
                "rsi_15m": market_features.rsi_15m,
                "rsi_30m": market_features.rsi_30m,
                "macd_signal": market_features.macd_signal,
                "macd_histogram": market_features.macd_histogram,
                "bb_position": market_features.bb_position,
                "bb_squeeze": market_features.bb_squeeze,
                # Market regime
                "vix_level": market_features.vix_level,
                "market_regime": 1 if market_features.market_regime == "normal" else 0,
                # Time features
                "time_of_day": market_features.time_of_day,
                "day_of_week": market_features.day_of_week,
                "time_to_expiry": market_features.time_to_expiry,
            }

            # Add trade-specific features if provided
            if trade_info:
                features.update(
                    {
                        "current_pnl_pct": trade_info.get("current_pnl_pct", 0),
                        "time_in_trade": trade_info.get("time_in_trade", 0),
                        "max_profit_pct": trade_info.get("max_profit_pct", 0),
                        "max_loss_pct": trade_info.get("max_loss_pct", 0),
                    }
                )

            # Create dataframe
            df = pd.DataFrame([features])

            # Add engineered features
            df = await self.feature_engineer.engineer_features(df)

            return df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    async def _train_model_if_needed(self, model: Any, model_type: str):
        """Train model if not already trained"""
        try:
            if model.is_trained:
                return

            logger.info(f"Training {model_type} model: {model.config.algorithm}")

            # Get training data for the last 30 days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            training_data = await self.feature_engineer.get_training_data(
                start_date=start_date, end_date=end_date, target_type=model_type
            )

            if training_data is None or len(training_data) < config.ml.min_training_samples:
                logger.warning(f"Insufficient training data for {model_type} model")
                return

            # Train model
            X = training_data.drop(columns=["target", "timestamp"], errors="ignore")
            y = training_data["target"]

            performance = await model.train(X, y)

            # Save model
            model_name = f"{model_type}_{model.config.algorithm}"
            model_path = self.model_dir / f"{model_name}.joblib"
            model.save_model(str(model_path))

            logger.info(f"Trained and saved {model_name}. Performance: {performance}")

        except Exception as e:
            logger.error(f"Error training model: {e}")

    def _round_to_strike(self, price: float) -> float:
        """Round price to nearest valid option strike"""
        # For MES, strikes are in increments of 5
        return round(price / 5) * 5

    def _log_prediction(
        self, prediction_type: str, ensemble_pred: float, individual_preds: List[float]
    ):
        """Log prediction for monitoring"""
        self.prediction_history.append(
            {
                "timestamp": datetime.utcnow(),
                "type": prediction_type,
                "ensemble_prediction": ensemble_pred,
                "individual_predictions": individual_preds,
                "std_dev": np.std(individual_preds) if len(individual_preds) > 1 else 0,
            }
        )

        # Keep only recent history
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            "entry_models": {},
            "exit_models": {},
            "total_predictions": len(self.prediction_history),
        }

        for name, model in self.entry_models.items():
            entry_models_dict = status["entry_models"]
            assert isinstance(entry_models_dict, dict)
            entry_models_dict[name] = {
                "trained": model.is_trained,
                "algorithm": model.config.algorithm,
                "performance": (
                    model.performance.__dict__
                    if hasattr(model, "performance") and model.performance
                    else None
                ),
            }

        for name, model in self.exit_models.items():
            exit_models_dict = status["exit_models"]
            assert isinstance(exit_models_dict, dict)
            exit_models_dict[name] = {
                "trained": model.is_trained,
                "algorithm": model.config.algorithm,
                "performance": (
                    model.performance.__dict__
                    if hasattr(model, "performance") and model.performance
                    else None
                ),
            }

        return status
