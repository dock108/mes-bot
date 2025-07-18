"""
ML-Enhanced Decision Engine for 0DTE Options Strategy
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.config import config
from app.market_indicators import MarketFeatures, MarketIndicatorEngine
from app.ml_ensemble import MLEnsembleImplementation

logger = logging.getLogger(__name__)


class DecisionConfidence(Enum):
    """Confidence levels for decisions"""

    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class TradingSignal:
    """Container for trading signals with confidence and reasoning"""

    action: str  # 'ENTER', 'EXIT', 'HOLD', 'REDUCE'
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]  # Human-readable reasons
    features_used: Dict[str, float]  # Feature values that influenced decision
    model_predictions: Dict[str, float]  # Individual model predictions
    optimal_strikes: Optional[Tuple[float, float]] = None  # (call_strike, put_strike)
    position_size_multiplier: float = 1.0  # Adjust position size based on confidence
    profit_target_multiplier: float = 4.0  # Dynamic profit target
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ExitSignal:
    """Container for exit signals"""

    should_exit: bool
    exit_type: str  # 'PROFIT_TARGET', 'STOP_LOSS', 'TIME_DECAY', 'MARKET_CHANGE'
    confidence: float
    reasoning: List[str]
    new_profit_target: Optional[float] = None  # Dynamic profit target adjustment
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class BaseDecisionModel(ABC):
    """Abstract base class for decision models"""

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.last_prediction = None
        self.performance_history = []

    @abstractmethod
    async def predict_entry_signal(
        self, features: MarketFeatures
    ) -> Tuple[float, Dict[str, float]]:
        """Predict entry signal strength (0-1) and feature importance"""
        pass

    @abstractmethod
    async def predict_exit_signal(
        self, features: MarketFeatures, trade_info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Predict exit signal strength (0-1) and reasoning"""
        pass

    @abstractmethod
    async def optimize_strikes(
        self, features: MarketFeatures, current_price: float, implied_move: float
    ) -> Tuple[float, float]:
        """Optimize strike selection based on features"""
        pass

    def record_performance(self, prediction: float, actual_outcome: float):
        """Record model performance for tracking"""
        self.performance_history.append(
            {
                "timestamp": datetime.utcnow(),
                "prediction": prediction,
                "actual": actual_outcome,
                "error": abs(prediction - actual_outcome),
            }
        )

        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]


class VolatilityBasedModel(BaseDecisionModel):
    """Enhanced volatility-based decision model (baseline)"""

    def __init__(self):
        super().__init__("VolatilityBasedModel")
        self.is_trained = True  # No training required for rule-based model

    async def predict_entry_signal(
        self, features: MarketFeatures
    ) -> Tuple[float, Dict[str, float]]:
        """Enhanced volatility-based entry signal"""
        signal_strength = 0.0
        feature_importance = {}

        # Multi-timeframe volatility analysis
        vol_signals = {
            "15m": self._calculate_vol_signal(features.realized_vol_15m, features.atm_iv, 0.15),
            "30m": self._calculate_vol_signal(features.realized_vol_30m, features.atm_iv, 0.25),
            "60m": self._calculate_vol_signal(features.realized_vol_60m, features.atm_iv, 0.35),
            "2h": self._calculate_vol_signal(features.realized_vol_2h, features.atm_iv, 0.25),
        }

        # Weighted average of volatility signals
        vol_signal = (
            vol_signals["15m"] * 0.2
            + vol_signals["30m"] * 0.3
            + vol_signals["60m"] * 0.4
            + vol_signals["2h"] * 0.1
        )

        signal_strength += vol_signal * 0.4
        feature_importance["volatility_signal"] = vol_signal

        # Technical indicator signals
        tech_signal = self._calculate_technical_signal(features)
        signal_strength += tech_signal * 0.3
        feature_importance["technical_signal"] = tech_signal

        # Market regime signal
        regime_signal = self._calculate_regime_signal(features)
        signal_strength += regime_signal * 0.2
        feature_importance["regime_signal"] = regime_signal

        # Time-based signal
        time_signal = self._calculate_time_signal(features)
        signal_strength += time_signal * 0.1
        feature_importance["time_signal"] = time_signal

        # Normalize signal strength
        signal_strength = max(0.0, min(1.0, signal_strength))

        return signal_strength, feature_importance

    async def predict_exit_signal(
        self, features: MarketFeatures, trade_info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Enhanced exit signal prediction"""
        exit_strength = 0.0
        reasoning = {}

        # Time decay signal
        time_to_expiry = features.time_to_expiry
        if time_to_expiry < 2.0:  # Less than 2 hours
            time_decay_signal = 1.0 - (time_to_expiry / 2.0)
            exit_strength += time_decay_signal * 0.4
            reasoning["time_decay"] = time_decay_signal

        # Volatility expansion signal
        if features.realized_vol_30m > features.atm_iv * 1.2:
            vol_expansion_signal = min(1.0, (features.realized_vol_30m / features.atm_iv - 1.0) * 2)
            exit_strength += vol_expansion_signal * 0.3
            reasoning["volatility_expansion"] = vol_expansion_signal

        # Technical reversal signal
        reversal_signal = self._calculate_reversal_signal(features)
        exit_strength += reversal_signal * 0.3
        reasoning["technical_reversal"] = reversal_signal

        return min(1.0, exit_strength), reasoning

    async def optimize_strikes(
        self, features: MarketFeatures, current_price: float, implied_move: float
    ) -> Tuple[float, float]:
        """Optimize strike selection based on market conditions"""
        base_multiplier_call = config.trading.implied_move_multiplier_1
        base_multiplier_put = config.trading.implied_move_multiplier_1

        # Adjust based on volatility regime
        if features.iv_rank < 30:  # Low IV environment
            base_multiplier_call *= 1.1
            base_multiplier_put *= 1.1
        elif features.iv_rank > 70:  # High IV environment
            base_multiplier_call *= 0.95
            base_multiplier_put *= 0.95

        # Adjust based on technical indicators
        if features.rsi_30m > 70:  # Overbought - favor put side
            base_multiplier_put *= 0.9
        elif features.rsi_30m < 30:  # Oversold - favor call side
            base_multiplier_call *= 0.9

        # Calculate strikes
        call_strike = self._round_to_strike(current_price + implied_move * base_multiplier_call)
        put_strike = self._round_to_strike(current_price - implied_move * base_multiplier_put)

        return call_strike, put_strike

    def _calculate_vol_signal(
        self, realized_vol: float, implied_vol: float, threshold: float
    ) -> float:
        """Calculate volatility signal for given timeframe"""
        if implied_vol == 0:
            return 0.0

        vol_ratio = realized_vol / implied_vol
        threshold_vol = config.trading.volatility_threshold * threshold

        if vol_ratio < threshold_vol:
            # Strong signal when realized vol is much lower than implied
            return min(1.0, (threshold_vol - vol_ratio) / threshold_vol * 2)
        else:
            return 0.0

    def _calculate_technical_signal(self, features: MarketFeatures) -> float:
        """Calculate technical indicator signal"""
        signal = 0.0

        # RSI signals
        if 30 < features.rsi_15m < 70 and 30 < features.rsi_30m < 70:
            signal += 0.3  # Neutral RSI is good for selling premium

        # MACD signal
        if abs(features.macd_histogram) < 0.1:  # MACD convergence
            signal += 0.2

        # Bollinger Band signal
        if 0.2 < features.bb_position < 0.8:  # Price in middle of bands
            signal += 0.3

        # Bollinger Band squeeze
        if features.bb_squeeze < 0.02:  # Low volatility squeeze
            signal += 0.2

        return signal

    def _calculate_regime_signal(self, features: MarketFeatures) -> float:
        """Calculate market regime signal"""
        signal = 0.0

        # VIX level signal
        if 15 < features.vix_level < 25:  # Normal volatility
            signal += 0.4
        elif features.vix_level < 15:  # Low volatility
            signal += 0.6
        elif features.vix_level > 30:  # High volatility - be cautious
            signal -= 0.2

        # Volume profile
        if 0.8 < features.volume_profile < 1.2:  # Normal volume
            signal += 0.3

        # Bid-ask spread
        if features.bid_ask_spread < 0.01:  # Tight spreads
            signal += 0.3

        return max(0.0, signal)

    def _calculate_time_signal(self, features: MarketFeatures) -> float:
        """Calculate time-based signal"""
        signal = 0.0

        # Time of day preference (avoid first and last hour)
        if 10.5 < features.time_of_day < 15.0:
            signal += 0.5

        # Day of week (avoid Mondays and Fridays for 0DTE)
        if 1 <= features.day_of_week <= 3:  # Tuesday to Thursday
            signal += 0.3

        # Time to expiry
        if 3 < features.time_to_expiry < 6:  # Sweet spot for entry
            signal += 0.2

        return signal

    def _calculate_reversal_signal(self, features: MarketFeatures) -> float:
        """Calculate technical reversal signal for exits"""
        signal = 0.0

        # RSI extremes
        if features.rsi_15m > 75 or features.rsi_15m < 25:
            signal += 0.4

        # Bollinger Band extremes
        if features.bb_position > 0.9 or features.bb_position < 0.1:
            signal += 0.3

        # MACD divergence
        if abs(features.macd_histogram) > 0.5:
            signal += 0.3

        return signal

    def _round_to_strike(self, price: float) -> float:
        """Round price to nearest valid option strike"""
        return round(price / 25) * 25


class MLEnsembleModel(BaseDecisionModel):
    """Machine Learning ensemble model with actual implementation"""

    def __init__(self, database_url: str):
        super().__init__("MLEnsembleModel")
        self.ml_implementation = MLEnsembleImplementation(database_url)
        self.last_prediction_cache = {}

    async def predict_entry_signal(
        self, features: MarketFeatures
    ) -> Tuple[float, Dict[str, float]]:
        """ML-based entry signal prediction using trained models"""
        try:
            # Get ML ensemble prediction
            signal, importance = await self.ml_implementation.predict_entry_signal(features)

            # Cache the prediction
            self.last_prediction_cache["entry"] = {
                "signal": signal,
                "importance": importance,
                "timestamp": datetime.utcnow(),
            }

            return signal, importance

        except Exception as e:
            logger.error(f"Error in ML entry prediction, falling back to baseline: {e}")
            # Fallback to volatility-based model
            baseline_model = VolatilityBasedModel()
            return await baseline_model.predict_entry_signal(features)

    async def predict_exit_signal(
        self, features: MarketFeatures, trade_info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """ML-based exit signal prediction using trained models"""
        try:
            # Get ML ensemble prediction
            signal, importance = await self.ml_implementation.predict_exit_signal(
                features, trade_info
            )

            # Cache the prediction
            self.last_prediction_cache["exit"] = {
                "signal": signal,
                "importance": importance,
                "timestamp": datetime.utcnow(),
            }

            return signal, importance

        except Exception as e:
            logger.error(f"Error in ML exit prediction, falling back to baseline: {e}")
            # Fallback to volatility-based model
            baseline_model = VolatilityBasedModel()
            return await baseline_model.predict_exit_signal(features, trade_info)

    async def optimize_strikes(
        self, features: MarketFeatures, current_price: float, implied_move: float
    ) -> Tuple[float, float]:
        """ML-optimized strike selection"""
        try:
            return await self.ml_implementation.optimize_strikes(
                features, current_price, implied_move
            )
        except Exception as e:
            logger.error(f"Error in ML strike optimization, falling back to baseline: {e}")
            baseline_model = VolatilityBasedModel()
            return await baseline_model.optimize_strikes(features, current_price, implied_move)

    def get_model_status(self) -> Dict[str, Any]:
        """Get ML model status"""
        return self.ml_implementation.get_model_status()


class DecisionEngine:
    """Main decision engine orchestrating multiple models"""

    def __init__(self, database_url: str = config.database.url):
        self.indicator_engine = MarketIndicatorEngine()
        self.models = {
            "volatility_based": VolatilityBasedModel(),
            "ml_ensemble": MLEnsembleModel(database_url),
        }

        self.model_weights = {
            "volatility_based": 0.7,  # Higher weight for proven model initially
            "ml_ensemble": 0.3,  # Lower weight until ML models are fully trained
        }

        self.performance_tracker = {}
        self.decision_history = []

        # Configuration
        self.min_confidence_threshold = 0.3
        self.ensemble_agreement_threshold = 0.6

    async def update_market_data(
        self,
        price: float,
        bid: float,
        ask: float,
        volume: float,
        atm_iv: float,
        vix_level: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Update market data for all indicators"""
        self.indicator_engine.update_market_data(price, bid, ask, volume, atm_iv, timestamp)

        if vix_level:
            self.indicator_engine.regime_detector.add_vix_data(
                vix_level, timestamp or datetime.utcnow()
            )

    async def generate_entry_signal(
        self, current_price: float, implied_move: float, vix_level: Optional[float] = None
    ) -> TradingSignal:
        """Generate comprehensive entry signal using all models"""
        # Calculate all market features
        features = self.indicator_engine.calculate_all_features(
            current_price, implied_move, vix_level
        )

        # Get predictions from all models
        model_predictions = {}
        model_importance = {}

        for model_name, model in self.models.items():
            try:
                signal, importance = await model.predict_entry_signal(features)
                model_predictions[model_name] = signal
                model_importance[model_name] = importance
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                model_predictions[model_name] = 0.0
                model_importance[model_name] = {}

        # Calculate ensemble prediction
        ensemble_signal = sum(
            model_predictions[name] * self.model_weights[name] for name in model_predictions
        )

        # Calculate confidence based on model agreement
        prediction_values = list(model_predictions.values())
        if len(prediction_values) > 1:
            prediction_std = np.std(prediction_values)
            confidence = max(0.1, 1.0 - prediction_std * 2)  # Lower std = higher confidence
        else:
            confidence = ensemble_signal

        # Generate reasoning
        reasoning = self._generate_reasoning(features, model_predictions, model_importance)

        # Determine action based on signal strength and confidence
        if ensemble_signal > 0.6 and confidence > self.min_confidence_threshold:
            action = "ENTER"
        elif ensemble_signal > 0.4 and confidence > 0.5:
            action = "ENTER"
        else:
            action = "HOLD"

        # Optimize strike selection
        optimal_strikes = None
        if action == "ENTER":
            try:
                # Use the best performing model for strike optimization
                best_model = max(self.models.values(), key=lambda m: len(m.performance_history))
                optimal_strikes = await best_model.optimize_strikes(
                    features, current_price, implied_move
                )
            except Exception as e:
                logger.error(f"Error optimizing strikes: {e}")

        # Calculate position size multiplier based on confidence
        position_size_multiplier = min(1.5, max(0.5, confidence * 1.5))

        # Calculate dynamic profit target
        profit_target_multiplier = self._calculate_dynamic_profit_target(features, confidence)

        signal = TradingSignal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            features_used=self._extract_key_features(features),
            model_predictions=model_predictions,
            optimal_strikes=optimal_strikes,
            position_size_multiplier=position_size_multiplier,
            profit_target_multiplier=profit_target_multiplier,
        )

        # Record decision for tracking
        self.decision_history.append(signal)

        return signal

    async def generate_exit_signal(
        self,
        trade_info: Dict,
        current_price: float,
        implied_move: float,
        vix_level: Optional[float] = None,
    ) -> ExitSignal:
        """Generate exit signal for existing position"""
        features = self.indicator_engine.calculate_all_features(
            current_price, implied_move, vix_level
        )

        # Get exit predictions from all models
        exit_predictions = {}
        exit_reasoning = {}

        for model_name, model in self.models.items():
            try:
                signal, reasoning = await model.predict_exit_signal(features, trade_info)
                exit_predictions[model_name] = signal
                exit_reasoning[model_name] = reasoning
            except Exception as e:
                logger.error(f"Error in {model_name} exit prediction: {e}")
                exit_predictions[model_name] = 0.0
                exit_reasoning[model_name] = {}

        # Calculate ensemble exit signal
        ensemble_exit = sum(
            exit_predictions[name] * self.model_weights[name] for name in exit_predictions
        )

        # Determine exit type and confidence
        should_exit = ensemble_exit > 0.5
        confidence = ensemble_exit

        # Determine exit type
        exit_type = "HOLD"
        combined_reasoning = []

        if should_exit:
            # Analyze reasons for exit
            if features.time_to_expiry < 1.0:
                exit_type = "TIME_DECAY"
                combined_reasoning.append("Time decay acceleration")
            elif any(
                "volatility_expansion" in r for r in exit_reasoning.values() if isinstance(r, dict)
            ):
                exit_type = "MARKET_CHANGE"
                combined_reasoning.append("Volatility expansion detected")
            elif any(
                "technical_reversal" in r for r in exit_reasoning.values() if isinstance(r, dict)
            ):
                exit_type = "MARKET_CHANGE"
                combined_reasoning.append("Technical reversal signals")
            else:
                exit_type = "PROFIT_TARGET"
                combined_reasoning.append("Optimal profit-taking opportunity")

        # Calculate new profit target if needed
        new_profit_target = None
        if not should_exit and confidence > 0.3:
            new_profit_target = self._calculate_dynamic_profit_target(features, confidence)

        return ExitSignal(
            should_exit=should_exit,
            exit_type=exit_type,
            confidence=confidence,
            reasoning=combined_reasoning,
            new_profit_target=new_profit_target,
        )

    def _generate_reasoning(
        self,
        features: MarketFeatures,
        model_predictions: Dict[str, float],
        model_importance: Dict[str, Dict],
    ) -> List[str]:
        """Generate human-readable reasoning for the decision"""
        reasoning = []

        # Volatility analysis
        vol_ratio = features.realized_vol_30m / features.atm_iv if features.atm_iv > 0 else 1.0
        if vol_ratio < 0.6:
            reasoning.append(
                f"Realized volatility ({vol_ratio:.1%}) significantly below implied volatility"
            )

        # Technical indicators
        if 30 < features.rsi_30m < 70:
            reasoning.append(
                f"RSI in neutral range ({features.rsi_30m:.1f}) - good for premium selling"
            )

        # Market regime
        if features.vix_level < 20:
            reasoning.append(
                f"Low VIX environment ({features.vix_level:.1f}) favorable for premium selling"
            )

        # Time factors
        if 10 < features.time_of_day < 15:
            reasoning.append("Optimal time window for 0DTE trades")

        # Model agreement
        predictions_range = max(model_predictions.values()) - min(model_predictions.values())
        if predictions_range < 0.3:
            reasoning.append("High model agreement increases confidence")

        return reasoning

    def _extract_key_features(self, features: MarketFeatures) -> Dict[str, float]:
        """Extract key features for logging and analysis"""
        return {
            "realized_vol_30m": features.realized_vol_30m,
            "atm_iv": features.atm_iv,
            "iv_rank": features.iv_rank,
            "rsi_30m": features.rsi_30m,
            "vix_level": features.vix_level,
            "time_of_day": features.time_of_day,
            "time_to_expiry": features.time_to_expiry,
            "bb_position": features.bb_position,
        }

    def _calculate_dynamic_profit_target(
        self, features: MarketFeatures, confidence: float
    ) -> float:
        """Calculate dynamic profit target based on market conditions"""
        base_target = config.trading.profit_target_multiplier

        # Adjust based on time to expiry
        if features.time_to_expiry < 2:
            base_target *= 0.8  # Take profits earlier when close to expiry
        elif features.time_to_expiry > 5:
            base_target *= 1.2  # Can afford to wait longer

        # Adjust based on volatility environment
        if features.iv_rank > 70:
            base_target *= 1.1  # High IV - can target higher profits
        elif features.iv_rank < 30:
            base_target *= 0.9  # Low IV - take profits sooner

        # Adjust based on confidence
        base_target *= 0.8 + confidence * 0.4  # Higher confidence = higher targets

        return max(2.0, min(6.0, base_target))  # Keep within reasonable bounds

    def update_model_performance(self, prediction_id: str, actual_outcome: float):
        """Update model performance tracking"""
        # Find the prediction in history and update model performance
        for decision in reversed(self.decision_history):
            if str(id(decision)) == prediction_id:
                for model_name, prediction in decision.model_predictions.items():
                    if model_name in self.models:
                        self.models[model_name].record_performance(prediction, actual_outcome)
                break

    def get_performance_summary(self) -> Dict:
        """Get performance summary for all models"""
        summary = {}

        for model_name, model in self.models.items():
            if model.performance_history:
                recent_errors = [p["error"] for p in model.performance_history[-50:]]
                summary[model_name] = {
                    "avg_error": np.mean(recent_errors),
                    "total_predictions": len(model.performance_history),
                    "recent_accuracy": 1.0 - np.mean(recent_errors),
                }
            else:
                summary[model_name] = {
                    "avg_error": 0.0,
                    "total_predictions": 0,
                    "recent_accuracy": 0.0,
                }

        return summary
