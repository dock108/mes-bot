"""
Predictive Risk Model for ML-based risk forecasting

This module implements GARCH volatility forecasting, regime change detection,
and pattern recognition for identifying dangerous market setups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class RiskPrediction:
    """Container for risk predictions"""
    timestamp: datetime
    drawdown_probability: float  # Probability of hitting max drawdown
    volatility_forecast: float  # Next period volatility forecast
    regime_change_probability: float
    risk_patterns_detected: List[str]
    recommended_actions: List[str]
    confidence_score: float


@dataclass
class VolatilityForecast:
    """Volatility forecast results"""
    current_volatility: float
    forecast_1h: float
    forecast_4h: float
    forecast_daily: float
    volatility_percentile: float
    is_clustering: bool


class RiskPredictor:
    """
    ML-based risk prediction using market microstructure
    """

    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.volatility_history = []
        self.regime_history = []
        self.pattern_library = self._initialize_pattern_library()

    def _initialize_pattern_library(self) -> Dict[str, Dict]:
        """Initialize library of dangerous market patterns"""
        return {
            "volatility_expansion": {
                "description": "Rapid volatility increase",
                "threshold": 2.0,  # 2x normal volatility
                "lookback": 10,
                "danger_level": "high"
            },
            "correlation_breakdown": {
                "description": "Normal correlations breaking down",
                "threshold": 0.3,  # Correlation drop
                "lookback": 20,
                "danger_level": "medium"
            },
            "liquidity_drain": {
                "description": "Bid-ask spreads widening",
                "threshold": 2.5,  # 2.5x normal spread
                "lookback": 5,
                "danger_level": "high"
            },
            "momentum_reversal": {
                "description": "Sharp trend reversal detected",
                "threshold": 3.0,  # 3 std dev move
                "lookback": 15,
                "danger_level": "medium"
            },
            "gap_risk": {
                "description": "Price gaps increasing",
                "threshold": 0.02,  # 2% gaps
                "lookback": 5,
                "danger_level": "high"
            },
            "volatility_smile_steepening": {
                "description": "Options skew increasing rapidly",
                "threshold": 0.2,  # 20% skew change
                "lookback": 10,
                "danger_level": "medium"
            }
        }

    def predict_drawdown_probability(self, current_drawdown: float,
                                    max_drawdown: float,
                                    returns: np.ndarray,
                                    horizon_minutes: int = 60) -> float:
        """
        Predict probability of hitting max drawdown in next N minutes

        Args:
            current_drawdown: Current drawdown level (negative value)
            max_drawdown: Maximum allowed drawdown (negative value)
            returns: Historical returns
            horizon_minutes: Prediction horizon

        Returns:
            Probability between 0 and 1
        """
        if len(returns) < 20:
            return 0.0

        # Calculate distance to max drawdown
        distance = abs(max_drawdown - current_drawdown)
        remaining_cushion = distance / abs(max_drawdown) if max_drawdown != 0 else 1

        # Estimate drift and volatility
        drift = np.mean(returns[-20:])
        volatility = np.std(returns[-20:])

        # Calculate probability using modified Black-Scholes approach
        if volatility == 0:
            return 0.0

        # Time scaling (assuming minute bars)
        time_factor = np.sqrt(horizon_minutes / 60)  # Scale to hours

        # Calculate z-score for hitting barrier
        z_score = (distance - drift * horizon_minutes) / (volatility * time_factor)

        # Get probability from normal distribution
        probability = 1 - stats.norm.cdf(z_score)

        # Adjust for momentum
        recent_momentum = np.mean(returns[-5:])
        if recent_momentum < 0:
            # Negative momentum increases probability
            momentum_adjustment = 1 + abs(recent_momentum) * 10
            probability = min(probability * momentum_adjustment, 1.0)

        # Adjust for volatility regime
        current_vol = np.std(returns[-10:])
        historical_vol = np.std(returns)
        if current_vol > historical_vol * 1.5:
            # High volatility increases probability
            probability = min(probability * 1.3, 1.0)

        return float(probability)

    def detect_regime_change(self, returns: np.ndarray,
                            volumes: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Identify market regime shifts affecting risk profile

        Args:
            returns: Price returns
            volumes: Trading volumes (optional)

        Returns:
            (new_regime, change_probability)
        """
        if len(returns) < 40:
            return "normal", 0.0

        # Calculate regime indicators
        recent_returns = returns[-20:]
        historical_returns = returns[-40:-20]

        # Volatility analysis
        recent_vol = np.std(recent_returns)
        hist_vol = np.std(historical_returns)
        vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1

        # Distribution analysis
        recent_skew = stats.skew(recent_returns)
        recent_kurt = stats.kurtosis(recent_returns)

        # Trend analysis
        recent_trend = np.mean(recent_returns)
        trend_strength = abs(recent_trend) / recent_vol if recent_vol > 0 else 0

        # Volume analysis if available
        volume_surge = 1.0
        if volumes is not None and len(volumes) >= 40:
            recent_volume = np.mean(volumes[-20:])
            hist_volume = np.mean(volumes[-40:-20])
            volume_surge = recent_volume / hist_volume if hist_volume > 0 else 1

        # Regime classification with probabilities
        regime_scores = {
            "crisis": 0.0,
            "volatile": 0.0,
            "trending": 0.0,
            "normal": 0.0,
            "quiet": 0.0
        }

        # Crisis indicators
        if vol_ratio > 2.0 and recent_kurt > 3:
            regime_scores["crisis"] = min(vol_ratio / 3, 1.0)

        # Volatile regime
        if vol_ratio > 1.5:
            regime_scores["volatile"] = min(vol_ratio / 2, 1.0)

        # Trending regime
        if trend_strength > 2:
            regime_scores["trending"] = min(trend_strength / 3, 1.0)

        # Quiet regime
        if vol_ratio < 0.7:
            regime_scores["quiet"] = min(1 / vol_ratio, 1.0)

        # Normal is default
        regime_scores["normal"] = 1 - max(regime_scores.values())

        # Get most likely regime
        current_regime = max(regime_scores, key=regime_scores.get)
        change_probability = regime_scores[current_regime]

        # Store in history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': current_regime,
            'probability': change_probability
        })

        return current_regime, change_probability

    def forecast_volatility_garch(self, returns: np.ndarray,
                                 periods_ahead: int = 24) -> VolatilityForecast:
        """
        GARCH(1,1) volatility forecasting

        Args:
            returns: Historical returns
            periods_ahead: Forecast horizon

        Returns:
            VolatilityForecast object
        """
        if len(returns) < 30:
            current_vol = np.std(returns) if len(returns) > 0 else 0
            return VolatilityForecast(
                current_volatility=current_vol,
                forecast_1h=current_vol,
                forecast_4h=current_vol,
                forecast_daily=current_vol,
                volatility_percentile=50,
                is_clustering=False
            )

        # Calculate current volatility
        current_vol = np.std(returns[-20:])

        # Simple GARCH(1,1) parameter estimation
        # Using method of moments for simplicity
        returns_sq = returns ** 2

        # Estimate parameters
        omega = np.var(returns) * 0.1  # Long-run variance weight
        alpha = 0.1  # ARCH parameter (reaction to shocks)
        beta = 0.85  # GARCH parameter (persistence)

        # Ensure stationarity
        if alpha + beta >= 1:
            beta = 0.99 - alpha

        # Initialize variance forecast
        variance_forecast = []
        current_variance = current_vol ** 2

        # Multi-step ahead forecast
        for t in range(periods_ahead):
            if t == 0:
                # One-step ahead
                next_var = omega + alpha * returns[-1]**2 + beta * current_variance
            else:
                # Multi-step (converges to long-run variance)
                long_run_var = omega / (1 - alpha - beta)
                next_var = variance_forecast[-1] + (long_run_var - variance_forecast[-1]) * 0.1

            variance_forecast.append(max(next_var, omega))

        # Convert variance to volatility
        volatility_forecast = np.sqrt(variance_forecast)

        # Calculate volatility percentile
        all_vols = [np.std(returns[i:i+20]) for i in range(len(returns)-20)]
        if all_vols:
            percentile = stats.percentileofscore(all_vols, current_vol)
        else:
            percentile = 50

        # Detect volatility clustering
        recent_vols = [np.std(returns[i:i+5]) for i in range(len(returns)-20, len(returns)-5)]
        is_clustering = False
        if len(recent_vols) > 2:
            vol_changes = np.diff(recent_vols)
            is_clustering = np.all(vol_changes > 0) or np.all(vol_changes < 0)

        return VolatilityForecast(
            current_volatility=current_vol,
            forecast_1h=float(volatility_forecast[0]) if len(volatility_forecast) > 0 else current_vol,
            forecast_4h=float(np.mean(volatility_forecast[:4])) if len(volatility_forecast) >= 4 else current_vol,
            forecast_daily=float(np.mean(volatility_forecast)) if volatility_forecast else current_vol,
            volatility_percentile=percentile,
            is_clustering=is_clustering
        )

    def identify_risk_patterns(self, market_data: Dict[str, np.ndarray]) -> List[str]:
        """
        Pattern recognition for dangerous market setups

        Args:
            market_data: Dictionary with price, volume, spread, etc.

        Returns:
            List of detected pattern names
        """
        detected_patterns = []

        if 'returns' not in market_data or len(market_data['returns']) < 20:
            return detected_patterns

        returns = market_data['returns']

        # Check each pattern in library
        for pattern_name, pattern_config in self.pattern_library.items():
            if self._check_pattern(pattern_name, pattern_config, market_data):
                detected_patterns.append(pattern_name)

        return detected_patterns

    def _check_pattern(self, pattern_name: str, config: Dict,
                      market_data: Dict[str, np.ndarray]) -> bool:
        """Check if a specific pattern is present"""

        lookback = config['lookback']
        threshold = config['threshold']

        if pattern_name == "volatility_expansion":
            if 'returns' in market_data and len(market_data['returns']) > lookback:
                current_vol = np.std(market_data['returns'][-lookback:])
                historical_vol = np.std(market_data['returns'][:-lookback])
                if historical_vol > 0:
                    return current_vol / historical_vol > threshold

        elif pattern_name == "liquidity_drain":
            if 'spreads' in market_data and len(market_data['spreads']) > lookback:
                current_spread = np.mean(market_data['spreads'][-lookback:])
                historical_spread = np.mean(market_data['spreads'][:-lookback])
                if historical_spread > 0:
                    return current_spread / historical_spread > threshold

        elif pattern_name == "momentum_reversal":
            if 'returns' in market_data and len(market_data['returns']) > lookback:
                recent_returns = market_data['returns'][-lookback:]
                if len(recent_returns) > 2:
                    # Check for sharp reversal
                    first_half = np.mean(recent_returns[:len(recent_returns)//2])
                    second_half = np.mean(recent_returns[len(recent_returns)//2:])
                    if first_half != 0:
                        reversal_magnitude = abs((second_half - first_half) / first_half)
                        return reversal_magnitude > threshold

        elif pattern_name == "gap_risk":
            if 'prices' in market_data and len(market_data['prices']) > lookback:
                prices = market_data['prices'][-lookback:]
                if len(prices) > 1:
                    gaps = np.abs(np.diff(prices) / prices[:-1])
                    max_gap = np.max(gaps) if len(gaps) > 0 else 0
                    return max_gap > threshold

        return False

    def detect_anomalies(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Detect anomalous market conditions using Isolation Forest

        Args:
            features: Array of market features

        Returns:
            (is_anomaly, anomaly_score)
        """
        if len(features) < 10:
            return False, 0.0

        # Reshape for sklearn
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)

        # Fit and predict
        try:
            self.anomaly_detector.fit(features[:-1])
            prediction = self.anomaly_detector.predict(features[-1:])
            score = self.anomaly_detector.score_samples(features[-1:])

            is_anomaly = prediction[0] == -1
            anomaly_score = float(1 - (score[0] + 1) / 2)  # Convert to 0-1 range

            return is_anomaly, anomaly_score
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return False, 0.0

    def generate_risk_recommendations(self, risk_metrics: Dict[str, float],
                                     patterns: List[str]) -> List[str]:
        """
        Generate actionable recommendations based on risk analysis

        Args:
            risk_metrics: Current risk metrics
            patterns: Detected risk patterns

        Returns:
            List of recommended actions
        """
        recommendations = []

        # Check risk score
        risk_score = risk_metrics.get('risk_score', 0)
        if risk_score > 80:
            recommendations.append("CRITICAL: Consider closing all positions immediately")
        elif risk_score > 60:
            recommendations.append("WARNING: Reduce position sizes by 50%")
        elif risk_score > 40:
            recommendations.append("CAUTION: Avoid new positions until risk normalizes")

        # Check drawdown proximity
        dd_probability = risk_metrics.get('drawdown_probability', 0)
        if dd_probability > 0.7:
            recommendations.append("HIGH RISK: Drawdown limit breach likely - reduce exposure")
        elif dd_probability > 0.5:
            recommendations.append("MODERATE RISK: Monitor positions closely")

        # Pattern-specific recommendations
        if "volatility_expansion" in patterns:
            recommendations.append("Volatility expanding - widen stop losses or reduce size")

        if "liquidity_drain" in patterns:
            recommendations.append("Liquidity deteriorating - use limit orders only")

        if "momentum_reversal" in patterns:
            recommendations.append("Trend reversal detected - review directional bias")

        if "gap_risk" in patterns:
            recommendations.append("Gap risk elevated - avoid holding overnight")

        # Regime-specific recommendations
        regime = risk_metrics.get('regime_state', 'normal')
        if regime == "crisis":
            recommendations.append("Crisis mode - defensive positioning only")
        elif regime == "volatile":
            recommendations.append("High volatility - reduce leverage and position size")
        elif regime == "trending":
            recommendations.append("Strong trend - consider trend-following strategies")
        elif regime == "quiet":
            recommendations.append("Low volatility - be prepared for volatility expansion")

        return recommendations

    def get_comprehensive_prediction(self, current_metrics: Dict[str, float],
                                    market_data: Dict[str, np.ndarray]) -> RiskPrediction:
        """
        Get all risk predictions in one call

        Args:
            current_metrics: Current risk metrics
            market_data: Market data arrays

        Returns:
            Comprehensive RiskPrediction object
        """
        returns = market_data.get('returns', np.array([]))

        # Drawdown probability
        current_dd = current_metrics.get('current_drawdown', 0)
        max_dd = current_metrics.get('max_drawdown_limit', -0.15)
        dd_probability = self.predict_drawdown_probability(current_dd, max_dd, returns)

        # Volatility forecast
        vol_forecast = self.forecast_volatility_garch(returns)

        # Regime change detection
        volumes = market_data.get('volumes', None)
        new_regime, regime_change_prob = self.detect_regime_change(returns, volumes)

        # Pattern detection
        patterns = self.identify_risk_patterns(market_data)

        # Generate recommendations
        risk_metrics_full = {**current_metrics, 'regime_state': new_regime}
        recommendations = self.generate_risk_recommendations(risk_metrics_full, patterns)

        # Calculate confidence score
        data_quality = min(len(returns) / 100, 1.0)  # More data = higher confidence
        model_agreement = 1.0 - np.std([dd_probability, regime_change_prob, vol_forecast.volatility_percentile/100])
        confidence = (data_quality + model_agreement) / 2

        return RiskPrediction(
            timestamp=datetime.now(),
            drawdown_probability=dd_probability,
            volatility_forecast=vol_forecast.forecast_1h,
            regime_change_probability=regime_change_prob,
            risk_patterns_detected=patterns,
            recommended_actions=recommendations,
            confidence_score=confidence
        )
