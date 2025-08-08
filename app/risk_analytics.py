"""
Risk Analytics Engine for Real-Time Risk Monitoring and Analysis

This module provides comprehensive risk metrics calculation including VaR, CVaR,
Greeks exposure, correlation analysis, and stress testing capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
    timestamp: datetime
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    portfolio_theta: float
    correlation_score: float
    risk_score: int  # 0-100 scale
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    kelly_fraction: float
    regime_state: str


@dataclass
class StressTestResult:
    """Results from stress testing scenario"""
    scenario_name: str
    probability: float
    expected_loss: float
    max_loss: float
    recovery_hours: float
    affected_positions: List[str]


class RiskAnalyticsEngine:
    """
    Calculates and tracks comprehensive risk metrics in real-time
    """

    def __init__(self, lookback_days: int = 30, confidence_level: float = 0.95):
        self.lookback_days = lookback_days
        self.confidence_level = confidence_level
        self.historical_returns = []
        self.current_positions = []
        self.risk_free_rate = 0.05  # 5% annual risk-free rate

    def calculate_var(self, returns: np.ndarray, method: str = "historical") -> float:
        """
        Calculate Value at Risk using specified method

        Args:
            returns: Array of historical returns
            method: "historical", "parametric", or "monte_carlo"

        Returns:
            VaR at specified confidence level
        """
        if len(returns) == 0:
            return 0.0

        if method == "historical":
            # Historical simulation
            var = np.percentile(returns, (1 - self.confidence_level) * 100)

        elif method == "parametric":
            # Parametric (variance-covariance) method
            mean = np.mean(returns)
            std = np.std(returns)
            var = mean - std * stats.norm.ppf(self.confidence_level)

        elif method == "monte_carlo":
            # Monte Carlo simulation
            mean = np.mean(returns)
            std = np.std(returns)
            simulations = np.random.normal(mean, std, 10000)
            var = np.percentile(simulations, (1 - self.confidence_level) * 100)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return abs(var)

    def calculate_cvar(self, returns: np.ndarray) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)

        Args:
            returns: Array of historical returns

        Returns:
            CVaR at specified confidence level
        """
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns)
        # Get returns worse than VaR
        tail_returns = returns[returns <= -var]

        if len(tail_returns) == 0:
            return var

        return abs(np.mean(tail_returns))

    def calculate_greeks_exposure(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate aggregate Greeks exposure across all positions

        Args:
            positions: List of position dictionaries with Greeks

        Returns:
            Dictionary of aggregated Greeks
        """
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }

        for position in positions:
            quantity = position.get('quantity', 0)
            for greek in total_greeks.keys():
                greek_value = position.get(greek, 0)
                total_greeks[greek] += quantity * greek_value

        return total_greeks

    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Calculate correlation matrix for positions

        Args:
            returns_df: DataFrame with returns for each position

        Returns:
            Correlation matrix and diversification score
        """
        if returns_df.empty or len(returns_df.columns) < 2:
            return pd.DataFrame(), 1.0

        correlation_matrix = returns_df.corr()

        # Calculate diversification score (0 = perfectly correlated, 1 = perfectly diversified)
        n = len(correlation_matrix)
        if n <= 1:
            return correlation_matrix, 1.0

        avg_correlation = (correlation_matrix.sum().sum() - n) / (n * (n - 1))
        diversification_score = 1 - abs(avg_correlation)

        return correlation_matrix, diversification_score

    def stress_test_scenarios(self, positions: List[Dict],
                            scenarios: Optional[Dict[str, Dict]] = None) -> List[StressTestResult]:
        """
        Run predefined stress scenarios

        Args:
            positions: Current positions
            scenarios: Custom scenarios or use defaults

        Returns:
            List of stress test results
        """
        if scenarios is None:
            scenarios = self._get_default_scenarios()

        results = []

        for scenario_name, params in scenarios.items():
            result = self._run_single_scenario(positions, scenario_name, params)
            results.append(result)

        return results

    def _get_default_scenarios(self) -> Dict[str, Dict]:
        """Get default stress test scenarios"""
        return {
            "flash_crash": {
                "price_shock": -0.05,  # 5% instant drop
                "iv_shock": 0.50,  # 50% IV increase
                "probability": 0.02,  # 2% probability
                "recovery_hours": 24
            },
            "volatility_spike": {
                "price_shock": 0,
                "iv_shock": 1.0,  # 100% IV increase
                "probability": 0.05,
                "recovery_hours": 8
            },
            "liquidity_crisis": {
                "price_shock": -0.02,
                "iv_shock": 0.30,
                "bid_ask_widening": 3.0,  # 3x wider spreads
                "probability": 0.03,
                "recovery_hours": 48
            },
            "fed_announcement": {
                "price_shock": 0.03,  # Could go either way
                "iv_shock": -0.20,  # IV crush
                "probability": 0.08,
                "recovery_hours": 2
            }
        }

    def _run_single_scenario(self, positions: List[Dict],
                            scenario_name: str, params: Dict) -> StressTestResult:
        """Run a single stress test scenario"""
        total_loss = 0
        max_loss = 0
        affected = []

        for position in positions:
            position_loss = self._calculate_position_loss(position, params)
            total_loss += position_loss
            max_loss = min(max_loss, position_loss)  # More negative is worse

            if abs(position_loss) > 0:
                affected.append(position.get('symbol', 'Unknown'))

        expected_loss = total_loss * params.get('probability', 0.01)

        return StressTestResult(
            scenario_name=scenario_name,
            probability=params.get('probability', 0.01),
            expected_loss=expected_loss,
            max_loss=max_loss,
            recovery_hours=params.get('recovery_hours', 24),
            affected_positions=affected
        )

    def _calculate_position_loss(self, position: Dict, scenario: Dict) -> float:
        """Calculate loss for a single position under stress scenario"""
        current_value = position.get('market_value', 0)

        # Apply price shock
        price_shock = scenario.get('price_shock', 0)
        delta = position.get('delta', 0)
        gamma = position.get('gamma', 0)

        # First-order approximation with gamma adjustment
        price_impact = current_value * (delta * price_shock + 0.5 * gamma * price_shock ** 2)

        # Apply IV shock
        iv_shock = scenario.get('iv_shock', 0)
        vega = position.get('vega', 0)
        iv_impact = vega * iv_shock * 100  # Vega is per 1% IV change

        # Total loss
        total_impact = price_impact + iv_impact

        return total_impact

    def calculate_kelly_criterion(self, win_rate: float, avg_win: float,
                                 avg_loss: float) -> float:
        """
        Calculate optimal position sizing using Kelly Criterion

        Args:
            win_rate: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)

        Returns:
            Kelly fraction (percentage of capital to risk)
        """
        if avg_loss == 0 or win_rate == 0 or win_rate == 1:
            return 0.0

        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        q = 1 - win_rate
        b = avg_win / avg_loss

        kelly = (win_rate * b - q) / b

        # Apply Kelly fraction limit (never more than 25%)
        return max(0, min(kelly, 0.25))

    def calculate_risk_adjusted_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk-adjusted return metrics

        Args:
            returns: Array of returns

        Returns:
            Dictionary with Sharpe, Sortino, and Calmar ratios
        """
        if len(returns) == 0:
            return {'sharpe': 0, 'sortino': 0, 'calmar': 0}

        # Annualized return and volatility
        periods_per_year = 252  # Trading days
        mean_return = np.mean(returns) * periods_per_year
        std_return = np.std(returns) * np.sqrt(periods_per_year)

        # Sharpe Ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return if std_return > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino = (mean_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Calmar Ratio (return / max drawdown)
        max_dd = self.calculate_max_drawdown(returns)
        calmar = mean_return / abs(max_dd) if max_dd != 0 else 0

        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar
        }

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from returns series

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown (negative value)
        """
        if len(returns) == 0:
            return 0.0

        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max

        return float(np.min(drawdown))

    def calculate_risk_score(self, metrics: Dict[str, Any]) -> int:
        """
        Calculate overall risk score from 0-100

        Args:
            metrics: Dictionary of risk metrics

        Returns:
            Risk score (0=low risk, 100=extreme risk)
        """
        score = 0
        weights = {
            'var_breach': 30,
            'drawdown': 25,
            'correlation': 20,
            'volatility': 15,
            'exposure': 10
        }

        # VaR breach risk
        current_loss = metrics.get('current_pnl', 0)
        var = metrics.get('var_95', 1)
        var_ratio = abs(current_loss / var) if var != 0 else 0
        score += weights['var_breach'] * min(var_ratio, 1)

        # Drawdown risk
        max_dd = metrics.get('max_drawdown_limit', 1)
        current_dd = abs(metrics.get('current_drawdown', 0))
        dd_ratio = current_dd / max_dd if max_dd != 0 else 0
        score += weights['drawdown'] * min(dd_ratio, 1)

        # Correlation risk
        correlation_score = 1 - metrics.get('diversification_score', 1)
        score += weights['correlation'] * correlation_score

        # Volatility risk
        current_vol = metrics.get('current_volatility', 0)
        normal_vol = metrics.get('average_volatility', 1)
        vol_ratio = current_vol / normal_vol if normal_vol != 0 else 0
        score += weights['volatility'] * min(vol_ratio - 1, 1) if vol_ratio > 1 else 0

        # Exposure risk
        exposure_ratio = metrics.get('exposure_ratio', 0)
        score += weights['exposure'] * min(exposure_ratio, 1)

        return int(min(max(score, 0), 100))

    def detect_regime(self, returns: np.ndarray, lookback: int = 20) -> str:
        """
        Detect current market regime

        Args:
            returns: Recent returns
            lookback: Number of periods to analyze

        Returns:
            Regime state: "normal", "trending", "volatile", "crisis"
        """
        if len(returns) < lookback:
            return "normal"

        recent_returns = returns[-lookback:]

        # Calculate regime indicators
        volatility = np.std(recent_returns)
        trend = np.mean(recent_returns)
        skewness = stats.skew(recent_returns)
        kurtosis = stats.kurtosis(recent_returns)

        # Historical comparison
        if len(returns) > lookback * 2:
            historical_vol = np.std(returns[:-lookback])
            vol_ratio = volatility / historical_vol if historical_vol > 0 else 1
        else:
            vol_ratio = 1

        # Regime classification
        if vol_ratio > 2 and kurtosis > 3:
            return "crisis"
        elif vol_ratio > 1.5:
            return "volatile"
        elif abs(trend) > 2 * volatility:
            return "trending"
        else:
            return "normal"

    def get_comprehensive_metrics(self, positions: List[Dict],
                                 historical_data: pd.DataFrame) -> RiskMetrics:
        """
        Get all risk metrics in one call

        Args:
            positions: Current positions
            historical_data: Historical price/return data

        Returns:
            Comprehensive RiskMetrics object
        """
        # Calculate returns if needed
        if 'returns' not in historical_data.columns:
            historical_data['returns'] = historical_data['close'].pct_change()

        returns = historical_data['returns'].dropna().values

        # Calculate all metrics
        var_95 = self.calculate_var(returns)
        cvar_95 = self.calculate_cvar(returns)

        greeks = self.calculate_greeks_exposure(positions)

        # Position returns for correlation
        position_returns = pd.DataFrame()
        if len(positions) > 1:
            # This would need actual position-level returns
            correlation_matrix, correlation_score = self.calculate_correlation_matrix(position_returns)
        else:
            correlation_score = 1.0

        risk_adjusted = self.calculate_risk_adjusted_returns(returns)
        max_drawdown = self.calculate_max_drawdown(returns)

        # Kelly criterion (would need actual win/loss stats)
        kelly_fraction = 0.02  # Conservative default

        # Risk score calculation
        risk_metrics_dict = {
            'current_pnl': sum(p.get('unrealized_pnl', 0) for p in positions),
            'var_95': var_95,
            'current_drawdown': max_drawdown,
            'max_drawdown_limit': 0.15,  # 15% limit
            'diversification_score': correlation_score,
            'current_volatility': np.std(returns[-20:]) if len(returns) > 20 else 0,
            'average_volatility': np.std(returns),
            'exposure_ratio': len(positions) / 15  # Max 15 positions
        }
        risk_score = self.calculate_risk_score(risk_metrics_dict)

        regime_state = self.detect_regime(returns)

        return RiskMetrics(
            timestamp=datetime.now(),
            var_95=var_95,
            cvar_95=cvar_95,
            portfolio_delta=greeks['delta'],
            portfolio_gamma=greeks['gamma'],
            portfolio_vega=greeks['vega'],
            portfolio_theta=greeks['theta'],
            correlation_score=correlation_score,
            risk_score=risk_score,
            max_drawdown=max_drawdown,
            sharpe_ratio=risk_adjusted['sharpe'],
            sortino_ratio=risk_adjusted['sortino'],
            calmar_ratio=risk_adjusted['calmar'],
            kelly_fraction=kelly_fraction,
            regime_state=regime_state
        )
