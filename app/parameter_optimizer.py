"""
Systematic Parameter Optimization System for Trading Strategy
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

from app.backtester import LottoGridBacktester
from app.config import config
from app.logging_service import get_logger, with_correlation_id

logger = get_logger(__name__)


@dataclass
class ParameterRange:
    """Define parameter range for optimization"""
    
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    discrete_values: Optional[List[Union[float, int, str]]] = None
    parameter_type: str = "continuous"  # continuous, discrete, categorical
    
    def get_values(self, n_samples: int = 10) -> List[Union[float, int, str]]:
        """Get parameter values for optimization"""
        if self.discrete_values is not None:
            return self.discrete_values
        elif self.parameter_type == "continuous":
            if self.step is not None:
                return list(np.arange(self.min_value, self.max_value + self.step, self.step))
            else:
                return list(np.linspace(self.min_value, self.max_value, n_samples))
        else:
            raise ValueError(f"Unknown parameter type: {self.parameter_type}")


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_backtests: int
    optimization_time: float
    convergence_info: Dict[str, Any]
    
    def save(self, filepath: str):
        """Save optimization result to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationResult':
        """Load optimization result from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ParameterOptimizer:
    """Advanced parameter optimization system"""
    
    def __init__(self, backtester: LottoGridBacktester):
        self.backtester = backtester
        self.optimization_cache = {}
        self.results_dir = Path("./optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Optimization settings
        self.max_concurrent_backtests = 4
        self.cache_results = True
        self.early_stopping_patience = 10
        self.early_stopping_threshold = 0.001
        
        # Default parameter ranges
        self.default_parameter_ranges = self._get_default_parameter_ranges()
    
    def _get_default_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Get default parameter ranges for optimization"""
        return {
            # Trading strategy parameters
            "implied_move_multiplier_1": ParameterRange(
                name="implied_move_multiplier_1",
                min_value=1.0,
                max_value=2.0,
                step=0.1
            ),
            "implied_move_multiplier_2": ParameterRange(
                name="implied_move_multiplier_2", 
                min_value=1.2,
                max_value=2.5,
                step=0.1
            ),
            "volatility_threshold": ParameterRange(
                name="volatility_threshold",
                min_value=0.5,
                max_value=0.9,
                step=0.05
            ),
            "profit_target_multiplier": ParameterRange(
                name="profit_target_multiplier",
                min_value=2.0,
                max_value=6.0,
                step=0.5
            ),
            "min_time_between_trades": ParameterRange(
                name="min_time_between_trades",
                min_value=15,
                max_value=120,
                discrete_values=[15, 30, 45, 60, 90, 120],
                parameter_type="discrete"
            ),
            
            # Risk management parameters
            "max_drawdown": ParameterRange(
                name="max_drawdown",
                min_value=500,
                max_value=1500,
                step=100
            ),
            "max_premium_per_strangle": ParameterRange(
                name="max_premium_per_strangle",
                min_value=15,
                max_value=40,
                step=5
            ),
            "max_open_trades": ParameterRange(
                name="max_open_trades",
                min_value=5,
                max_value=25,
                discrete_values=[5, 10, 15, 20, 25],
                parameter_type="discrete"
            ),
            
            # ML model parameters
            "ml_confidence_threshold": ParameterRange(
                name="ml_confidence_threshold",
                min_value=0.4,
                max_value=0.8,
                step=0.05
            ),
            "ensemble_weight_ml": ParameterRange(
                name="ensemble_weight_ml",
                min_value=0.1,
                max_value=0.7,
                step=0.1
            ),
        }
    
    def _calculate_objective_score(self, backtest_results: Dict[str, Any]) -> float:
        """Calculate objective score for optimization"""
        # Multi-objective optimization combining multiple metrics
        metrics = backtest_results.get("summary", {})
        
        # Primary objectives (higher is better)
        total_return = metrics.get("total_return", 0)
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 1)
        
        # Risk metrics (lower is better for max_drawdown)
        max_drawdown = abs(metrics.get("max_drawdown", 0))
        
        # Sharpe ratio (higher is better)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        
        # Weighted combination of metrics
        score = (
            total_return * 0.3 +
            win_rate * 0.2 +
            profit_factor * 0.2 +
            sharpe_ratio * 0.15 +
            max(0, 1000 - max_drawdown) / 1000 * 0.15  # Normalize max_drawdown
        )
        
        return score
    
    @with_correlation_id()
    async def grid_search_optimization(
        self,
        parameter_ranges: Optional[Dict[str, ParameterRange]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_combinations: int = 100
    ) -> OptimizationResult:
        """Perform grid search parameter optimization"""
        logger.info("Starting grid search optimization",
                   optimization_type="grid_search",
                   max_combinations=max_combinations)
        
        # Use default parameter ranges if none provided
        if parameter_ranges is None:
            parameter_ranges = self.default_parameter_ranges
        
        # Generate parameter grid
        param_grid = {}
        for param_name, param_range in parameter_ranges.items():
            param_grid[param_name] = param_range.get_values(10)
        
        # Limit combinations if too many
        all_combinations = list(ParameterGrid(param_grid))
        if len(all_combinations) > max_combinations:
            # Sample random combinations
            np.random.seed(42)  # For reproducibility
            combinations = np.random.choice(
                all_combinations, 
                size=max_combinations, 
                replace=False
            ).tolist()
        else:
            combinations = all_combinations
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        # Run backtests for all combinations
        start_time = datetime.now()
        results = await self._run_parallel_backtests(
            combinations, start_date, end_date
        )
        
        # Find best parameters
        best_score = float('-inf')
        best_params = None
        optimization_history = []
        
        for i, (params, backtest_result) in enumerate(results):
            if backtest_result is None:
                continue
                
            score = self._calculate_objective_score(backtest_result)
            
            optimization_history.append({
                "iteration": i,
                "parameters": params.copy(),
                "score": score,
                "metrics": backtest_result.get("summary", {})
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                logger.info(f"New best score: {best_score:.4f}", 
                           optimization_best_score=best_score,
                           optimization_best_params=best_params)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            total_backtests=len(results),
            optimization_time=optimization_time,
            convergence_info={"method": "grid_search"}
        )
        
        # Save results
        result_file = self.results_dir / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        result.save(result_file)
        
        logger.info("Grid search optimization completed",
                   optimization_total_time=optimization_time,
                   optimization_best_score=best_score,
                   optimization_total_backtests=len(results))
        
        return result
    
    @with_correlation_id()
    async def bayesian_optimization(
        self,
        parameter_ranges: Optional[Dict[str, ParameterRange]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        n_iterations: int = 50,
        n_initial_points: int = 10
    ) -> OptimizationResult:
        """Perform Bayesian optimization (simplified version)"""
        logger.info("Starting Bayesian optimization",
                   optimization_type="bayesian",
                   n_iterations=n_iterations)
        
        # Use default parameter ranges if none provided
        if parameter_ranges is None:
            parameter_ranges = self.default_parameter_ranges
        
        # Get parameter bounds
        param_names = list(parameter_ranges.keys())
        bounds = []
        for param_name in param_names:
            param_range = parameter_ranges[param_name]
            bounds.append((param_range.min_value, param_range.max_value))
        
        # Generate initial random points
        np.random.seed(42)
        initial_points = []
        for _ in range(n_initial_points):
            point = {}
            for param_name, param_range in parameter_ranges.items():
                if param_range.parameter_type == "discrete":
                    point[param_name] = np.random.choice(param_range.discrete_values)
                else:
                    point[param_name] = np.random.uniform(
                        param_range.min_value, param_range.max_value
                    )
            initial_points.append(point)
        
        # Evaluate initial points
        start_time = datetime.now()
        initial_results = await self._run_parallel_backtests(
            initial_points, start_date, end_date
        )
        
        # Track optimization history
        optimization_history = []
        best_score = float('-inf')
        best_params = None
        
        for i, (params, backtest_result) in enumerate(initial_results):
            if backtest_result is None:
                continue
                
            score = self._calculate_objective_score(backtest_result)
            optimization_history.append({
                "iteration": i,
                "parameters": params.copy(),
                "score": score,
                "metrics": backtest_result.get("summary", {})
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        # Simplified Bayesian optimization (random search with adaptive sampling)
        for iteration in range(n_iterations - n_initial_points):
            # Generate new candidate points (simplified - should use Gaussian Process)
            candidate_points = []
            for _ in range(5):  # Generate 5 candidates
                point = {}
                for param_name, param_range in parameter_ranges.items():
                    if param_range.parameter_type == "discrete":
                        point[param_name] = np.random.choice(param_range.discrete_values)
                    else:
                        # Add some gaussian noise around best parameters
                        if best_params and param_name in best_params:
                            center = best_params[param_name]
                            noise = np.random.normal(0, 0.1 * (param_range.max_value - param_range.min_value))
                            value = np.clip(
                                center + noise,
                                param_range.min_value,
                                param_range.max_value
                            )
                        else:
                            value = np.random.uniform(param_range.min_value, param_range.max_value)
                        point[param_name] = value
                candidate_points.append(point)
            
            # Evaluate candidates
            candidate_results = await self._run_parallel_backtests(
                candidate_points, start_date, end_date
            )
            
            # Update best parameters
            for params, backtest_result in candidate_results:
                if backtest_result is None:
                    continue
                    
                score = self._calculate_objective_score(backtest_result)
                optimization_history.append({
                    "iteration": len(optimization_history),
                    "parameters": params.copy(),
                    "score": score,
                    "metrics": backtest_result.get("summary", {})
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"New best score: {best_score:.4f}", 
                               optimization_best_score=best_score,
                               optimization_iteration=iteration)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            total_backtests=len(optimization_history),
            optimization_time=optimization_time,
            convergence_info={"method": "bayesian_simplified"}
        )
        
        # Save results
        result_file = self.results_dir / f"bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        result.save(result_file)
        
        logger.info("Bayesian optimization completed",
                   optimization_total_time=optimization_time,
                   optimization_best_score=best_score,
                   optimization_total_backtests=len(optimization_history))
        
        return result
    
    async def _run_parallel_backtests(
        self,
        parameter_combinations: List[Dict[str, Any]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """Run backtests in parallel for parameter combinations"""
        
        # Create semaphore to limit concurrent backtests
        semaphore = asyncio.Semaphore(self.max_concurrent_backtests)
        
        async def run_single_backtest(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
            async with semaphore:
                try:
                    # Check cache first
                    cache_key = self._get_cache_key(params, start_date, end_date)
                    if self.cache_results and cache_key in self.optimization_cache:
                        return params, self.optimization_cache[cache_key]
                    
                    # Update backtester configuration
                    self._update_backtester_config(params)
                    
                    # Run backtest
                    backtest_result = await self.backtester.run_backtest(
                        start_date=start_date or datetime.now() - timedelta(days=90),
                        end_date=end_date or datetime.now() - timedelta(days=1),
                        symbol="MES"
                    )
                    
                    # Cache result
                    if self.cache_results:
                        self.optimization_cache[cache_key] = backtest_result
                    
                    return params, backtest_result
                    
                except Exception as e:
                    logger.error(f"Error in backtest: {e}", 
                               optimization_error=str(e),
                               optimization_params=params)
                    return params, None
        
        # Run all backtests in parallel
        tasks = [run_single_backtest(params) for params in parameter_combinations]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def _update_backtester_config(self, params: Dict[str, Any]):
        """Update backtester configuration with new parameters"""
        # Update trading configuration
        for param_name, value in params.items():
            if hasattr(config.trading, param_name):
                setattr(config.trading, param_name, value)
            elif hasattr(config.ml, param_name):
                setattr(config.ml, param_name, value)
        
        # Update backtester with new config
        self.backtester.config = config
    
    def _get_cache_key(
        self,
        params: Dict[str, Any],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> str:
        """Generate cache key for parameter combination"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        date_str = f"{start_date}_{end_date}" if start_date and end_date else "default"
        return f"{param_str}_{date_str}"
    
    def analyze_optimization_results(self, result: OptimizationResult) -> Dict[str, Any]:
        """Analyze optimization results and provide insights"""
        history = result.optimization_history
        
        if not history:
            return {"error": "No optimization history available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(history)
        
        # Parameter sensitivity analysis
        parameter_sensitivity = {}
        for param_name in result.best_parameters.keys():
            if param_name in df.columns:
                continue  # Skip if parameter is not in parameters dict
            
            param_values = [entry["parameters"].get(param_name) for entry in history]
            scores = [entry["score"] for entry in history]
            
            # Calculate correlation
            correlation = np.corrcoef(param_values, scores)[0, 1] if len(set(param_values)) > 1 else 0
            parameter_sensitivity[param_name] = {
                "correlation": correlation,
                "best_value": result.best_parameters[param_name],
                "value_range": [min(param_values), max(param_values)]
            }
        
        # Convergence analysis
        scores = [entry["score"] for entry in history]
        best_scores = []
        current_best = float('-inf')
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        # Calculate improvement rate
        if len(best_scores) > 1:
            improvement_rate = (best_scores[-1] - best_scores[0]) / len(best_scores)
        else:
            improvement_rate = 0
        
        analysis = {
            "total_iterations": len(history),
            "best_score": result.best_score,
            "best_parameters": result.best_parameters,
            "parameter_sensitivity": parameter_sensitivity,
            "convergence_info": {
                "improvement_rate": improvement_rate,
                "final_score": best_scores[-1],
                "iterations_to_best": len(best_scores)
            },
            "optimization_time": result.optimization_time,
            "recommendations": self._generate_recommendations(result)
        }
        
        return analysis
    
    def _generate_recommendations(self, result: OptimizationResult) -> List[str]:
        """Generate recommendations based on optimization results"""
        recommendations = []
        
        # Check if optimization improved significantly
        if result.best_score > 0.7:
            recommendations.append("Optimization found good parameters - consider using these settings")
        elif result.best_score > 0.5:
            recommendations.append("Moderate improvement found - may need more iterations or different parameter ranges")
        else:
            recommendations.append("Limited improvement - consider expanding parameter ranges or checking data quality")
        
        # Check optimization time
        if result.optimization_time > 3600:  # 1 hour
            recommendations.append("Optimization took significant time - consider reducing parameter ranges or using fewer iterations")
        
        return recommendations
    
    def save_optimization_report(self, result: OptimizationResult, analysis: Dict[str, Any], filename: str):
        """Save comprehensive optimization report"""
        report = {
            "optimization_result": {
                "best_parameters": result.best_parameters,
                "best_score": result.best_score,
                "total_backtests": result.total_backtests,
                "optimization_time": result.optimization_time
            },
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "max_concurrent_backtests": self.max_concurrent_backtests,
                "cache_enabled": self.cache_results
            }
        }
        
        report_file = self.results_dir / f"{filename}.json"
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report saved to {report_file}")


# Helper function to run optimization
async def optimize_parameters(
    method: str = "grid_search",
    parameter_ranges: Optional[Dict[str, ParameterRange]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    **kwargs
) -> OptimizationResult:
    """
    Run parameter optimization
    
    Args:
        method: Optimization method ('grid_search', 'bayesian')
        parameter_ranges: Dictionary of parameter ranges
        start_date: Start date for backtesting
        end_date: End date for backtesting
        **kwargs: Additional optimization parameters
    
    Returns:
        OptimizationResult object with best parameters and analysis
    """
    # Initialize backtester
    backtester = LottoGridBacktester(database_url=config.database.url)
    
    # Create optimizer
    optimizer = ParameterOptimizer(backtester)
    
    # Run optimization
    if method == "grid_search":
        result = await optimizer.grid_search_optimization(
            parameter_ranges=parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
    elif method == "bayesian":
        result = await optimizer.bayesian_optimization(
            parameter_ranges=parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Analyze results
    analysis = optimizer.analyze_optimization_results(result)
    
    # Save report
    report_name = f"optimization_report_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    optimizer.save_optimization_report(result, analysis, report_name)
    
    return result