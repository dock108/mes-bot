"""
Tests for parameter optimization system
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.backtester import LottoGridBacktester
from app.parameter_optimizer import (
    OptimizationResult,
    ParameterOptimizer,
    ParameterRange,
    optimize_parameters,
)


class TestParameterRange:
    """Test parameter range functionality"""

    def test_continuous_parameter_range(self):
        """Test continuous parameter range"""
        param_range = ParameterRange(name="test_param", min_value=1.0, max_value=2.0, step=0.1)

        values = param_range.get_values()
        expected_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        assert len(values) == len(expected_values)
        assert values[0] == 1.0
        assert abs(values[-1] - 2.0) < 1e-10  # Account for floating point precision

    def test_discrete_parameter_range(self):
        """Test discrete parameter range"""
        param_range = ParameterRange(
            name="test_param",
            min_value=0,
            max_value=100,
            discrete_values=[10, 20, 30, 40],
            parameter_type="discrete",
        )

        values = param_range.get_values()
        assert values == [10, 20, 30, 40]

    def test_continuous_parameter_range_with_samples(self):
        """Test continuous parameter range with specific number of samples"""
        param_range = ParameterRange(name="test_param", min_value=0.0, max_value=1.0)

        values = param_range.get_values(n_samples=5)
        assert len(values) == 5
        assert values[0] == 0.0
        assert values[-1] == 1.0


class TestOptimizationResult:
    """Test optimization result functionality"""

    def test_optimization_result_creation(self):
        """Test creating optimization result"""
        result = OptimizationResult(
            best_parameters={"param1": 1.5, "param2": 2.0},
            best_score=0.85,
            optimization_history=[],
            total_backtests=100,
            optimization_time=300.0,
            convergence_info={"method": "grid_search"},
        )

        assert result.best_parameters["param1"] == 1.5
        assert result.best_score == 0.85
        assert result.total_backtests == 100

    def test_optimization_result_save_load(self, tmp_path):
        """Test saving and loading optimization result"""
        result = OptimizationResult(
            best_parameters={"param1": 1.5},
            best_score=0.85,
            optimization_history=[],
            total_backtests=50,
            optimization_time=150.0,
            convergence_info={"method": "test"},
        )

        # Save result
        filepath = tmp_path / "test_result.pkl"
        result.save(str(filepath))

        # Load result
        loaded_result = OptimizationResult.load(str(filepath))

        assert loaded_result.best_parameters == result.best_parameters
        assert loaded_result.best_score == result.best_score
        assert loaded_result.total_backtests == result.total_backtests


class TestParameterOptimizer:
    """Test parameter optimizer functionality"""

    @pytest.fixture
    def mock_backtester(self):
        """Create mock backtester"""
        backtester = MagicMock(spec=LottoGridBacktester)
        backtester.run_backtest = AsyncMock()
        return backtester

    @pytest.fixture
    def optimizer(self, mock_backtester):
        """Create parameter optimizer with mock backtester"""
        optimizer = ParameterOptimizer(mock_backtester)
        optimizer.max_concurrent_backtests = 2  # Limit for testing
        return optimizer

    def test_optimizer_initialization(self, mock_backtester):
        """Test optimizer initialization"""
        optimizer = ParameterOptimizer(mock_backtester)

        assert optimizer.backtester == mock_backtester
        assert optimizer.max_concurrent_backtests == 4
        assert optimizer.cache_results is True
        assert len(optimizer.default_parameter_ranges) > 0

    def test_default_parameter_ranges(self, optimizer):
        """Test default parameter ranges"""
        ranges = optimizer.default_parameter_ranges

        # Check that key parameters are included
        assert "implied_move_multiplier_1" in ranges
        assert "volatility_threshold" in ranges
        assert "profit_target_multiplier" in ranges
        assert "ml_confidence_threshold" in ranges

        # Check parameter range properties
        im_range = ranges["implied_move_multiplier_1"]
        assert im_range.min_value == 1.0
        assert im_range.max_value == 2.0
        assert im_range.step == 0.1

    def test_calculate_objective_score(self, optimizer):
        """Test objective score calculation"""
        # Mock backtest results
        backtest_results = {
            "summary": {
                "total_return": 1000.0,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "max_drawdown": 500.0,
                "sharpe_ratio": 1.2,
            }
        }

        score = optimizer._calculate_objective_score(backtest_results)

        # Score should be positive and reasonable
        assert score > 0
        assert score < 10  # Reasonable upper bound

    def test_calculate_objective_score_empty_results(self, optimizer):
        """Test objective score with empty results"""
        backtest_results = {}
        score = optimizer._calculate_objective_score(backtest_results)

        # Should handle empty results gracefully
        assert score >= 0

    def test_update_backtester_config(self, optimizer):
        """Test updating backtester configuration"""
        params = {
            "implied_move_multiplier_1": 1.5,
            "volatility_threshold": 0.75,
            "ml_confidence_threshold": 0.6,
        }

        optimizer._update_backtester_config(params)

        # Should update the backtester config
        assert optimizer.backtester.config is not None

    def test_get_cache_key(self, optimizer):
        """Test cache key generation"""
        params = {"param1": 1.5, "param2": 2.0}
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        cache_key = optimizer._get_cache_key(params, start_date, end_date)

        assert isinstance(cache_key, str)
        assert "param1=1.5" in cache_key
        assert "param2=2.0" in cache_key

    @pytest.mark.asyncio
    async def test_run_parallel_backtests(self, optimizer):
        """Test running parallel backtests"""
        # Mock successful backtest results
        mock_result = {
            "summary": {
                "total_return": 500.0,
                "win_rate": 0.55,
                "profit_factor": 1.3,
                "max_drawdown": 300.0,
                "sharpe_ratio": 0.8,
            }
        }

        optimizer.backtester.run_backtest.return_value = mock_result

        # Test parameters
        param_combinations = [{"param1": 1.0, "param2": 2.0}, {"param1": 1.5, "param2": 2.5}]

        results = await optimizer._run_parallel_backtests(param_combinations)

        assert len(results) == 2
        assert results[0][0] == param_combinations[0]
        assert results[0][1] == mock_result
        assert results[1][0] == param_combinations[1]
        assert results[1][1] == mock_result

    @pytest.mark.asyncio
    async def test_run_parallel_backtests_with_error(self, optimizer):
        """Test parallel backtests with error handling"""
        # Mock backtest that raises an error
        optimizer.backtester.run_backtest.side_effect = Exception("Backtest failed")

        param_combinations = [{"param1": 1.0}]

        results = await optimizer._run_parallel_backtests(param_combinations)

        assert len(results) == 1
        assert results[0][0] == param_combinations[0]
        assert results[0][1] is None  # Should be None on error

    @pytest.mark.asyncio
    async def test_grid_search_optimization(self, optimizer):
        """Test grid search optimization"""
        # Mock backtest results
        mock_result = {
            "summary": {
                "total_return": 600.0,
                "win_rate": 0.58,
                "profit_factor": 1.4,
                "max_drawdown": 400.0,
                "sharpe_ratio": 0.9,
            }
        }

        optimizer.backtester.run_backtest.return_value = mock_result

        # Simple parameter ranges for testing
        parameter_ranges = {
            "param1": ParameterRange(name="param1", min_value=1.0, max_value=2.0, step=0.5),
            "param2": ParameterRange(
                name="param2", discrete_values=[10, 20], parameter_type="discrete"
            ),
        }

        result = await optimizer.grid_search_optimization(
            parameter_ranges=parameter_ranges, max_combinations=10
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_parameters is not None
        assert result.best_score > 0
        assert result.total_backtests > 0
        assert result.optimization_time > 0

    @pytest.mark.asyncio
    async def test_bayesian_optimization(self, optimizer):
        """Test Bayesian optimization"""
        # Mock backtest results
        mock_result = {
            "summary": {
                "total_return": 700.0,
                "win_rate": 0.62,
                "profit_factor": 1.6,
                "max_drawdown": 350.0,
                "sharpe_ratio": 1.1,
            }
        }

        optimizer.backtester.run_backtest.return_value = mock_result

        # Simple parameter ranges for testing
        parameter_ranges = {"param1": ParameterRange(name="param1", min_value=1.0, max_value=2.0)}

        result = await optimizer.bayesian_optimization(
            parameter_ranges=parameter_ranges, n_iterations=10, n_initial_points=5
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_parameters is not None
        assert result.best_score > 0
        assert result.total_backtests > 0
        assert result.convergence_info["method"] == "bayesian_simplified"

    def test_analyze_optimization_results(self, optimizer):
        """Test optimization results analysis"""
        # Create mock optimization result
        result = OptimizationResult(
            best_parameters={"param1": 1.5, "param2": 20},
            best_score=0.85,
            optimization_history=[
                {
                    "iteration": 0,
                    "parameters": {"param1": 1.0, "param2": 10},
                    "score": 0.7,
                    "metrics": {"total_return": 500},
                },
                {
                    "iteration": 1,
                    "parameters": {"param1": 1.5, "param2": 20},
                    "score": 0.85,
                    "metrics": {"total_return": 700},
                },
            ],
            total_backtests=2,
            optimization_time=120.0,
            convergence_info={"method": "grid_search"},
        )

        analysis = optimizer.analyze_optimization_results(result)

        assert "total_iterations" in analysis
        assert "best_score" in analysis
        assert "best_parameters" in analysis
        assert "convergence_info" in analysis
        assert "recommendations" in analysis

        assert analysis["total_iterations"] == 2
        assert analysis["best_score"] == 0.85
        assert analysis["best_parameters"] == {"param1": 1.5, "param2": 20}

    def test_generate_recommendations(self, optimizer):
        """Test recommendation generation"""
        # High score result
        high_score_result = OptimizationResult(
            best_parameters={},
            best_score=0.8,
            optimization_history=[],
            total_backtests=10,
            optimization_time=100.0,
            convergence_info={},
        )

        recommendations = optimizer._generate_recommendations(high_score_result)
        assert len(recommendations) > 0
        assert any("good parameters" in rec for rec in recommendations)

        # Low score result
        low_score_result = OptimizationResult(
            best_parameters={},
            best_score=0.3,
            optimization_history=[],
            total_backtests=10,
            optimization_time=100.0,
            convergence_info={},
        )

        recommendations = optimizer._generate_recommendations(low_score_result)
        assert len(recommendations) > 0
        assert any("Limited improvement" in rec for rec in recommendations)

    def test_save_optimization_report(self, optimizer, tmp_path):
        """Test saving optimization report"""
        result = OptimizationResult(
            best_parameters={"param1": 1.5},
            best_score=0.75,
            optimization_history=[],
            total_backtests=20,
            optimization_time=200.0,
            convergence_info={"method": "test"},
        )

        analysis = {"test": "analysis"}

        # Temporarily change results directory
        optimizer.results_dir = tmp_path

        optimizer.save_optimization_report(result, analysis, "test_report")

        # Check that file was created
        report_file = tmp_path / "test_report.json"
        assert report_file.exists()

        # Check file content
        import json

        with open(report_file, "r") as f:
            report_data = json.load(f)

        assert "optimization_result" in report_data
        assert "analysis" in report_data
        assert "timestamp" in report_data
        assert report_data["optimization_result"]["best_score"] == 0.75


class TestOptimizeParametersFunction:
    """Test the optimize_parameters helper function"""

    @pytest.mark.asyncio
    async def test_optimize_parameters_grid_search(self):
        """Test optimize_parameters function with grid search"""
        with patch("app.parameter_optimizer.LottoGridBacktester") as mock_backtester_class:
            mock_backtester = MagicMock()
            mock_backtester.run_backtest = AsyncMock(
                return_value={
                    "summary": {
                        "total_return": 500.0,
                        "win_rate": 0.55,
                        "profit_factor": 1.3,
                        "max_drawdown": 300.0,
                        "sharpe_ratio": 0.8,
                    }
                }
            )
            mock_backtester_class.return_value = mock_backtester

            # Simple parameter ranges
            parameter_ranges = {
                "param1": ParameterRange(name="param1", min_value=1.0, max_value=1.5, step=0.5)
            }

            result = await optimize_parameters(
                method="grid_search", parameter_ranges=parameter_ranges, max_combinations=5
            )

            assert isinstance(result, OptimizationResult)
            assert result.best_parameters is not None
            assert result.best_score > 0

    @pytest.mark.asyncio
    async def test_optimize_parameters_bayesian(self):
        """Test optimize_parameters function with Bayesian optimization"""
        with patch("app.parameter_optimizer.LottoGridBacktester") as mock_backtester_class:
            mock_backtester = MagicMock()
            mock_backtester.run_backtest = AsyncMock(
                return_value={
                    "summary": {
                        "total_return": 600.0,
                        "win_rate": 0.58,
                        "profit_factor": 1.4,
                        "max_drawdown": 400.0,
                        "sharpe_ratio": 0.9,
                    }
                }
            )
            mock_backtester_class.return_value = mock_backtester

            parameter_ranges = {
                "param1": ParameterRange(name="param1", min_value=1.0, max_value=2.0)
            }

            result = await optimize_parameters(
                method="bayesian",
                parameter_ranges=parameter_ranges,
                n_iterations=5,
                n_initial_points=3,
            )

            assert isinstance(result, OptimizationResult)
            assert result.best_parameters is not None
            assert result.best_score > 0

    @pytest.mark.asyncio
    async def test_optimize_parameters_invalid_method(self):
        """Test optimize_parameters with invalid method"""
        with pytest.raises(ValueError, match="Unknown optimization method"):
            await optimize_parameters(method="invalid_method")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
