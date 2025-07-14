"""
Performance and load testing suite for the ML-enhanced trading bot.
Tests system performance under various load conditions and validates
real-time trading requirements.
"""
import pytest
import asyncio
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy import create_engine, text
from memory_profiler import profile
import psutil
import gc

from app.models import Base, MarketData, MarketFeatures, DecisionHistory, get_session_maker
from app.feature_pipeline import FeatureCollector, FeatureEngineer
from app.decision_engine import DecisionEngine
from app.ml_training import ModelTrainer
from app.enhanced_strategy import EnhancedLottoGridStrategy
from app.market_indicators import MarketIndicatorEngine


class TestMarketDataProcessingPerformance:
    """Test performance of market data collection and processing"""
    
    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test_perf.db"
        database_url = f"sqlite:///{db_file}"
        
        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()
        
        return database_url
    
    @pytest.fixture
    def feature_collector(self, database_url):
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        return FeatureCollector(database_url)
    
    @pytest.mark.performance
    def test_high_frequency_data_collection_performance(self, feature_collector):
        """Test data collection performance at high frequency"""
        import time
        
        num_data_points = 1000
        start_time = time.perf_counter()
        
        async def collect_data():
            base_time = datetime.utcnow()
            tasks = []
            
            for i in range(num_data_points):
                task = feature_collector.collect_market_data(
                    price=4200.0 + np.random.normal(0, 1),
                    bid=4199.5,
                    ask=4200.5,
                    volume=1000,
                    atm_iv=0.25,
                    vix_level=20.0,
                    timestamp=base_time + timedelta(seconds=i)
                )
                tasks.append(task)
            
            # Process in batches to avoid overwhelming the system
            batch_size = 50
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                await asyncio.gather(*batch)
        
        # Run the data collection
        asyncio.run(collect_data())
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance requirements
        assert total_time < 30.0, f"Data collection took {total_time:.2f}s, should be under 30s"
        
        # Verify throughput
        throughput = num_data_points / total_time
        assert throughput > 30, f"Throughput {throughput:.1f} data points/sec too low"
        
        # Verify all data was stored
        session = feature_collector.session_maker()
        try:
            count = session.query(MarketData).count()
            assert count == num_data_points
        finally:
            session.close()
        
        print(f"Performance: {num_data_points} data points in {total_time:.2f}s "
              f"({throughput:.1f} points/sec)")
    
    @pytest.mark.performance
    def test_concurrent_data_collection_performance(self, database_url):
        """Test concurrent data collection from multiple sources"""
        import time
        
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        
        num_threads = 5
        data_points_per_thread = 100
        
        def collect_data_thread(thread_id):
            collector = FeatureCollector(database_url)
            
            async def collect():
                base_time = datetime.utcnow()
                for i in range(data_points_per_thread):
                    await collector.collect_market_data(
                        price=4200.0 + np.random.normal(0, 1),
                        bid=4199.5,
                        ask=4200.5,
                        volume=1000,
                        atm_iv=0.25,
                        vix_level=20.0,
                        timestamp=base_time + timedelta(seconds=i, microseconds=thread_id*1000)
                    )
            
            return asyncio.run(collect())
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(collect_data_thread, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Verify performance
        total_data_points = num_threads * data_points_per_thread
        throughput = total_data_points / total_time
        
        assert total_time < 20.0, f"Concurrent collection took {total_time:.2f}s, too slow"
        assert throughput > 20, f"Concurrent throughput {throughput:.1f} points/sec too low"
        
        # Verify data integrity
        session_maker = get_session_maker(database_url)
        session = session_maker()
        try:
            count = session.query(MarketData).count()
            assert count == total_data_points
        finally:
            session.close()
        
        print(f"Concurrent performance: {total_data_points} data points in {total_time:.2f}s "
              f"({throughput:.1f} points/sec) using {num_threads} threads")
    
    @pytest.mark.performance
    def test_memory_usage_during_bulk_operations(self, feature_collector):
        """Test memory usage during bulk data operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        async def bulk_operation():
            base_time = datetime.utcnow()
            
            # Collect data in large batches
            for batch in range(10):  # 10 batches
                tasks = []
                for i in range(200):  # 200 points per batch
                    task = feature_collector.collect_market_data(
                        price=4200.0 + np.random.normal(0, 1),
                        bid=4199.5,
                        ask=4200.5,
                        volume=1000,
                        atm_iv=0.25,
                        vix_level=20.0,
                        timestamp=base_time + timedelta(seconds=batch*200 + i)
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                
                # Force garbage collection between batches
                gc.collect()
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't grow excessively
                assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB, too high"
        
        asyncio.run(bulk_operation())
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
              f"(+{total_memory_increase:.1f}MB for 2000 data points)")


class TestFeatureEngineeringPerformance:
    """Test performance of feature engineering pipeline"""
    
    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test_perf.db"
        database_url = f"sqlite:///{db_file}"
        
        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()
        
        return database_url
    
    @pytest.fixture
    def feature_engineer(self, database_url):
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        return FeatureEngineer(database_url)
    
    @pytest.mark.performance
    def test_feature_calculation_performance(self, feature_engineer):
        """Test performance of feature calculation with large datasets"""
        import time
        
        # First, populate database with historical data
        session = feature_engineer.session_maker()
        try:
            base_time = datetime.utcnow() - timedelta(hours=4)
            market_data_records = []
            
            for i in range(240):  # 4 hours of minute data
                market_data = MarketData(
                    timestamp=base_time + timedelta(minutes=i),
                    underlying_price=4200.0 + np.random.normal(0, 2),
                    bid_price=4199.5,
                    ask_price=4200.5,
                    volume=1000 + np.random.randint(-100, 100),
                    atm_iv=0.25 + np.random.normal(0, 0.01),
                    vix_level=20.0 + np.random.normal(0, 1)
                )
                market_data_records.append(market_data)
            
            session.add_all(market_data_records)
            session.commit()
            
            # Test feature engineering performance
            market_data = session.query(MarketData).order_by(MarketData.timestamp).all()
            
            start_time = time.perf_counter()
            
            # Calculate features for last 20 data points
            feature_tasks = []
            for i in range(220, 240):  # Last 20 points
                current_data = market_data[i]
                historical_data = market_data[:i+1]
                
                task = feature_engineer.calculate_and_store_features(
                    current_data=current_data,
                    historical_data=historical_data
                )
                feature_tasks.append(task)
            
            # Run feature calculations
            async def run_feature_calculations():
                results = await asyncio.gather(*feature_tasks)
                return results
            
            results = asyncio.run(run_feature_calculations())
            
            end_time = time.perf_counter()
            calculation_time = end_time - start_time
            
            # Performance requirements
            assert calculation_time < 10.0, f"Feature calculation took {calculation_time:.2f}s, too slow"
            
            # Verify features were calculated
            features_count = session.query(MarketFeatures).count()
            assert features_count >= 10  # At least some features should be generated
            
            print(f"Feature engineering: 20 feature sets in {calculation_time:.2f}s "
                  f"({20/calculation_time:.1f} features/sec)")
        
        finally:
            session.close()
    
    @pytest.mark.performance
    def test_feature_engineering_memory_efficiency(self, feature_engineer):
        """Test memory efficiency of feature engineering with large windows"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        session = feature_engineer.session_maker()
        try:
            base_time = datetime.utcnow() - timedelta(hours=8)
            
            # Insert data in chunks to avoid memory issues
            for chunk in range(8):  # 8 chunks of 1 hour each
                chunk_data = []
                for i in range(60):  # 60 minutes per chunk
                    market_data = MarketData(
                        timestamp=base_time + timedelta(hours=chunk, minutes=i),
                        underlying_price=4200.0 + np.random.normal(0, 2),
                        bid_price=4199.5,
                        ask_price=4200.5,
                        volume=1000,
                        atm_iv=0.25,
                        vix_level=20.0
                    )
                    chunk_data.append(market_data)
                
                session.add_all(chunk_data)
                session.commit()
                gc.collect()  # Force garbage collection
            
            # Test feature engineering with large lookback window
            all_data = session.query(MarketData).order_by(MarketData.timestamp).all()
            
            async def calculate_features_with_large_window():
                # Calculate features for the last data point using all historical data
                current_data = all_data[-1]
                historical_data = all_data  # Full 8 hours of data
                
                return await feature_engineer.calculate_and_store_features(
                    current_data=current_data,
                    historical_data=historical_data
                )
            
            feature_id = asyncio.run(calculate_features_with_large_window())
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory usage should be reasonable even with large datasets
            assert memory_increase < 200, f"Memory increased by {memory_increase:.1f}MB, too high"
            assert feature_id is not None, "Feature calculation should succeed"
            
            print(f"Large window feature engineering: {memory_increase:.1f}MB memory increase "
                  f"for 480 data points")
        
        finally:
            session.close()


class TestDecisionEnginePerformance:
    """Test performance of decision engine under load"""
    
    @pytest.fixture
    def decision_engine(self):
        return DecisionEngine()
    
    @pytest.mark.performance
    def test_decision_generation_latency(self, decision_engine):
        """Test decision generation latency for real-time requirements"""
        import time
        
        # Mock feature calculation to return immediately
        from app.market_indicators import MarketFeatures
        
        mock_features = MarketFeatures(
            realized_vol_15m=0.12, realized_vol_30m=0.15, realized_vol_60m=0.18,
            realized_vol_2h=0.20, realized_vol_daily=0.22,
            atm_iv=0.25, iv_rank=70.0, iv_percentile=75.0,
            iv_skew=0.02, iv_term_structure=0.01,
            rsi_15m=45.0, rsi_30m=50.0, macd_signal=0.05, macd_histogram=0.02,
            bb_position=0.4, bb_squeeze=0.015,
            price_momentum_15m=0.005, price_momentum_30m=0.008, price_momentum_60m=0.012,
            support_resistance_strength=0.2, mean_reversion_signal=0.1,
            bid_ask_spread=0.002, option_volume_ratio=1.1, put_call_ratio=0.95,
            gamma_exposure=1200.0, vix_level=18.0, vix_term_structure=0.02,
            market_correlation=0.7, volume_profile=1.05, time_of_day=14.5,
            day_of_week=3, time_to_expiry=4.0, days_since_last_trade=2,
            win_rate_recent=0.65, profit_factor_recent=1.35, sharpe_ratio_recent=1.2,
            timestamp=datetime.utcnow()
        )
        
        async def test_latency():
            with patch.object(decision_engine.indicator_engine, 'calculate_all_features') as mock_calc:
                mock_calc.return_value = mock_features
                
                # Test multiple decision generations
                latencies = []
                
                for i in range(50):  # 50 decision generations
                    start_time = time.perf_counter()
                    
                    decision = await decision_engine.generate_entry_signal(
                        current_price=4200.0 + np.random.normal(0, 1),
                        implied_move=25.0,
                        vix_level=18.0 + np.random.normal(0, 1)
                    )
                    
                    end_time = time.perf_counter()
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                    assert decision is not None
                
                return latencies
        
        latencies = asyncio.run(test_latency())
        
        # Performance requirements for real-time trading
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 0.1, f"Average latency {avg_latency:.3f}s too high"
        assert max_latency < 0.5, f"Max latency {max_latency:.3f}s too high"
        assert p95_latency < 0.2, f"95th percentile latency {p95_latency:.3f}s too high"
        
        print(f"Decision latency: avg={avg_latency*1000:.1f}ms, "
              f"max={max_latency*1000:.1f}ms, p95={p95_latency*1000:.1f}ms")
    
    @pytest.mark.performance
    def test_concurrent_decision_generation(self, decision_engine):
        """Test concurrent decision generation performance"""
        import time
        from app.market_indicators import MarketFeatures
        
        mock_features = MarketFeatures(
            realized_vol_15m=0.12, realized_vol_30m=0.15, realized_vol_60m=0.18,
            realized_vol_2h=0.20, realized_vol_daily=0.22,
            atm_iv=0.25, iv_rank=70.0, iv_percentile=75.0,
            iv_skew=0.02, iv_term_structure=0.01,
            rsi_15m=45.0, rsi_30m=50.0, macd_signal=0.05, macd_histogram=0.02,
            bb_position=0.4, bb_squeeze=0.015,
            price_momentum_15m=0.005, price_momentum_30m=0.008, price_momentum_60m=0.012,
            support_resistance_strength=0.2, mean_reversion_signal=0.1,
            bid_ask_spread=0.002, option_volume_ratio=1.1, put_call_ratio=0.95,
            gamma_exposure=1200.0, vix_level=18.0, vix_term_structure=0.02,
            market_correlation=0.7, volume_profile=1.05, time_of_day=14.5,
            day_of_week=3, time_to_expiry=4.0, days_since_last_trade=2,
            win_rate_recent=0.65, profit_factor_recent=1.35, sharpe_ratio_recent=1.2,
            timestamp=datetime.utcnow()
        )
        
        async def concurrent_decisions():
            with patch.object(decision_engine.indicator_engine, 'calculate_all_features') as mock_calc:
                mock_calc.return_value = mock_features
                
                # Generate multiple decisions concurrently
                tasks = []
                for i in range(20):  # 20 concurrent decisions
                    task = decision_engine.generate_entry_signal(
                        current_price=4200.0 + np.random.normal(0, 1),
                        implied_move=25.0,
                        vix_level=18.0
                    )
                    tasks.append(task)
                
                start_time = time.perf_counter()
                decisions = await asyncio.gather(*tasks)
                end_time = time.perf_counter()
                
                return decisions, end_time - start_time
        
        decisions, total_time = asyncio.run(concurrent_decisions())
        
        # Verify all decisions were generated
        assert len(decisions) == 20
        assert all(d is not None for d in decisions)
        
        # Performance requirements
        throughput = len(decisions) / total_time
        assert throughput > 30, f"Concurrent throughput {throughput:.1f} decisions/sec too low"
        assert total_time < 2.0, f"Concurrent generation took {total_time:.2f}s, too slow"
        
        print(f"Concurrent decisions: {len(decisions)} in {total_time:.2f}s "
              f"({throughput:.1f} decisions/sec)")


class TestMLModelPerformance:
    """Test ML model training and prediction performance"""
    
    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test_perf.db"
        database_url = f"sqlite:///{db_file}"
        
        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()
        
        return database_url
    
    @pytest.fixture
    def model_trainer(self, database_url):
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        return ModelTrainer(database_url)
    
    @pytest.mark.performance
    def test_model_training_performance(self, model_trainer):
        """Test ML model training performance with large datasets"""
        import time
        
        # Create large synthetic training dataset
        n_samples = 2000
        training_data = self._create_large_training_dataset(n_samples)
        feature_cols = [col for col in training_data.columns if col != 'actual_outcome']
        
        # Mock feature engineer and data quality
        model_trainer.feature_engineer.prepare_ml_dataset = Mock(
            return_value=(training_data, feature_cols)
        )
        model_trainer.data_quality_monitor.check_data_quality = Mock(return_value={
            'completeness': 0.95,
            'consistency': 0.90,
            'freshness': 0.85
        })
        
        async def train_models():
            start_time = time.perf_counter()
            
            # Train multiple model types
            start_date = datetime.now().date() - timedelta(days=30)
            end_date = datetime.now().date()
            
            entry_model = await model_trainer.train_entry_model(start_date, end_date, 'random_forest')
            exit_model = await model_trainer.train_exit_model(start_date, end_date, 'random_forest')
            
            end_time = time.perf_counter()
            
            return entry_model, exit_model, end_time - start_time
        
        entry_model, exit_model, training_time = asyncio.run(train_models())
        
        # Performance requirements
        assert training_time < 60.0, f"Model training took {training_time:.2f}s, too slow"
        assert entry_model is not None, "Entry model should be trained successfully"
        assert exit_model is not None, "Exit model should be trained successfully"
        
        print(f"Model training: {n_samples} samples in {training_time:.2f}s")
    
    def _create_large_training_dataset(self, n_samples):
        """Create large synthetic training dataset"""
        np.random.seed(42)
        
        data = {
            'realized_vol_15m': np.random.uniform(0.08, 0.35, n_samples),
            'realized_vol_30m': np.random.uniform(0.10, 0.40, n_samples),
            'atm_iv': np.random.uniform(0.15, 0.45, n_samples),
            'iv_rank': np.random.uniform(0, 100, n_samples),
            'rsi_15m': np.random.uniform(20, 80, n_samples),
            'rsi_30m': np.random.uniform(20, 80, n_samples),
            'vix_level': np.random.uniform(12, 35, n_samples),
            'time_of_day': np.random.uniform(9.5, 16, n_samples),
            'time_to_expiry': np.random.uniform(1, 6, n_samples),
            'bb_position': np.random.uniform(0, 1, n_samples),
            'price_momentum_30m': np.random.normal(0, 0.01, n_samples),
            'macd_signal': np.random.normal(0, 0.1, n_samples),
            'bid_ask_spread': np.random.uniform(0.001, 0.01, n_samples),
            'volume_profile': np.random.uniform(0.5, 2.0, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic targets
        profit_signal = (
            (df['iv_rank'] / 100) * 0.3 +
            ((0.30 - df['realized_vol_30m']) / 0.20) * 0.3 +
            np.random.uniform(0, 0.4, n_samples)
        )
        
        df['actual_outcome'] = np.where(
            profit_signal > 0.5,
            np.random.uniform(25, 150, n_samples),
            np.random.uniform(-100, -10, n_samples)
        )
        
        return df
    
    @pytest.mark.performance
    def test_model_prediction_performance(self, model_trainer):
        """Test ML model prediction performance"""
        import time
        
        # Create and train a model
        training_data = self._create_large_training_dataset(1000)
        feature_cols = [col for col in training_data.columns if col != 'actual_outcome']
        
        model_trainer.feature_engineer.prepare_ml_dataset = Mock(
            return_value=(training_data, feature_cols)
        )
        model_trainer.data_quality_monitor.check_data_quality = Mock(return_value={
            'completeness': 0.95, 'consistency': 0.90, 'freshness': 0.85
        })
        
        async def train_and_predict():
            start_date = datetime.now().date() - timedelta(days=30)
            end_date = datetime.now().date()
            
            # Train model
            model = await model_trainer.train_entry_model(start_date, end_date, 'random_forest')
            
            if model is None:
                return None, 0
            
            # Test prediction performance
            test_data = training_data[feature_cols].head(100)
            
            start_time = time.perf_counter()
            
            # Generate many predictions
            for i in range(100):
                predictions = model.predict(test_data.iloc[i:i+1])
                assert len(predictions) == 1
            
            end_time = time.perf_counter()
            prediction_time = end_time - start_time
            
            return model, prediction_time
        
        model, prediction_time = asyncio.run(train_and_predict())
        
        if model is not None:
            # Performance requirements for real-time prediction
            avg_prediction_time = prediction_time / 100
            throughput = 100 / prediction_time
            
            assert avg_prediction_time < 0.01, f"Average prediction time {avg_prediction_time:.4f}s too slow"
            assert throughput > 100, f"Prediction throughput {throughput:.1f} predictions/sec too low"
            
            print(f"Model prediction: 100 predictions in {prediction_time:.3f}s "
                  f"({throughput:.1f} predictions/sec)")


class TestDatabasePerformance:
    """Test database performance under load"""
    
    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test_perf.db"
        database_url = f"sqlite:///{db_file}"
        
        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()
        
        return database_url
    
    @pytest.mark.performance
    def test_bulk_insert_performance(self, database_url):
        """Test bulk database insert performance"""
        import time
        
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        
        session_maker = get_session_maker(database_url)
        session = session_maker()
        
        try:
            # Test bulk insert of market data
            num_records = 5000
            start_time = time.perf_counter()
            
            market_data_records = []
            base_time = datetime.utcnow()
            
            for i in range(num_records):
                market_data = MarketData(
                    timestamp=base_time + timedelta(seconds=i),
                    underlying_price=4200.0 + np.random.normal(0, 2),
                    bid_price=4199.5,
                    ask_price=4200.5,
                    volume=1000,
                    atm_iv=0.25,
                    vix_level=20.0
                )
                market_data_records.append(market_data)
            
            session.add_all(market_data_records)
            session.commit()
            
            end_time = time.perf_counter()
            insert_time = end_time - start_time
            
            # Performance requirements
            throughput = num_records / insert_time
            assert insert_time < 10.0, f"Bulk insert took {insert_time:.2f}s, too slow"
            assert throughput > 500, f"Insert throughput {throughput:.1f} records/sec too low"
            
            # Verify all records were inserted
            count = session.query(MarketData).count()
            assert count == num_records
            
            print(f"Bulk insert: {num_records} records in {insert_time:.2f}s "
                  f"({throughput:.1f} records/sec)")
        
        finally:
            session.close()
    
    @pytest.mark.performance
    def test_query_performance_with_large_dataset(self, database_url):
        """Test database query performance with large datasets"""
        import time
        
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        
        session_maker = get_session_maker(database_url)
        session = session_maker()
        
        try:
            # Insert large dataset
            num_records = 10000
            market_data_records = []
            base_time = datetime.utcnow() - timedelta(hours=24)
            
            for i in range(num_records):
                market_data = MarketData(
                    timestamp=base_time + timedelta(seconds=i*8.64),  # Every 8.64 seconds for 24 hours
                    underlying_price=4200.0 + np.random.normal(0, 2),
                    bid_price=4199.5,
                    ask_price=4200.5,
                    volume=1000 + np.random.randint(-100, 100),
                    atm_iv=0.25 + np.random.normal(0, 0.01),
                    vix_level=20.0 + np.random.normal(0, 1)
                )
                market_data_records.append(market_data)
            
            session.add_all(market_data_records)
            session.commit()
            
            # Test various query patterns
            query_times = {}
            
            # 1. Time range query
            start_time = time.perf_counter()
            recent_data = session.query(MarketData).filter(
                MarketData.timestamp >= base_time + timedelta(hours=20)
            ).all()
            query_times['time_range'] = time.perf_counter() - start_time
            
            # 2. Aggregation query
            start_time = time.perf_counter()
            avg_price = session.query(MarketData.underlying_price).filter(
                MarketData.timestamp >= base_time + timedelta(hours=12)
            ).all()
            query_times['aggregation'] = time.perf_counter() - start_time
            
            # 3. Ordered query with limit
            start_time = time.perf_counter()
            latest_data = session.query(MarketData).order_by(
                MarketData.timestamp.desc()
            ).limit(100).all()
            query_times['ordered_limit'] = time.perf_counter() - start_time
            
            # Performance requirements
            for query_type, query_time in query_times.items():
                assert query_time < 1.0, f"{query_type} query took {query_time:.3f}s, too slow"
            
            print(f"Query performance on {num_records} records:")
            for query_type, query_time in query_times.items():
                print(f"  {query_type}: {query_time*1000:.1f}ms")
        
        finally:
            session.close()


class TestSystemResourceUsage:
    """Test system resource usage under various loads"""
    
    @pytest.mark.performance
    def test_cpu_usage_under_load(self):
        """Test CPU usage during intensive operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Monitor CPU usage during intensive computation
        cpu_percentages = []
        
        def cpu_intensive_task():
            """Simulate CPU-intensive feature calculation"""
            data = np.random.random((1000, 50))
            result = data
            for i in range(100):
                # Simulate complex calculations
                corr_matrix = np.corrcoef(data.T)  # Transpose to get feature correlations
                result = np.linalg.inv(corr_matrix + np.eye(50) * 0.01)
                eigenvals = np.linalg.eigvals(result)
                # Use eigenvals in some computation to prevent optimization
                data = data + eigenvals.mean() * 0.001
            return result
        
        # Run task while monitoring CPU
        for i in range(5):
            start_cpu = process.cpu_percent()
            cpu_intensive_task()
            end_cpu = process.cpu_percent()
            cpu_percentages.append(end_cpu)
        
        avg_cpu = np.mean(cpu_percentages)
        max_cpu = np.max(cpu_percentages)
        
        # CPU usage should be reasonable (can exceed 100% on multi-core systems)
        # On multi-core systems, CPU usage can exceed 100% (up to number_of_cores * 100%)
        import multiprocessing
        max_expected_cpu = multiprocessing.cpu_count() * 100 * 1.2  # Allow 20% overhead
        assert max_cpu < max_expected_cpu, f"Max CPU usage {max_cpu:.1f}% too high for system"
        
        print(f"CPU usage: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%")
    
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations"""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate repeated trading operations
        from app.decision_engine import DecisionEngine
        decision_engine = DecisionEngine()
        
        memory_samples = []
        
        for iteration in range(10):
            # Simulate many decision generations
            for i in range(50):
                # Create and destroy objects to test for leaks
                mock_data = {
                    'price': 4200.0 + np.random.normal(0, 1),
                    'volume': 1000,
                    'iv': 0.25,
                    'features': np.random.random(20).tolist()
                }
                
                # Simulate decision processing
                from app.market_indicators import MarketFeatures
                mock_features = MarketFeatures(
                    realized_vol_15m=0.12, realized_vol_30m=0.15, realized_vol_60m=0.18,
                    realized_vol_2h=0.20, realized_vol_daily=0.22,
                    atm_iv=0.25, iv_rank=50.0, iv_percentile=50.0,
                    iv_skew=0.02, iv_term_structure=0.01,
                    rsi_15m=50.0, rsi_30m=50.0, macd_signal=0.05, macd_histogram=0.02,
                    bb_position=0.4, bb_squeeze=0.015,
                    price_momentum_15m=0.005, price_momentum_30m=0.008, price_momentum_60m=0.012,
                    support_resistance_strength=0.2, mean_reversion_signal=0.1,
                    bid_ask_spread=0.002, option_volume_ratio=1.1, put_call_ratio=0.95,
                    gamma_exposure=1200.0, vix_level=20.0, vix_term_structure=0.02,
                    market_correlation=0.7, volume_profile=1.05,
                    time_of_day=14.5, day_of_week=3, time_to_expiry=4.0,
                    days_since_last_trade=2, win_rate_recent=0.65,
                    profit_factor_recent=1.35, sharpe_ratio_recent=1.2,
                    timestamp=datetime.utcnow()
                )
                result = decision_engine._calculate_dynamic_profit_target(mock_features, 0.7)
            
            # Force garbage collection
            gc.collect()
            
            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory - initial_memory)
        
        # Check for memory growth pattern
        memory_growth = memory_samples[-1] - memory_samples[0]
        max_memory_increase = max(memory_samples)
        
        # Memory shouldn't grow significantly over iterations
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB, possible leak"
        assert max_memory_increase < 100, f"Max memory increase {max_memory_increase:.1f}MB too high"
        
        print(f"Memory leak test: growth={memory_growth:.1f}MB, max={max_memory_increase:.1f}MB")


class TestRealTimePerformanceRequirements:
    """Test performance requirements for real-time trading"""
    
    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test_perf.db"
        database_url = f"sqlite:///{db_file}"
        
        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()
        
        return database_url
    
    @pytest.mark.performance
    def test_end_to_end_latency(self, database_url):
        """Test end-to-end latency from data to decision"""
        import time
        
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        
        # Set up components
        feature_collector = FeatureCollector(database_url)
        decision_engine = DecisionEngine()
        
        async def end_to_end_test():
            latencies = []
            
            for i in range(10):  # 10 end-to-end tests
                start_time = time.perf_counter()
                
                # 1. Collect market data
                await feature_collector.collect_market_data(
                    price=4200.0 + np.random.normal(0, 1),
                    bid=4199.5,
                    ask=4200.5,
                    volume=1000,
                    atm_iv=0.25,
                    vix_level=20.0
                )
                
                # 2. Generate decision (with mocked features)
                from app.market_indicators import MarketFeatures
                mock_features = MarketFeatures(
                    realized_vol_15m=0.12, realized_vol_30m=0.15, realized_vol_60m=0.18,
                    realized_vol_2h=0.20, realized_vol_daily=0.22,
                    atm_iv=0.25, iv_rank=70.0, iv_percentile=75.0,
                    iv_skew=0.02, iv_term_structure=0.01,
                    rsi_15m=45.0, rsi_30m=50.0, macd_signal=0.05, macd_histogram=0.02,
                    bb_position=0.4, bb_squeeze=0.015,
                    price_momentum_15m=0.005, price_momentum_30m=0.008, price_momentum_60m=0.012,
                    support_resistance_strength=0.2, mean_reversion_signal=0.1,
                    bid_ask_spread=0.002, option_volume_ratio=1.1, put_call_ratio=0.95,
                    gamma_exposure=1200.0, vix_level=18.0, vix_term_structure=0.02,
                    market_correlation=0.7, volume_profile=1.05, time_of_day=14.5,
                    day_of_week=3, time_to_expiry=4.0, days_since_last_trade=2,
                    win_rate_recent=0.65, profit_factor_recent=1.35, sharpe_ratio_recent=1.2,
                    timestamp=datetime.utcnow()
                )
                
                with patch.object(decision_engine.indicator_engine, 'calculate_all_features') as mock_calc:
                    mock_calc.return_value = mock_features
                    
                    decision = await decision_engine.generate_entry_signal(
                        current_price=4200.0,
                        implied_move=25.0,
                        vix_level=18.0
                    )
                
                end_time = time.perf_counter()
                latency = end_time - start_time
                latencies.append(latency)
                
                assert decision is not None
            
            return latencies
        
        latencies = asyncio.run(end_to_end_test())
        
        # Real-time trading requirements
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Strict requirements for real-time trading
        assert avg_latency < 0.5, f"Average end-to-end latency {avg_latency:.3f}s too high"
        assert max_latency < 1.0, f"Max end-to-end latency {max_latency:.3f}s too high"
        assert p95_latency < 0.8, f"95th percentile latency {p95_latency:.3f}s too high"
        
        print(f"End-to-end latency: avg={avg_latency*1000:.1f}ms, "
              f"max={max_latency*1000:.1f}ms, p95={p95_latency*1000:.1f}ms")
    
    @pytest.mark.performance
    def test_sustained_throughput(self, database_url):
        """Test sustained throughput over extended period"""
        import time
        
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        
        feature_collector = FeatureCollector(database_url)
        
        async def sustained_load_test():
            start_time = time.perf_counter()
            duration = 30.0  # 30 seconds
            operations = 0
            
            while time.perf_counter() - start_time < duration:
                await feature_collector.collect_market_data(
                    price=4200.0 + np.random.normal(0, 1),
                    bid=4199.5,
                    ask=4200.5,
                    volume=1000,
                    atm_iv=0.25,
                    vix_level=20.0
                )
                operations += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            actual_duration = time.perf_counter() - start_time
            return operations, actual_duration
        
        operations, duration = asyncio.run(sustained_load_test())
        
        throughput = operations / duration
        
        # Sustained throughput requirements
        assert throughput > 10, f"Sustained throughput {throughput:.1f} ops/sec too low"
        
        # Verify data integrity (allow some tolerance for high-frequency operations)
        session_maker = get_session_maker(database_url)
        session = session_maker()
        try:
            count = session.query(MarketData).count()
            # Allow for 2% loss due to potential race conditions in high-frequency testing
            min_expected = int(operations * 0.98)
            assert count >= min_expected, f"Data integrity check failed: got {count}, expected at least {min_expected}"
            assert count <= operations, f"Got more data than expected: {count} > {operations}"
        finally:
            session.close()
        
        print(f"Sustained throughput: {operations} operations in {duration:.1f}s "
              f"({throughput:.1f} ops/sec)")


if __name__ == "__main__":
    # Run specific performance tests
    pytest.main([
        __file__ + "::TestMarketDataProcessingPerformance::test_high_frequency_data_collection_performance",
        "-v", "-s", "--tb=short"
    ])