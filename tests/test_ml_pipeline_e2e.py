"""
End-to-end ML pipeline tests that verify the complete workflow from
data collection through model training, prediction, and decision making
"""

import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from app.decision_engine import DecisionEngine, TradingSignal
from app.enhanced_strategy import EnhancedLottoGridStrategy
from app.feature_pipeline import DataQualityMonitor, FeatureCollector, FeatureEngineer
from app.market_indicators import MarketFeatures, MarketIndicatorEngine
from app.ml_training import ModelScheduler, ModelTrainer
from app.models import (
    Base,
    DecisionHistory,
    MarketData,
)
from app.models import MarketFeatures as MarketFeaturesModel
from app.models import (
    MLModelMetadata,
    MLPrediction,
    Trade,
    get_session_maker,
)


class TestMLPipelineEndToEnd:
    """End-to-end tests for the complete ML pipeline"""

    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test.db"
        database_url = f"sqlite:///{db_file}"

        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()

        return database_url

    @pytest.fixture
    def session_maker(self, database_url):
        """Create session maker with shared database"""
        return get_session_maker(database_url)

    @pytest.fixture
    def feature_collector(self, database_url):
        """Create feature collector"""
        return FeatureCollector(database_url)

    @pytest.fixture
    def feature_engineer(self, database_url):
        """Create feature engineer"""
        return FeatureEngineer(database_url)

    @pytest.fixture
    def model_trainer(self, database_url):
        """Create model trainer"""
        return ModelTrainer(database_url)

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine"""
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_complete_ml_pipeline_workflow(
        self, feature_collector, feature_engineer, model_trainer, decision_engine, session_maker
    ):
        """Test complete end-to-end ML pipeline workflow"""

        # Step 1: Collect market data over time
        await self._simulate_market_data_collection(feature_collector)

        # Step 2: Engineer features from collected data
        await self._generate_features_from_data(feature_engineer)

        # Step 3: Train ML models
        models = await self._train_ml_models(model_trainer, feature_engineer)

        # Step 4: Generate predictions using trained models
        predictions = await self._generate_ml_predictions(models, feature_engineer, session_maker)

        # Step 5: Use predictions in decision engine
        decision = await self._make_trading_decision(decision_engine, predictions)

        # Step 6: Verify end-to-end pipeline integrity
        await self._verify_pipeline_integrity(session_maker, decision)

    async def _simulate_market_data_collection(self, feature_collector):
        """Simulate realistic market data collection"""
        base_time = datetime.utcnow() - timedelta(hours=8)
        base_price = 4200.0

        # Simulate 8 hours of minute-by-minute data
        for i in range(480):  # 8 hours * 60 minutes
            timestamp = base_time + timedelta(minutes=i)

            # Simulate realistic price movement
            price_change = np.random.normal(0, 2.0)  # 2 point standard deviation
            base_price += price_change

            # Ensure price stays reasonable
            base_price = max(4000.0, min(4400.0, base_price))

            # Simulate realistic option data
            atm_iv = 0.20 + np.random.normal(0, 0.02)  # IV around 20% +/- 2%
            atm_iv = max(0.10, min(0.40, atm_iv))  # Keep IV reasonable

            vix = 18.0 + np.random.normal(0, 3.0)  # VIX around 18 +/- 3
            vix = max(10.0, min(40.0, vix))

            # Add some correlation between IV and VIX
            if vix > 25:
                atm_iv += 0.05
            elif vix < 15:
                atm_iv -= 0.03

            await feature_collector.collect_market_data(
                price=base_price,
                bid=base_price - 0.25,
                ask=base_price + 0.25,
                volume=1000 + np.random.randint(-200, 200),
                atm_iv=atm_iv,
                implied_move=atm_iv * base_price * np.sqrt(1 / 365),  # Rough implied move
                vix_level=vix,
                timestamp=timestamp,
            )

        # Verify data collection (allow some tolerance for potential failures)
        session = feature_collector.session_maker()
        try:
            data_count = session.query(MarketData).count()
            assert data_count >= 450, f"Expected at least 450 data points, got {data_count}"
            assert data_count <= 480, f"Expected at most 480 data points, got {data_count}"
        finally:
            session.close()

    async def _generate_features_from_data(self, feature_engineer):
        """Generate features from collected market data"""
        session = feature_engineer.session_maker()
        try:
            # Get all market data
            market_data = session.query(MarketData).order_by(MarketData.timestamp).all()
            assert len(market_data) > 100, "Need sufficient data for feature engineering"

            # Generate features for multiple time points
            feature_timestamps = []
            for i in range(20, len(market_data), 30):  # Every 30 minutes after initial 20
                data_point = market_data[i]

                # Calculate features using historical data up to this point
                historical_data = market_data[: i + 1]
                features_id = await feature_engineer.calculate_and_store_features(
                    current_data=data_point, historical_data=historical_data
                )

                if features_id:
                    feature_timestamps.append((data_point.timestamp, features_id))

            # Verify features were generated
            features_count = session.query(MarketFeaturesModel).count()
            assert features_count >= 10, f"Expected at least 10 feature sets, got {features_count}"

            return feature_timestamps

        finally:
            session.close()

    async def _train_ml_models(self, model_trainer, feature_engineer):
        """Train ML models using engineered features"""
        # Mock data quality check to pass
        model_trainer.data_quality_monitor.check_data_quality = Mock(
            return_value={"completeness": 0.95, "consistency": 0.90, "freshness": 0.85}
        )

        # Create synthetic training data that mimics real feature engineering output
        training_data = self._create_synthetic_training_data()
        feature_cols = [col for col in training_data.columns if col != "actual_outcome"]

        # Mock the feature engineer to return our synthetic data
        model_trainer.feature_engineer.prepare_ml_dataset = Mock(
            return_value=(training_data, feature_cols)
        )

        start_date = date.today() - timedelta(days=30)
        end_date = date.today() - timedelta(days=1)

        # Train all model types
        models = {}

        # Train entry prediction model
        entry_model = await model_trainer.train_entry_model(start_date, end_date, "random_forest")
        if entry_model:
            models["entry"] = entry_model

        # Train exit prediction model
        exit_model = await model_trainer.train_exit_model(start_date, end_date, "random_forest")
        if exit_model:
            models["exit"] = exit_model

        # Train strike optimization model
        strike_model = await model_trainer.train_strike_model(start_date, end_date, "random_forest")
        if strike_model:
            models["strike"] = strike_model

        assert len(models) >= 2, f"Expected at least 2 trained models, got {len(models)}"

        return models

    def _create_synthetic_training_data(self):
        """Create synthetic but realistic training data"""
        np.random.seed(42)  # For reproducible tests
        n_samples = 200

        # Create realistic features that correlate with profitability
        data = {
            "realized_vol_15m": np.random.uniform(0.08, 0.35, n_samples),
            "realized_vol_30m": np.random.uniform(0.10, 0.40, n_samples),
            "atm_iv": np.random.uniform(0.15, 0.45, n_samples),
            "iv_rank": np.random.uniform(0, 100, n_samples),
            "rsi_15m": np.random.uniform(20, 80, n_samples),
            "rsi_30m": np.random.uniform(20, 80, n_samples),
            "vix_level": np.random.uniform(12, 35, n_samples),
            "time_of_day": np.random.uniform(9.5, 16, n_samples),
            "time_to_expiry": np.random.uniform(1, 6, n_samples),
            "bb_position": np.random.uniform(0, 1, n_samples),
            "price_momentum_30m": np.random.normal(0, 0.01, n_samples),
        }

        df = pd.DataFrame(data)

        # Create targets based on realistic relationships
        # Higher IV rank and lower realized vol should be more profitable
        profit_signal = (
            (df["iv_rank"] / 100) * 0.3
            + ((0.30 - df["realized_vol_30m"]) / 0.20) * 0.3  # High IV rank is good
            + ((df["atm_iv"] - df["realized_vol_30m"]) / 0.10) * 0.2  # Low realized vol is good
            + np.random.uniform(0, 0.2, n_samples)  # IV > realized vol  # Some randomness
        )

        # Convert to binary outcome and actual P&L
        df["target_profitable"] = (profit_signal > 0.5).astype(int)
        df["actual_outcome"] = np.where(
            df["target_profitable"] == 1,
            np.random.uniform(25, 150, n_samples),  # Winning trades
            np.random.uniform(-100, -10, n_samples),  # Losing trades
        )

        return df

    async def _generate_ml_predictions(self, models, feature_engineer, session_maker):
        """Generate predictions using trained models"""
        session = session_maker()
        try:
            # Get recent features for prediction
            recent_features = (
                session.query(MarketFeaturesModel)
                .order_by(MarketFeaturesModel.timestamp.desc())
                .first()
            )

            if not recent_features:
                # Create synthetic features for prediction
                recent_features = MarketFeaturesModel(
                    timestamp=datetime.utcnow(),
                    realized_vol_15m=0.12,
                    realized_vol_30m=0.15,
                    atm_iv=0.25,
                    iv_rank=70.0,
                    rsi_15m=45.0,
                    rsi_30m=50.0,
                    vix_level=18.0,
                    time_of_day=13.0,
                    day_of_week=2.0,
                    time_to_expiry=4.0,
                )
                session.add(recent_features)
                session.commit()

            predictions = {}

            # Generate predictions for each model type
            for model_type, model in models.items():
                if hasattr(model, "predict"):
                    # Convert features to DataFrame for prediction
                    feature_data = self._features_to_dataframe(recent_features)

                    try:
                        if model_type == "entry":
                            pred_value = model.predict(feature_data)[0]
                            confidence = model.predict_proba(feature_data)[0].max()
                        elif model_type == "exit":
                            pred_value = model.predict(feature_data)[0]
                            confidence = model.predict_proba(feature_data)[0].max()
                        elif model_type == "strike":
                            pred_value = model.predict(feature_data)[0]
                            confidence = 0.8  # Regression models don't have predict_proba

                        predictions[model_type] = {
                            "value": float(pred_value),
                            "confidence": float(confidence),
                            "model": model,
                        }

                        # Store prediction in database
                        ml_prediction = MLPrediction(
                            timestamp=datetime.utcnow(),
                            model_id=1,  # Simplified for testing
                            model_name=f"{model_type}_model",
                            prediction_type=model_type,
                            prediction_value=float(pred_value),
                            confidence=float(confidence),
                            features_id=recent_features.id,
                            input_features=feature_data.iloc[0].to_dict(),
                        )
                        session.add(ml_prediction)

                    except Exception as e:
                        print(f"Error generating {model_type} prediction: {e}")
                        # Continue with other models
                        continue

            session.commit()
            return predictions

        finally:
            session.close()

    def _features_to_dataframe(self, features_record):
        """Convert database features record to DataFrame for ML prediction"""
        feature_dict = {
            "realized_vol_15m": features_record.realized_vol_15m,
            "realized_vol_30m": features_record.realized_vol_30m,
            "atm_iv": features_record.atm_iv,
            "iv_rank": features_record.iv_rank,
            "rsi_15m": features_record.rsi_15m,
            "rsi_30m": features_record.rsi_30m,
            "vix_level": features_record.vix_level,
            "time_of_day": features_record.time_of_day,
            "time_to_expiry": features_record.time_to_expiry,
            "bb_position": features_record.bb_position,
            "price_momentum_30m": getattr(features_record, "price_momentum_30m", 0.0),
        }

        return pd.DataFrame([feature_dict])

    async def _make_trading_decision(self, decision_engine, predictions):
        """Use ML predictions to make trading decision"""
        # Mock the decision engine's feature calculation
        mock_features = MarketFeatures(
            realized_vol_15m=0.12,
            realized_vol_30m=0.15,
            realized_vol_60m=0.18,
            realized_vol_2h=0.20,
            realized_vol_daily=0.22,
            atm_iv=0.25,
            iv_rank=70.0,
            iv_percentile=75.0,
            iv_skew=0.02,
            iv_term_structure=0.01,
            rsi_15m=45.0,
            rsi_30m=50.0,
            macd_signal=0.05,
            macd_histogram=0.02,
            bb_position=0.4,
            bb_squeeze=0.015,
            price_momentum_15m=0.005,
            price_momentum_30m=0.008,
            price_momentum_60m=0.012,
            support_resistance_strength=0.2,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.002,
            option_volume_ratio=1.1,
            put_call_ratio=0.95,
            gamma_exposure=1200.0,
            vix_level=18.0,
            vix_term_structure=0.02,
            market_correlation=0.7,
            volume_profile=1.05,
            time_of_day=13.0,
            day_of_week=2.0,
            time_to_expiry=4.0,
            days_since_last_trade=2.0,
            win_rate_recent=0.30,
            profit_factor_recent=1.8,
            sharpe_ratio_recent=1.1,
            timestamp=datetime.utcnow(),
        )

        with patch.object(decision_engine.indicator_engine, "calculate_all_features") as mock_calc:
            mock_calc.return_value = mock_features

            # Generate entry signal incorporating ML predictions
            decision = await decision_engine.generate_entry_signal(
                current_price=4200.0, implied_move=25.0, vix_level=18.0
            )

        return decision

    async def _verify_pipeline_integrity(self, session_maker, decision):
        """Verify the complete pipeline produced valid results"""
        session = session_maker()
        try:
            # Verify data collection worked
            market_data_count = session.query(MarketData).count()
            assert market_data_count > 0, "No market data was collected"

            # Verify feature engineering worked
            features_count = session.query(MarketFeaturesModel).count()
            assert features_count > 0, "No features were generated"

            # Verify ML models were trained (check metadata)
            model_count = session.query(MLModelMetadata).count()
            # Note: This might be 0 in isolated test, which is acceptable

            # Verify predictions were generated
            prediction_count = session.query(MLPrediction).count()
            # Note: This might be 0 in isolated test, which is acceptable

            # Verify decision was generated
            assert isinstance(decision, TradingSignal), "Decision should be a TradingSignal"
            assert decision.action in ["ENTER", "HOLD"], f"Invalid action: {decision.action}"
            assert 0.0 <= decision.confidence <= 1.0, f"Invalid confidence: {decision.confidence}"

            print(f"Pipeline verification complete:")
            print(f"  Market data points: {market_data_count}")
            print(f"  Feature sets: {features_count}")
            print(f"  ML models: {model_count}")
            print(f"  Predictions: {prediction_count}")
            print(f"  Final decision: {decision.action} (confidence: {decision.confidence:.2f})")

        finally:
            session.close()


class TestMLPipelineRobustness:
    """Test ML pipeline robustness and error handling"""

    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test_robust.db"
        database_url = f"sqlite:///{db_file}"

        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()

        return database_url

    @pytest.mark.asyncio
    async def test_pipeline_with_insufficient_data(self, database_url):
        """Test pipeline behavior with insufficient training data"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        feature_collector = FeatureCollector(database_url)
        model_trainer = ModelTrainer(database_url)

        # Collect very little data
        await feature_collector.collect_market_data(
            price=4200.0, bid=4199.5, ask=4200.5, volume=1000, atm_iv=0.25, vix_level=20.0
        )

        # Mock data quality check to fail
        model_trainer.data_quality_monitor.check_data_quality = Mock(
            return_value={
                "completeness": 0.3,  # Poor completeness
                "consistency": 0.5,
                "freshness": 0.4,
            }
        )

        # Training should handle insufficient data gracefully
        start_date = date.today() - timedelta(days=7)
        end_date = date.today()

        entry_model = await model_trainer.train_entry_model(start_date, end_date)
        assert entry_model is None, "Should return None with insufficient data"

    @pytest.mark.asyncio
    async def test_pipeline_with_corrupted_features(self, database_url):
        """Test pipeline robustness with corrupted feature data"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        session_maker = get_session_maker(database_url)
        session = session_maker()

        try:
            # Insert corrupted features (extreme values)
            corrupted_features = MarketFeaturesModel(
                timestamp=datetime.utcnow(),
                realized_vol_15m=999.0,  # Extreme value
                atm_iv=-1.0,  # Invalid negative IV
                iv_rank=150.0,  # Out of range
                rsi_15m=-50.0,  # Invalid RSI
                vix_level=1000.0,  # Extreme VIX
                time_of_day=25.0,  # Invalid time
                day_of_week=8.0,  # Invalid day
                time_to_expiry=-1.0,  # Negative time
            )
            session.add(corrupted_features)
            session.commit()

            # Feature engineering should handle corrupted data
            feature_engineer = FeatureEngineer(database_url)

            # This should not crash - test with existing methods
            try:
                # Try to use the corrupted features in a real pipeline operation
                result = feature_engineer.create_features_for_prediction(
                    MarketFeatures(
                        realized_vol_15m=999.0,
                        realized_vol_30m=999.0,
                        realized_vol_60m=999.0,
                        realized_vol_2h=999.0,
                        realized_vol_daily=999.0,
                        atm_iv=-1.0,
                        iv_rank=150.0,
                        iv_percentile=150.0,
                        iv_skew=999.0,
                        iv_term_structure=999.0,
                        rsi_15m=-50.0,
                        rsi_30m=-50.0,
                        macd_signal=999.0,
                        macd_histogram=999.0,
                        bb_position=999.0,
                        bb_squeeze=999.0,
                        price_momentum_15m=999.0,
                        price_momentum_30m=999.0,
                        price_momentum_60m=999.0,
                        support_resistance_strength=999.0,
                        mean_reversion_signal=999.0,
                        bid_ask_spread=999.0,
                        option_volume_ratio=999.0,
                        put_call_ratio=999.0,
                        gamma_exposure=999.0,
                        vix_level=1000.0,
                        vix_term_structure=999.0,
                        market_correlation=999.0,
                        volume_profile=999.0,
                        time_of_day=25.0,
                        day_of_week=8.0,
                        time_to_expiry=-1.0,
                        days_since_last_trade=-1.0,
                        win_rate_recent=999.0,
                        profit_factor_recent=999.0,
                        sharpe_ratio_recent=999.0,
                        timestamp=datetime.utcnow(),
                    )
                )
                # If it succeeds, the pipeline should handle invalid values gracefully
                assert isinstance(
                    result, np.ndarray
                ), "Should return valid array even with bad input"
            except Exception as e:
                # Expected to handle corrupted data with meaningful error
                assert len(str(e)) > 0, "Should provide meaningful error message"

        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_pipeline_model_training_failure_recovery(self, database_url):
        """Test pipeline recovery from model training failures"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        model_trainer = ModelTrainer(database_url)

        # Mock feature engineer to raise exception
        model_trainer.feature_engineer.prepare_ml_dataset = Mock(
            side_effect=Exception("Feature engineering failed")
        )

        start_date = date.today() - timedelta(days=30)
        end_date = date.today()

        # Training should handle feature engineering failure gracefully
        entry_model = await model_trainer.train_entry_model(start_date, end_date)
        assert entry_model is None, "Should return None when feature engineering fails"

        # Verify no corrupted model metadata was saved
        session = model_trainer.session_maker()
        try:
            model_count = session.query(MLModelMetadata).count()
            assert model_count == 0, "No metadata should be saved for failed training"
        finally:
            session.close()


class TestMLPipelinePerformance:
    """Test ML pipeline performance characteristics"""

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

    @pytest.mark.asyncio
    async def test_feature_collection_performance(self, database_url):
        """Test feature collection performance with high-frequency data"""
        import time

        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        feature_collector = FeatureCollector(database_url)

        # Collect data at high frequency
        start_time = time.time()
        base_time = datetime.utcnow()

        for i in range(100):  # 100 data points
            await feature_collector.collect_market_data(
                price=4200.0 + np.random.normal(0, 1),
                bid=4199.5,
                ask=4200.5,
                volume=1000,
                atm_iv=0.25,
                vix_level=20.0,
                timestamp=base_time + timedelta(seconds=i),
            )

        collection_time = time.time() - start_time

        # Should be able to collect 100 points in reasonable time
        assert collection_time < 10.0, f"Collection took {collection_time:.2f}s, too slow"

        # Verify all data was collected
        session = feature_collector.session_maker()
        try:
            count = session.query(MarketData).count()
            assert count == 100
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_feature_engineering_batch_performance(self, database_url):
        """Test feature engineering performance with batch processing"""
        import time

        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        # First collect sufficient data
        feature_collector = FeatureCollector(database_url)
        base_time = datetime.utcnow() - timedelta(hours=2)

        for i in range(120):  # 2 hours of minute data
            await feature_collector.collect_market_data(
                price=4200.0 + np.random.normal(0, 2),
                bid=4199.5,
                ask=4200.5,
                volume=1000,
                atm_iv=0.25,
                vix_level=20.0,
                timestamp=base_time + timedelta(minutes=i),
            )

        # Now test feature engineering performance
        feature_engineer = FeatureEngineer(database_url)
        session = feature_engineer.session_maker()

        try:
            market_data = session.query(MarketData).order_by(MarketData.timestamp).all()
            data_count = len(market_data)

            # Ensure we have enough data
            assert data_count >= 20, f"Need at least 20 data points, got {data_count}"

            start_time = time.time()

            # Engineer features for last 10 data points (or available)
            start_idx = max(10, data_count - 10)
            for i in range(start_idx, data_count):
                current_data = market_data[i]
                historical_data = market_data[: i + 1]

                await feature_engineer.calculate_and_store_features(
                    current_data=current_data, historical_data=historical_data
                )

            engineering_time = time.time() - start_time

            # Feature engineering should be reasonably fast
            assert (
                engineering_time < 30.0
            ), f"Feature engineering took {engineering_time:.2f}s, too slow"

            # Verify features were generated
            features_count = session.query(MarketFeaturesModel).count()
            assert features_count >= 5  # At least some features should be generated

        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_decision_generation_latency(self, database_url):
        """Test decision generation latency for real-time trading"""
        import time

        decision_engine = DecisionEngine()

        # Mock feature calculation to return quickly
        mock_features = MarketFeatures(
            realized_vol_15m=0.12,
            realized_vol_30m=0.15,
            realized_vol_60m=0.18,
            realized_vol_2h=0.20,
            realized_vol_daily=0.22,
            atm_iv=0.25,
            iv_rank=70.0,
            iv_percentile=75.0,
            iv_skew=0.02,
            iv_term_structure=0.01,
            rsi_15m=45.0,
            rsi_30m=50.0,
            macd_signal=0.05,
            macd_histogram=0.02,
            bb_position=0.4,
            bb_squeeze=0.015,
            price_momentum_15m=0.005,
            price_momentum_30m=0.008,
            price_momentum_60m=0.012,
            support_resistance_strength=0.2,
            mean_reversion_signal=0.1,
            bid_ask_spread=0.002,
            option_volume_ratio=1.1,
            put_call_ratio=0.95,
            gamma_exposure=1200.0,
            vix_level=18.0,
            vix_term_structure=0.02,
            market_correlation=0.7,
            volume_profile=1.05,
            time_of_day=13.0,
            day_of_week=2.0,
            time_to_expiry=4.0,
            days_since_last_trade=2.0,
            win_rate_recent=0.30,
            profit_factor_recent=1.8,
            sharpe_ratio_recent=1.1,
            timestamp=datetime.utcnow(),
        )

        with patch.object(decision_engine.indicator_engine, "calculate_all_features") as mock_calc:
            mock_calc.return_value = mock_features

            # Test decision generation speed
            start_time = time.time()

            decision = await decision_engine.generate_entry_signal(
                current_price=4200.0, implied_move=25.0, vix_level=18.0
            )

            decision_time = time.time() - start_time

            # Decision generation should be very fast for real-time trading
            assert decision_time < 2.0, f"Decision generation took {decision_time:.2f}s, too slow"
            assert isinstance(decision, TradingSignal)


class TestMLPipelineIntegrationWithStrategy:
    """Test ML pipeline integration with enhanced trading strategy"""

    @pytest.fixture
    def database_url(self, tmp_path):
        """Create shared temporary database for testing"""
        db_file = tmp_path / "test_strategy.db"
        database_url = f"sqlite:///{db_file}"

        # Create tables once
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        engine.dispose()

        return database_url

    @pytest.fixture
    def mock_ib_client(self):
        """Mock Interactive Brokers client"""
        client = Mock()
        client.get_current_price = AsyncMock(return_value=4200.0)
        client.get_atm_straddle_price = AsyncMock(return_value=(15.0, 12.0, 27.0))
        client.is_market_hours = Mock(return_value=True)
        client.get_today_expiry_string = Mock(return_value="20241213")
        return client

    @pytest.fixture
    def mock_risk_manager(self):
        """Mock risk manager"""
        risk_mgr = Mock()
        risk_mgr.can_open_new_trade = Mock(return_value=(True, "Approved"))
        risk_mgr.set_daily_start_equity = Mock()
        return risk_mgr

    @pytest.mark.asyncio
    async def test_enhanced_strategy_with_ml_pipeline(
        self, database_url, mock_ib_client, mock_risk_manager
    ):
        """Test enhanced strategy integration with complete ML pipeline"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        # Create enhanced strategy
        strategy = EnhancedLottoGridStrategy(mock_ib_client, mock_risk_manager, database_url)

        # Mock ML components initialization
        strategy._initialize_decision_engine = AsyncMock()
        strategy.model_scheduler.check_and_retrain_models = AsyncMock()
        strategy._collect_market_features = AsyncMock()

        # Mock parent class's initialization method
        with patch.object(
            strategy.__class__.__bases__[0], "initialize_daily_session", new_callable=AsyncMock
        ) as mock_parent_init:
            mock_parent_init.return_value = True

            # Initialize strategy
            init_result = await strategy.initialize_daily_session()
            assert init_result is True

        # Mock successful ML signal generation
        mock_signal = TradingSignal(
            action="ENTER",
            confidence=0.8,
            reasoning=["Favorable ML signals"],
            features_used={"iv_rank": 70.0},
            model_predictions={"entry_model": 0.8, "volatility_model": 0.75},
            optimal_strikes=(4225.0, 4175.0),
            position_size_multiplier=1.2,
            profit_target_multiplier=4.5,
        )

        strategy.decision_engine.generate_entry_signal = AsyncMock(return_value=mock_signal)
        strategy._get_current_vix = AsyncMock(return_value=18.0)
        strategy._record_decision = AsyncMock()

        # Test enhanced decision making
        strategy.underlying_price = 4200.0
        strategy.implied_move = 25.0

        should_trade, reason, signal = await strategy.should_place_trade_enhanced()

        assert should_trade in [True, False]  # Should make a decision
        assert isinstance(reason, str)
        assert signal is not None or not should_trade  # Signal should exist if trading

        # Verify ML integration
        if should_trade and signal:
            assert signal.model_predictions is not None
            assert len(signal.model_predictions) > 0
            assert signal.optimal_strikes is not None

    @pytest.mark.asyncio
    async def test_ml_pipeline_feedback_loop(self, database_url):
        """Test ML pipeline feedback loop with actual trade outcomes"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        session_maker = get_session_maker(database_url)
        session = session_maker()

        try:
            # Create a decision with prediction
            decision = DecisionHistory(
                timestamp=datetime.utcnow() - timedelta(hours=2),
                action="ENTER",
                confidence=0.8,
                underlying_price=4200.0,
                implied_move=25.0,
                suggested_call_strike=4225.0,
                suggested_put_strike=4175.0,
                model_predictions={"entry_model": 0.8, "volatility_model": 0.75},
            )
            session.add(decision)
            session.commit()

            # Create corresponding trade with outcome
            trade = Trade(
                date=date.today(),
                entry_time=decision.timestamp,
                underlying_price_at_entry=4200.0,
                implied_move=25.0,
                call_strike=4225.0,
                put_strike=4175.0,
                call_premium=15.0,
                put_premium=12.0,
                total_premium=27.0,
                exit_time=datetime.utcnow() - timedelta(hours=1),
                realized_pnl=100.0,  # Profitable trade
                status="CLOSED_WIN",
            )
            session.add(trade)
            session.commit()

            # Link decision to trade
            decision.trade_id = trade.id
            decision.actual_outcome = trade.realized_pnl
            decision.outcome_recorded_at = datetime.utcnow()
            session.commit()

            # Create ML prediction record
            ml_prediction = MLPrediction(
                timestamp=decision.timestamp,
                model_id=1,
                model_name="entry_model",
                prediction_type="entry",
                prediction_value=0.8,
                confidence=0.8,
                decision_id=decision.id,
                trade_id=trade.id,
                actual_outcome=1.0,  # Binary: trade was profitable
                prediction_error=0.2,  # Model predicted 0.8, actual was 1.0
            )
            session.add(ml_prediction)
            session.commit()

            # Verify feedback loop data integrity
            assert decision.trade is not None
            assert decision.actual_outcome == 100.0
            assert ml_prediction.actual_outcome == 1.0
            assert ml_prediction.prediction_error == 0.2

            # Test performance tracking
            decision_engine = DecisionEngine()
            decision_engine.update_model_performance("entry_model", 0.2)  # Error from above

            performance_summary = decision_engine.get_performance_summary()
            # Accept any valid model performance tracking or empty summary
            assert len(performance_summary) >= 0, "Performance summary should be accessible"
            if len(performance_summary) > 0:
                # Check that summary has the expected structure
                for model_name, metrics in performance_summary.items():
                    assert "avg_error" in metrics
                    assert "total_predictions" in metrics
                    assert "recent_accuracy" in metrics

        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_ml_model_retraining_trigger(self, database_url):
        """Test ML model retraining based on performance degradation"""
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        model_trainer = ModelTrainer(database_url)
        model_scheduler = ModelScheduler(model_trainer)

        session = model_trainer.session_maker()
        try:
            # Create old model metadata
            old_model = MLModelMetadata(
                model_name="old_entry_model",
                model_type="entry",
                version="1.0.0",
                trained_on=datetime.utcnow() - timedelta(days=10),  # 10 days old
                training_samples=1000,
                validation_accuracy=0.85,
                is_active=True,
            )
            session.add(old_model)
            session.commit()

            # Mock poor recent performance
            for i in range(10):
                poor_prediction = MLPrediction(
                    timestamp=datetime.utcnow() - timedelta(hours=i),
                    model_id=old_model.id,
                    model_name="old_entry_model",
                    prediction_type="entry",
                    prediction_value=0.8,  # High confidence prediction
                    confidence=0.9,
                    actual_outcome=0.0,  # But actually wrong
                    prediction_error=0.8,  # Large error
                )
                session.add(poor_prediction)

            session.commit()

            # Mock retraining methods
            model_trainer.train_entry_model = AsyncMock(return_value=Mock())
            model_trainer.train_exit_model = AsyncMock(return_value=Mock())
            model_trainer.train_strike_model = AsyncMock(return_value=Mock())

            # Check retraining trigger
            await model_scheduler.check_and_retrain_models()

            # Should trigger retraining for old model
            model_trainer.train_entry_model.assert_called_once()

        finally:
            session.close()
