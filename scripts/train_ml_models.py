#!/usr/bin/env python
"""
Script to train ML models using available training data
"""

import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config  # noqa: E402
from app.ml_training import ModelTrainer  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Train ML models"""
    try:
        logger.info("Starting ML model training...")

        # Create trainer
        trainer = ModelTrainer(config.database.url)

        # Use last 60 days of data for training (or all available data)
        end_date = date.today()
        start_date = end_date - timedelta(days=60)

        # Train entry model
        logger.info(f"\nTraining entry prediction model...")
        entry_model = await trainer.train_entry_model(
            start_date, end_date, algorithm="random_forest"
        )

        if entry_model:
            logger.info(f"Entry model trained successfully!")
            logger.info(f"  Accuracy: {entry_model.performance.accuracy:.3f}")
            logger.info(f"  Precision: {entry_model.performance.precision:.3f}")
            logger.info(f"  Recall: {entry_model.performance.recall:.3f}")
            logger.info(f"  F1 Score: {entry_model.performance.f1_score:.3f}")

            # Show top features
            if entry_model.performance.feature_importance:
                logger.info("\nTop 10 Most Important Features:")
                sorted_features = sorted(
                    entry_model.performance.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
                for feature, importance in sorted_features:
                    logger.info(f"  {feature}: {importance:.4f}")
        else:
            logger.error("Failed to train entry model")

        # Train exit model
        logger.info(f"\nTraining exit prediction model...")
        exit_model = await trainer.train_exit_model(start_date, end_date, algorithm="random_forest")

        if exit_model:
            logger.info(f"Exit model trained successfully!")
            logger.info(f"  Accuracy: {exit_model.performance.accuracy:.3f}")
            logger.info(f"  Cross-validation mean: {exit_model.performance.cross_val_mean:.3f}")
        else:
            logger.warning("Exit model training skipped (insufficient data)")

        # Train strike optimization model
        logger.info(f"\nTraining strike optimization model...")
        strike_model = await trainer.train_strike_model(
            start_date, end_date, algorithm="random_forest"
        )

        if strike_model:
            logger.info(f"Strike optimization model trained successfully!")
            logger.info(f"  MSE: {strike_model.performance.mse:.3f}")
            logger.info(f"  R2 Score: {strike_model.performance.r2_score:.3f}")
        else:
            logger.warning("Strike model training skipped (insufficient data)")

        # Perform hyperparameter optimization
        logger.info(f"\nPerforming hyperparameter optimization for entry model...")
        best_params = await trainer.hyperparameter_optimization(
            "entry", "random_forest", start_date, end_date
        )
        logger.info(f"Best hyperparameters: {best_params}")

        logger.info("\n=== ML Model Training Complete ===")
        logger.info("Models are now available for enhanced trading decisions!")

    except Exception as e:
        logger.error(f"Error in ML model training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
