#!/usr/bin/env python
"""
Script to generate synthetic training data for ML models
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config  # noqa: E402
from app.synthetic_data_generator import SyntheticDataGenerator  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Generate synthetic training data"""
    try:
        logger.info("Starting synthetic data generation...")

        # Create generator
        generator = SyntheticDataGenerator(config.database.url)

        # Generate dataset (default 1200 records)
        success = generator.create_synthetic_dataset(num_records=1200)

        if success:
            # Validate generated data
            validation = generator.validate_generated_data()

            logger.info("\n=== Synthetic Data Generation Complete ===")
            logger.info(f"Generated Records:")
            logger.info(f"  Features: {validation['data_counts']['features']}")
            logger.info(f"  Decisions: {validation['data_counts']['decisions']}")
            logger.info(f"  Trades: {validation['data_counts']['trades']}")

            logger.info(f"\nTrade Performance:")
            logger.info(f"  Win Rate: {validation['trade_metrics']['win_rate']:.1%}")
            logger.info(f"  Avg Profit: ${validation['trade_metrics']['avg_profit']:.2f}")
            logger.info(f"  Avg Loss: ${validation['trade_metrics']['avg_loss']:.2f}")
            logger.info(f"  Total P&L: ${validation['trade_metrics']['total_pnl']:.2f}")

            logger.info(f"\nMarket Characteristics:")
            logger.info(f"  Avg IV Rank: {validation['market_characteristics']['avg_iv_rank']:.1f}")
            logger.info(f"  Avg VIX: {validation['market_characteristics']['avg_vix']:.1f}")
            logger.info(
                f"  Regime Distribution: {validation['market_characteristics']['regime_distribution']}"
            )

            logger.info("\nSynthetic training data is now available for ML model training!")

        else:
            logger.error("Failed to generate synthetic data")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in synthetic data generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
