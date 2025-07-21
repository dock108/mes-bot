"""
Synthetic Training Data Generator for ML Model Training

This module generates realistic synthetic market data and trading scenarios
to enable immediate ML model training when historical data is limited.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import config
from app.market_indicators import MarketFeatures
from app.models import DecisionHistory
from app.models import MarketFeatures as MarketFeaturesModel
from app.models import Trade, get_session_maker

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic market data and trading scenarios for ML training"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.session_maker = get_session_maker(database_url)

        # Market regime parameters
        self.market_regimes = {
            "low_vol": {"vix_range": (12, 18), "vol_multiplier": 0.7, "trend_strength": 0.3},
            "normal": {"vix_range": (18, 25), "vol_multiplier": 1.0, "trend_strength": 0.5},
            "high_vol": {"vix_range": (25, 35), "vol_multiplier": 1.4, "trend_strength": 0.7},
            "crisis": {"vix_range": (35, 60), "vol_multiplier": 2.0, "trend_strength": 0.9},
        }

        # Trading outcome probabilities based on market conditions
        self.outcome_probabilities = {
            "favorable": {"win_rate": 0.65, "avg_profit": 3.2, "avg_loss": -1.8},
            "neutral": {"win_rate": 0.45, "avg_profit": 2.8, "avg_loss": -2.1},
            "unfavorable": {"win_rate": 0.25, "avg_profit": 2.4, "avg_loss": -2.5},
        }

    def generate_market_features(
        self, timestamp: datetime, regime: str = "normal", base_price: float = 4500.0
    ) -> MarketFeatures:
        """Generate realistic market features for a given timestamp and regime"""

        regime_params = self.market_regimes[regime]

        # Generate correlated volatility measures
        base_vol = np.random.uniform(0.15, 0.35) * regime_params["vol_multiplier"]
        realized_vol_15m = base_vol * np.random.uniform(0.8, 1.2)
        realized_vol_30m = base_vol * np.random.uniform(0.85, 1.15)
        realized_vol_60m = base_vol * np.random.uniform(0.9, 1.1)
        realized_vol_2h = base_vol * np.random.uniform(0.95, 1.05)
        realized_vol_daily = base_vol

        # Implied volatility with typical premium to realized
        atm_iv = base_vol * np.random.uniform(1.1, 1.4)

        # IV rank and percentile
        iv_rank = np.random.uniform(10, 90)
        iv_percentile = iv_rank + np.random.uniform(-5, 5)

        # Technical indicators with realistic correlations
        trend_strength = regime_params["trend_strength"]
        rsi_base = 50 + (np.random.random() - 0.5) * 40 * trend_strength
        rsi_5m = np.clip(rsi_base + np.random.uniform(-5, 5), 5, 95)
        rsi_15m = np.clip(rsi_base + np.random.uniform(-3, 3), 5, 95)
        rsi_30m = np.clip(rsi_base + np.random.uniform(-2, 2), 5, 95)

        # MACD signals
        macd_signal = (rsi_base - 50) * 0.02 + np.random.uniform(-0.1, 0.1)
        macd_histogram = macd_signal * 0.7 + np.random.uniform(-0.05, 0.05)

        # Bollinger Band position
        bb_position = 0.5 + (rsi_base - 50) * 0.008 + np.random.uniform(-0.1, 0.1)
        bb_position = np.clip(bb_position, 0, 1)

        # Price momentum correlated with RSI
        momentum_factor = (rsi_base - 50) * 0.001
        price_momentum_15m = momentum_factor + np.random.uniform(-0.005, 0.005)
        price_momentum_30m = momentum_factor * 0.8 + np.random.uniform(-0.003, 0.003)
        price_momentum_60m = momentum_factor * 0.6 + np.random.uniform(-0.002, 0.002)

        # VIX and market regime
        vix_level = np.random.uniform(*regime_params["vix_range"])
        vix_term_structure = np.random.uniform(-0.1, 0.2)  # Usually positive

        # Market microstructure
        bid_ask_spread = base_vol * np.random.uniform(0.01, 0.03)
        volume_profile = np.random.uniform(0.5, 1.5)

        # Time-based features
        hour = timestamp.hour + timestamp.minute / 60.0
        time_of_day = hour
        day_of_week = timestamp.weekday()

        # Options-specific features
        put_call_ratio = np.random.uniform(0.8, 1.3)
        gamma_exposure = np.random.uniform(-0.1, 0.1) * regime_params["vol_multiplier"]

        # Generate price based on momentum and volatility
        price_change = price_momentum_15m * base_price + np.random.normal(
            0, base_vol * base_price * 0.01
        )
        current_price = base_price + price_change

        return MarketFeatures(
            realized_vol_15m=realized_vol_15m,
            realized_vol_30m=realized_vol_30m,
            realized_vol_60m=realized_vol_60m,
            realized_vol_2h=realized_vol_2h,
            realized_vol_daily=realized_vol_daily,
            atm_iv=atm_iv,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_skew=np.random.uniform(-0.1, 0.1),
            iv_term_structure=np.random.uniform(-0.05, 0.05),
            rsi_5m=rsi_5m,
            rsi_15m=rsi_15m,
            rsi_30m=rsi_30m,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_position=bb_position,
            bb_squeeze=np.random.uniform(0, 0.05),
            price_momentum_15m=price_momentum_15m,
            price_momentum_30m=price_momentum_30m,
            price_momentum_60m=price_momentum_60m,
            support_resistance_strength=np.random.uniform(0, 1),
            mean_reversion_signal=np.random.uniform(-0.5, 0.5),
            bid_ask_spread=bid_ask_spread,
            option_volume_ratio=np.random.uniform(0.8, 1.2),
            put_call_ratio=put_call_ratio,
            gamma_exposure=gamma_exposure,
            vix_level=vix_level,
            vix_term_structure=vix_term_structure,
            market_correlation=np.random.uniform(0.3, 0.9),
            volume_profile=volume_profile,
            market_regime=regime,
            time_of_day=time_of_day,
            day_of_week=float(day_of_week),
            time_to_expiry=np.random.uniform(1, 8),  # Hours to expiry
            days_since_last_trade=np.random.uniform(0, 5),
            win_rate_recent=np.random.uniform(0.2, 0.7),
            profit_factor_recent=np.random.uniform(0.8, 2.5),
            sharpe_ratio_recent=np.random.uniform(-1.0, 2.0),
            price=current_price,
            volume=np.random.uniform(100000, 500000),
            timestamp=timestamp,
        )

    def classify_market_condition(self, features: MarketFeatures) -> str:
        """Classify market condition based on features"""

        # Favorable conditions: low realized vol, high IV, neutral RSI
        vol_ratio = features.realized_vol_30m / features.atm_iv if features.atm_iv > 0 else 1.0
        rsi_neutral = 30 < features.rsi_30m < 70
        low_vol_premium = vol_ratio < 0.7
        high_iv_rank = features.iv_rank > 60

        score = 0
        if low_vol_premium:
            score += 2
        if rsi_neutral:
            score += 1
        if high_iv_rank:
            score += 1
        if features.vix_level < 20:
            score += 1
        if 10 < features.time_of_day < 15:  # Good trading hours
            score += 1

        if score >= 4:
            return "favorable"
        elif score >= 2:
            return "neutral"
        else:
            return "unfavorable"

    def generate_trade_outcome(
        self, condition: str, features: MarketFeatures
    ) -> Tuple[bool, float]:
        """Generate realistic trade outcome based on market condition"""

        prob_params = self.outcome_probabilities[condition]

        # Determine if trade wins
        is_winner = np.random.random() < prob_params["win_rate"]

        if is_winner:
            # Winning trade - profit target hit
            base_profit = prob_params["avg_profit"]
            profit_multiplier = base_profit + np.random.uniform(-0.5, 0.5)
            return True, profit_multiplier
        else:
            # Losing trade
            base_loss = prob_params["avg_loss"]
            loss_multiplier = base_loss + np.random.uniform(-0.3, 0.3)
            return False, loss_multiplier

    def create_synthetic_dataset(self, num_records: int = 1000) -> bool:
        """Create a complete synthetic dataset for ML training"""

        logger.info(f"Generating {num_records} synthetic training records...")

        session = self.session_maker()
        try:
            # Generate data over the past 6 months
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=180)

            generated_features = []
            generated_decisions = []
            generated_trades = []

            for i in range(num_records):
                # Random timestamp in the range
                timestamp = start_date + timedelta(
                    seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
                )

                # Skip weekends
                if timestamp.weekday() >= 5:
                    continue

                # Skip outside market hours (9:30 AM - 4:00 PM ET)
                if timestamp.hour < 9 or timestamp.hour >= 16:
                    continue

                # Choose market regime
                regime_weights = [0.3, 0.4, 0.25, 0.05]  # low_vol, normal, high_vol, crisis
                regime = np.random.choice(list(self.market_regimes.keys()), p=regime_weights)

                # Generate market features
                features = self.generate_market_features(timestamp, regime)

                # Store market features
                features_model = MarketFeaturesModel(
                    timestamp=timestamp,
                    realized_vol_15m=features.realized_vol_15m,
                    realized_vol_30m=features.realized_vol_30m,
                    realized_vol_60m=features.realized_vol_60m,
                    realized_vol_2h=features.realized_vol_2h,
                    realized_vol_daily=features.realized_vol_daily,
                    atm_iv=features.atm_iv,
                    iv_rank=features.iv_rank,
                    iv_percentile=features.iv_percentile,
                    iv_skew=features.iv_skew,
                    iv_term_structure=features.iv_term_structure,
                    rsi_5m=features.rsi_5m,
                    rsi_15m=features.rsi_15m,
                    rsi_30m=features.rsi_30m,
                    macd_signal=features.macd_signal,
                    macd_histogram=features.macd_histogram,
                    bb_position=features.bb_position,
                    bb_squeeze=features.bb_squeeze,
                    price_momentum_15m=features.price_momentum_15m,
                    price_momentum_30m=features.price_momentum_30m,
                    price_momentum_60m=features.price_momentum_60m,
                    support_resistance_strength=features.support_resistance_strength,
                    mean_reversion_signal=features.mean_reversion_signal,
                    bid_ask_spread=features.bid_ask_spread,
                    option_volume_ratio=features.option_volume_ratio,
                    put_call_ratio=features.put_call_ratio,
                    gamma_exposure=features.gamma_exposure,
                    vix_level=features.vix_level,
                    vix_term_structure=features.vix_term_structure,
                    market_correlation=features.market_correlation,
                    volume_profile=features.volume_profile,
                    market_regime=features.market_regime,
                    time_of_day=features.time_of_day,
                    day_of_week=features.day_of_week,
                    time_to_expiry=features.time_to_expiry,
                    days_since_last_trade=features.days_since_last_trade,
                    win_rate_recent=features.win_rate_recent,
                    profit_factor_recent=features.profit_factor_recent,
                    sharpe_ratio_recent=features.sharpe_ratio_recent,
                    price=features.price,
                    volume=features.volume,
                )
                generated_features.append(features_model)

                # Generate decision
                condition = self.classify_market_condition(features)
                should_enter = condition in ["favorable", "neutral"] and np.random.random() > 0.3

                confidence = {
                    "favorable": np.random.uniform(0.7, 0.95),
                    "neutral": np.random.uniform(0.4, 0.7),
                    "unfavorable": np.random.uniform(0.1, 0.4),
                }[condition]

                # Store the features first to get an ID
                session.add(features_model)
                session.flush()  # Get the ID without committing

                decision = DecisionHistory(
                    timestamp=timestamp,
                    action="ENTER" if should_enter else "HOLD",
                    confidence=confidence,
                    underlying_price=features.price,
                    implied_move=features.price * 0.02,  # Typical 2% implied move
                    features_id=features_model.id,  # Link to features
                    reasoning=[f"Market condition: {condition}", f"Confidence: {confidence:.2f}"],
                    model_predictions={
                        "volatility_based": confidence * 0.8,
                        "ml_ensemble": confidence * 1.2,
                    },
                )
                generated_decisions.append(decision)

                # Generate trade if decision was to enter
                if (
                    should_enter and np.random.random() > 0.1
                ):  # Some decisions don't result in trades
                    is_winner, pnl_multiplier = self.generate_trade_outcome(condition, features)

                    # Calculate trade details
                    premium_collected = np.random.uniform(
                        15, 40
                    )  # Typical premium for 0DTE strangles
                    pnl = premium_collected * pnl_multiplier

                    exit_time = timestamp + timedelta(hours=np.random.uniform(1, 6))
                    call_strike = features.price + np.random.uniform(25, 75)
                    put_strike = features.price - np.random.uniform(25, 75)

                    trade = Trade(
                        date=timestamp.date(),
                        entry_time=timestamp,
                        exit_time=exit_time,
                        underlying_symbol="MES",
                        underlying_price_at_entry=features.price,
                        implied_move=features.price * 0.02,
                        call_strike=call_strike,
                        put_strike=put_strike,
                        call_premium=premium_collected * 0.5,
                        put_premium=premium_collected * 0.5,
                        total_premium=premium_collected,
                        call_exit_price=0.01 if is_winner else premium_collected * 0.6,
                        put_exit_price=0.01 if is_winner else premium_collected * 0.6,
                        realized_pnl=pnl,
                        status="CLOSED_WIN" if is_winner else "CLOSED_LOSS",
                        call_status="CLOSED_PROFIT" if is_winner else "EXPIRED",
                        put_status="CLOSED_PROFIT" if is_winner else "EXPIRED",
                    )
                    session.add(trade)
                    session.flush()  # Get trade ID

                    # Update decision with trade outcome
                    decision.trade_id = trade.id
                    decision.actual_outcome = pnl

                    generated_trades.append(trade)

            # Add remaining decisions (features and trades already added individually)
            logger.info("Finalizing synthetic data insertion...")
            session.add_all(generated_decisions)
            session.commit()

            logger.info(f"Successfully generated synthetic dataset:")
            logger.info(f"  - {len(generated_features)} market feature records")
            logger.info(f"  - {len(generated_decisions)} decision records")
            logger.info(f"  - {len(generated_trades)} trade records")

            return True

        except Exception as e:
            logger.error(f"Error generating synthetic dataset: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def validate_generated_data(self) -> Dict[str, Any]:
        """Validate the quality and realism of generated data"""

        session = self.session_maker()
        try:
            # Get basic counts
            features_count = session.query(MarketFeaturesModel).count()
            decisions_count = session.query(DecisionHistory).count()
            trades_count = session.query(Trade).count()

            # Analyze trade outcomes
            trades = session.query(Trade).all()
            if trades:
                win_rate = sum(1 for t in trades if t.realized_pnl > 0) / len(trades)
                avg_profit = np.mean([t.realized_pnl for t in trades if t.realized_pnl > 0])
                avg_loss = np.mean([t.realized_pnl for t in trades if t.realized_pnl < 0])
                total_pnl = sum(t.realized_pnl for t in trades)
            else:
                win_rate = avg_profit = avg_loss = total_pnl = 0

            # Analyze market features
            features = session.query(MarketFeaturesModel).all()
            if features:
                avg_iv_rank = np.mean([f.iv_rank for f in features])
                avg_vix = np.mean([f.vix_level for f in features])
                regime_distribution = {}
                for f in features:
                    regime_distribution[f.market_regime] = (
                        regime_distribution.get(f.market_regime, 0) + 1
                    )
            else:
                avg_iv_rank = avg_vix = 0
                regime_distribution = {}

            validation_results = {
                "data_counts": {
                    "features": features_count,
                    "decisions": decisions_count,
                    "trades": trades_count,
                },
                "trade_metrics": {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "avg_loss": avg_loss,
                    "total_pnl": total_pnl,
                },
                "market_characteristics": {
                    "avg_iv_rank": avg_iv_rank,
                    "avg_vix": avg_vix,
                    "regime_distribution": regime_distribution,
                },
            }

            return validation_results

        except Exception as e:
            logger.error(f"Error validating synthetic data: {e}")
            return {}
        finally:
            session.close()


def main():
    """Generate synthetic training data"""
    generator = SyntheticDataGenerator(config.database.url)

    # Generate dataset
    success = generator.create_synthetic_dataset(num_records=1200)

    if success:
        # Validate generated data
        validation = generator.validate_generated_data()
        print("\n=== Synthetic Data Generation Complete ===")
        print(f"Generated Records:")
        print(f"  Features: {validation['data_counts']['features']}")
        print(f"  Decisions: {validation['data_counts']['decisions']}")
        print(f"  Trades: {validation['data_counts']['trades']}")
        print(f"\nTrade Performance:")
        print(f"  Win Rate: {validation['trade_metrics']['win_rate']:.1%}")
        print(f"  Avg Profit: ${validation['trade_metrics']['avg_profit']:.2f}")
        print(f"  Avg Loss: ${validation['trade_metrics']['avg_loss']:.2f}")
        print(f"  Total P&L: ${validation['trade_metrics']['total_pnl']:.2f}")
        print(f"\nMarket Characteristics:")
        print(f"  Avg IV Rank: {validation['market_characteristics']['avg_iv_rank']:.1f}")
        print(f"  Avg VIX: {validation['market_characteristics']['avg_vix']:.1f}")
        print(
            f"  Regime Distribution: {validation['market_characteristics']['regime_distribution']}"
        )
    else:
        print("Failed to generate synthetic data")


if __name__ == "__main__":
    main()
