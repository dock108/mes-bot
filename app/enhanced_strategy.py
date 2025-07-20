"""
Enhanced ML-Powered Trading Strategy for MES 0DTE Lotto-Grid Options Bot
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.config import config
from app.data_providers.vix_provider import VIXProvider
from app.decision_engine import DecisionEngine, ExitSignal, TradingSignal
from app.feature_pipeline import FeatureCollector
from app.ib_client import IBClient
from app.ml_training import ModelScheduler, ModelTrainer
from app.models import DecisionHistory, MarketData, Trade, get_session_maker
from app.risk_manager import RiskManager
from app.strategy import LottoGridStrategy  # Import base strategy

logger = logging.getLogger(__name__)


class EnhancedLottoGridStrategy(LottoGridStrategy):
    """
    Enhanced strategy that combines the original volatility-based approach
    with advanced ML-powered decision making
    """

    def __init__(self, ib_client: IBClient, risk_manager: RiskManager, database_url: str):
        # Initialize base strategy
        super().__init__(ib_client, risk_manager, database_url)

        # Enhanced components
        self.decision_engine = DecisionEngine(database_url)
        self.feature_collector = FeatureCollector(database_url)
        self.model_trainer = ModelTrainer(database_url)
        self.model_scheduler = ModelScheduler(self.model_trainer)

        # Configuration
        self.ml_enabled = getattr(config.trading, "ml_enabled", True)
        self.ml_confidence_threshold = getattr(config.trading, "ml_confidence_threshold", 0.4)
        self.fallback_to_basic = getattr(config.trading, "fallback_to_basic", True)

        # Performance tracking
        self.ml_decisions = []
        self.basic_decisions = []
        self.decision_accuracy = {"ml": [], "basic": []}

        # Enhanced state tracking
        self.last_feature_collection = None
        self.last_ml_training = None
        self.current_market_regime = None
        self.last_option_chain_update = None
        self.option_chain_data = None

        # VIX data provider
        self.vix_provider = None  # Initialize when needed

        logger.info(f"Enhanced strategy initialized. ML enabled: {self.ml_enabled}")

    async def initialize_daily_session(self) -> bool:
        """Enhanced daily initialization with ML components"""
        try:
            # Call base initialization
            base_success = await super().initialize_daily_session()
            if not base_success:
                return False

            logger.info("Initializing enhanced ML components...")

            # Initialize VIX provider
            try:
                self.vix_provider = VIXProvider()
                logger.info("VIX data provider initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize VIX provider: {e}")
                self.vix_provider = None

            # Initialize decision engine with current market data
            await self._initialize_decision_engine()

            # Check and potentially retrain models
            await self.model_scheduler.check_and_retrain_models()

            # Collect initial market features
            await self._collect_market_features()

            logger.info("Enhanced daily initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Enhanced daily initialization failed: {e}")
            # Fall back to basic strategy if ML initialization fails
            if self.fallback_to_basic:
                logger.warning("Falling back to basic strategy mode")
                self.ml_enabled = False
                return await super().initialize_daily_session()
            return False

    async def should_place_trade_enhanced(self) -> Tuple[bool, str, Optional[TradingSignal]]:
        """Enhanced trade decision using ML models"""

        # Always check basic conditions first
        basic_should_trade, basic_reason = self.should_place_trade()

        if not self.ml_enabled:
            return basic_should_trade, basic_reason, None

        try:
            # Collect current market data for ML
            await self._collect_market_features()

            # Get ML signal
            ml_signal = await self.decision_engine.generate_entry_signal(
                current_price=self.underlying_price,
                implied_move=self.implied_move,
                vix_level=await self._get_current_vix(),
            )

            # Record decision for tracking
            await self._record_decision(ml_signal)

            # Decision logic combining basic and ML signals
            should_trade, reason = self._combine_signals(
                basic_should_trade, basic_reason, ml_signal
            )

            return should_trade, reason, ml_signal if should_trade else None

        except Exception as e:
            logger.error(f"Error in enhanced trade decision: {e}")

            # Fall back to basic strategy
            if self.fallback_to_basic:
                logger.warning("ML decision failed, using basic strategy")
                return basic_should_trade, f"ML fallback: {basic_reason}", None
            else:
                return False, f"ML error: {str(e)}", None

    async def execute_trading_cycle_enhanced(self) -> bool:
        """Enhanced trading cycle with ML decision making"""
        try:
            # Update current price and features
            mes_contract = await self.ib_client.get_mes_contract()
            current_price = await self.ib_client.get_current_price(mes_contract)

            if not current_price:
                logger.warning("Could not get current price")
                return False

            self.underlying_price = current_price
            self.update_price_history(current_price)

            # Enhanced decision making
            should_trade, reason, ml_signal = await self.should_place_trade_enhanced()

            if not should_trade:
                logger.debug(f"Not placing trade: {reason}")
                return False

            # Determine strike levels (ML-optimized if available)
            if ml_signal and ml_signal.optimal_strikes:
                call_strike, put_strike = ml_signal.optimal_strikes
                logger.info(f"Using ML-optimized strikes: {call_strike}C / {put_strike}P")
            else:
                # Fall back to basic strike calculation
                strike_pairs = self.calculate_strike_levels(current_price)
                if not strike_pairs:
                    logger.warning("No strike levels calculated")
                    return False
                call_strike, put_strike = strike_pairs[0]
                logger.info(f"Using basic strike calculation: {call_strike}C / {put_strike}P")

            # Adjust position size based on ML confidence
            position_multiplier = 1.0
            if ml_signal:
                position_multiplier = ml_signal.position_size_multiplier
                logger.info(f"ML position size multiplier: {position_multiplier:.2f}")

            # Place the enhanced strangle
            result = await self.place_enhanced_strangle_trade(
                call_strike=call_strike,
                put_strike=put_strike,
                position_multiplier=position_multiplier,
                profit_target_multiplier=ml_signal.profit_target_multiplier if ml_signal else 4.0,
                ml_signal=ml_signal,
            )

            return result is not None

        except Exception as e:
            logger.error(f"Error in enhanced trading cycle: {e}")
            # Fall back to basic trading cycle
            if self.fallback_to_basic:
                return await super().execute_trading_cycle()
            return False

    async def place_enhanced_strangle_trade(
        self,
        call_strike: float,
        put_strike: float,
        position_multiplier: float = 1.0,
        profit_target_multiplier: float = 4.0,
        ml_signal: Optional[TradingSignal] = None,
    ) -> Optional[Dict]:
        """Place strangle with enhanced ML-based parameters"""
        try:
            logger.info(f"Placing enhanced strangle: {call_strike}C / {put_strike}P")
            logger.info(
                f"Position multiplier: {position_multiplier:.2f}, Profit target: {profit_target_multiplier:.1f}x"
            )

            # Get current account equity
            account_values = await self.ib_client.get_account_values()
            current_equity = account_values.get("NetLiquidation", 0)

            # Get estimated premium cost
            expiry = self.ib_client.get_today_expiry_string()
            call_contract = await self.ib_client.get_mes_option_contract(expiry, call_strike, "C")
            put_contract = await self.ib_client.get_mes_option_contract(expiry, put_strike, "P")

            call_price = await self.ib_client.get_current_price(call_contract)
            put_price = await self.ib_client.get_current_price(put_contract)

            if not call_price or not put_price:
                logger.warning("Could not get option prices for enhanced strangle")
                return None

            # Calculate adjusted premium with position multiplier
            base_premium = (call_price + put_price) * 5  # MES multiplier
            adjusted_premium = base_premium * position_multiplier

            # Enhanced risk check
            can_trade, reason = self.risk_manager.can_open_new_trade(
                adjusted_premium, current_equity
            )
            if not can_trade:
                logger.warning(f"Enhanced risk check failed: {reason}")
                return None

            # Place the strangle with enhanced parameters
            strangle_result = await self.ib_client.place_strangle_legacy(
                self.underlying_price,
                call_strike,
                put_strike,
                expiry,
                config.trading.max_premium_per_strangle
                * position_multiplier,  # Adjusted max premium
            )

            # Record enhanced trade in database
            trade_record = await self._record_enhanced_trade(
                strangle_result=strangle_result,
                position_multiplier=position_multiplier,
                profit_target_multiplier=profit_target_multiplier,
                ml_signal=ml_signal,
            )

            self.last_trade_time = datetime.utcnow()

            logger.info(f"Enhanced strangle placed successfully:")
            logger.info(f"  Call: {call_strike} @ ${call_price:.2f}")
            logger.info(f"  Put: {put_strike} @ ${put_price:.2f}")
            logger.info(f"  Total Premium: ${adjusted_premium:.2f}")
            logger.info(f"  Profit Target: {profit_target_multiplier:.1f}x")
            logger.info(f"  Trade ID: {trade_record.id}")

            return {
                "trade_record": trade_record,
                "strangle_result": strangle_result,
                "ml_enhanced": ml_signal is not None,
            }

        except Exception as e:
            logger.error(f"Error placing enhanced strangle trade: {e}")
            return None

    async def update_open_positions_enhanced(self):
        """Enhanced position updates with ML exit signals"""
        session = self.session_maker()
        try:
            # Get open trades from database
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()

            for trade in open_trades:
                # Update basic P&L
                await self._update_trade_pnl(trade, session)

                if self.ml_enabled:
                    # Check for ML exit signals
                    await self._check_ml_exit_signal(trade, session)

            session.commit()

        except Exception as e:
            logger.error(f"Error updating enhanced positions: {e}")
            session.rollback()
        finally:
            session.close()

    async def _initialize_decision_engine(self):
        """Initialize decision engine with current market state"""
        try:
            # Get current market data
            mes_contract = await self.ib_client.get_mes_contract()
            market_data = await self.ib_client.get_market_data(mes_contract)

            if market_data:
                # Get ATM IV
                atm_strike = self._round_to_strike(market_data["mid"])
                expiry = self.ib_client.get_today_expiry_string()
                atm_call_contract = await self.ib_client.get_mes_option_contract(
                    expiry, atm_strike, "C"
                )
                atm_iv = await self.ib_client.get_option_implied_volatility(atm_call_contract)
                if not atm_iv:
                    atm_iv = 0.2  # Fallback default

                # Initialize with real market data
                await self.decision_engine.update_market_data(
                    price=market_data["mid"],
                    bid=market_data["bid"],
                    ask=market_data["ask"],
                    volume=market_data["volume"],
                    atm_iv=atm_iv,
                    vix_level=await self._get_current_vix(),
                )

                logger.info("Decision engine initialized with real market data")
                logger.debug(
                    f"Market data: bid={market_data['bid']:.2f}, ask={market_data['ask']:.2f}, "
                    f"mid={market_data['mid']:.2f}, volume={market_data['volume']}, "
                    f"ATM IV={atm_iv:.3f}"
                )
        except Exception as e:
            logger.error(f"Error initializing decision engine: {e}")

    async def _collect_market_features(self):
        """Collect current market features for ML models"""
        try:
            current_time = datetime.utcnow()

            # Rate limit feature collection (every 5 minutes)
            if (
                self.last_feature_collection
                and (current_time - self.last_feature_collection).total_seconds() < 300
            ):
                return

            if not self.underlying_price or not self.implied_move:
                return

            # Get real market data
            mes_contract = await self.ib_client.get_mes_contract()
            market_data = await self.ib_client.get_market_data(mes_contract)

            if not market_data:
                logger.warning("Could not get market data for feature collection")
                return

            # Get ATM IV
            atm_strike = self._round_to_strike(market_data["mid"])
            expiry = self.ib_client.get_today_expiry_string()
            atm_call_contract = await self.ib_client.get_mes_option_contract(
                expiry, atm_strike, "C"
            )
            atm_iv = await self.ib_client.get_option_implied_volatility(atm_call_contract)
            if not atm_iv:
                atm_iv = 0.2  # Fallback default
                logger.warning("Could not get ATM IV, using default 0.2")

            # Get option chain data periodically (every 15 minutes)
            if (
                not self.last_option_chain_update
                or (current_time - self.last_option_chain_update).total_seconds() > 900
            ):
                self.option_chain_data = await self._collect_option_chain_data()
                self.last_option_chain_update = current_time

            # Collect real market data
            await self.feature_collector.collect_market_data(
                price=market_data["mid"],
                bid=market_data["bid"],
                ask=market_data["ask"],
                volume=market_data["volume"],
                atm_iv=atm_iv,
                implied_move=self.implied_move,
                vix_level=await self._get_current_vix(),
                timestamp=current_time,
            )

            # Calculate and store features with option chain data
            features_id = await self.feature_collector.calculate_and_store_features(
                current_price=market_data["mid"],
                implied_move=self.implied_move,
                vix_level=await self._get_current_vix(),
                timestamp=current_time,
                option_chain_data=self.option_chain_data,
            )

            self.last_feature_collection = current_time
            logger.info(
                f"Collected real market features: bid={market_data['bid']:.2f}, "
                f"ask={market_data['ask']:.2f}, volume={market_data['volume']}, "
                f"ATM IV={atm_iv:.3f}, features_id={features_id}"
            )

            if self.option_chain_data:
                logger.info(
                    f"Option chain metrics: P/C ratio={self.option_chain_data.get('put_call_ratio', 0):.2f}, "
                    f"Net gamma={self.option_chain_data.get('net_gamma', 0):.0f}"
                )

        except Exception as e:
            logger.error(f"Error collecting market features: {e}")

    async def _get_current_vix(self) -> float:
        """Get current VIX level from FRED API"""
        if self.vix_provider:
            try:
                vix_value = self.vix_provider.get_latest_vix()
                logger.debug(f"Current VIX: {vix_value}")
                return vix_value
            except Exception as e:
                logger.warning(f"Failed to get VIX data: {e}")

        # Fallback to default value
        logger.debug("Using default VIX value: 20.0")
        return 20.0

    async def _collect_option_chain_data(self) -> Optional[Dict]:
        """Collect option chain data for market microstructure features"""
        try:
            expiry = self.ib_client.get_today_expiry_string()
            chain_data = await self.ib_client.get_option_chain_data("MES", expiry, num_strikes=10)

            if chain_data:
                logger.info(
                    f"Collected option chain data: {len(chain_data.get('strikes', {}))} strikes, "
                    f"P/C ratio: {chain_data.get('put_call_ratio', 0):.2f}"
                )
            return chain_data

        except Exception as e:
            logger.error(f"Error collecting option chain data: {e}")
            return None

    def _combine_signals(
        self, basic_should_trade: bool, basic_reason: str, ml_signal: TradingSignal
    ) -> Tuple[bool, str]:
        """Combine basic and ML signals for final decision"""

        # If ML confidence is very high, trust ML
        if ml_signal.confidence > 0.8:
            should_trade = ml_signal.action == "ENTER"
            reason = (
                f"High ML confidence ({ml_signal.confidence:.2f}): {', '.join(ml_signal.reasoning)}"
            )

        # If ML confidence is moderate, require agreement with basic signal
        elif ml_signal.confidence > self.ml_confidence_threshold:
            ml_agrees = ml_signal.action == "ENTER"
            should_trade = basic_should_trade and ml_agrees

            if should_trade:
                reason = f"ML+Basic agreement ({ml_signal.confidence:.2f}): {basic_reason}"
            else:
                disagreement = "ML disagrees" if not ml_agrees else "Basic disagrees"
                reason = f"No consensus ({disagreement}): {basic_reason}"

        # Low ML confidence - defer to basic strategy
        else:
            should_trade = basic_should_trade
            reason = f"Low ML confidence ({ml_signal.confidence:.2f}), using basic: {basic_reason}"

        return should_trade, reason

    async def _record_decision(self, ml_signal: TradingSignal):
        """Record decision in database for performance tracking"""
        session = self.session_maker()
        try:
            decision = DecisionHistory(
                action=ml_signal.action,
                confidence=ml_signal.confidence,
                underlying_price=self.underlying_price,
                implied_move=self.implied_move,
                reasoning=ml_signal.reasoning,
                model_predictions=ml_signal.model_predictions,
                suggested_call_strike=(
                    ml_signal.optimal_strikes[0] if ml_signal.optimal_strikes else None
                ),
                suggested_put_strike=(
                    ml_signal.optimal_strikes[1] if ml_signal.optimal_strikes else None
                ),
                position_size_multiplier=ml_signal.position_size_multiplier,
                profit_target_multiplier=ml_signal.profit_target_multiplier,
            )

            session.add(decision)
            session.commit()

            logger.debug(
                f"Recorded decision: {ml_signal.action} with confidence {ml_signal.confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"Error recording decision: {e}")
            session.rollback()
        finally:
            session.close()

    async def _record_enhanced_trade(
        self,
        strangle_result: Dict,
        position_multiplier: float,
        profit_target_multiplier: float,
        ml_signal: Optional[TradingSignal],
    ) -> Trade:
        """Record enhanced trade with ML metadata"""
        session = self.session_maker()
        try:
            trade = Trade(
                date=datetime.utcnow().date(),
                entry_time=datetime.utcnow(),
                underlying_symbol="MES",
                underlying_price_at_entry=self.underlying_price,
                implied_move=self.implied_move,
                call_strike=strangle_result["call_strike"],
                put_strike=strangle_result["put_strike"],
                call_premium=strangle_result["call_price"],
                put_premium=strangle_result["put_price"],
                total_premium=strangle_result["total_premium"] * position_multiplier,
                status="OPEN",
                call_status="OPEN",
                put_status="OPEN",
            )

            # Store IB order IDs
            if strangle_result.get("call_trades"):
                call_entry_trade = strangle_result["call_trades"][0]
                trade.call_order_id = call_entry_trade.order.orderId
                if len(strangle_result["call_trades"]) > 1:
                    trade.call_tp_order_id = strangle_result["call_trades"][1].order.orderId

            if strangle_result.get("put_trades"):
                put_entry_trade = strangle_result["put_trades"][0]
                trade.put_order_id = put_entry_trade.order.orderId
                if len(strangle_result["put_trades"]) > 1:
                    trade.put_tp_order_id = strangle_result["put_trades"][1].order.orderId

            session.add(trade)
            session.commit()

            # Link decision to trade if available
            if ml_signal:
                await self._link_decision_to_trade(trade.id, ml_signal)

            logger.info(f"Enhanced trade recorded with ID: {trade.id}")
            return trade

        except Exception as e:
            logger.error(f"Error recording enhanced trade: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    async def _link_decision_to_trade(self, trade_id: int, ml_signal: TradingSignal):
        """Link decision record to completed trade"""
        session = self.session_maker()
        try:
            # Find recent decision that led to this trade
            recent_decision = (
                session.query(DecisionHistory)
                .filter(
                    DecisionHistory.action == "ENTER",
                    DecisionHistory.trade_id.is_(None),
                    DecisionHistory.timestamp >= datetime.utcnow() - timedelta(minutes=5),
                )
                .order_by(DecisionHistory.timestamp.desc())
                .first()
            )

            if recent_decision:
                recent_decision.trade_id = trade_id
                session.commit()
                logger.debug(f"Linked decision to trade {trade_id}")

        except Exception as e:
            logger.error(f"Error linking decision to trade: {e}")
            session.rollback()
        finally:
            session.close()

    async def _check_ml_exit_signal(self, trade: Trade, session: Session):
        """Check for ML-based exit signals on open trade"""
        try:
            if not self.ml_enabled:
                return

            # Prepare trade info for exit signal
            trade_info = {
                "entry_time": trade.entry_time,
                "entry_price": trade.underlying_price_at_entry,
                "call_strike": trade.call_strike,
                "put_strike": trade.put_strike,
                "current_pnl": trade.unrealized_pnl or 0,
                "time_in_trade": (datetime.utcnow() - trade.entry_time).total_seconds() / 3600,
            }

            # Get ML exit signal
            exit_signal = await self.decision_engine.generate_exit_signal(
                trade_info=trade_info,
                current_price=self.underlying_price,
                implied_move=self.implied_move,
                vix_level=await self._get_current_vix(),
            )

            # Act on strong exit signals
            if exit_signal.should_exit and exit_signal.confidence > 0.7:
                logger.info(f"ML exit signal for trade {trade.id}: {exit_signal.reasoning}")

                # Could implement automatic exit here
                # For now, just log the signal
                await self._log_exit_signal(trade.id, exit_signal)

        except Exception as e:
            logger.error(f"Error checking ML exit signal for trade {trade.id}: {e}")

    async def _log_exit_signal(self, trade_id: int, exit_signal: ExitSignal):
        """Log exit signal for analysis"""
        session = self.session_maker()
        try:
            decision = DecisionHistory(
                action="EXIT",
                confidence=exit_signal.confidence,
                underlying_price=self.underlying_price,
                implied_move=self.implied_move,
                reasoning=exit_signal.reasoning,
                trade_id=trade_id,
            )

            session.add(decision)
            session.commit()

        except Exception as e:
            logger.error(f"Error logging exit signal: {e}")
            session.rollback()
        finally:
            session.close()

    def get_enhanced_strategy_status(self) -> Dict:
        """Get enhanced strategy status for monitoring"""
        base_status = self.get_strategy_status()

        enhanced_status = {
            **base_status,
            "ml_enabled": self.ml_enabled,
            "ml_confidence_threshold": self.ml_confidence_threshold,
            "last_feature_collection": self.last_feature_collection,
            "last_ml_training": self.last_ml_training,
            "ml_decisions_count": len(self.ml_decisions),
            "basic_decisions_count": len(self.basic_decisions),
            "current_market_regime": self.current_market_regime,
            "decision_engine_performance": self.decision_engine.get_performance_summary(),
        }

        return enhanced_status

    async def perform_ml_maintenance(self):
        """Perform periodic ML model maintenance"""
        try:
            logger.info("Performing ML maintenance...")

            # Check model performance and retrain if needed
            await self.model_scheduler.check_and_retrain_models()

            # Update model weights based on recent performance
            await self._update_model_weights()

            # Clean up old predictions and features
            await self._cleanup_old_data()

            self.last_ml_training = datetime.utcnow()
            logger.info("ML maintenance completed")

        except Exception as e:
            logger.error(f"Error in ML maintenance: {e}")

    async def _update_model_weights(self):
        """Update ensemble model weights based on recent performance"""
        try:
            performance_summary = self.decision_engine.get_performance_summary()

            # Adjust weights based on accuracy
            for model_name, metrics in performance_summary.items():
                if metrics["recent_accuracy"] > 0.6:
                    # Increase weight for well-performing models
                    current_weight = self.decision_engine.model_weights.get(model_name, 0.5)
                    self.decision_engine.model_weights[model_name] = min(0.8, current_weight * 1.1)
                elif metrics["recent_accuracy"] < 0.4:
                    # Decrease weight for poorly performing models
                    current_weight = self.decision_engine.model_weights.get(model_name, 0.5)
                    self.decision_engine.model_weights[model_name] = max(0.2, current_weight * 0.9)

            # Normalize weights
            total_weight = sum(self.decision_engine.model_weights.values())
            if total_weight > 0:
                for model_name in self.decision_engine.model_weights:
                    self.decision_engine.model_weights[model_name] /= total_weight

            logger.info(f"Updated model weights: {self.decision_engine.model_weights}")

        except Exception as e:
            logger.error(f"Error updating model weights: {e}")

    async def _cleanup_old_data(self):
        """Clean up old ML data to prevent database bloat"""
        session = self.session_maker()
        try:
            # Remove market data older than 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)

            deleted_market_data = (
                session.query(MarketData).filter(MarketData.timestamp < cutoff_date).delete()
            )

            # Remove decision history older than 1 year
            cutoff_date_decisions = datetime.utcnow() - timedelta(days=365)
            deleted_decisions = (
                session.query(DecisionHistory)
                .filter(DecisionHistory.timestamp < cutoff_date_decisions)
                .delete()
            )

            session.commit()

            logger.info(
                f"Cleaned up {deleted_market_data} old market data records and {deleted_decisions} old decision records"
            )

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            session.rollback()
        finally:
            session.close()
