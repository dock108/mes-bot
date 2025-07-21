"""
Core trading strategy logic for 0DTE Lotto-Grid Options Bot
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.base_strategy import MultiInstrumentStrategy
from app.config import config
from app.ib_client import IBClient
from app.models import Trade, get_session_maker
from app.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class LottoGridStrategy(MultiInstrumentStrategy):
    """
    Implements the 0DTE options strangle strategy for multiple instruments:
    - Calculate implied move from ATM straddle
    - Place OTM strangles when realized volatility < implied
    - Target 4x profit, accept 100% loss
    - Manage risk with position limits
    """

    def __init__(self, ib_client: IBClient, risk_manager: RiskManager, database_url: str):
        super().__init__(ib_client, risk_manager, database_url)

    async def initialize_daily_session(self) -> bool:
        """Initialize strategy for the trading day"""
        try:
            logger.info("Initializing daily trading session...")

            # Initialize each active instrument
            successful_inits = 0
            for symbol in self.get_active_symbols():
                try:
                    success = await self.initialize_instrument_session(symbol)
                    if success:
                        successful_inits += 1
                    else:
                        logger.error(f"Failed to initialize {symbol}")
                except Exception as e:
                    logger.error(f"Error initializing {symbol}: {e}")

            if successful_inits == 0:
                logger.error("Failed to initialize any instruments")
                return False

            logger.info(
                f"Successfully initialized {successful_inits}/{len(self.get_active_symbols())} instruments"
            )

            # Set starting equity for risk management
            account_values = await self.ib_client.get_account_values()
            if "NetLiquidation" in account_values:
                self.risk_manager.set_daily_start_equity(account_values["NetLiquidation"])

            return True

        except Exception as e:
            logger.error(f"Failed to initialize daily session: {e}")
            return False

    async def initialize_instrument_session(self, symbol: str) -> bool:
        """Initialize session for a specific instrument"""
        try:
            logger.info(f"Initializing {symbol} session...")

            # Get current price
            current_price = await self.get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current {symbol} price")
                return False

            # Update instrument state
            self.update_instrument_state(
                symbol,
                underlying_price=current_price,
                daily_high=current_price,
                daily_low=current_price,
            )

            # Update price history
            self.update_price_history(symbol, current_price)

            # Calculate implied move from ATM straddle
            expiry = self.ib_client.get_today_expiry_string()
            call_price, put_price, implied_move = await self.get_atm_straddle_price(
                symbol, current_price, expiry
            )

            self.update_instrument_state(symbol, implied_move=implied_move)

            logger.info(f"{symbol} session initialized:")
            logger.info(f"  {symbol} Price: ${current_price:.2f}")
            logger.info(f"  ATM Call: ${call_price:.2f}")
            logger.info(f"  ATM Put: ${put_price:.2f}")
            logger.info(f"  Implied Move: ${implied_move:.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize {symbol} session: {e}")
            return False

    async def execute_instrument_cycle(self, symbol: str) -> bool:
        """Execute trading cycle for specific instrument"""
        try:
            # Update current price
            current_price = await self.get_current_price(symbol)
            if current_price:
                self.update_price_history(symbol, current_price)

            # Check if we should place a trade
            should_trade, reason = self.should_place_trade_for_symbol(symbol)
            if not should_trade:
                logger.debug(f"{symbol}: {reason}")
                return False

            # Calculate strike levels
            strike_levels = self.calculate_strike_levels(symbol, current_price)
            if not strike_levels:
                logger.warning(f"{symbol}: No strike levels calculated")
                return False

            # Try to place a strangle trade
            call_strike, put_strike = strike_levels[0]  # Use first level
            trade_result = await self.place_strangle_trade_for_symbol(
                symbol, call_strike, put_strike
            )

            if trade_result:
                logger.info(f"{symbol}: Successfully placed strangle trade")
                self.update_instrument_state(symbol, last_trade_time=datetime.utcnow())
                return True
            else:
                logger.warning(f"{symbol}: Failed to place strangle trade")
                return False

        except Exception as e:
            logger.error(f"Error executing trading cycle for {symbol}: {e}")
            return False

    def should_place_trade_for_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Determine if conditions are met to place a new strangle for instrument"""
        state = self.get_instrument_state(symbol)
        spec = self.get_instrument_spec(symbol)

        if not state["implied_move"]:
            return False, f"{symbol}: Implied move not calculated"

        # Check minimum time between trades for this instrument
        if state["last_trade_time"]:
            time_since_last = datetime.utcnow() - state["last_trade_time"]
            min_gap = timedelta(minutes=config.trading.min_time_between_trades)
            if time_since_last < min_gap:
                remaining = min_gap - time_since_last
                return (
                    False,
                    f"{symbol}: Too soon since last trade (wait {remaining.seconds // 60} more minutes)",
                )

        # Check if realized volatility is low enough using instrument-specific threshold
        realized_range = self.calculate_realized_range(symbol, 60)  # Last 60 minutes
        volatility_threshold = state["implied_move"] * spec.volatility_threshold

        if realized_range >= volatility_threshold:
            return (
                False,
                f"{symbol}: Realized range {realized_range:.2f} >= threshold {volatility_threshold:.2f}",
            )

        # Check market hours
        if not self.ib_client.is_market_hours():
            return False, f"{symbol}: Outside market hours"

        logger.info(f"{symbol}: Trade conditions met:")
        logger.info(f"  Realized range (60m): {realized_range:.2f}")
        logger.info(f"  Volatility threshold: {volatility_threshold:.2f}")
        logger.info(f"  Implied move: {state['implied_move']:.2f}")

        return True, f"{symbol}: Conditions met for trade"

    async def place_strangle_trade_for_symbol(
        self, symbol: str, call_strike: float, put_strike: float
    ) -> Optional[Dict]:
        """Place a single strangle trade for specific instrument"""
        try:
            logger.info(f"Attempting to place {symbol} strangle: {call_strike}C / {put_strike}P")

            spec = self.get_instrument_spec(symbol)
            state = self.get_instrument_state(symbol)

            # Get current account equity
            account_values = await self.ib_client.get_account_values()
            current_equity = account_values.get("NetLiquidation", 0)

            # Get estimated premium cost
            expiry = self.ib_client.get_today_expiry_string()
            call_contract = await self.ib_client.get_option_contract(
                symbol, expiry, call_strike, "C"
            )
            put_contract = await self.ib_client.get_option_contract(symbol, expiry, put_strike, "P")

            call_price = await self.ib_client.get_current_price(call_contract)
            put_price = await self.ib_client.get_current_price(put_contract)

            if not call_price or not put_price:
                logger.warning(f"Could not get option prices for {symbol} strangle")
                return None

            estimated_premium = (call_price + put_price) * spec.option_multiplier

            # Risk check
            can_trade, reason = self.risk_manager.can_open_new_trade(
                estimated_premium, current_equity
            )
            if not can_trade:
                logger.warning(f"{symbol}: Risk check failed: {reason}")
                return None

            # Place the strangle
            strangle_result = await self.ib_client.place_strangle(
                symbol,
                state["underlying_price"],
                call_strike,
                put_strike,
                expiry,
                spec.max_option_premium * spec.option_multiplier,
            )

            # Record the trade
            trade_record = await self.record_trade(symbol, strangle_result)

            return {
                "trade_record": trade_record,
                "strangle_result": strangle_result,
            }

        except Exception as e:
            logger.error(f"Error placing {symbol} strangle trade: {e}")
            return None

    async def record_trade(self, symbol: str, strangle_result: Dict) -> Trade:
        """Record a trade in the database"""
        try:
            state = self.get_instrument_state(symbol)
            spec = self.get_instrument_spec(symbol)

            session = self.session_maker()
            try:
                trade = Trade(
                    date=datetime.utcnow().date(),
                    entry_time=datetime.utcnow(),
                    underlying_symbol=symbol,
                    underlying_price_at_entry=state["underlying_price"],
                    implied_move=state["implied_move"],
                    call_strike=strangle_result["call_strike"],
                    put_strike=strangle_result["put_strike"],
                    call_premium=strangle_result["call_price"],
                    put_premium=strangle_result["put_price"],
                    total_premium=strangle_result["total_premium"],
                    call_order_id=(
                        strangle_result["call_trades"][0].order.orderId
                        if strangle_result["call_trades"]
                        else None
                    ),
                    put_order_id=(
                        strangle_result["put_trades"][0].order.orderId
                        if strangle_result["put_trades"]
                        else None
                    ),
                    status="OPEN",
                )

                session.add(trade)
                session.commit()

                logger.info(f"Recorded {symbol} trade: ID {trade.id}")
                return trade

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error recording {symbol} trade: {e}")
            raise

    # Note: execute_trading_cycle is now handled by the base class MultiInstrumentStrategy

    # Backward compatibility properties and methods for existing tests
    @property
    def underlying_price(self) -> Optional[float]:
        """Get underlying price for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            return self.instrument_state[primary_symbol]["underlying_price"]
        return None

    @underlying_price.setter
    def underlying_price(self, value: float):
        """Set underlying price for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            self.instrument_state[primary_symbol]["underlying_price"] = value

    @property
    def implied_move(self) -> Optional[float]:
        """Get implied move for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            return self.instrument_state[primary_symbol]["implied_move"]
        return None

    @implied_move.setter
    def implied_move(self, value: float):
        """Set implied move for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            self.instrument_state[primary_symbol]["implied_move"] = value

    @property
    def daily_high(self) -> Optional[float]:
        """Get daily high for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            return self.instrument_state[primary_symbol]["daily_high"]
        return None

    @daily_high.setter
    def daily_high(self, value: float):
        """Set daily high for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            self.instrument_state[primary_symbol]["daily_high"] = value

    @property
    def daily_low(self) -> Optional[float]:
        """Get daily low for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            return self.instrument_state[primary_symbol]["daily_low"]
        return None

    @daily_low.setter
    def daily_low(self, value: float):
        """Set daily low for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            self.instrument_state[primary_symbol]["daily_low"] = value

    @property
    def price_history(self) -> List:
        """Get price history for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            return self.instrument_state[primary_symbol]["price_history"]
        return []

    @price_history.setter
    def price_history(self, value: List):
        """Set price history for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            self.instrument_state[primary_symbol]["price_history"] = value

    @property  # type: ignore[override]
    def last_trade_time(self) -> Optional[datetime]:
        """Get last trade time for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            return self.instrument_state[primary_symbol]["last_trade_time"]
        return None

    @last_trade_time.setter
    def last_trade_time(self, value: Optional[datetime]):
        """Set last trade time for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        if primary_symbol in self.instrument_state:
            self.instrument_state[primary_symbol]["last_trade_time"] = value

    def update_price_history(self, symbol_or_price, price=None):
        """Update price history (backward compatibility)"""
        if price is None:
            # Old API: update_price_history(price)
            primary_symbol = config.trading.primary_instrument
            super().update_price_history(primary_symbol, symbol_or_price)
        else:
            # New API: update_price_history(symbol, price)
            super().update_price_history(symbol_or_price, price)

    def calculate_realized_range(
        self, symbol_or_lookback=None, lookback_minutes: int = 60
    ) -> float:
        """Calculate realized range for primary instrument (backward compatibility)"""
        if symbol_or_lookback is None:
            # Old API: calculate_realized_range() with default lookback
            primary_symbol = config.trading.primary_instrument
            return super().calculate_realized_range(primary_symbol, lookback_minutes)
        elif isinstance(symbol_or_lookback, str):
            # New API: calculate_realized_range(symbol, lookback_minutes)
            return super().calculate_realized_range(symbol_or_lookback, lookback_minutes)
        else:
            # Old API: calculate_realized_range(lookback_minutes)
            primary_symbol = config.trading.primary_instrument
            return super().calculate_realized_range(primary_symbol, symbol_or_lookback)

    def calculate_strike_levels(
        self, symbol_or_price, underlying_price: float = None
    ) -> List[Tuple[float, float]]:
        """Calculate strike levels for primary instrument (backward compatibility)"""
        if underlying_price is None:
            # Old API: calculate_strike_levels(underlying_price)
            primary_symbol = config.trading.primary_instrument
            return super().calculate_strike_levels(primary_symbol, symbol_or_price)
        else:
            # New API: calculate_strike_levels(symbol, underlying_price)
            return super().calculate_strike_levels(symbol_or_price, underlying_price)

    def _round_to_strike(self, price: float) -> float:
        """Round to strike for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        return self.instrument_manager.round_to_strike(primary_symbol, price)

    def should_place_trade(self, symbol: str = None) -> Tuple[bool, str]:
        """Determine if conditions are met to place trade (backward compatibility)"""
        if symbol is None:
            symbol = config.trading.primary_instrument
        return self.should_place_trade_for_symbol(symbol)

    async def place_strangle_trade(self, call_strike: float, put_strike: float) -> Optional[Dict]:
        """Place strangle trade for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument
        return await self.place_strangle_trade_for_symbol(primary_symbol, call_strike, put_strike)

    async def execute_trading_cycle(self) -> bool:
        """Execute trading cycle for primary instrument (backward compatibility)"""
        primary_symbol = config.trading.primary_instrument

        # For backward compatibility, use the underlying_price if available
        # instead of calling get_current_price
        try:
            current_price = self.underlying_price
            if current_price:
                self.update_price_history(current_price)

            # Check if we should place a trade
            should_trade, reason = self.should_place_trade()
            if not should_trade:
                logger.debug(f"Trade conditions not met: {reason}")
                return False

            # Calculate strike levels using current price
            strike_levels = self.calculate_strike_levels(current_price)
            if not strike_levels:
                logger.warning("No strike levels calculated")
                return False

            # Try to place a strangle trade
            call_strike, put_strike = strike_levels[0]  # Use first level
            trade_result = await self.place_strangle_trade(call_strike, put_strike)

            if trade_result:
                logger.info("Successfully placed strangle trade")
                self.last_trade_time = datetime.utcnow()
                return True
            else:
                logger.warning("Failed to place strangle trade")
                return False

        except Exception as e:
            logger.error(f"Error executing trading cycle: {e}")
            return False

    def get_strategy_status(self) -> Dict:
        """Get strategy status with backward compatibility"""
        primary_symbol = config.trading.primary_instrument

        # Get base status
        base_status = super().get_strategy_status()

        # Add backward compatibility fields
        if primary_symbol in self.instrument_state:
            state = self.instrument_state[primary_symbol]
            base_status.update(
                {
                    "underlying_price": state["underlying_price"],
                    "implied_move": state["implied_move"],
                    "daily_high": state["daily_high"],
                    "daily_low": state["daily_low"],
                    "daily_range": (
                        state["daily_high"] - state["daily_low"]
                        if state["daily_high"] and state["daily_low"]
                        else None
                    ),
                    "realized_range_60m": self.calculate_realized_range(60),
                    "volatility_threshold": (
                        state["implied_move"]
                        * self.get_instrument_spec(primary_symbol).volatility_threshold
                        if state["implied_move"]
                        else None
                    ),
                    "price_history_length": len(state["price_history"]),
                }
            )

        return base_status

    async def update_open_positions(self):
        """Update P&L and status of open positions"""
        session = self.session_maker()
        try:
            # Get open trades from database
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()

            for trade in open_trades:
                await self._update_trade_pnl(trade, session)

            session.commit()

        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            session.rollback()
        finally:
            session.close()

    async def _update_trade_pnl(self, trade: Trade, session: Session):
        """Update P&L for a single trade"""
        try:
            expiry = self.ib_client.get_today_expiry_string()

            # Get current option prices
            call_contract = await self.ib_client.get_mes_option_contract(
                expiry, trade.call_strike, "C"
            )
            put_contract = await self.ib_client.get_mes_option_contract(
                expiry, trade.put_strike, "P"
            )

            current_call_price = await self.ib_client.get_current_price(call_contract)
            current_put_price = await self.ib_client.get_current_price(put_contract)

            if current_call_price is not None and current_put_price is not None:
                # Calculate unrealized P&L
                call_pnl = (current_call_price - trade.call_premium) * 5  # MES multiplier
                put_pnl = (current_put_price - trade.put_premium) * 5
                total_unrealized_pnl = call_pnl + put_pnl

                trade.unrealized_pnl = total_unrealized_pnl

                logger.debug(
                    f"Trade {trade.id} P&L: Call ${call_pnl:.2f}, Put ${put_pnl:.2f}, Total ${total_unrealized_pnl:.2f}"
                )

        except Exception as e:
            logger.error(f"Error updating trade {trade.id} P&L: {e}")

    async def flatten_all_positions(self) -> bool:
        """Close all open positions at market (end-of-day routine)"""
        logger.info("Flattening all open positions...")

        session = self.session_maker()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()

            if not open_trades:
                logger.info("No open positions to flatten")
                return True

            for trade in open_trades:
                await self._close_trade_at_market(trade, session)

            session.commit()
            logger.info(f"Flattened {len(open_trades)} positions")
            return True

        except Exception as e:
            logger.error(f"Error flattening positions: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    async def _close_trade_at_market(self, trade: Trade, session: Session):
        """Close a single trade at market prices"""
        try:
            expiry = self.ib_client.get_today_expiry_string()

            # Cancel any pending profit-taking orders
            if trade.call_tp_order_id:
                await self.ib_client.cancel_order(trade.call_tp_order_id)
            if trade.put_tp_order_id:
                await self.ib_client.cancel_order(trade.put_tp_order_id)

            # Close positions at market if they have value
            total_pnl = 0

            if trade.call_status == "OPEN":
                call_contract = await self.ib_client.get_mes_option_contract(
                    expiry, trade.call_strike, "C"
                )
                call_close_trade = await self.ib_client.close_position_at_market(call_contract, 1)
                if call_close_trade:
                    # Would need to wait for fill and record exit price
                    trade.call_status = "EXPIRED"

            if trade.put_status == "OPEN":
                put_contract = await self.ib_client.get_mes_option_contract(
                    expiry, trade.put_strike, "P"
                )
                put_close_trade = await self.ib_client.close_position_at_market(put_contract, 1)
                if put_close_trade:
                    trade.put_status = "EXPIRED"

            # Mark trade as closed
            trade.status = "EXPIRED"
            trade.exit_time = datetime.utcnow()
            trade.realized_pnl = trade.unrealized_pnl or -trade.total_premium

            logger.info(f"Closed trade {trade.id} with P&L: ${trade.realized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error closing trade {trade.id}: {e}")
