"""
Abstract base strategy class for multi-instrument trading
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.config import config
from app.ib_client import IBClient
from app.instruments import InstrumentManager, InstrumentSpec, instrument_manager
from app.models import Trade, get_session_maker
from app.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for multi-instrument trading strategies
    """

    def __init__(self, ib_client: IBClient, risk_manager: RiskManager, database_url: str):
        self.ib_client = ib_client
        self.risk_manager = risk_manager
        self.session_maker = get_session_maker(database_url)
        self.instrument_manager = instrument_manager

        # Strategy state per instrument
        self.instrument_state: Dict[str, Dict[str, Any]] = {}

        # Initialize instruments
        self.initialize_instruments()

        # Common strategy state
        self.session_start_time = datetime.utcnow()
        self.last_trade_time = None

    def initialize_instruments(self):
        """Initialize instrument configurations"""
        # Set active instruments from config
        active_instruments = config.trading.active_instruments

        # Validate instruments exist
        for symbol in active_instruments:
            spec = self.instrument_manager.get_instrument(symbol)
            if not spec:
                raise ValueError(f"Unknown instrument: {symbol}")

        self.instrument_manager.set_active_instruments(active_instruments)

        # Initialize state for each instrument
        for symbol in active_instruments:
            self.instrument_state[symbol] = {
                "underlying_price": None,
                "implied_move": None,
                "daily_high": None,
                "daily_low": None,
                "price_history": [],
                "last_trade_time": None,
            }

        logger.info(f"Initialized strategy for instruments: {active_instruments}")

    def get_instrument_spec(self, symbol: str) -> InstrumentSpec:
        """Get instrument specification"""
        spec = self.instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")
        return spec

    def get_instrument_state(self, symbol: str) -> Dict[str, Any]:
        """Get state for specific instrument"""
        if symbol not in self.instrument_state:
            raise ValueError(f"Instrument {symbol} not initialized")
        return self.instrument_state[symbol]

    def update_instrument_state(self, symbol: str, **kwargs):
        """Update state for specific instrument"""
        if symbol not in self.instrument_state:
            raise ValueError(f"Instrument {symbol} not initialized")

        self.instrument_state[symbol].update(kwargs)

    @abstractmethod
    async def initialize_daily_session(self) -> bool:
        """Initialize strategy for the trading day"""
        pass

    @abstractmethod
    async def execute_trading_cycle(self) -> bool:
        """Execute one complete trading cycle"""
        pass

    @abstractmethod
    def should_place_trade(self, symbol: str) -> Tuple[bool, str]:
        """Determine if conditions are met to place a trade for instrument"""
        pass

    def update_price_history(self, symbol: str, price: float):
        """Update price history for instrument"""
        state = self.get_instrument_state(symbol)
        timestamp = datetime.utcnow()

        # Add to price history
        state["price_history"].append((timestamp, price))

        # Keep only recent history (last 100 points)
        max_history = 100
        if len(state["price_history"]) > max_history:
            state["price_history"] = state["price_history"][-max_history:]

        # Update daily high/low
        if state["daily_high"] is None or price > state["daily_high"]:
            state["daily_high"] = price
        if state["daily_low"] is None or price < state["daily_low"]:
            state["daily_low"] = price

        # Update current price
        state["underlying_price"] = price

    def calculate_realized_range(self, symbol: str, lookback_minutes: int = 60) -> float:
        """Calculate realized price range for instrument"""
        state = self.get_instrument_state(symbol)
        price_history = state["price_history"]

        if len(price_history) < 2:
            return 0.0

        cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        recent_prices = [price for timestamp, price in price_history if timestamp >= cutoff_time]

        if len(recent_prices) < 2:
            return 0.0

        return max(recent_prices) - min(recent_prices)

    def calculate_strike_levels(
        self, symbol: str, underlying_price: float
    ) -> List[Tuple[float, float]]:
        """Calculate strike levels for instrument"""
        state = self.get_instrument_state(symbol)
        spec = self.get_instrument_spec(symbol)
        implied_move = state["implied_move"]

        if not implied_move:
            return []

        # Calculate OTM strikes at different multipliers
        call_strike_1 = underlying_price + (implied_move * config.trading.implied_move_multiplier_1)
        put_strike_1 = underlying_price - (implied_move * config.trading.implied_move_multiplier_1)

        call_strike_2 = underlying_price + (implied_move * config.trading.implied_move_multiplier_2)
        put_strike_2 = underlying_price - (implied_move * config.trading.implied_move_multiplier_2)

        # Round to valid strikes using instrument-specific strike increment
        call_strike_1 = self.instrument_manager.round_to_strike(symbol, call_strike_1)
        put_strike_1 = self.instrument_manager.round_to_strike(symbol, put_strike_1)
        call_strike_2 = self.instrument_manager.round_to_strike(symbol, call_strike_2)
        put_strike_2 = self.instrument_manager.round_to_strike(symbol, put_strike_2)

        return [
            (call_strike_1, put_strike_1),
            (call_strike_2, put_strike_2),
        ]

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        active_instruments = self.instrument_manager.get_active_instruments()

        status: Dict[str, Any] = {
            "timestamp": datetime.utcnow(),
            "session_start_time": self.session_start_time,
            "active_instruments": [spec.symbol for spec in active_instruments],
            "instruments": {},
        }

        for symbol in self.instrument_state:
            state = self.instrument_state[symbol]
            daily_range = None
            if state["daily_high"] is not None and state["daily_low"] is not None:
                daily_range = state["daily_high"] - state["daily_low"]

            status["instruments"][symbol] = {
                "underlying_price": state["underlying_price"],
                "implied_move": state["implied_move"],
                "daily_high": state["daily_high"],
                "daily_low": state["daily_low"],
                "daily_range": daily_range,
                "last_trade_time": state["last_trade_time"],
                "price_history_length": len(state["price_history"]),
            }

        return status

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for instrument"""
        try:
            contract = await self.ib_client.get_contract_for_symbol(symbol)
            return await self.ib_client.get_current_price(contract)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def get_atm_straddle_price(
        self, symbol: str, underlying_price: float, expiry: str
    ) -> Tuple[float, float, float]:
        """Get ATM straddle price for instrument"""
        try:
            return await self.ib_client.get_atm_straddle_price(symbol, underlying_price, expiry)
        except Exception as e:
            logger.error(f"Error getting ATM straddle price for {symbol}: {e}")
            raise

    def get_active_symbols(self) -> List[str]:
        """Get list of active instrument symbols"""
        return [spec.symbol for spec in self.instrument_manager.get_active_instruments()]

    def is_instrument_active(self, symbol: str) -> bool:
        """Check if instrument is active for trading"""
        return symbol in self.get_active_symbols()


class MultiInstrumentStrategy(BaseStrategy):
    """
    Base class for strategies that trade multiple instruments simultaneously
    """

    async def execute_trading_cycle(self) -> bool:
        """Execute trading cycle for all active instruments"""
        results = []

        for symbol in self.get_active_symbols():
            try:
                result = await self.execute_instrument_cycle(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing cycle for {symbol}: {e}")
                results.append(False)

        return any(results)  # Return True if any instrument had activity

    @abstractmethod
    async def execute_instrument_cycle(self, symbol: str) -> bool:
        """Execute trading cycle for specific instrument"""
        pass
