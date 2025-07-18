"""
Instrument configuration and specifications for multi-asset trading
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InstrumentType(Enum):
    """Types of tradable instruments"""

    FUTURES = "futures"
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"


@dataclass
class InstrumentSpec:
    """Configuration for a tradable instrument"""

    # Basic identification
    symbol: str
    name: str
    instrument_type: InstrumentType
    exchange: str

    # Contract specifications
    tick_size: float  # Minimum price increment
    multiplier: float  # Dollar value per point
    strike_increment: float  # Option strike spacing

    # Trading parameters
    min_option_premium: float  # Minimum option premium to trade
    max_option_premium: float  # Maximum option premium to trade
    volatility_threshold: float  # Realized vol threshold as % of implied

    # Risk parameters
    max_position_size: int  # Maximum contracts per trade
    max_daily_trades: int  # Maximum trades per day
    position_limit: int  # Maximum open positions

    # Market data
    market_hours_start: str  # Market open time (HH:MM)
    market_hours_end: str  # Market close time (HH:MM)
    timezone: str  # Market timezone

    # Option specifications
    option_multiplier: float  # Option contract multiplier
    option_symbol: str  # Option symbol (may differ from underlying)

    def __post_init__(self):
        """Validate instrument configuration"""
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")
        if self.strike_increment <= 0:
            raise ValueError(f"strike_increment must be positive, got {self.strike_increment}")
        if not (0 < self.volatility_threshold <= 1):
            raise ValueError(
                f"volatility_threshold must be between 0 and 1, got {self.volatility_threshold}"
            )


# Default instrument configurations
DEFAULT_INSTRUMENTS: Dict[str, InstrumentSpec] = {
    "MES": InstrumentSpec(
        symbol="MES",
        name="Micro E-mini S&P 500",
        instrument_type=InstrumentType.FUTURES,
        exchange="GLOBEX",
        tick_size=0.25,
        multiplier=5.0,
        strike_increment=25.0,
        min_option_premium=0.25,
        max_option_premium=5.0,
        volatility_threshold=0.67,
        max_position_size=10,
        max_daily_trades=20,
        position_limit=50,
        market_hours_start="09:30",
        market_hours_end="16:00",
        timezone="US/Eastern",
        option_multiplier=5.0,
        option_symbol="MES",
    ),
    "ES": InstrumentSpec(
        symbol="ES",
        name="E-mini S&P 500",
        instrument_type=InstrumentType.FUTURES,
        exchange="GLOBEX",
        tick_size=0.25,
        multiplier=50.0,
        strike_increment=25.0,
        min_option_premium=0.50,
        max_option_premium=10.0,
        volatility_threshold=0.67,
        max_position_size=5,
        max_daily_trades=15,
        position_limit=25,
        market_hours_start="09:30",
        market_hours_end="16:00",
        timezone="US/Eastern",
        option_multiplier=50.0,
        option_symbol="ES",
    ),
    "NQ": InstrumentSpec(
        symbol="NQ",
        name="E-mini NASDAQ 100",
        instrument_type=InstrumentType.FUTURES,
        exchange="GLOBEX",
        tick_size=0.25,
        multiplier=20.0,
        strike_increment=25.0,
        min_option_premium=0.50,
        max_option_premium=15.0,
        volatility_threshold=0.65,  # Slightly more volatile
        max_position_size=5,
        max_daily_trades=15,
        position_limit=25,
        market_hours_start="09:30",
        market_hours_end="16:00",
        timezone="US/Eastern",
        option_multiplier=20.0,
        option_symbol="NQ",
    ),
    "M2K": InstrumentSpec(
        symbol="M2K",
        name="Micro E-mini Russell 2000",
        instrument_type=InstrumentType.FUTURES,
        exchange="GLOBEX",
        tick_size=0.10,
        multiplier=5.0,
        strike_increment=5.0,
        min_option_premium=0.25,
        max_option_premium=8.0,
        volatility_threshold=0.60,  # Russell 2000 is more volatile
        max_position_size=10,
        max_daily_trades=20,
        position_limit=50,
        market_hours_start="09:30",
        market_hours_end="16:00",
        timezone="US/Eastern",
        option_multiplier=5.0,
        option_symbol="M2K",
    ),
}


class InstrumentManager:
    """Manages instrument configurations and selection"""

    def __init__(self, instruments: Optional[Dict[str, InstrumentSpec]] = None):
        self.instruments = instruments or DEFAULT_INSTRUMENTS.copy()
        self.active_instruments: List[str] = []

    def add_instrument(self, symbol: str, spec: InstrumentSpec):
        """Add or update an instrument configuration"""
        self.instruments[symbol] = spec
        logger.info(f"Added instrument configuration for {symbol}")

    def get_instrument(self, symbol: str) -> Optional[InstrumentSpec]:
        """Get instrument specification by symbol"""
        return self.instruments.get(symbol)

    def get_active_instruments(self) -> List[InstrumentSpec]:
        """Get all active instrument specifications"""
        return [
            self.instruments[symbol]
            for symbol in self.active_instruments
            if symbol in self.instruments
        ]

    def set_active_instruments(self, symbols: List[str]):
        """Set which instruments should be actively traded"""
        invalid_symbols = [s for s in symbols if s not in self.instruments]
        if invalid_symbols:
            raise ValueError(f"Unknown instrument symbols: {invalid_symbols}")

        self.active_instruments = symbols
        logger.info(f"Set active instruments: {symbols}")

    def get_available_instruments(self) -> List[str]:
        """Get list of all available instrument symbols"""
        return list(self.instruments.keys())

    def validate_instrument_config(self, symbol: str) -> bool:
        """Validate instrument configuration"""
        spec = self.get_instrument(symbol)
        if not spec:
            logger.error(f"Instrument {symbol} not found")
            return False

        try:
            # Validation is done in __post_init__
            return True
        except ValueError as e:
            logger.error(f"Invalid configuration for {symbol}: {e}")
            return False

    def get_instrument_by_type(self, instrument_type: InstrumentType) -> List[InstrumentSpec]:
        """Get all instruments of a specific type"""
        return [
            spec for spec in self.instruments.values() if spec.instrument_type == instrument_type
        ]

    def round_to_tick(self, symbol: str, price: float) -> float:
        """Round price to nearest tick for the instrument"""
        spec = self.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        return round(price / spec.tick_size) * spec.tick_size

    def round_to_strike(self, symbol: str, price: float) -> float:
        """Round price to nearest strike increment for the instrument"""
        spec = self.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        return round(price / spec.strike_increment) * spec.strike_increment

    def calculate_option_value(self, symbol: str, premium: float) -> float:
        """Calculate dollar value of option premium"""
        spec = self.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        return premium * spec.option_multiplier

    def get_max_premium(self, symbol: str) -> float:
        """Get maximum allowed option premium for instrument"""
        spec = self.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        return spec.max_option_premium

    def get_volatility_threshold(self, symbol: str) -> float:
        """Get volatility threshold for instrument"""
        spec = self.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        return spec.volatility_threshold


# Global instrument manager instance
instrument_manager = InstrumentManager()
