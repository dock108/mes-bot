"""
Comprehensive tests for instrument configuration and management
"""

from dataclasses import asdict
from datetime import datetime, time

import pytest

from app.instruments import DEFAULT_INSTRUMENTS, InstrumentManager, InstrumentSpec, InstrumentType


@pytest.mark.unit
class TestInstrumentType:
    """Test InstrumentType enum"""

    def test_instrument_types(self):
        """Test all instrument types are defined"""
        assert InstrumentType.FUTURES.value == "futures"
        assert InstrumentType.STOCK.value == "stock"
        assert InstrumentType.ETF.value == "etf"
        assert InstrumentType.INDEX.value == "index"

    def test_instrument_type_values(self):
        """Test instrument type string values"""
        all_types = [e.value for e in InstrumentType]
        expected_types = ["futures", "stock", "etf", "index"]
        assert all_types == expected_types


@pytest.mark.unit
class TestInstrumentSpec:
    """Test InstrumentSpec dataclass"""

    @pytest.fixture
    def sample_spec(self):
        """Create sample instrument specification"""
        return InstrumentSpec(
            symbol="ES",
            name="E-mini S&P 500",
            instrument_type=InstrumentType.FUTURES,
            exchange="CME",
            tick_size=0.25,
            multiplier=50.0,
            strike_increment=5.0,
            min_option_premium=0.05,
            max_option_premium=100.0,
            volatility_threshold=0.8,
            max_position_size=10,
            max_daily_trades=20,
            position_limit=50,
            market_hours_start="09:30",
            market_hours_end="16:00",
            timezone="America/Chicago",
            option_multiplier=50.0,
            option_symbol="ES",
        )

    def test_instrument_spec_creation(self, sample_spec):
        """Test InstrumentSpec creation and attributes"""
        assert sample_spec.symbol == "ES"
        assert sample_spec.name == "E-mini S&P 500"
        assert sample_spec.instrument_type == InstrumentType.FUTURES
        assert sample_spec.exchange == "CME"
        assert sample_spec.tick_size == 0.25
        assert sample_spec.multiplier == 50.0
        assert sample_spec.strike_increment == 5.0

    def test_instrument_spec_risk_parameters(self, sample_spec):
        """Test risk parameter attributes"""
        assert sample_spec.max_position_size == 10
        assert sample_spec.max_daily_trades == 20
        assert sample_spec.position_limit == 50
        assert sample_spec.min_option_premium == 0.05
        assert sample_spec.max_option_premium == 100.0
        assert sample_spec.volatility_threshold == 0.8

    def test_instrument_spec_market_hours(self, sample_spec):
        """Test market hours attributes"""
        assert sample_spec.market_hours_start == "09:30"
        assert sample_spec.market_hours_end == "16:00"
        assert sample_spec.timezone == "America/Chicago"

    def test_instrument_spec_trading_parameters(self, sample_spec):
        """Test trading-related parameters"""
        assert sample_spec.option_multiplier == 50.0
        assert sample_spec.option_symbol == "ES"

    def test_instrument_spec_dataclass_features(self, sample_spec):
        """Test dataclass functionality"""
        # Test conversion to dict
        spec_dict = asdict(sample_spec)
        assert spec_dict["symbol"] == "ES"
        assert spec_dict["instrument_type"] == InstrumentType.FUTURES

        # Test equality
        other_spec = InstrumentSpec(
            symbol="ES",
            name="E-mini S&P 500",
            instrument_type=InstrumentType.FUTURES,
            exchange="CME",
            tick_size=0.25,
            multiplier=50.0,
            strike_increment=5.0,
            min_option_premium=0.05,
            max_option_premium=100.0,
            volatility_threshold=0.8,
            max_position_size=10,
            max_daily_trades=20,
            position_limit=50,
            market_hours_start="09:30",
            market_hours_end="16:00",
            timezone="America/Chicago",
            option_multiplier=50.0,
            option_symbol="ES",
        )
        assert sample_spec == other_spec


@pytest.mark.unit
class TestDefaultInstruments:
    """Test default instrument configurations"""

    def test_default_instruments_exist(self):
        """Test that default instruments are defined"""
        assert len(DEFAULT_INSTRUMENTS) > 0
        assert "MES" in DEFAULT_INSTRUMENTS
        assert "ES" in DEFAULT_INSTRUMENTS
        assert "NQ" in DEFAULT_INSTRUMENTS

    def test_default_instrument_specifications(self):
        """Test default instrument specifications are valid"""
        for symbol, spec in DEFAULT_INSTRUMENTS.items():
            assert isinstance(spec, InstrumentSpec)
            assert spec.symbol == symbol
            assert spec.tick_size > 0
            assert spec.multiplier > 0
            assert spec.strike_increment > 0
            assert 0 < spec.volatility_threshold <= 1


@pytest.mark.unit
class TestInstrumentManager:
    """Test InstrumentManager functionality"""

    @pytest.fixture
    def instrument_manager(self):
        """Create InstrumentManager with default instruments"""
        return InstrumentManager()

    def test_instrument_manager_initialization(self, instrument_manager):
        """Test InstrumentManager initialization"""
        assert len(instrument_manager.instruments) >= 3  # At least MES, ES, NQ
        assert "ES" in instrument_manager.instruments
        assert "NQ" in instrument_manager.instruments
        assert "MES" in instrument_manager.instruments

    def test_get_instrument_success(self, instrument_manager):
        """Test getting existing instrument"""
        es_spec = instrument_manager.get_instrument("ES")
        assert es_spec is not None
        assert es_spec.symbol == "ES"
        assert es_spec.name == "E-mini S&P 500"
        assert es_spec.instrument_type == InstrumentType.FUTURES

    def test_get_instrument_not_found(self, instrument_manager):
        """Test getting non-existent instrument"""
        spec = instrument_manager.get_instrument("INVALID")
        assert spec is None

    def test_get_available_instruments(self, instrument_manager):
        """Test getting all available instruments"""
        available = instrument_manager.get_available_instruments()
        assert len(available) >= 3
        assert "ES" in available
        assert "NQ" in available
        assert "MES" in available

    def test_add_instrument(self, instrument_manager):
        """Test adding new instrument"""
        initial_count = len(instrument_manager.instruments)

        new_spec = InstrumentSpec(
            symbol="QQQ",
            name="Invesco QQQ Trust",
            instrument_type=InstrumentType.ETF,
            exchange="NASDAQ",
            tick_size=0.01,
            multiplier=100.0,
            strike_increment=1.0,
            min_option_premium=0.01,
            max_option_premium=30.0,
            volatility_threshold=0.80,
            max_position_size=50,
            max_daily_trades=30,
            position_limit=200,
            market_hours_start="09:30",
            market_hours_end="16:00",
            timezone="America/New_York",
            option_multiplier=100.0,
            option_symbol="QQQ",
        )

        instrument_manager.add_instrument("QQQ", new_spec)

        assert len(instrument_manager.instruments) == initial_count + 1
        retrieved_spec = instrument_manager.get_instrument("QQQ")
        assert retrieved_spec is not None
        assert retrieved_spec.symbol == "QQQ"

    def test_set_active_instruments(self, instrument_manager):
        """Test setting active instruments"""
        available = instrument_manager.get_available_instruments()
        test_symbols = available[:2]  # Take first 2 available

        instrument_manager.set_active_instruments(test_symbols)

        assert instrument_manager.active_instruments == test_symbols

        active_specs = instrument_manager.get_active_instruments()
        assert len(active_specs) == 2
        assert all(isinstance(spec, InstrumentSpec) for spec in active_specs)

    def test_set_active_instruments_invalid(self, instrument_manager):
        """Test setting invalid active instruments"""
        with pytest.raises(ValueError, match="Unknown instrument symbols"):
            instrument_manager.set_active_instruments(["INVALID", "ALSO_INVALID"])

    def test_validate_instrument_config(self, instrument_manager):
        """Test instrument configuration validation"""
        # Valid instrument
        assert instrument_manager.validate_instrument_config("ES") is True

        # Invalid instrument
        assert instrument_manager.validate_instrument_config("INVALID") is False


@pytest.mark.unit
class TestInstrumentSpecValidation:
    """Test InstrumentSpec validation"""

    def test_valid_instrument_spec(self):
        """Test creating valid instrument spec"""
        spec = InstrumentSpec(
            symbol="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.FUTURES,
            exchange="TEST",
            tick_size=0.25,
            multiplier=50.0,
            strike_increment=5.0,
            min_option_premium=0.05,
            max_option_premium=100.0,
            volatility_threshold=0.8,
            max_position_size=10,
            max_daily_trades=20,
            position_limit=50,
            market_hours_start="09:30",
            market_hours_end="16:00",
            timezone="America/Chicago",
            option_multiplier=50.0,
            option_symbol="TEST",
        )

        assert spec.symbol == "TEST"
        assert spec.tick_size == 0.25
        assert spec.volatility_threshold == 0.8

    def test_invalid_tick_size(self):
        """Test validation fails for invalid tick size"""
        with pytest.raises(ValueError, match="tick_size must be positive"):
            InstrumentSpec(
                symbol="TEST",
                name="Test",
                instrument_type=InstrumentType.FUTURES,
                exchange="TEST",
                tick_size=-0.1,  # Invalid
                multiplier=50.0,
                strike_increment=5.0,
                min_option_premium=0.05,
                max_option_premium=100.0,
                volatility_threshold=0.8,
                max_position_size=10,
                max_daily_trades=20,
                position_limit=50,
                market_hours_start="09:30",
                market_hours_end="16:00",
                timezone="America/Chicago",
                option_multiplier=50.0,
                option_symbol="TEST",
            )

    def test_invalid_volatility_threshold(self):
        """Test validation fails for invalid volatility threshold"""
        with pytest.raises(ValueError, match="volatility_threshold must be between 0 and 1"):
            InstrumentSpec(
                symbol="TEST",
                name="Test",
                instrument_type=InstrumentType.FUTURES,
                exchange="TEST",
                tick_size=0.25,
                multiplier=50.0,
                strike_increment=5.0,
                min_option_premium=0.05,
                max_option_premium=100.0,
                volatility_threshold=1.5,  # Invalid - greater than 1
                max_position_size=10,
                max_daily_trades=20,
                position_limit=50,
                market_hours_start="09:30",
                market_hours_end="16:00",
                timezone="America/Chicago",
                option_multiplier=50.0,
                option_symbol="TEST",
            )


@pytest.mark.unit
class TestInstrumentErrorHandling:
    """Test error handling and edge cases"""

    def test_instrument_manager_empty_initialization(self):
        """Test InstrumentManager with empty instruments"""
        manager = InstrumentManager({})
        # With empty dict, it should still load defaults
        assert len(manager.instruments) >= 3  # Default instruments loaded
        assert manager.get_instrument("ES") is not None  # Default should exist

    def test_instrument_manager_none_initialization(self):
        """Test InstrumentManager with None"""
        manager = InstrumentManager(None)
        # With None, it should load defaults
        assert len(manager.instruments) >= 3  # Default instruments loaded

    def test_invalid_instrument_operations(self):
        """Test operations with invalid instruments"""
        manager = InstrumentManager({})

        # Test operations that should handle invalid instruments gracefully
        assert manager.validate_instrument_config("INVALID") is False

        # Test operations that should raise ValueError for invalid instruments
        with pytest.raises(ValueError):
            manager.round_to_tick("INVALID", 100.0)

        with pytest.raises(ValueError):
            manager.round_to_strike("INVALID", 100.0)

        with pytest.raises(ValueError):
            manager.calculate_option_value("INVALID", 10.0)

    def test_instrument_spec_with_minimal_data(self):
        """Test InstrumentSpec with minimal required data"""
        # Test that all required fields must be provided
        with pytest.raises(TypeError):
            InstrumentSpec(symbol="TEST")  # Missing required fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
