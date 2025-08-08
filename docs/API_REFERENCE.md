# API Reference

Complete API documentation for the MES 0DTE Options Trading Bot.

## Table of Contents

- [Core Modules](#core-modules)
  - [Bot Engine](#bot-engine)
  - [Strategy](#strategy)
  - [Risk Manager](#risk-manager)
- [Trading Integration](#trading-integration)
  - [IB Client](#ib-client)
  - [Order Management](#order-management)
- [ML Components](#ml-components)
  - [Decision Engine](#decision-engine)
  - [Feature Pipeline](#feature-pipeline)
  - [Market Indicators](#market-indicators)
- [Data Layer](#data-layer)
  - [Database Models](#database-models)
  - [Data Providers](#data-providers)
- [User Interface](#user-interface)
  - [Streamlit Components](#streamlit-components)
- [Utilities](#utilities)

## Core Modules

### Bot Engine

#### `app.bot.TradingBot`

Main orchestrator for the trading system.

```python
class TradingBot:
    def __init__(self, config: dict = None):
        """Initialize trading bot with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
    
    async def start(self) -> None:
        """Start the trading bot and all components."""
    
    async def stop(self) -> None:
        """Gracefully stop the trading bot."""
    
    async def run_trading_cycle(self) -> None:
        """Execute one complete trading cycle."""
```

**Example Usage:**
```python
from app.bot import TradingBot

bot = TradingBot()
await bot.start()
```

### Strategy

#### `app.strategy.Strategy`

Core trading strategy implementation.

```python
class Strategy:
    def __init__(self, risk_manager: RiskManager):
        """Initialize strategy with risk manager.
        
        Args:
            risk_manager: Risk management component
        """
    
    def should_open_strangle(
        self, 
        current_price: float,
        implied_move: float,
        realized_vol: float
    ) -> bool:
        """Determine if conditions are met for opening a strangle.
        
        Args:
            current_price: Current underlying price
            implied_move: Expected daily move percentage
            realized_vol: Realized volatility over lookback period
            
        Returns:
            True if strangle should be opened
        """
    
    def calculate_strikes(
        self,
        spot_price: float,
        implied_move: float
    ) -> tuple[float, float]:
        """Calculate put and call strikes for strangle.
        
        Args:
            spot_price: Current underlying price
            implied_move: Expected daily move
            
        Returns:
            Tuple of (put_strike, call_strike)
        """
```

#### `app.enhanced_strategy.EnhancedStrategy`

ML-enhanced strategy with decision engine integration.

```python
class EnhancedStrategy(Strategy):
    def __init__(
        self,
        risk_manager: RiskManager,
        decision_engine: DecisionEngine = None
    ):
        """Initialize enhanced strategy.
        
        Args:
            risk_manager: Risk management component
            decision_engine: Optional ML decision engine
        """
    
    async def get_ml_signal(
        self,
        market_data: dict
    ) -> TradingSignal:
        """Get ML-based trading signal.
        
        Args:
            market_data: Current market data dictionary
            
        Returns:
            TradingSignal with strength and confidence
        """
```

### Risk Manager

#### `app.risk_manager.RiskManager`

Manages position and account-level risk.

```python
class RiskManager:
    def __init__(self, config: dict):
        """Initialize risk manager with configuration.
        
        Args:
            config: Risk management configuration
        """
    
    def can_open_position(
        self,
        premium: float,
        current_positions: int
    ) -> bool:
        """Check if new position can be opened.
        
        Args:
            premium: Premium required for position
            current_positions: Number of open positions
            
        Returns:
            True if position can be opened
        """
    
    def check_drawdown(self) -> bool:
        """Check if drawdown limit has been reached.
        
        Returns:
            True if within drawdown limits
        """
    
    def calculate_position_size(
        self,
        account_balance: float,
        premium: float
    ) -> int:
        """Calculate appropriate position size.
        
        Args:
            account_balance: Current account balance
            premium: Premium per contract
            
        Returns:
            Number of contracts to trade
        """
```

## Trading Integration

### IB Client

#### `app.ib_client.IBClient`

Interactive Brokers API integration.

```python
class IBClient:
    def __init__(self):
        """Initialize IB client."""
    
    async def connect(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1
    ) -> bool:
        """Connect to IB Gateway/TWS.
        
        Args:
            host: IB Gateway host
            port: IB Gateway port (7497 paper, 7496 live)
            client_id: Unique client identifier
            
        Returns:
            True if connection successful
        """
    
    async def place_strangle(
        self,
        symbol: str,
        expiry: str,
        put_strike: float,
        call_strike: float,
        quantity: int = 1
    ) -> tuple[Trade, Trade]:
        """Place a strangle order.
        
        Args:
            symbol: Underlying symbol (e.g., "MES")
            expiry: Expiration date (YYYYMMDD)
            put_strike: Put option strike price
            call_strike: Call option strike price
            quantity: Number of contracts
            
        Returns:
            Tuple of (put_trade, call_trade)
        """
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price.
        
        Args:
            symbol: Contract symbol
            
        Returns:
            Current market price
        """
    
    async def close_position(
        self,
        contract: Contract,
        quantity: int
    ) -> Trade:
        """Close an existing position.
        
        Args:
            contract: IB contract object
            quantity: Number of contracts to close
            
        Returns:
            Trade object for the closing order
        """
```

#### `app.ib_connection_manager.IBConnectionManager`

Manages IB connection resilience.

```python
class IBConnectionManager:
    def __init__(self, ib_client: IBClient):
        """Initialize connection manager.
        
        Args:
            ib_client: IB client instance
        """
    
    async def maintain_connection(self) -> None:
        """Maintain connection with automatic recovery."""
    
    def get_connection_status(self) -> dict:
        """Get detailed connection status.
        
        Returns:
            Dictionary with connection metrics
        """
```

### Order Management

#### `app.order_manager.OrderManager`

Manages order lifecycle and execution.

```python
class OrderManager:
    def __init__(self, ib_client: IBClient):
        """Initialize order manager.
        
        Args:
            ib_client: IB client for order execution
        """
    
    async def place_bracket_order(
        self,
        contract: Contract,
        quantity: int,
        entry_price: float,
        profit_target: float,
        stop_loss: float
    ) -> BracketOrder:
        """Place bracket order with profit target and stop loss.
        
        Args:
            contract: Contract to trade
            quantity: Number of contracts
            entry_price: Entry limit price
            profit_target: Take profit price
            stop_loss: Stop loss price
            
        Returns:
            BracketOrder object with all orders
        """
```

## ML Components

### Decision Engine

#### `app.ml.decision_engine.DecisionEngine`

ML-based decision making system.

```python
class DecisionEngine:
    def __init__(self, models: list = None):
        """Initialize decision engine.
        
        Args:
            models: List of trained models
        """
    
    def predict(
        self,
        features: pd.DataFrame
    ) -> TradingSignal:
        """Generate trading signal from features.
        
        Args:
            features: Feature dataframe
            
        Returns:
            TradingSignal with prediction
        """
    
    def update_model(
        self,
        new_model: Any,
        model_name: str
    ) -> None:
        """Update a model in the ensemble.
        
        Args:
            new_model: New trained model
            model_name: Name of model to update
        """
```

### Feature Pipeline

#### `app.feature_pipeline.FeaturePipeline`

Feature engineering and data preprocessing.

```python
class FeaturePipeline:
    def __init__(self):
        """Initialize feature pipeline."""
    
    def calculate_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Calculate all features for ML models.
        
        Args:
            price_data: Price history dataframe
            volume_data: Optional volume data
            
        Returns:
            DataFrame with engineered features
        """
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature importance
        """
```

### Market Indicators

#### `app.market_indicators.MarketIndicatorEngine`

Technical indicator calculations.

```python
class MarketIndicatorEngine:
    def __init__(self):
        """Initialize indicator engine."""
    
    def calculate_rsi(
        self,
        prices: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Price array
            period: RSI period
            
        Returns:
            RSI value (0-100)
        """
    
    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Price array
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
    
    def calculate_implied_volatility(
        self,
        option_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate implied volatility.
        
        Args:
            option_price: Current option price
            spot: Spot price of underlying
            strike: Strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Implied volatility as decimal
        """
```

## Data Layer

### Database Models

#### `app.models.Trade`

Trade record model.

```python
@dataclass
class Trade:
    id: int
    timestamp: datetime
    symbol: str
    trade_type: str  # "STRANGLE", "CLOSE"
    put_strike: float
    call_strike: float
    premium_paid: float
    quantity: int
    status: str  # "OPEN", "CLOSED", "EXPIRED"
    pnl: float = None
    close_timestamp: datetime = None
    
    def calculate_pnl(self, close_price: float) -> float:
        """Calculate P&L for the trade."""
```

#### `app.models.DailySummary`

Daily performance summary.

```python
@dataclass
class DailySummary:
    date: date
    trades_opened: int
    trades_closed: int
    gross_pnl: float
    net_pnl: float
    win_rate: float
    max_drawdown: float
    ending_balance: float
```

### Data Providers

#### `app.data_providers.vix_provider.VIXDataProvider`

VIX data provider for volatility analysis.

```python
class VIXDataProvider:
    def __init__(self):
        """Initialize VIX data provider."""
    
    async def get_current_vix(self) -> float:
        """Get current VIX value.
        
        Returns:
            Current VIX level
        """
    
    async def get_vix_percentile(
        self,
        lookback_days: int = 252
    ) -> float:
        """Get VIX percentile rank.
        
        Args:
            lookback_days: Historical lookback period
            
        Returns:
            Percentile rank (0-100)
        """
```

## User Interface

### Streamlit Components

#### `app.ui.DashboardUI`

Main dashboard interface.

```python
class DashboardUI:
    def __init__(self):
        """Initialize dashboard UI."""
    
    def render_live_monitor(self) -> None:
        """Render live monitoring section."""
    
    def render_performance_analytics(self) -> None:
        """Render performance analytics section."""
    
    def render_backtesting(self) -> None:
        """Render backtesting interface."""
    
    def render_configuration(self) -> None:
        """Render configuration management interface."""
```

## Utilities

### Circuit Breaker

#### `app.circuit_breaker.CircuitBreaker`

Fault tolerance implementation.

```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before retry
            expected_exception: Exception type to catch
        """
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
        """
```

### Logging

#### `app.utils.logging.setup_logging`

Configure application logging.

```python
def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/bot.log"
) -> logging.Logger:
    """Set up application logging.
    
    Args:
        log_level: Logging level
        log_file: Log file path
        
    Returns:
        Configured logger instance
    """
```

### Configuration

#### `app.config.Config`

Application configuration management.

```python
class Config:
    def __init__(self, env_file: str = ".env"):
        """Load configuration from environment.
        
        Args:
            env_file: Path to environment file
        """
    
    @property
    def trade_mode(self) -> str:
        """Get trading mode (paper/live)."""
    
    @property
    def max_open_trades(self) -> int:
        """Get maximum open trades limit."""
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
```

## Error Handling

### Custom Exceptions

```python
class TradingBotError(Exception):
    """Base exception for trading bot."""

class ConnectionError(TradingBotError):
    """Raised when connection fails."""

class RiskLimitError(TradingBotError):
    """Raised when risk limits are exceeded."""

class StrategyError(TradingBotError):
    """Raised for strategy-related errors."""

class ConfigurationError(TradingBotError):
    """Raised for configuration errors."""
```

## Testing Utilities

### Mock Objects

```python
from app.testing import MockIBClient, MockStrategy

# Use in tests
mock_client = MockIBClient()
mock_strategy = MockStrategy()
```

## Environment Variables

See [Configuration Reference](CONFIG_REFERENCE.md) for complete list of environment variables.

## Rate Limits

- IB API: 50 messages per second
- Market data: 100 requests per second
- Order placement: 10 orders per second

## Best Practices

1. **Always use async/await** for I/O operations
2. **Implement proper error handling** with specific exceptions
3. **Use type hints** for all function parameters
4. **Add docstrings** to all public methods
5. **Log important events** at appropriate levels
6. **Validate inputs** before processing
7. **Use connection pooling** for database operations
8. **Implement circuit breakers** for external calls

## Examples

### Complete Trading Flow

```python
import asyncio
from app.bot import TradingBot
from app.config import Config

async def main():
    # Load configuration
    config = Config()
    
    # Initialize bot
    bot = TradingBot(config)
    
    # Start trading
    try:
        await bot.start()
        
        # Run until interrupted
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Strategy Implementation

```python
from app.strategy import Strategy

class CustomStrategy(Strategy):
    def should_open_strangle(
        self,
        current_price: float,
        implied_move: float,
        realized_vol: float
    ) -> bool:
        # Custom logic here
        if realized_vol < implied_move * 0.5:
            return True
        return False
```

---

For more examples and detailed usage, see the [examples/](../examples/) directory.