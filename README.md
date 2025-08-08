# MES 0DTE Options Trading Bot

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-255%20passing-brightgreen)](docs/testing/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-yellow)](docs/testing/coverage.md)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](docs/DEPLOYMENT.md#docker)

A production-grade automated trading system for Micro E-mini S&P 500 (MES) 0-day-to-expiry (0DTE) options using systematic strangle strategies with ML-enhanced decision making.

## 🎯 Overview

This bot systematically buys deep out-of-the-money (OTM) same-day strangles on MES options, aiming to capture occasional bursts of intraday volatility while strictly limiting risk. The strategy is designed for small accounts and circumvents pattern-day-trading rules by trading futures options.

### Key Features

- **Automated Trading**: Connects to Interactive Brokers and places OTM strangles based on market conditions
- **Risk Management**: Strict position limits, drawdown controls, and automated profit-taking
- **Real-time Monitoring**: Live Streamlit dashboard with P&L tracking and position management
- **Backtesting**: Synthetic option pricing engine for historical strategy validation
- **Production Ready**: Docker containerization, systemd integration, and comprehensive logging

## 📊 Strategy Logic

### Core Concept

- **Entry**: Place strangles when realized 60-minute volatility < 67% of implied daily move
- **Strikes**: Use 1.25x and 1.5x implied move distances from current price
- **Profit Target**: 400% of premium paid per leg
- **Risk Management**: Accept 100% loss (let options expire worthless)
- **Frequency**: Maximum ~10-12 strangles per day, minimum 30 minutes between trades

### Risk Controls

- Max $25 premium per strangle
- Max 15 concurrent open trades
- Max $750 daily drawdown (15% of $5k account)
- Auto-flatten all positions at 15:58 ET

## 🚀 Quick Start

### Prerequisites

- Interactive Brokers account with API access
- CME Micro E-mini futures and options market data subscription
- Python 3.10+ or Docker
- 4GB+ RAM for production deployment

### Installation

```bash
# Clone repository
git clone https://github.com/dock108/mes-bot.git
cd mes-bot

# Install with Poetry (recommended)
curl -sSL https://install.python-poetry.org | python3 -
poetry install

# Or with pip
pip install -r requirements.txt
```

### Configuration

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings (see docs/CONFIG_REFERENCE.md for all options)
nano .env
```

Key settings to configure:
- `TRADE_MODE`: paper or live
- `IB_GATEWAY_PORT`: 7497 (paper) or 7496 (live)
- `MAX_DRAW`: Maximum daily drawdown allowed
- `MAX_OPEN_TRADES`: Concurrent position limit

### Running the Bot

```bash
# Start IB Gateway/TWS first, then:

# Run trading bot
poetry run python -m app.bot

# Start web dashboard (separate terminal)
poetry run streamlit run app/ui.py

# Or use Docker
docker-compose up -d
```

Access dashboard at http://localhost:8501

## 📚 Documentation

### Essential Guides
- [User Guide](docs/USER_GUIDE.md) - Comprehensive setup and operation guide
- [Configuration Reference](docs/CONFIG_REFERENCE.md) - All configuration options
- [API Reference](docs/API_REFERENCE.md) - Module and function documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

### Deployment & Operations
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Testing Guide](docs/testing/README.md) - Testing strategies and execution
- [Architecture](docs/architecture/README.md) - System design and components

### Development
- [Contributing](CONTRIBUTING.md) - How to contribute to the project
- [Changelog](CHANGELOG.md) - Version history and updates

## 🏗️ Architecture

### Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IB Gateway    │    │   Bot Engine    │    │  Streamlit UI   │
│                 │    │                 │    │                 │
│ • Market Data   │◄──►│ • Strategy      │    │ • Live Monitor  │
│ • Order Exec    │    │ • Risk Mgmt     │    │ • Backtesting   │
│ • API Server    │    │ • Scheduling    │    │ • Configuration │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   SQLite DB     │    │   Log Files     │
                       │                 │    │                 │
                       │ • Trades        │    │ • Bot Events    │
                       │ • Daily P&L     │    │ • Errors        │
                       │ • Backtests     │    │ • IB Messages   │
                       └─────────────────┘    └─────────────────┘
```

### Key Modules

- **`app/bot.py`**: Main orchestrator with scheduling and event handling
- **`app/strategy.py`**: Core trading logic and signal generation
- **`app/ib_client.py`**: Interactive Brokers API integration
- **`app/risk_manager.py`**: Position limits and drawdown monitoring
- **`app/backtester.py`**: Historical simulation with synthetic pricing
- **`app/ui.py`**: Streamlit dashboard for monitoring and configuration

## 📈 Performance Monitoring

### Live Dashboard Features

- **Real-time Metrics**: Daily P&L, win rate, active trades
- **Position Tracking**: Live updates on open strangles with unrealized P&L
- **Risk Monitoring**: Drawdown tracking, position limits, exposure metrics
- **Historical Analysis**: Equity curves, performance statistics, trade history

### Backtesting

The bot includes a comprehensive backtesting engine that:

- Fetches historical price data from Yahoo Finance
- Uses Black-Scholes model for synthetic option pricing
- Simulates exact strategy logic and risk controls
- Provides detailed performance metrics and trade-by-trade analysis

## ⚙️ Configuration

### Key Parameters (`.env` file)

```bash
# Trading Configuration
TRADE_MODE=paper                    # paper or live
MAX_OPEN_TRADES=15                 # Maximum concurrent strangles
MAX_PREMIUM_PER_STRANGLE=25        # Max $ risk per trade
PROFIT_TARGET_MULTIPLIER=4         # Take profit at 4x premium
MAX_DRAW=750                       # Max daily drawdown ($)

# Strategy Parameters
IMPLIED_MOVE_MULTIPLIER_1=1.25     # First strike level
IMPLIED_MOVE_MULTIPLIER_2=1.5      # Second strike level
VOLATILITY_THRESHOLD=0.67          # Entry condition threshold
MIN_TIME_BETWEEN_TRADES=30         # Minutes between trades

# Market Hours (ET)
FLATTEN_HOUR=15                    # Auto-flatten hour
FLATTEN_MINUTE=58                  # Auto-flatten minute
```

### Risk Management

The bot implements multiple layers of risk control:

1. **Trade-Level**: Premium limits, strike selection rules
2. **Position-Level**: Maximum concurrent trades, exposure limits
3. **Account-Level**: Daily drawdown limits, equity monitoring
4. **Time-Based**: Market hours enforcement, end-of-day flattening

## 🔧 Features

### Trading Capabilities
- ✅ Automated 0DTE strangle execution
- ✅ Dynamic strike selection based on implied move
- ✅ Volatility-based entry signals
- ✅ Profit target and stop-loss management
- ✅ End-of-day position flattening

### Risk Management
- ✅ Position-level premium limits
- ✅ Account-level drawdown controls
- ✅ Maximum concurrent position limits
- ✅ Time-based trading restrictions
- ✅ Emergency stop functionality

### ML Enhancement (Optional)
- ✅ Feature engineering pipeline
- ✅ Multiple model ensemble
- ✅ Real-time prediction serving
- ✅ Automatic model retraining
- ✅ Fallback to rule-based system

### Monitoring & Analysis
- ✅ Real-time P&L tracking
- ✅ Position monitoring dashboard
- ✅ Historical performance analytics
- ✅ Comprehensive backtesting engine
- ✅ Trade execution logs

### Infrastructure
- ✅ Automatic connection recovery
- ✅ Circuit breaker pattern
- ✅ Docker containerization
- ✅ Systemd service integration
- ✅ Comprehensive test coverage

## 📊 Performance

### Backtesting Results (2024 Q1)
```
Period: 2024-01-01 to 2024-03-31
Initial Capital: $5,000
Final Capital: $5,847
Total Return: +16.9%
Max Drawdown: -8.5%
Win Rate: 23.4%
Sharpe Ratio: 1.34
```

*Note: Past performance does not guarantee future results.*

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m performance   # Performance tests

# Run UAT tests (requires Streamlit running)
streamlit run app/ui.py &
pytest tests/uat/
```

Current test coverage: **83.1%** across 255 tests

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop services
docker-compose down
```

See [Deployment Guide](docs/DEPLOYMENT.md) for production deployment options.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code style and standards
- Development workflow
- Testing requirements
- Pull request process

## ⚠️ Risk Disclaimer

**IMPORTANT**: This software is for educational purposes only. Trading options involves substantial risk of loss and is not suitable for all investors. 

- Options trading can result in total loss of investment
- 0DTE options are particularly risky and volatile
- Past performance does not guarantee future results
- Always test thoroughly in paper trading first
- Never trade with money you cannot afford to lose

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](docs/)
- [Issue Tracker](https://github.com/dock108/mes-bot/issues)
- [Discussions](https://github.com/dock108/mes-bot/discussions)

## 🙏 Acknowledgments

- Interactive Brokers for API access
- ib_insync for Python IB integration
- Streamlit for dashboard framework
- The open-source trading community

---

**Built for systematic traders who value robust risk management and production-grade reliability.**

*For detailed setup and usage instructions, see the [User Guide](docs/USER_GUIDE.md).*

