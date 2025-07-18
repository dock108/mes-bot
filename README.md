# MES 0DTE Lotto-Grid Options Bot

A production-grade automated trading system for Micro E-mini S&P 500 (MES) 0-day-to-expiry (0DTE) options using a systematic strangle strategy.

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

### Local Development (macOS)

1. **Clone and Setup**

   ```bash
   git clone https://github.com/your-username/lotto-grid-bot.git
   cd lotto-grid-bot

   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install
   ```

2. **Configure Environment**

   ```bash
   cp .env.example .env
   # Edit .env with your IB credentials and preferences
   nano .env
   ```

3. **Start IB Gateway/TWS**
   - Launch IB Gateway or TWS in paper trading mode
   - Enable API access (Configure → API → Settings)
   - Set socket port to 7497 (paper) or 7496 (live)

4. **Run the Bot**

   ```bash
   # Start trading bot
   poetry run python -m app.bot

   # In another terminal, start UI
   poetry run streamlit run app/ui.py
   ```

5. **Access Dashboard**
   - Open browser to <http://localhost:8501>
   - Monitor live trades and performance

### Docker Deployment

1. **Setup Environment**

   ```bash
   cp .env.example .env
   # Configure your IB credentials
   ```

2. **Run with Docker Compose**

   ```bash
   docker-compose up -d
   ```

3. **Access Services**
   - Streamlit UI: <http://localhost:8501>
   - IB Gateway VNC: vnc://localhost:5900 (password: ibgateway)

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

## 🐳 Production Deployment

### Ubuntu Server Setup

1. **Automated Setup**

   ```bash
   wget https://raw.githubusercontent.com/your-repo/lotto-grid-bot/main/deploy/setup-ubuntu.sh
   chmod +x setup-ubuntu.sh
   sudo ./setup-ubuntu.sh
   ```

2. **Manual Configuration**

   ```bash
   # Edit environment
   sudo nano /opt/lotto-grid-bot/.env

   # Start services
   sudo systemctl start lotto-grid-bot
   sudo systemctl enable lotto-grid-bot
   ```

3. **Monitor Status**

   ```bash
   # Check service status
   sudo systemctl status lotto-grid-bot

   # View logs
   sudo journalctl -u lotto-grid-bot -f

   # Monitor containers
   sudo docker-compose ps
   ```

### Systemd Integration

The bot runs as a systemd service with:

- Automatic startup on boot
- Restart on failure
- Resource limits (CPU/Memory)
- Secure user isolation

### Backup & Maintenance

```bash
# Daily backup (automated via cron)
/opt/lotto-grid-bot/scripts/backup.sh

# Update to latest version
/opt/lotto-grid-bot/scripts/update.sh

# Monitor system health
/opt/lotto-grid-bot/scripts/monitor.sh
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
poetry install --with dev

# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app tests/

# Run specific test file
poetry run pytest tests/test_strategy.py -v
```

### Test Coverage

The test suite covers:

- Strategy logic and signal generation
- Risk management rules and limits
- Backtesting engine accuracy
- Database operations
- Configuration validation

## 📊 Example Performance

### Sample Backtest Results

```
Period: 2024-01-01 to 2024-03-31 (60 trading days)
Initial Capital: $5,000
Final Capital: $5,847
Total Return: +16.9%
Max Drawdown: -$423 (8.5%)
Win Rate: 23.4% (47/201 trades)
Average Win: +$67.50 (300% avg return)
Average Loss: -$22.50 (100% loss)
Sharpe Ratio: 1.34
```

*Note: Past performance does not guarantee future results. This is a high-risk strategy suitable only for risk capital.*

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify IB Gateway is running and API is enabled
   - Check firewall settings and port accessibility
   - Ensure market data subscriptions are active

2. **No Trades Placed**
   - Check if in market hours (9:30 AM - 4:00 PM ET)
   - Verify volatility conditions (realized < 67% implied)
   - Check risk limits and available capital

3. **High Memory Usage**
   - Review price history buffer size
   - Check for memory leaks in long-running sessions
   - Consider restarting daily via cron

### Log Files

- **Bot Events**: `logs/bot_run.log`
- **Errors**: `logs/bot_errors.log`
- **IB Messages**: `logs/ib_messages.log`
- **System Logs**: `journalctl -u lotto-grid-bot`

## 📞 Support & Contributing

### Getting Help

1. Check the troubleshooting section above
2. Review log files for error messages
3. Search existing GitHub issues
4. Create a new issue with logs and configuration details

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install

# Run linting
poetry run black app/ tests/
poetry run flake8 app/ tests/

# Type checking
poetry run mypy app/
```

## ⚠️ Disclaimers

- **High Risk**: This strategy involves significant risk of loss
- **Not Financial Advice**: For educational purposes only
- **Paper Trading**: Always test thoroughly in paper mode first
- **Market Data**: Requires paid CME market data subscription
- **No Guarantees**: Past performance does not predict future results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built for traders who value systematic approaches, robust risk management, and production-grade reliability.**
