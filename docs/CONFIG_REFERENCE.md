# Configuration Reference

Complete configuration guide for the MES 0DTE Options Trading Bot.

## Table of Contents

- [Environment Variables](#environment-variables)
  - [Core Settings](#core-settings)
  - [Trading Parameters](#trading-parameters)
  - [Risk Management](#risk-management)
  - [Interactive Brokers](#interactive-brokers)
  - [Market Hours](#market-hours)
  - [ML Configuration](#ml-configuration)
  - [Database](#database)
  - [Logging](#logging)
  - [Monitoring](#monitoring)
- [Configuration Files](#configuration-files)
- [Parameter Ranges](#parameter-ranges)
- [Configuration Profiles](#configuration-profiles)
- [Validation Rules](#validation-rules)

## Environment Variables

All configuration is managed through environment variables. Create a `.env` file in the project root based on `.env.example`.

### Core Settings

| Variable | Type | Default | Description | Valid Values |
|----------|------|---------|-------------|--------------|
| `TRADE_MODE` | string | `paper` | Trading mode | `paper`, `live` |
| `START_CASH` | float | `5000` | Starting capital in USD | `1000` - `1000000` |
| `ACCOUNT_ID` | string | - | IB account ID | Any valid IB account |
| `CLIENT_ID` | int | `1` | IB client ID | `1` - `32` |
| `ENABLE_ML` | boolean | `false` | Enable ML decision engine | `true`, `false` |
| `DEBUG_MODE` | boolean | `false` | Enable debug logging | `true`, `false` |

### Trading Parameters

| Variable | Type | Default | Description | Valid Range |
|----------|------|---------|-------------|-------------|
| `SYMBOL` | string | `MES` | Trading symbol | `MES`, `ES` |
| `CONTRACT_MULTIPLIER` | int | `5` | Contract multiplier | Symbol-dependent |
| `MAX_OPEN_TRADES` | int | `15` | Maximum concurrent positions | `1` - `50` |
| `MAX_PREMIUM_PER_STRANGLE` | float | `25` | Max premium per strangle ($) | `5` - `500` |
| `MIN_PREMIUM_PER_STRANGLE` | float | `5` | Min premium per strangle ($) | `1` - `100` |
| `PROFIT_TARGET_MULTIPLIER` | float | `4.0` | Profit target as multiplier of premium | `2.0` - `10.0` |
| `MIN_TIME_BETWEEN_TRADES` | int | `30` | Minutes between trades | `5` - `120` |
| `TRADE_QUANTITY` | int | `1` | Contracts per trade | `1` - `10` |

### Risk Management

| Variable | Type | Default | Description | Valid Range |
|----------|------|---------|-------------|-------------|
| `MAX_DRAW` | float | `750` | Maximum daily drawdown ($) | `100` - `10000` |
| `MAX_DRAW_PERCENT` | float | `15` | Maximum drawdown (% of capital) | `5` - `50` |
| `POSITION_SIZE_PERCENT` | float | `2` | Position size (% of capital) | `0.5` - `10` |
| `MAX_LOSS_PER_TRADE` | float | `100` | Max loss per trade ($) | `10` - `1000` |
| `RISK_CHECK_INTERVAL` | int | `60` | Risk check interval (seconds) | `10` - `300` |
| `EMERGENCY_STOP_ENABLED` | boolean | `true` | Enable emergency stop | `true`, `false` |
| `AUTO_FLATTEN_ON_DISCONNECT` | boolean | `true` | Flatten positions on disconnect | `true`, `false` |

### Interactive Brokers

| Variable | Type | Default | Description | Valid Values |
|----------|------|---------|-------------|--------------|
| `IB_GATEWAY_HOST` | string | `127.0.0.1` | IB Gateway host | IP address or hostname |
| `IB_GATEWAY_PORT` | int | `7497` | IB Gateway port | `7496` (live), `7497` (paper) |
| `IB_USERNAME` | string | - | IB username | Valid IB username |
| `IB_PASSWORD` | string | - | IB password | Valid IB password |
| `IB_TRADING_MODE` | string | `paper` | IB trading mode | `paper`, `live` |
| `IB_MARKET_DATA_TYPE` | int | `3` | Market data type | `1` (live), `2` (frozen), `3` (delayed), `4` (delayed frozen) |
| `IB_CONNECTION_TIMEOUT` | int | `30` | Connection timeout (seconds) | `10` - `120` |
| `IB_REQUEST_TIMEOUT` | int | `10` | Request timeout (seconds) | `5` - `60` |
| `IB_MAX_RETRY_ATTEMPTS` | int | `3` | Max connection retries | `1` - `10` |

### Market Hours

All times in Eastern Time (ET).

| Variable | Type | Default | Description | Valid Range |
|----------|------|---------|-------------|-------------|
| `MARKET_OPEN_HOUR` | int | `9` | Market open hour | `0` - `23` |
| `MARKET_OPEN_MINUTE` | int | `30` | Market open minute | `0` - `59` |
| `MARKET_CLOSE_HOUR` | int | `16` | Market close hour | `0` - `23` |
| `MARKET_CLOSE_MINUTE` | int | `0` | Market close minute | `0` - `59` |
| `TRADING_START_HOUR` | int | `9` | Trading start hour | `0` - `23` |
| `TRADING_START_MINUTE` | int | `35` | Trading start minute | `0` - `59` |
| `TRADING_END_HOUR` | int | `15` | Trading end hour | `0` - `23` |
| `TRADING_END_MINUTE` | int | `55` | Trading end minute | `0` - `59` |
| `FLATTEN_HOUR` | int | `15` | Position flatten hour | `0` - `23` |
| `FLATTEN_MINUTE` | int | `58` | Position flatten minute | `0` - `59` |

### Strategy Parameters

| Variable | Type | Default | Description | Valid Range |
|----------|------|---------|-------------|-------------|
| `IMPLIED_MOVE_MULTIPLIER_1` | float | `1.25` | First strike distance multiplier | `0.5` - `3.0` |
| `IMPLIED_MOVE_MULTIPLIER_2` | float | `1.5` | Second strike distance multiplier | `0.5` - `3.0` |
| `VOLATILITY_THRESHOLD` | float | `0.67` | Volatility entry threshold | `0.3` - `1.0` |
| `VOLATILITY_LOOKBACK_MINUTES` | int | `60` | Volatility calculation period | `15` - `240` |
| `REALIZED_VOL_PERIOD` | int | `20` | Realized volatility period (bars) | `10` - `100` |
| `IMPLIED_VOL_ADJUSTMENT` | float | `1.0` | Implied volatility adjustment | `0.5` - `2.0` |
| `ENTRY_SIGNAL_THRESHOLD` | float | `0.7` | ML signal threshold for entry | `0.5` - `0.95` |
| `EXIT_SIGNAL_THRESHOLD` | float | `0.3` | ML signal threshold for exit | `0.1` - `0.5` |

### ML Configuration

| Variable | Type | Default | Description | Valid Values |
|----------|------|---------|-------------|--------------|
| `ML_ENABLED` | boolean | `false` | Enable ML predictions | `true`, `false` |
| `ML_MODEL_PATH` | string | `models/` | Path to ML models | Valid directory path |
| `ML_FEATURE_SET` | string | `standard` | Feature set to use | `minimal`, `standard`, `full` |
| `ML_PREDICTION_THRESHOLD` | float | `0.6` | Prediction confidence threshold | `0.5` - `0.95` |
| `ML_ENSEMBLE_ENABLED` | boolean | `true` | Use ensemble of models | `true`, `false` |
| `ML_RETRAIN_INTERVAL_DAYS` | int | `7` | Days between model retraining | `1` - `30` |
| `ML_MIN_TRAINING_SAMPLES` | int | `1000` | Minimum samples for training | `100` - `10000` |
| `ML_FALLBACK_TO_RULES` | boolean | `true` | Fallback to rules if ML fails | `true`, `false` |

### Database

| Variable | Type | Default | Description | Valid Values |
|----------|------|---------|-------------|--------------|
| `DATABASE_URL` | string | `sqlite:///mes_bot.db` | Database connection URL | Valid database URL |
| `DB_POOL_SIZE` | int | `5` | Database connection pool size | `1` - `20` |
| `DB_MAX_OVERFLOW` | int | `10` | Max overflow connections | `0` - `50` |
| `DB_POOL_TIMEOUT` | int | `30` | Pool timeout (seconds) | `10` - `120` |
| `DB_ECHO` | boolean | `false` | Echo SQL statements | `true`, `false` |
| `DB_BACKUP_ENABLED` | boolean | `true` | Enable database backups | `true`, `false` |
| `DB_BACKUP_INTERVAL_HOURS` | int | `24` | Hours between backups | `1` - `168` |

### Logging

| Variable | Type | Default | Description | Valid Values |
|----------|------|---------|-------------|--------------|
| `LOG_LEVEL` | string | `INFO` | Logging level | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FILE` | string | `logs/bot.log` | Main log file path | Valid file path |
| `LOG_ERROR_FILE` | string | `logs/errors.log` | Error log file path | Valid file path |
| `LOG_FORMAT` | string | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Log format | Valid format string |
| `LOG_ROTATION` | string | `daily` | Log rotation schedule | `daily`, `weekly`, `size` |
| `LOG_MAX_SIZE_MB` | int | `100` | Max log file size (MB) | `10` - `1000` |
| `LOG_BACKUP_COUNT` | int | `7` | Number of log backups | `1` - `30` |
| `LOG_TO_CONSOLE` | boolean | `true` | Log to console | `true`, `false` |

### Monitoring

| Variable | Type | Default | Description | Valid Values |
|----------|------|---------|-------------|--------------|
| `MONITORING_ENABLED` | boolean | `false` | Enable monitoring | `true`, `false` |
| `METRICS_PORT` | int | `8080` | Metrics server port | `1024` - `65535` |
| `HEALTH_CHECK_INTERVAL` | int | `60` | Health check interval (seconds) | `10` - `300` |
| `ALERT_EMAIL` | string | - | Email for alerts | Valid email address |
| `SLACK_WEBHOOK_URL` | string | - | Slack webhook for alerts | Valid webhook URL |
| `DISCORD_WEBHOOK_URL` | string | - | Discord webhook for alerts | Valid webhook URL |
| `ALERT_ON_ERROR` | boolean | `true` | Send alerts on errors | `true`, `false` |
| `ALERT_ON_TRADE` | boolean | `false` | Send alerts on trades | `true`, `false` |

### Performance

| Variable | Type | Default | Description | Valid Range |
|----------|------|---------|-------------|-------------|
| `CACHE_ENABLED` | boolean | `true` | Enable caching | `true`, `false` |
| `CACHE_TTL_SECONDS` | int | `300` | Cache TTL (seconds) | `60` - `3600` |
| `MAX_WORKERS` | int | `4` | Max worker threads | `1` - `16` |
| `BATCH_SIZE` | int | `100` | Batch processing size | `10` - `1000` |
| `RATE_LIMIT_ENABLED` | boolean | `true` | Enable rate limiting | `true`, `false` |
| `RATE_LIMIT_REQUESTS` | int | `100` | Requests per interval | `10` - `1000` |
| `RATE_LIMIT_INTERVAL` | int | `60` | Rate limit interval (seconds) | `1` - `300` |

## Configuration Files

### .env.example

Template configuration file with all available options:

```bash
# Core Settings
TRADE_MODE=paper
START_CASH=5000
ACCOUNT_ID=DU123456

# Trading Parameters
MAX_OPEN_TRADES=15
MAX_PREMIUM_PER_STRANGLE=25
PROFIT_TARGET_MULTIPLIER=4.0

# Risk Management
MAX_DRAW=750
MAX_DRAW_PERCENT=15

# Interactive Brokers
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=7497
IB_USERNAME=your_username
IB_PASSWORD=your_password

# Strategy
IMPLIED_MOVE_MULTIPLIER_1=1.25
IMPLIED_MOVE_MULTIPLIER_2=1.5
VOLATILITY_THRESHOLD=0.67

# Market Hours (ET)
FLATTEN_HOUR=15
FLATTEN_MINUTE=58
```

### config.yaml (Alternative)

Optional YAML configuration format:

```yaml
trading:
  mode: paper
  start_cash: 5000
  max_open_trades: 15

risk:
  max_draw: 750
  max_draw_percent: 15

strategy:
  implied_move_multipliers:
    - 1.25
    - 1.5
  volatility_threshold: 0.67

ib:
  gateway:
    host: 127.0.0.1
    port: 7497
  credentials:
    username: ${IB_USERNAME}
    password: ${IB_PASSWORD}
```

## Parameter Ranges

### Risk Profiles

#### Conservative
```bash
MAX_OPEN_TRADES=5
MAX_DRAW_PERCENT=10
PROFIT_TARGET_MULTIPLIER=3.0
IMPLIED_MOVE_MULTIPLIER_1=1.5
IMPLIED_MOVE_MULTIPLIER_2=1.75
VOLATILITY_THRESHOLD=0.5
```

#### Moderate
```bash
MAX_OPEN_TRADES=10
MAX_DRAW_PERCENT=15
PROFIT_TARGET_MULTIPLIER=4.0
IMPLIED_MOVE_MULTIPLIER_1=1.25
IMPLIED_MOVE_MULTIPLIER_2=1.5
VOLATILITY_THRESHOLD=0.67
```

#### Aggressive
```bash
MAX_OPEN_TRADES=20
MAX_DRAW_PERCENT=25
PROFIT_TARGET_MULTIPLIER=5.0
IMPLIED_MOVE_MULTIPLIER_1=1.0
IMPLIED_MOVE_MULTIPLIER_2=1.25
VOLATILITY_THRESHOLD=0.8
```

### Account Size Configurations

#### Small Account ($5,000)
```bash
START_CASH=5000
MAX_OPEN_TRADES=5
MAX_PREMIUM_PER_STRANGLE=15
MAX_DRAW=500
```

#### Medium Account ($25,000)
```bash
START_CASH=25000
MAX_OPEN_TRADES=12
MAX_PREMIUM_PER_STRANGLE=50
MAX_DRAW=2500
```

#### Large Account ($100,000+)
```bash
START_CASH=100000
MAX_OPEN_TRADES=20
MAX_PREMIUM_PER_STRANGLE=200
MAX_DRAW=10000
```

## Configuration Profiles

### Development
```bash
# .env.development
TRADE_MODE=paper
DEBUG_MODE=true
LOG_LEVEL=DEBUG
DB_ECHO=true
MONITORING_ENABLED=false
```

### Testing
```bash
# .env.test
TRADE_MODE=paper
DATABASE_URL=sqlite:///test.db
LOG_LEVEL=WARNING
CACHE_ENABLED=false
```

### Production
```bash
# .env.production
TRADE_MODE=live
DEBUG_MODE=false
LOG_LEVEL=INFO
MONITORING_ENABLED=true
ALERT_ON_ERROR=true
DB_BACKUP_ENABLED=true
```

## Validation Rules

### Required Variables

The following variables MUST be set:
- `TRADE_MODE`
- `IB_USERNAME` (if not using IB Gateway docker)
- `IB_PASSWORD` (if not using IB Gateway docker)

### Conditional Requirements

- If `TRADE_MODE=live`:
  - `ACCOUNT_ID` must be set
  - `IB_GATEWAY_PORT` must be `7496`
  - `EMERGENCY_STOP_ENABLED` should be `true`

- If `ML_ENABLED=true`:
  - `ML_MODEL_PATH` must exist
  - `ML_FEATURE_SET` must be valid

- If `MONITORING_ENABLED=true`:
  - At least one alert method must be configured

### Value Constraints

- `IMPLIED_MOVE_MULTIPLIER_2` > `IMPLIED_MOVE_MULTIPLIER_1`
- `TRADING_END_HOUR:MINUTE` < `FLATTEN_HOUR:MINUTE`
- `MAX_DRAW` < `START_CASH`
- `MIN_PREMIUM_PER_STRANGLE` < `MAX_PREMIUM_PER_STRANGLE`

## Environment-Specific Settings

### Local Development
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export ENV=development
```

### Docker
```bash
# docker-compose.yml
environment:
  - TRADE_MODE=${TRADE_MODE}
  - IB_GATEWAY_HOST=ib-gateway
```

### Kubernetes
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mes-bot-config
data:
  TRADE_MODE: "paper"
  LOG_LEVEL: "INFO"
```

## Security Considerations

### Sensitive Variables

Never commit these to version control:
- `IB_USERNAME`
- `IB_PASSWORD`
- `ACCOUNT_ID`
- `DATABASE_URL` (if contains credentials)
- `SLACK_WEBHOOK_URL`
- `DISCORD_WEBHOOK_URL`
- `ALERT_EMAIL`

### Best Practices

1. Use `.env` files for local development
2. Use environment variables in production
3. Use secrets management in cloud deployments
4. Rotate credentials regularly
5. Use read-only database users where possible
6. Enable encryption for sensitive data

## Troubleshooting Configuration

### Common Issues

1. **Missing Variables**: Check `.env` file exists and is loaded
2. **Type Errors**: Ensure numeric values are not quoted
3. **Path Issues**: Use absolute paths or verify working directory
4. **Permission Errors**: Check file/directory permissions
5. **Connection Issues**: Verify network settings and firewall rules

### Validation Script

```bash
# Validate configuration
python scripts/validate_config.py

# Test with specific env file
python scripts/validate_config.py --env-file .env.production
```

---

For more information, see the [User Guide](USER_GUIDE.md) and [Deployment Guide](DEPLOYMENT.md).
