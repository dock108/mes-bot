# MES 0DTE Lotto-Grid Options Bot - User Decision Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Critical Configuration Decisions](#critical-configuration-decisions)
4. [Getting Started Workflow](#getting-started-workflow)
5. [Key Decision Points](#key-decision-points)
6. [Testing & Validation](#testing--validation)
7. [Monitoring & Management](#monitoring--management)
8. [Common Scenarios](#common-scenarios)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

**New to 0DTE options trading?** Start here with conservative settings:

```bash
# 1. Install and setup
make dev-setup

# 2. Configure for paper trading (edit .env)
TRADE_MODE=paper
START_CASH=5000
MAX_DRAW=500
MAX_OPEN_TRADES=5

# 3. Test with backtest
make run-backtest

# 4. Start paper trading
make run-bot
make run-ui  # In another terminal
```

**Experienced trader?** Jump to [Configuration Decisions](#critical-configuration-decisions)

---

## Prerequisites & Setup

### System Requirements

- **Python 3.10+** (with Poetry package manager)
- **Interactive Brokers Account** (paper or live)
- **IB Gateway or TWS** (Trader Workstation)
- **Minimum 8GB RAM** (for data processing)
- **Stable Internet** (for real-time market data)

### Interactive Brokers Setup

#### 1. Account Requirements

- ✅ **Options Trading Permissions**: Level 2 minimum for spreads
- ✅ **Futures Options**: MES options trading enabled
- ✅ **API Access**: Enable in Account Management
- ✅ **Market Data**: Real-time ES/MES data subscription

#### 2. IB Gateway/TWS Configuration

```
API Settings:
├── Enable ActiveX and Socket Clients: ✅
├── Socket Port: 7497 (paper) / 7496 (live)
├── Master API Client ID: 0
├── Read-Only API: ❌ (must be disabled)
└── Download open orders on connection: ✅
```

#### 3. Market Data Subscriptions Needed

- **US Equity and Options Add-On Streaming Bundle**
- **US Futures and Futures Options Bundle**

### Installation

#### Option 1: Standard Setup

```bash
# Clone the repository
git clone <repository-url>
cd lotto_grid_bot

# Install with Poetry
make dev-setup

# This will:
# - Install all dependencies
# - Create .env from .env.example
# - Validate configuration
```

#### Option 2: Docker Setup

```bash
# Setup environment
make setup-env

# Edit .env with your settings
# Then start with Docker
make docker-up
```

---

## Critical Configuration Decisions

### 📊 Risk Management Parameters

#### Starting Capital (`START_CASH`)

- **Conservative**: $5,000 - $10,000
- **Moderate**: $10,000 - $25,000
- **Aggressive**: $25,000+

**Decision Guide:**

- Start with amount you can afford to lose completely
- 0DTE options can move fast - expect volatility
- Consider this "risk capital" separate from main portfolio

#### Maximum Drawdown (`MAX_DRAW`)

```
Conservative: 10-15% of starting capital
Moderate:     15-20% of starting capital
Aggressive:   20-25% of starting capital

Examples:
$5,000 account  → $500-750 max drawdown
$10,000 account → $1,500-2,000 max drawdown
$25,000 account → $3,750-5,000 max drawdown
```

**Key Decision**: Bot stops trading when this level is hit

#### Maximum Open Trades (`MAX_OPEN_TRADES`)

```
Beginner:     5-10 positions
Intermediate: 10-15 positions
Advanced:     15-20 positions
```

**Consider:**

- More positions = more diversification but harder to manage
- Each position requires margin (typically $400-600 per MES strangle)
- Risk increases with more concurrent positions

### ⚡ Strategy Parameters

#### Profit Target (`PROFIT_TARGET_MULTIPLIER`)

```
Conservative: 3.0x (300% profit)
Standard:     4.0x (400% profit)
Aggressive:   5.0x (500% profit)
```

**Trade-off Analysis:**

- Higher targets = more profit per winner but lower win rate
- Lower targets = higher win rate but less profit per winner
- 4.0x is historically optimal for 0DTE strategies

#### Strike Selection (`IMPLIED_MOVE_MULTIPLIER_1` & `_2`)

```
Strike Distance = Implied Move × Multiplier

Conservative: 1.5x and 1.75x (far OTM, safer)
Standard:     1.25x and 1.5x (balanced)
Aggressive:   1.0x and 1.25x (closer ATM, riskier)
```

**Decision Impact:**

- Further strikes = lower premium, higher win rate, lower max profit
- Closer strikes = higher premium, lower win rate, higher max profit

#### Volatility Threshold (`VOLATILITY_THRESHOLD`)

```
Conservative: 0.5 (trade when realized vol < 50% of implied)
Standard:     0.67 (trade when realized vol < 67% of implied)
Aggressive:   0.8 (trade when realized vol < 80% of implied)
```

**Strategy Logic:**

- Lower threshold = fewer but higher-quality trade signals
- Higher threshold = more trades but potentially lower edge

### 📅 Timing Parameters

#### Market Hours

```
MARKET_OPEN_HOUR=9      # 9:30 AM ET (after futures open)
MARKET_OPEN_MINUTE=30
MARKET_CLOSE_HOUR=16    # 4:00 PM ET (equity close)
MARKET_CLOSE_MINUTE=0
FLATTEN_HOUR=15         # 3:58 PM ET (close before expiry)
FLATTEN_MINUTE=58
```

**Decision Considerations:**

- **Opening Time**: Avoid first 30 minutes (high volatility)
- **Closing Time**: Close positions before 4PM expiry
- **Flatten Time**: Critical - must exit before expiration

#### Trade Frequency (`MIN_TIME_BETWEEN_TRADES`)

```
Conservative: 60 minutes (slower, quality over quantity)
Standard:     30 minutes (balanced)
Aggressive:   15 minutes (higher frequency)
```

---

## Getting Started Workflow

### Phase 1: Setup & Validation (Day 1)

#### Step 1: Environment Configuration

```bash
# Copy and edit configuration
make setup-env
nano .env  # Edit with your settings
```

**Critical Settings to Configure:**

```bash
# Start conservative
TRADE_MODE=paper
START_CASH=5000
MAX_DRAW=500
MAX_OPEN_TRADES=5
PROFIT_TARGET_MULTIPLIER=4.0

# IB Credentials
IB_USERNAME=your_paper_trading_username
IB_PASSWORD=your_paper_trading_password
```

#### Step 2: Validate Setup

```bash
# Check configuration
make validate-env

# Test database connection
make validate-db

# Check overall health
make health-check
```

#### Step 3: Run Initial Backtest

```bash
# Run 30-day backtest with your settings
make run-backtest

# Review results - look for:
# - Positive total return
# - Win rate around 20-30%
# - Max drawdown within tolerance
```

### Phase 2: Paper Trading (Weeks 1-2)

#### Step 1: Start IB Gateway

1. Open IB Gateway or TWS
2. Login to paper trading account
3. Verify API is enabled (port 7497)
4. Ensure MES market data is active

#### Step 2: Launch Bot

```bash
# Terminal 1: Start trading bot
make run-bot

# Terminal 2: Start dashboard
make run-ui
```

#### Step 3: Monitor First Day

1. **Dashboard**: Open <http://localhost:8501>
2. **Watch for**:
   - Bot connects to IB successfully
   - Implied move calculation works
   - First trade signals appear
   - Positions open and close correctly

#### Step 4: Daily Review Process

1. **End of Day**: Review dashboard performance tab
2. **Check**:
   - Total P&L for the day
   - Number of trades executed
   - Win/loss ratio
   - Any errors in logs
3. **Adjust**: Based on performance, consider parameter tweaks

### Phase 3: Optimization (Weeks 3-4)

#### Performance Analysis

```bash
# After 1-2 weeks, analyze results
# Access Performance tab in dashboard

Key Metrics to Track:
├── Win Rate: Target 20-30%
├── Average Win: Should be 3-4x average loss
├── Max Drawdown: Stay within limits
├── Sharpe Ratio: Aim for > 1.0
└── Total Return: Positive over time
```

#### Parameter Tuning Decisions

**If Win Rate Too Low (<15%)**

```bash
# Make strikes further OTM
IMPLIED_MOVE_MULTIPLIER_1=1.5  # from 1.25
IMPLIED_MOVE_MULTIPLIER_2=1.75 # from 1.5

# Or reduce profit target
PROFIT_TARGET_MULTIPLIER=3.0   # from 4.0
```

**If Win Rate Too High (>40%)**

```bash
# Strategy may be too conservative
IMPLIED_MOVE_MULTIPLIER_1=1.0  # from 1.25
IMPLIED_MOVE_MULTIPLIER_2=1.25 # from 1.5

# Or increase profit target
PROFIT_TARGET_MULTIPLIER=5.0   # from 4.0
```

**If Too Many/Few Trades**

```bash
# Adjust volatility threshold
VOLATILITY_THRESHOLD=0.5   # Fewer trades (from 0.67)
VOLATILITY_THRESHOLD=0.8   # More trades (from 0.67)

# Adjust trade frequency
MIN_TIME_BETWEEN_TRADES=60 # Slower (from 30)
MIN_TIME_BETWEEN_TRADES=15 # Faster (from 30)
```

### Phase 4: Live Trading Decision

#### Prerequisites for Going Live

- ✅ **2+ weeks successful paper trading**
- ✅ **Positive overall P&L in paper**
- ✅ **Comfortable with strategy behavior**
- ✅ **No significant technical issues**
- ✅ **Live trading account approved for options**

#### Going Live Checklist

```bash
# 1. Update configuration
TRADE_MODE=live
IB_GATEWAY_PORT=7496           # Live port
IB_USERNAME=your_live_username
IB_PASSWORD=your_live_password

# 2. Start with reduced size
MAX_OPEN_TRADES=3              # Reduce from paper trading
START_CASH=<your_actual_balance>

# 3. Set conservative drawdown
MAX_DRAW=<10% of account>      # Be conservative initially
```

---

## Key Decision Points

### Decision Tree: Position Sizing

```
Account Size → Max Positions → Risk per Trade
└── $5,000    → 3-5 trades   → $200-400 per strangle
└── $10,000   → 5-8 trades   → $400-600 per strangle
└── $25,000   → 8-12 trades  → $600-1000 per strangle
└── $50,000+  → 12-15 trades → $1000+ per strangle
```

### Decision Tree: Risk Tolerance

```
Risk Appetite → Settings Profile
├── Conservative
│   ├── MAX_DRAW: 10% of account
│   ├── PROFIT_TARGET: 3.0x
│   ├── STRIKE_DISTANCE: 1.5x, 1.75x
│   └── VOL_THRESHOLD: 0.5
├── Moderate
│   ├── MAX_DRAW: 15% of account
│   ├── PROFIT_TARGET: 4.0x
│   ├── STRIKE_DISTANCE: 1.25x, 1.5x
│   └── VOL_THRESHOLD: 0.67
└── Aggressive
    ├── MAX_DRAW: 20% of account
    ├── PROFIT_TARGET: 5.0x
    ├── STRIKE_DISTANCE: 1.0x, 1.25x
    └── VOL_THRESHOLD: 0.8
```

### When to Stop Trading

#### Automatic Stops (Bot Handles)

- Maximum drawdown reached
- Market hours outside trading window
- IB connection issues
- Insufficient margin

#### Manual Stop Considerations

- **Market Conditions**: Unusual volatility events (FOMC, earnings)
- **Technical Issues**: Repeated errors or connection problems
- **Performance**: Consistent losses beyond expectations
- **Personal**: Stress levels, vacation, etc.

### When to Adjust Parameters

#### Weekly Review Triggers

```
Metric                  Action Needed
────────────────────────────────────────
Win Rate < 10%         → Increase strike distance
Win Rate > 50%         → Decrease strike distance
Avg Loss > 2x Premium → Review position sizing
Max DD > Target        → Reduce position size
No trades for days     → Lower vol threshold
Too many trades/day    → Raise vol threshold
```

---

## Testing & Validation

### Backtesting Strategy

#### 1. Historical Performance Test

```bash
# Test different parameter sets
make run-backtest

# Test scenarios:
# - Different volatility thresholds
# - Different strike distances
# - Different profit targets
```

#### 2. Stress Testing

```bash
# Edit config for stress test
MAX_OPEN_TRADES=20        # Test high load
VOLATILITY_THRESHOLD=0.9  # Test high frequency

# Run and observe:
# - System performance
# - Memory usage
# - Error handling
```

#### 3. Walk-Forward Analysis

- Test on multiple time periods
- Ensure consistent performance
- Identify market regime sensitivity

### Paper Trading Validation

#### Week 1 Checklist

- [ ] Bot connects reliably to IB
- [ ] Trades execute as expected
- [ ] Risk management works (stops)
- [ ] Dashboard shows accurate data
- [ ] No critical errors in logs

#### Week 2 Checklist

- [ ] Performance meets backtested expectations
- [ ] Drawdown stays within limits
- [ ] Win rate in expected range (20-30%)
- [ ] Comfortable with trade frequency
- [ ] Ready for position size scaling

---

## Monitoring & Management

### Daily Monitoring Routine

#### Morning (Pre-Market)

1. **Check System Health**

   ```bash
   make health-check
   ```

2. **Review Overnight Logs**
   - Check for any errors
   - Verify positions closed properly
   - Note any system messages

3. **Market Preparation**
   - Check economic calendar
   - Note any special events (FOMC, earnings)
   - Consider stopping bot for high-volatility events

#### During Trading Hours

##### Dashboard Monitoring

Access dashboard at `http://localhost:8501`

**Live Monitor Tab - Check Every 30-60 Minutes:**

- Current open positions
- P&L progression
- Bot status indicators
- Any error messages

**Key Metrics to Watch:**

```
Metric                 Normal Range        Action if Outside
─────────────────────────────────────────────────────────────
Win Rate               20-30%             Adjust strikes
Avg Win/Loss Ratio     3:1 to 5:1         Review profit target
Daily Drawdown         <5% of account     Consider stopping
Open Positions         Within max limit   Check margin usage
```

#### End of Day

1. **Performance Review**
   - Total P&L for day
   - Number of trades executed
   - Win/loss breakdown
   - Drawdown levels

2. **Risk Assessment**
   - Current account balance
   - Available margin
   - Maximum drawdown distance

3. **Planning Next Day**
   - Any parameter adjustments needed
   - Market events to consider
   - System maintenance required

### Weekly Analysis

#### Performance Tab Review

1. **Return Analysis**
   - Weekly P&L trend
   - Cumulative returns
   - Sharpe ratio development

2. **Trade Analysis**
   - Win rate stability
   - Average trade duration
   - Profit per trade trends

3. **Risk Analysis**
   - Maximum drawdown periods
   - Volatility of returns
   - Correlation with market moves

#### Parameter Optimization

```python
# Weekly review checklist
Review Frequency: Every Sunday
Data Period: Previous week + month-to-date

Questions to Ask:
├── Is performance meeting expectations?
├── Are drawdowns manageable?
├── Is trade frequency appropriate?
├── Any unusual market behavior to note?
└── Should parameters be adjusted?
```

---

## Common Scenarios

### Scenario 1: New Trader (Conservative Start)

**Profile**: New to options, wants to learn 0DTE strategies

**Recommended Settings:**

```bash
TRADE_MODE=paper
START_CASH=5000
MAX_DRAW=500
MAX_OPEN_TRADES=3
PROFIT_TARGET_MULTIPLIER=3.0
IMPLIED_MOVE_MULTIPLIER_1=1.5
IMPLIED_MOVE_MULTIPLIER_2=1.75
VOLATILITY_THRESHOLD=0.5
MIN_TIME_BETWEEN_TRADES=60
```

**Timeline:**

- Week 1-2: Paper trade with these settings
- Week 3-4: If comfortable, increase to 5 max trades
- Month 2: Consider going live with $2,500 real capital
- Month 3+: Gradually scale position size

### Scenario 2: Experienced Options Trader

**Profile**: Familiar with options, wants to automate 0DTE strategies

**Recommended Settings:**

```bash
TRADE_MODE=paper  # Still start with paper
START_CASH=10000
MAX_DRAW=1500
MAX_OPEN_TRADES=8
PROFIT_TARGET_MULTIPLIER=4.0
IMPLIED_MOVE_MULTIPLIER_1=1.25
IMPLIED_MOVE_MULTIPLIER_2=1.5
VOLATILITY_THRESHOLD=0.67
MIN_TIME_BETWEEN_TRADES=30
```

**Timeline:**

- Week 1: Paper trade, focus on automation vs manual differences
- Week 2: Optimize parameters based on style preferences
- Week 3-4: Go live with 50% of intended size
- Month 2: Scale to full size

### Scenario 3: High-Frequency Trader

**Profile**: Wants maximum trade frequency and automation

**Recommended Settings:**

```bash
TRADE_MODE=paper
START_CASH=25000
MAX_DRAW=3750
MAX_OPEN_TRADES=15
PROFIT_TARGET_MULTIPLIER=4.0
IMPLIED_MOVE_MULTIPLIER_1=1.0
IMPLIED_MOVE_MULTIPLIER_2=1.25
VOLATILITY_THRESHOLD=0.8
MIN_TIME_BETWEEN_TRADES=15
```

**Considerations:**

- Higher system resource requirements
- More market data costs
- Increased commission costs
- Need robust risk management

### Scenario 4: Part-Time Trader

**Profile**: Can't monitor constantly, wants set-and-forget

**Recommended Settings:**

```bash
TRADE_MODE=paper
START_CASH=7500
MAX_DRAW=750  # Lower due to less monitoring
MAX_OPEN_TRADES=5
PROFIT_TARGET_MULTIPLIER=3.0  # Take profits quicker
IMPLIED_MOVE_MULTIPLIER_1=1.5  # Further strikes for safety
IMPLIED_MOVE_MULTIPLIER_2=1.75
VOLATILITY_THRESHOLD=0.5  # Fewer, higher-quality trades
MIN_TIME_BETWEEN_TRADES=60
```

**Special Considerations:**

- Set lower drawdown limits
- Use more conservative parameters
- Check performance daily at minimum
- Have mobile alerts set up

---

## Troubleshooting

### Common Issues & Solutions

#### Bot Won't Connect to IB

```
Symptoms: "Connection failed" errors
Solutions:
├── Verify IB Gateway/TWS is running
├── Check API is enabled in IB settings
├── Confirm correct port (7497 paper / 7496 live)
├── Verify username/password in .env
└── Check firewall settings
```

#### No Trades Being Executed

```
Symptoms: Bot runs but no positions opened
Possible Causes:
├── Volatility threshold too low (no signals)
├── Max open trades reached
├── Insufficient margin
├── Market data issues
└── Outside trading hours

Solutions:
├── Check volatility threshold setting
├── Review account balance and margin
├── Verify market data subscriptions
└── Check market hours configuration
```

#### Unexpected Losses

```
Symptoms: Losses exceed expectations
Analysis Steps:
├── Review win rate vs expected (20-30%)
├── Check if profit targets being hit
├── Analyze market conditions during losses
├── Verify risk management settings
└── Consider if parameters need adjustment

Actions:
├── Reduce position size temporarily
├── Increase strike distances
├── Lower profit targets
├── Pause during high volatility events
└── Review and adjust parameters
```

#### Performance Degradation

```
Symptoms: Bot was profitable, now losing
Investigation:
├── Market regime change?
├── Increased competition in 0DTE space?
├── Parameters need reoptimization?
├── System/data issues?
└── Increased volatility environment?

Response:
├── Pause trading to analyze
├── Run fresh backtests on recent data
├── Consider parameter adjustments
├── Review market conditions
└── Possibly reduce size until resolved
```

### Emergency Procedures

#### Emergency Stop

```bash
# Stop bot immediately
pkill -f "python.*bot"

# Or use dashboard emergency stop button
# Access: http://localhost:8501 → Live Monitor → Emergency Stop
```

#### Position Management

- **Close All Positions**: Use IB TWS to manually close if needed
- **Monitor Margin**: Ensure sufficient margin for existing positions
- **Risk Assessment**: Calculate maximum loss exposure

#### Data Recovery

```bash
# Backup current data
make backup-data

# Restore from backup if needed
tar -xzf backups/lotto_grid_backup_YYYYMMDD_HHMMSS.tar.gz
```

### Support Resources

#### Log Files

```
Location: ./logs/
├── bot_run.log     # Main bot activity
├── bot_errors.log  # Error messages
└── ib_messages.log # IB API communication
```

#### Health Check

```bash
# Comprehensive system check
make health-check

# Check specific components
make validate-env    # Configuration
make validate-db     # Database
make status         # Overall system
```

#### Performance Analysis

```bash
# Run performance test
make perf-test

# Generate test coverage
make test-cov
```

---

## Conclusion

This bot implements a sophisticated 0DTE options strategy with professional risk management. Success depends on:

1. **Conservative Start**: Begin with paper trading and small sizes
2. **Continuous Monitoring**: Daily performance review and adjustment
3. **Risk Management**: Strict adherence to drawdown limits
4. **Parameter Optimization**: Regular review and adjustment of settings
5. **Market Awareness**: Understanding when to pause during unusual conditions

Remember: 0DTE options trading carries significant risk. Never trade with money you cannot afford to lose, and always start with paper trading to understand the strategy behavior.

**Key Success Factors:**

- Start small and scale gradually
- Monitor performance daily
- Adjust parameters based on market conditions
- Maintain strict risk discipline
- Have realistic expectations (aim for 15-25% annual returns)

For additional support, refer to the test results, logs, and dashboard analytics to understand your bot's performance and make informed decisions.
