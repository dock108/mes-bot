# UAT (User Acceptance Testing) Tests

## Overview

UAT tests are browser-based integration tests that verify the Streamlit dashboard UI functionality using Playwright.

## Running UAT Tests Locally

### Prerequisites

1. Install Playwright browsers:

   ```bash
   playwright install chromium
   ```

2. Start the Streamlit application:

   ```bash
   streamlit run app/ui.py
   ```

3. Run UAT tests in a separate terminal:

   ```bash
   pytest tests/uat/ -v
   ```

## CI/CD Configuration

UAT tests are **excluded from CI** because they require:

- A running Streamlit server
- Browser automation (Playwright)
- Interactive UI components
- Longer execution time

To run UAT tests in CI, you would need:

1. Start Streamlit server in background
2. Wait for server to be ready
3. Run Playwright tests
4. Handle teardown properly

## Test Categories

- `test_backtest.py` - Backtesting workflow validation
- `test_configuration.py` - Configuration management tests
- `test_dashboard_ui.py` - Main dashboard UI tests
- `test_live_monitor.py` - Live monitoring features
- `test_manual_trading.py` - Manual trading mode tests
- `test_performance.py` - Performance analytics tests
- `test_trading_controls.py` - Trading control features

## Common Issues

1. **Port conflicts**: Ensure Streamlit is running on expected port (default: 8501)
2. **Timeout errors**: Increase timeout in conftest.py if needed
3. **Browser issues**: Run `playwright install` to update browsers
