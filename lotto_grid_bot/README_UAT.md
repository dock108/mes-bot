# UAT Testing Guide for MES 0DTE Lotto-Grid Options Bot

## Overview

This document describes the User Acceptance Testing (UAT) framework implemented for the MES 0DTE "Lotto-Grid" Options Bot. The UAT tests are built using Playwright and provide comprehensive end-to-end testing of the Streamlit dashboard interface.

## Test Structure

### Test Suites

1. **Dashboard UI Tests** (`test_dashboard_ui.py`)
   - Dashboard loading and component visibility
   - Navigation between sections
   - Metrics display and formatting
   - Emergency stop functionality
   - Responsive layout testing
   - Error handling verification

2. **Live Monitoring Tests** (`test_live_monitor.py`)
   - Real-time position display
   - Market status indicators
   - P&L calculations accuracy
   - Emergency controls
   - Data refresh cycles

3. **Performance Analytics Tests** (`test_performance.py`)
   - Performance metrics display
   - Chart visualizations
   - Trades history table
   - Period filtering
   - Data consistency checks
   - Export functionality

4. **Backtesting Tests** (`test_backtest.py`)
   - Parameter input interface
   - Backtest execution workflow
   - Results display and formatting
   - Chart visualizations
   - Parameter validation
   - History management

5. **Configuration Tests** (`test_configuration.py`)
   - Settings display and modification
   - Save/reset functionality
   - Value validation
   - Import/export capabilities
   - Real-time updates

### Page Object Model

The tests use a Page Object Model pattern with dedicated classes:

- `BasePage`: Common functionality for all pages
- `DashboardPage`: Main dashboard interactions
- `LiveMonitorPage`: Live monitoring specific actions
- `PerformancePage`: Performance analytics interactions
- `BacktestPage`: Backtesting workflow actions
- `ConfigurationPage`: Configuration management

## Running UAT Tests

### Quick Start

```bash
# Run all UAT tests
./scripts/run_uat_tests.sh

# Run smoke tests only (quick validation)
./scripts/run_uat_tests.sh smoke

# Run specific test suite
./scripts/run_uat_tests.sh ui
./scripts/run_uat_tests.sh monitor
./scripts/run_uat_tests.sh performance
```

### Manual Execution

```bash
# Install dependencies
poetry install
playwright install chromium

# Run specific test file
pytest tests/uat/test_dashboard_ui.py -v

# Run with different browsers
pytest tests/uat/ --browser firefox --headed

# Run with screenshots and videos
pytest tests/uat/ --screenshot only-on-failure --video retain-on-failure
```

### Environment Variables

- `DATABASE_URL`: Test database connection string
- `TRADE_MODE`: Set to "paper" for testing
- `HEADLESS`: true/false for browser display
- `SLOW_MO`: Milliseconds to slow down actions

## Test Data

The UAT tests use the `TestDataGenerator` class to create realistic test data:

- **Trades**: 30 days of historical trades with ~25% win rate
- **Daily Summaries**: Aggregate daily statistics
- **Backtest Results**: Sample backtest scenarios
- **Market Data**: Realistic MES price movements

## CI/CD Integration

### GitHub Actions Workflow

The UAT tests are integrated into CI/CD pipeline:

- **On Push/PR**: Smoke tests across Python 3.10/3.11 and multiple browsers
- **Scheduled**: Daily full test suite execution
- **Manual**: Workflow dispatch for on-demand testing
- **Performance**: PR-specific performance validation

### Test Artifacts

- Screenshots saved on test failures
- Videos recorded for failed test runs
- Test results uploaded as GitHub artifacts
- Coverage reports integration

## Test Markers

Use pytest markers to categorize tests:

```bash
# Run only smoke tests
pytest tests/uat/ -m "smoke"

# Skip slow tests
pytest tests/uat/ -m "not slow"

# Run UI tests only
pytest tests/uat/ -m "ui"
```

## Screenshots and Debugging

Screenshots are automatically taken:
- On test failures
- At key verification points
- During error conditions

Screenshots are saved to `test-results/screenshots/` with descriptive names.

## Best Practices

### Writing UAT Tests

1. **Use descriptive test names** that explain the user scenario
2. **Include realistic wait times** for dynamic content loading
3. **Take screenshots at key points** for debugging
4. **Test error conditions** not just happy paths
5. **Verify data consistency** across page refreshes

### Debugging Failed Tests

1. **Check screenshots** in test-results/screenshots/
2. **Run tests with --headed** to see browser actions
3. **Use --slowmo** to slow down execution
4. **Enable debug logging** with appropriate log levels

### Maintenance

1. **Update selectors** if UI changes
2. **Adjust wait times** based on performance
3. **Review test data** for realistic scenarios
4. **Monitor CI failures** and address flaky tests

## Configuration

### pytest-uat.ini

UAT-specific pytest configuration:
- Browser settings (headless, slowmo)
- Screenshot and video capture
- Timeout configurations
- Parallel execution settings

### Test Database

Each test run uses an isolated SQLite database with:
- Fresh test data generation
- Cleanup after test completion
- No interference with production data

## Troubleshooting

### Common Issues

1. **Streamlit server startup failures**
   - Check port availability
   - Verify environment variables
   - Review server logs

2. **Element not found errors**
   - Increase wait timeouts
   - Verify selector accuracy
   - Check page loading states

3. **Database connection issues**
   - Verify DATABASE_URL setting
   - Check file permissions
   - Ensure test database cleanup

### Browser Compatibility

Tests are validated across:
- Chromium (primary)
- Firefox
- Different viewport sizes
- Various operating systems (via CI)

## Reporting

### Test Results

UAT test results include:
- Pass/fail status for each test suite
- Execution time metrics
- Screenshot evidence
- Error details and stack traces

### Metrics Tracked

- Test execution time
- Browser compatibility
- Screenshot count
- Failure patterns
- Coverage across UI components

## Future Enhancements

Planned improvements for UAT testing:

1. **Mobile Testing**: Add mobile browser testing
2. **Accessibility**: Include accessibility validation
3. **Performance**: Load testing scenarios
4. **Integration**: API endpoint validation
5. **Visual Regression**: Screenshot comparison testing

## Support

For UAT testing issues:

1. Check this documentation
2. Review test artifacts in CI
3. Examine screenshot evidence
4. Run tests locally with debugging enabled
5. Update test data or selectors as needed