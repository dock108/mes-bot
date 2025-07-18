# Testing Guide for MES Bot

## Test Categories

### 1. Unit Tests (CI)

- **Location**: `tests/` (excluding `tests/uat/`)
- **Coverage**: 80.87% (target: 80%)
- **Run Command**: `pytest tests/ -v --cov=app --cov-config=.coveragerc --ignore=tests/uat/`
- **Excluded from Coverage**:
  - `app/bot.py` - Main bot runner (integration)
  - `app/ui.py` - Streamlit UI
  - `app/ib_client.py` - IB Gateway integration

### 2. UAT Tests (Local Only)

- **Location**: `tests/uat/`
- **Requirements**:
  - Running Streamlit server
  - Playwright browsers installed
- **Setup**:

  ```bash
  # Install Playwright browsers
  playwright install chromium

  # Start Streamlit (separate terminal)
  streamlit run app/ui.py

  # Run UAT tests
  pytest tests/uat/ -v
  ```

- **Note**: UAT tests are excluded from CI due to browser/server requirements

### 3. Integration Tests

- Currently no tests are marked with `@pytest.mark.integration`
- Integration testing is done through:
  - Feature pipeline tests (`test_feature_pipeline.py`)
  - Enhanced strategy tests (`test_enhanced_strategy.py`)
  - ML training tests (`test_ml_training.py`)

## Running Tests Locally

### Full Test Suite

```bash
# Unit tests only (what CI runs)
pytest tests/ -v --ignore=tests/uat/

# All tests including UAT (requires Streamlit)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-config=.coveragerc --cov-report=html --ignore=tests/uat/
```

### Specific Test Categories

```bash
# Performance tests
pytest tests/ -v -m performance

# Database tests
pytest tests/ -v -m db

# ML tests
pytest tests/ -v -m ml
```

## Known Issues

1. UAT tests require Streamlit server to be running manually
2. Some feature pipeline tests have data fixture timing issues
3. Performance test thresholds are adjusted for realistic expectations

## Coverage Report

After running tests with coverage:

- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml`
- Terminal report shows missing lines

## CI/CD Configuration

- GitHub Actions runs unit tests only
- Coverage requirement: 80% (excluding integration files)
- UAT tests must be run locally before releases
