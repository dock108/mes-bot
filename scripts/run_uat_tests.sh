#!/bin/bash

# UAT Test Runner Script for MES 0DTE Lotto-Grid Options Bot
# This script sets up the environment and runs UAT tests with Playwright

set -e  # Exit on error

echo "🚀 Starting UAT Tests for MES 0DTE Lotto-Grid Options Bot"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from project root."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Create test results directory
echo "📁 Creating test results directory..."
mkdir -p test-results/screenshots

# Set environment variables for testing
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRADE_MODE="paper"
export LOG_LEVEL="INFO"

# Function to run specific test suites
run_test_suite() {
    local test_file=$1
    local description=$2

    echo ""
    echo "🧪 Running $description..."
    echo "----------------------------------------"

    if pytest tests/uat/$test_file -v --tb=short; then
        echo "✅ $description passed"
    else
        echo "❌ $description failed"
        return 1
    fi
}

# Function to run all UAT tests
run_all_tests() {
    echo ""
    echo "🧪 Running All UAT Tests..."
    echo "----------------------------------------"

    if pytest tests/uat/ -v --tb=short --maxfail=5; then
        echo "✅ All UAT tests passed"
    else
        echo "❌ Some UAT tests failed"
        return 1
    fi
}

# Function to run smoke tests only
run_smoke_tests() {
    echo ""
    echo "🧪 Running Smoke Tests..."
    echo "----------------------------------------"

    # Run basic navigation and UI tests
    if pytest tests/uat/test_dashboard_ui.py::TestDashboardUI::test_dashboard_loads_successfully \
             tests/uat/test_dashboard_ui.py::TestDashboardUI::test_navigation_between_sections \
             tests/uat/test_live_monitor.py::TestLiveMonitoring::test_live_monitor_navigation \
             -v --tb=short; then
        echo "✅ Smoke tests passed"
    else
        echo "❌ Smoke tests failed"
        return 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    "smoke")
        run_smoke_tests
        ;;
    "ui")
        run_test_suite "test_dashboard_ui.py" "Dashboard UI Tests"
        ;;
    "monitor")
        run_test_suite "test_live_monitor.py" "Live Monitoring Tests"
        ;;
    "performance")
        run_test_suite "test_performance.py" "Performance Analytics Tests"
        ;;
    "backtest")
        run_test_suite "test_backtest.py" "Backtesting Tests"
        ;;
    "config")
        run_test_suite "test_configuration.py" "Configuration Tests"
        ;;
    "all")
        run_all_tests
        ;;
    "help")
        echo "Usage: $0 [test_suite]"
        echo ""
        echo "Available test suites:"
        echo "  smoke       - Run basic smoke tests only"
        echo "  ui          - Run dashboard UI tests"
        echo "  monitor     - Run live monitoring tests"
        echo "  performance - Run performance analytics tests"
        echo "  backtest    - Run backtesting tests"
        echo "  config      - Run configuration tests"
        echo "  all         - Run all UAT tests (default)"
        echo "  help        - Show this help message"
        exit 0
        ;;
    *)
        echo "❌ Unknown test suite: $1"
        echo "Run '$0 help' for available options"
        exit 1
        ;;
esac

# Report test results
echo ""
echo "📊 Test Results Summary"
echo "======================"

if [ -d "test-results" ]; then
    screenshot_count=$(find test-results/screenshots -name "*.png" 2>/dev/null | wc -l)
    echo "📸 Screenshots taken: $screenshot_count"

    if [ $screenshot_count -gt 0 ]; then
        echo "🔍 Screenshots location: test-results/screenshots/"
    fi
fi

echo ""
echo "✨ UAT testing completed!"
echo ""
echo "📝 Tips:"
echo "   - Screenshots are saved to test-results/screenshots/"
echo "   - Run '$0 smoke' for quick validation"
echo "   - Run '$0 help' for all available options"
echo "   - Check test-results/ for detailed test artifacts"
