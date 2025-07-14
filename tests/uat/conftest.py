"""
Pytest configuration and fixtures for UAT testing with Playwright
"""

import asyncio
import subprocess
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from app.config import config
from app.models import DailySummary, Trade, create_database, get_session_maker
from tests.uat.helpers.test_data import TestDataGenerator


@pytest.fixture(scope="session")
def uat_database() -> str:
    """Create a test database with sample data for UAT tests"""
    # Create temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    db_url = f"sqlite:///{db_path}"

    # Create database and tables
    create_database(db_url)

    # Populate with test data
    test_data = TestDataGenerator()
    test_data.populate_database(db_url)

    yield db_url

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def streamlit_server(uat_database) -> Generator[str, None, None]:
    """Start Streamlit server for UAT testing"""
    import os

    # Set test database URL
    os.environ["DATABASE_URL"] = uat_database
    os.environ["TRADE_MODE"] = "paper"

    # Start Streamlit server in background
    port = 8502  # Different port to avoid conflicts
    cmd = [
        "streamlit",
        "run",
        "app/ui.py",
        "--server.port",
        str(port),
        "--server.address",
        "localhost",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=Path(__file__).parent.parent.parent
    )

    # Wait for server to start
    server_url = f"http://localhost:{port}"
    max_wait = 30  # seconds
    for _ in range(max_wait):
        try:
            import requests

            response = requests.get(server_url, timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        # Kill process if it didn't start
        process.terminate()
        raise RuntimeError("Streamlit server failed to start")

    yield server_url

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


@pytest.fixture
def page_with_server(page, streamlit_server):
    """Page fixture with server setup"""
    # Navigate to the dashboard
    page.goto(streamlit_server)

    # Wait for initial load
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=30000)

    return page


@pytest.fixture
def dashboard_with_data(page_with_server, uat_database):
    """Dashboard page with test data loaded"""
    page = page_with_server

    # Wait for Streamlit app to fully load
    try:
        # Wait for sidebar to appear (most reliable indicator)
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=30000)
        # Then wait a bit more for content to populate
        page.wait_for_timeout(3000)
    except Exception as e:
        # If sidebar doesn't appear, try waiting for main content
        try:
            page.wait_for_selector('div[data-stale="false"]', timeout=20000)
        except:
            # Final fallback - just wait for any Streamlit app container
            page.wait_for_selector(".main", timeout=15000)

    return page


# Playwright configuration
def pytest_configure(config):
    """Configure Playwright settings"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for testing"""
    return {
        **browser_context_args,
        "viewport": {"width": 1920, "height": 1080},
        "ignore_https_errors": True,
    }


@pytest.fixture
def screenshot_on_failure(request, page):
    """Take screenshot on test failure"""
    yield

    if request.node.rep_call.failed:
        # Create screenshots directory
        screenshot_dir = Path("test-results/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Take screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = request.node.name
        screenshot_path = screenshot_dir / f"{test_name}_{timestamp}.png"

        page.screenshot(path=str(screenshot_path))
        print(f"Screenshot saved: {screenshot_path}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for screenshot on failure"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)
