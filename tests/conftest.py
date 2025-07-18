"""
Pytest configuration and fixtures for MES-Bot tests
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="function")
def test_db_url():
    """Provide a temporary database URL for tests"""
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{db_path}"
    
    yield db_url
    
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def test_data_dir():
    """Provide a temporary data directory for tests"""
    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="session", autouse=True)
def ensure_data_directory():
    """Ensure the data directory exists for tests that need it"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Note: We don't remove the data directory after tests
    # as it might contain important files


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to easily set environment variables for tests"""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env


# Configure asyncio for tests
pytest_plugins = ('pytest_asyncio',)