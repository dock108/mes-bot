"""Tests for VIX data provider"""

import os
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import requests

from app.data_providers.vix_provider import VIXProvider


class TestVIXProvider:
    """Test cases for VIX data provider"""

    def test_init_without_api_key(self):
        """Test initialization fails without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="FRED_API_KEY not found"):
                VIXProvider()

    def test_init_with_api_key(self):
        """Test successful initialization with API key"""
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            assert provider.api_key == "test_key"
            assert provider.series_id == "VIXCLS"
            assert "fred.stlouisfed.org" in provider.base_url

    @patch("requests.get")
    def test_get_vix_value_success(self, mock_get):
        """Test successful VIX value retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "observations": [{"date": "2025-07-17", "value": "15.25"}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            vix_value = provider.get_vix_value(date(2025, 7, 17))

            assert vix_value == 15.25
            mock_get.assert_called_once()
            assert "VIXCLS" in mock_get.call_args[1]["params"]["series_id"]

    @patch("requests.get")
    def test_get_vix_value_no_data_falls_back_to_latest(self, mock_get):
        """Test fallback to latest VIX when specific date has no data"""
        # First call returns no data
        mock_response_1 = Mock()
        mock_response_1.json.return_value = {"observations": []}
        mock_response_1.raise_for_status = Mock()

        # Second call (get_latest_vix) returns data
        mock_response_2 = Mock()
        mock_response_2.json.return_value = {
            "observations": [{"date": "2025-07-16", "value": "16.50"}]
        }
        mock_response_2.raise_for_status = Mock()

        mock_get.side_effect = [mock_response_1, mock_response_2]

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            vix_value = provider.get_vix_value(date(2025, 7, 17))

            assert vix_value == 16.50
            assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_vix_value_api_error(self, mock_get):
        """Test error handling for API failures"""
        mock_get.side_effect = requests.RequestException("API error")

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            with pytest.raises(ValueError, match="Failed to fetch VIX data"):
                provider.get_vix_value(date(2025, 7, 17))

    @patch("requests.get")
    def test_get_latest_vix_success(self, mock_get):
        """Test successful latest VIX retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2025-07-17", "value": "17.25"},
                {"date": "2025-07-16", "value": "16.50"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            vix_value = provider.get_latest_vix()

            assert vix_value == 17.25  # Should return first (most recent) value

    @patch("requests.get")
    def test_get_latest_vix_no_valid_data(self, mock_get):
        """Test error when no recent VIX data available"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2025-07-17", "value": "."},  # Invalid value
                {"date": "2025-07-16", "value": "."},  # Invalid value
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            with pytest.raises(ValueError, match="No recent VIX data available"):
                provider.get_latest_vix()

    @patch("requests.get")
    def test_get_vix_range_success(self, mock_get):
        """Test successful VIX range retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2025-07-15", "value": "15.00"},
                {"date": "2025-07-16", "value": "16.00"},
                {"date": "2025-07-17", "value": "17.00"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            df = provider.get_vix_range(date(2025, 7, 15), date(2025, 7, 17))

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert "vix" in df.columns
            assert df["vix"].tolist() == [15.00, 16.00, 17.00]
            assert df.index.name == "date"

    @patch("requests.get")
    def test_get_vix_range_no_data(self, mock_get):
        """Test error when no data in range"""
        mock_response = Mock()
        mock_response.json.return_value = {"observations": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            with pytest.raises(ValueError, match="No VIX data available between"):
                provider.get_vix_range(date(2025, 7, 15), date(2025, 7, 17))

    @patch("requests.get")
    def test_get_vix_range_api_error(self, mock_get):
        """Test error handling for range API failures"""
        mock_get.side_effect = requests.RequestException("API error")

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            with pytest.raises(ValueError, match="Failed to fetch VIX range data"):
                provider.get_vix_range(date(2025, 7, 15), date(2025, 7, 17))

    def test_clear_cache(self):
        """Test cache clearing functionality"""
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            
            # Mock the cache_clear method
            provider.get_vix_value.cache_clear = Mock()
            
            provider.clear_cache()
            provider.get_vix_value.cache_clear.assert_called_once()

    @patch("requests.get")
    def test_caching_behavior(self, mock_get):
        """Test that VIX values are cached properly"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "observations": [{"date": "2025-07-17", "value": "15.25"}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            
            # First call
            vix1 = provider.get_vix_value(date(2025, 7, 17))
            # Second call - should use cache
            vix2 = provider.get_vix_value(date(2025, 7, 17))
            
            assert vix1 == vix2 == 15.25
            # Should only call API once due to caching
            mock_get.assert_called_once()

    @patch("requests.get")
    def test_skip_invalid_observations(self, mock_get):
        """Test that invalid observations are skipped"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2025-07-15", "value": "."},  # Invalid
                {"date": "2025-07-16", "value": "16.00"},
                {"date": "2025-07-17", "value": "."},  # Invalid
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            provider = VIXProvider()
            df = provider.get_vix_range(date(2025, 7, 15), date(2025, 7, 17))

            assert len(df) == 1  # Only one valid observation
            assert df["vix"].iloc[0] == 16.00