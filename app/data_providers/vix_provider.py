"""
VIX Data Provider using FRED API
Fetches CBOE Volatility Index (VIX) data from Federal Reserve Economic Data
"""

import logging
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class VIXProvider:
    """Fetches VIX data from FRED API"""

    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY not found in environment variables")

        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.series_id = "VIXCLS"  # CBOE Volatility Index
        logger.info("VIX Provider initialized with FRED API")

    @lru_cache(maxsize=32)
    def get_vix_value(self, target_date: date) -> float:
        """Get VIX value for a specific date"""
        try:
            params = {
                "series_id": self.series_id,
                "api_key": self.api_key,
                "observation_start": target_date.isoformat(),
                "observation_end": target_date.isoformat(),
                "file_type": "json",
            }

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            if observations and observations[0]["value"] != ".":
                vix_value = float(observations[0]["value"])
                logger.debug(f"VIX value for {target_date}: {vix_value}")
                return vix_value

            # If no data for exact date, get most recent
            logger.warning(f"No VIX data for {target_date}, fetching most recent")
            return self.get_latest_vix()

        except Exception as e:
            logger.error(f"Error fetching VIX data for {target_date}: {e}")
            raise ValueError(f"Failed to fetch VIX data: {str(e)}")

    def get_latest_vix(self) -> float:
        """Get most recent VIX value"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=7)

            params = {
                "series_id": self.series_id,
                "api_key": self.api_key,
                "observation_start": start_date.isoformat(),
                "observation_end": end_date.isoformat(),
                "sort_order": "desc",
                "file_type": "json",
            }

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            for obs in observations:
                if obs["value"] != ".":
                    vix_value = float(obs["value"])
                    logger.info(f"Latest VIX value: {vix_value} (from {obs['date']})")
                    return vix_value

            raise ValueError("No recent VIX data available")

        except Exception as e:
            logger.error(f"Error fetching latest VIX: {e}")
            raise

    def get_vix_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get VIX values for a date range"""
        try:
            params = {
                "series_id": self.series_id,
                "api_key": self.api_key,
                "observation_start": start_date.isoformat(),
                "observation_end": end_date.isoformat(),
                "file_type": "json",
            }

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            # Convert to DataFrame
            records = []
            for obs in observations:
                if obs["value"] != ".":
                    records.append(
                        {"date": pd.to_datetime(obs["date"]), "vix": float(obs["value"])}
                    )

            if not records:
                raise ValueError(f"No VIX data available between {start_date} and {end_date}")

            df = pd.DataFrame(records)
            df.set_index("date", inplace=True)
            logger.info(f"Retrieved {len(df)} VIX observations from {start_date} to {end_date}")

            return df

        except Exception as e:
            logger.error(f"Error fetching VIX range data: {e}")
            raise ValueError(f"Failed to fetch VIX range data: {str(e)}")

    def clear_cache(self):
        """Clear the LRU cache"""
        self.get_vix_value.cache_clear()
        logger.info("VIX cache cleared")
