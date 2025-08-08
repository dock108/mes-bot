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
    """Fetches VIX data from FRED API with fallback when API key unavailable"""

    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")
        self.fallback_mode = False

        if not self.api_key:
            logger.warning("FRED_API_KEY not found - using fallback VIX values for backtesting")
            self.fallback_mode = True
        else:
            logger.info("VIX Provider initialized with FRED API")

        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.series_id = "VIXCLS"  # CBOE Volatility Index

    @lru_cache(maxsize=32)
    def get_vix_value(self, target_date: date) -> float:
        """Get VIX value for a specific date"""
        if self.fallback_mode:
            return self._get_fallback_vix_value(target_date)

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
            # Fall back to synthetic VIX if API fails
            logger.warning("Falling back to synthetic VIX value")
            return self._get_fallback_vix_value(target_date)

    def get_latest_vix(self) -> float:
        """Get most recent VIX value"""
        if self.fallback_mode:
            return self._get_fallback_vix_value(date.today())

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
            logger.warning("Falling back to synthetic VIX value")
            return self._get_fallback_vix_value(date.today())

    def get_vix_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get VIX values for a date range"""
        if self.fallback_mode:
            return self._get_fallback_vix_range(start_date, end_date)

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
            logger.warning("Falling back to synthetic VIX range data")
            return self._get_fallback_vix_range(start_date, end_date)

    def _get_fallback_vix_value(self, target_date: date) -> float:
        """
        Generate synthetic VIX value for backtesting when FRED API unavailable.
        Returns reasonable VIX estimates based on historical patterns.
        """
        import math

        # Base VIX level (historical average around 19-20)
        base_vix = 19.5

        # Add seasonal variation (higher volatility in fall/winter)
        month = target_date.month
        seasonal_adjustment = 0.0
        if month in [9, 10, 11]:  # Fall months typically higher volatility
            seasonal_adjustment = 2.0
        elif month in [12, 1, 2]:  # Winter months
            seasonal_adjustment = 1.5
        elif month in [6, 7, 8]:  # Summer months typically lower
            seasonal_adjustment = -1.0

        # Add some pseudo-random variation based on date
        # Use date as seed for consistent but varied values
        date_hash = hash(target_date.isoformat()) % 1000
        noise = (date_hash / 1000.0 - 0.5) * 6.0  # ±3 point variation

        # Add day-of-week effect (Mondays typically higher)
        weekday_adjustment = 0.0
        if target_date.weekday() == 0:  # Monday
            weekday_adjustment = 1.0
        elif target_date.weekday() == 4:  # Friday
            weekday_adjustment = -0.5

        synthetic_vix = base_vix + seasonal_adjustment + noise + weekday_adjustment

        # Clamp to reasonable bounds (VIX rarely below 10 or above 80)
        synthetic_vix = max(10.0, min(80.0, synthetic_vix))

        logger.debug(f"Generated fallback VIX value for {target_date}: {synthetic_vix:.2f}")
        return synthetic_vix

    def _get_fallback_vix_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Generate synthetic VIX range data for backtesting when FRED API unavailable.
        Creates daily VIX values for the specified date range.
        """
        from datetime import timedelta

        records = []
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends (VIX is only available on trading days)
            if current_date.weekday() < 5:  # Monday=0, Friday=4
                vix_value = self._get_fallback_vix_value(current_date)
                records.append({
                    "date": pd.to_datetime(current_date),
                    "vix": vix_value
                })
            current_date += timedelta(days=1)

        if not records:
            raise ValueError(f"No trading days found between {start_date} and {end_date}")

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        logger.info(f"Generated {len(df)} synthetic VIX observations from {start_date} to {end_date}")

        return df

    def clear_cache(self):
        """Clear the LRU cache"""
        self.get_vix_value.cache_clear()
        logger.info("VIX cache cleared")
