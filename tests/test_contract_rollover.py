"""
Tests for automatic contract rollover functionality
"""

import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from ib_insync import Contract, Future

from app.config import config
from app.ib_client import IBClient


@pytest.mark.integration
@pytest.mark.db
class TestContractRollover:
    """Test automatic contract rollover logic"""

    @pytest.fixture
    def mock_ib(self):
        """Mock IB connection"""
        ib = Mock()
        ib.qualifyContractsAsync = AsyncMock()
        ib.reqContractDetailsAsync = AsyncMock()
        return ib

    @pytest.fixture
    def ib_client(self, mock_ib):
        """Create IBClient with mocked IB connection"""
        client = IBClient()
        client.ib = mock_ib
        client.connected = True
        return client

    @pytest.mark.asyncio
    async def test_get_front_month_contract_with_config(self, ib_client):
        """Test front month detection when specific month is configured"""
        # Mock configured contract month
        with patch.object(config.ib, "mes_contract_month", "202501"):
            # Mock contract qualification
            mock_contract = Future("MES", "202501", "GLOBEX")
            ib_client.ib.qualifyContractsAsync.return_value = [mock_contract]

            result = await ib_client.get_front_month_contract()

            assert result == mock_contract
            ib_client.ib.qualifyContractsAsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_front_month_contract_auto_detection(self, ib_client):
        """Test automatic front month detection via IB API"""
        # Mock no configured month (use auto-detection)
        with patch.object(config.ib, "mes_contract_month", None):
            with patch.object(config.ib, "contract_rollover_days", 3):
                # Mock contract details response
                mock_contract_details = []

                # Create contracts for current and next month
                current_date = datetime.now().date()
                next_month = current_date.replace(day=1) + timedelta(days=32)
                next_month = next_month.replace(day=15)  # 15th of next month

                for i, expiry_date in enumerate(
                    [
                        current_date + timedelta(days=1),  # Expires tomorrow (too soon)
                        next_month,  # Next month (good)
                        next_month + timedelta(days=30),  # Month after (future)
                    ]
                ):
                    contract_detail = Mock()
                    contract_detail.contract = Future("MES", expiry_date.strftime("%Y%m"), "GLOBEX")
                    contract_detail.contract.lastTradeDateOrContractMonth = expiry_date.strftime(
                        "%Y%m%d"
                    )
                    mock_contract_details.append(contract_detail)

                ib_client.ib.reqContractDetailsAsync.return_value = mock_contract_details

                result = await ib_client.get_front_month_contract()

                # Should select the next month contract (index 1) since current month expires too soon
                assert result.lastTradeDateOrContractMonth == next_month.strftime("%Y%m%d")

    @pytest.mark.asyncio
    async def test_get_front_month_contract_fallback_estimation(self, ib_client):
        """Test fallback to estimation when IB API fails"""
        with patch.object(config.ib, "mes_contract_month", None):
            # Mock API failure
            ib_client.ib.reqContractDetailsAsync.return_value = []

            # Mock successful qualification of estimated contract
            current_month = datetime.now().strftime("%Y%m")
            mock_contract = Future("MES", current_month, "GLOBEX")
            ib_client.ib.qualifyContractsAsync.return_value = [mock_contract]

            result = await ib_client.get_front_month_contract()

            assert result == mock_contract
            # Should have tried to get contract details first, then fallen back
            ib_client.ib.reqContractDetailsAsync.assert_called_once()
            ib_client.ib.qualifyContractsAsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_estimated_contract_next_month_rollover(self, ib_client):
        """Test estimation logic when we're near month end"""
        with patch.object(config.ib, "contract_rollover_days", 5):
            # Mock being near end of month (within rollover threshold)
            near_month_end = datetime(2024, 12, 28)  # Dec 28th

            with patch("app.ib_client.datetime") as mock_datetime:
                mock_datetime.now.return_value = near_month_end
                mock_datetime.strptime = datetime.strptime  # Keep real strptime

                # Mock successful qualification
                expected_contract_month = "202501"  # Should roll to January
                mock_contract = Future("MES", expected_contract_month, "GLOBEX")
                ib_client.ib.qualifyContractsAsync.return_value = [mock_contract]

                result = await ib_client._get_estimated_front_month_contract()

                # Should qualify January contract since we're near December end
                call_args = ib_client.ib.qualifyContractsAsync.call_args[0][0]
                assert call_args.lastTradeDateOrContractMonth == expected_contract_month

    @pytest.mark.asyncio
    async def test_get_mes_contract_with_expiry(self, ib_client):
        """Test getting specific contract by expiry"""
        expiry = "202501"
        mock_contract = Future("MES", expiry, "GLOBEX")
        ib_client.ib.qualifyContractsAsync.return_value = [mock_contract]

        result = await ib_client.get_mes_contract(expiry)

        assert result == mock_contract
        ib_client.ib.qualifyContractsAsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_mes_contract_without_expiry_uses_front_month(self, ib_client):
        """Test that get_mes_contract without expiry uses front month logic"""
        with patch.object(ib_client, "get_front_month_contract") as mock_front_month:
            mock_contract = Future("MES", "202501", "GLOBEX")
            mock_front_month.return_value = mock_contract

            result = await ib_client.get_mes_contract()

            assert result == mock_contract
            mock_front_month.assert_called_once()

    @pytest.mark.asyncio
    async def test_contract_qualification_failure(self, ib_client):
        """Test error handling when contract qualification fails"""
        expiry = "999999"  # Invalid expiry
        ib_client.ib.qualifyContractsAsync.return_value = []

        with pytest.raises(ValueError, match="Could not qualify MES contract"):
            await ib_client.get_mes_contract(expiry)

    @pytest.mark.asyncio
    async def test_api_error_handling(self, ib_client):
        """Test error handling when IB API raises exceptions"""
        with patch.object(config.ib, "mes_contract_month", None):
            # Mock API exception
            ib_client.ib.reqContractDetailsAsync.side_effect = Exception("IB API Error")

            # Mock successful fallback estimation
            mock_contract = Future("MES", "202412", "GLOBEX")
            ib_client.ib.qualifyContractsAsync.return_value = [mock_contract]

            result = await ib_client.get_front_month_contract()

            # Should fall back to estimation and succeed
            assert result == mock_contract

    def test_contract_expiry_date_parsing(self, ib_client):
        """Test parsing of contract expiry dates"""
        # Test valid date formats
        test_cases = [
            ("20241215", date(2024, 12, 15)),
            ("20250117", date(2025, 1, 17)),
        ]

        for date_str, expected_date in test_cases:
            parsed_date = datetime.strptime(date_str, "%Y%m%d").date()
            assert parsed_date == expected_date

    @pytest.mark.asyncio
    async def test_rollover_threshold_logic(self, ib_client):
        """Test that rollover threshold is properly applied"""
        with patch.object(config.ib, "mes_contract_month", None):
            with patch.object(config.ib, "contract_rollover_days", 7):
                # Mock current date and contract details
                current_date = date(2024, 12, 10)
                rollover_threshold = current_date + timedelta(days=7)  # Dec 17

                mock_contract_details = []

                # Contract expiring Dec 15 (before threshold) - should be skipped
                early_contract = Mock()
                early_contract.contract = Future("MES", "202412", "GLOBEX")
                early_contract.contract.lastTradeDateOrContractMonth = "20241215"
                mock_contract_details.append(early_contract)

                # Contract expiring Dec 20 (after threshold) - should be selected
                good_contract = Mock()
                good_contract.contract = Future("MES", "202501", "GLOBEX")
                good_contract.contract.lastTradeDateOrContractMonth = "20241220"
                mock_contract_details.append(good_contract)

                ib_client.ib.reqContractDetailsAsync.return_value = mock_contract_details

                with patch("app.ib_client.datetime") as mock_datetime:
                    # Mock datetime.now() to return a datetime object with the date we want
                    mock_now = Mock()
                    mock_now.date.return_value = current_date
                    mock_datetime.now.return_value = mock_now

                    # Keep the real datetime.strptime method
                    mock_datetime.strptime = datetime.strptime

                    result = await ib_client.get_front_month_contract()

                    # Should select the contract expiring after rollover threshold
                    assert result.lastTradeDateOrContractMonth == "20241220"
