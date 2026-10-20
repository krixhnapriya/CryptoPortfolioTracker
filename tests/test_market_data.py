"""Unit tests for CoinGeckoMarketService using mocks to avoid live network dependencies."""

from unittest.mock import patch, MagicMock
import requests
from src.market_data import CoinGeckoMarketService


class TestCoinGeckoMarketService:
    """Test suite for CoinGecko API client."""

    def test_empty_coins_list(self):
        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_prices([])
        assert res.is_success is True
        assert res.prices == {}
        assert res.is_cached is False
        assert res.error_message is None

    @patch("src.market_data.requests.get")
    def test_successful_price_fetch(self, mock_get):
        # Mock CoinGecko /simple/price response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bitcoin": {"inr": 5800000.0},
            "ethereum": {"inr": 290000.0},
        }
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_prices(["bitcoin", "ethereum"])

        assert res.is_success is True
        assert res.is_cached is False
        assert res.prices == {"bitcoin": 5800000.0, "ethereum": 290000.0}
        assert res.error_message is None
        mock_get.assert_called_once()

    @patch("src.market_data.requests.get")
    def test_cache_hit_prevents_network_call(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bitcoin": {"inr": 5800000.0}}
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        # First call hits network
        res1 = service.get_prices(["bitcoin"])
        assert res1.is_cached is False
        assert mock_get.call_count == 1

        # Second immediate call should hit cache and NOT invoke requests.get again
        res2 = service.get_prices(["bitcoin"])
        assert res2.is_cached is True
        assert res2.prices == {"bitcoin": 5800000.0}
        assert mock_get.call_count == 1  # Still 1!

    @patch("src.market_data.requests.get")
    def test_rate_limit_429_without_cache(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_prices(["bitcoin"])

        assert res.is_success is False
        assert res.prices == {}
        assert "rate limit reached (HTTP 429)" in res.error_message

    @patch("src.market_data.requests.get")
    def test_rate_limit_429_with_existing_cache_fallback(self, mock_get):
        # First call succeeds
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"bitcoin": {"inr": 5800000.0}}

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Too Many Requests"

        mock_get.side_effect = [mock_ok, mock_429]

        service = CoinGeckoMarketService(cache_ttl=0.001)  # tiny TTL so it attempts refresh
        res1 = service.get_prices(["bitcoin"])
        assert res1.is_success is True

        import time
        time.sleep(0.01)  # cache expired, force refresh attempt

        res2 = service.get_prices(["bitcoin"])
        # Should gracefully return previously cached price with notice
        assert res2.is_success is False
        assert res2.is_cached is True
        assert res2.prices == {"bitcoin": 5800000.0}
        assert "rate limit reached" in res2.error_message
        assert "cached" in res2.error_message

    @patch("src.market_data.requests.get")
    def test_network_timeout_handling(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_prices(["bitcoin"])

        assert res.is_success is False
        assert res.prices == {}
        assert "timed out" in res.error_message

    @patch("src.market_data.requests.get")
    def test_connection_error_handling(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("DNS failure")

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_prices(["bitcoin"])

        assert res.is_success is False
        assert res.prices == {}
        assert "internet connection" in res.error_message.lower()

    @patch("src.market_data.requests.get")
    def test_missing_coin_in_api_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # API returns bitcoin, but solana is missing
        mock_response.json.return_value = {"bitcoin": {"inr": 5800000.0}}
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_prices(["bitcoin", "solana"])

        assert res.is_success is True
        assert "bitcoin" in res.prices
        assert "solana" not in res.prices

    @patch("src.market_data.requests.get")
    def test_historical_chart_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulated CoinGecko response: 3 days of prices
        mock_response.json.return_value = {
            "prices": [
                [1767225600000, 7000000.0],  # 2026-01-01
                [1767312000000, 7100000.0],  # 2026-01-02
                [1767398400000, 7200000.0],  # 2026-01-03
            ]
        }
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_historical_market_chart("bitcoin", days=3)

        assert res.is_success is True
        assert len(res.records) == 3
        assert res.records[0]["date"] == "2026-01-01"
        assert res.records[0]["price_inr"] == 7000000.0
        assert res.records[2]["date"] == "2026-01-03"
        assert res.records[2]["price_inr"] == 7200000.0

    @patch("src.market_data.requests.get")
    def test_historical_chart_multiple_points_same_day_normalization(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Two observations on the same UTC day (hourly data)
        # 1767225600000 is 2026-01-01 00:00:00 UTC
        # 1767268800000 is 2026-01-01 12:00:00 UTC
        mock_response.json.return_value = {
            "prices": [
                [1767225600000, 7000000.0],
                [1767268800000, 7050000.0],
                [1767312000000, 7100000.0],  # 2026-01-02
            ]
        }
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_historical_market_chart("bitcoin", days=2)

        assert res.is_success is True
        # Should be deduplicated into 2 unique daily records
        assert len(res.records) == 2
        assert res.records[0]["date"] == "2026-01-01"
        # Should keep the later price 7050000.0 as the closing price
        assert res.records[0]["price_inr"] == 7050000.0
        assert res.records[1]["date"] == "2026-01-02"

    @patch("src.market_data.requests.get")
    def test_historical_chart_429_rate_limit(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_historical_market_chart("bitcoin", days=30)

        assert res.is_success is False
        assert res.records == []
        assert "rate limit reached (HTTP 429)" in res.error_message

    @patch("src.market_data.requests.get")
    def test_historical_chart_404_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Coin not found"
        mock_get.return_value = mock_response

        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_historical_market_chart("invalid_coin_id", days=30)

        assert res.is_success is False
        assert "not found" in res.error_message.lower()

    def test_historical_chart_empty_coin_id(self):
        service = CoinGeckoMarketService(cache_ttl=60.0)
        res = service.get_historical_market_chart("", days=30)
        assert res.is_success is False
        assert "cannot be empty" in res.error_message

    def test_historical_range_invalid_timestamps(self):
        service = CoinGeckoMarketService(cache_ttl=60.0)
        # from_timestamp >= to_timestamp
        res = service.get_historical_range("bitcoin", from_timestamp=2000, to_timestamp=1000)
        assert res.is_success is False
        assert "must precede" in res.error_message
