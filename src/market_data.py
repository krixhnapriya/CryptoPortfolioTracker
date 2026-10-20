"""Market data service for fetching cryptocurrency prices from CoinGecko API."""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Optional
import requests

from src.config import (
    COINGECKO_BASE_URL,
    COINGECKO_API_KEY,
    DEFAULT_CURRENCY,
    PRICE_CACHE_TTL_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)


@dataclass
class MarketPriceResponse:
    """Structured response container for market price queries."""

    prices: dict[str, float] = field(default_factory=dict)
    is_cached: bool = False
    cache_timestamp: Optional[float] = None
    error_message: Optional[str] = None
    is_success: bool = True


@dataclass
class HistoricalDataResponse:
    """Structured response container for historical market price queries."""

    records: list[dict[str, Any]] = field(default_factory=list)
    is_cached: bool = False
    error_message: Optional[str] = None
    is_success: bool = True


class CoinGeckoMarketService:
    """
    CoinGecko API service client with caching, rate limit resilience, and error handling.
    """

    def __init__(
        self,
        base_url: str = COINGECKO_BASE_URL,
        api_key: str = COINGECKO_API_KEY,
        currency: str = DEFAULT_CURRENCY,
        cache_ttl: float = PRICE_CACHE_TTL_SECONDS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.currency = currency.lower()
        self.cache_ttl = cache_ttl
        self.timeout = timeout

        # In-memory cache: {coin_id: {"price": float, "timestamp": float}}
        self._price_cache: dict[str, dict[str, float]] = {}
        # In-memory historical cache: {cache_key: {"records": list, "timestamp": float}}
        self._historical_cache: dict[str, dict[str, Any]] = {}

    def _get_headers(self) -> dict[str, str]:
        """Construct request headers including demo API key if configured."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "CryptoPortfolioTracker/1.0",
        }
        if self.api_key:
            # CoinGecko Demo keys use 'x-cg-demo-api-key', Pro keys use 'x-cg-pro-api-key'
            # If the user has a pro key (often starting with CG-PRO or configured as such),
            # provide header. Free tier does not require any header.
            headers["x-cg-demo-api-key"] = self.api_key
        return headers

    def get_prices(self, coin_ids: list[str]) -> MarketPriceResponse:
        """
        Fetch current market prices for a list of CoinGecko coin IDs in the target currency.

        Args:
            coin_ids: List of valid CoinGecko coin IDs (e.g., ['bitcoin', 'ethereum']).

        Returns:
            MarketPriceResponse containing price dictionary, cache status, and any error message.
        """
        if not coin_ids:
            return MarketPriceResponse(prices={}, is_cached=False, is_success=True)

        now = time.time()
        normalized_ids = [cid.strip().lower() for cid in coin_ids if cid.strip()]

        # Check if all requested coins exist in our cache and are fresh
        all_cached_fresh = True
        cached_result: dict[str, float] = {}
        oldest_cache_ts = now

        for cid in normalized_ids:
            if cid in self._price_cache:
                entry = self._price_cache[cid]
                if now - entry["timestamp"] < self.cache_ttl:
                    cached_result[cid] = entry["price"]
                    oldest_cache_ts = min(oldest_cache_ts, entry["timestamp"])
                else:
                    all_cached_fresh = False
            else:
                all_cached_fresh = False

        if all_cached_fresh and len(cached_result) == len(normalized_ids):
            return MarketPriceResponse(
                prices=cached_result,
                is_cached=True,
                cache_timestamp=oldest_cache_ts,
                is_success=True,
            )

        # Make live HTTP request to CoinGecko
        endpoint = f"{self.base_url}/simple/price"
        params = {
            "ids": ",".join(normalized_ids),
            "vs_currencies": self.currency,
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout,
            )

            # Handle Rate Limiting (HTTP 429)
            if response.status_code == 429:
                return self._fallback_or_error(
                    normalized_ids,
                    "CoinGecko API rate limit reached (HTTP 429). Please wait a moment before refreshing.",
                )

            # Handle other HTTP error codes
            if response.status_code != 200:
                return self._fallback_or_error(
                    normalized_ids,
                    f"CoinGecko API returned HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            fresh_prices: dict[str, float] = {}

            for cid in normalized_ids:
                if cid in data and self.currency in data[cid]:
                    price_val = float(data[cid][self.currency])
                    fresh_prices[cid] = price_val
                    # Store in cache
                    self._price_cache[cid] = {
                        "price": price_val,
                        "timestamp": now,
                    }

            return MarketPriceResponse(
                prices=fresh_prices,
                is_cached=False,
                cache_timestamp=now,
                is_success=True,
            )

        except requests.exceptions.Timeout:
            return self._fallback_or_error(
                normalized_ids,
                f"Network request to CoinGecko timed out after {self.timeout}s.",
            )
        except requests.exceptions.ConnectionError:
            return self._fallback_or_error(
                normalized_ids,
                "Unable to connect to CoinGecko API. Please check your internet connection.",
            )
        except Exception as exc:
            return self._fallback_or_error(
                normalized_ids,
                f"Unexpected error communicating with CoinGecko: {exc}",
            )

    def _fallback_or_error(
        self, coin_ids: list[str], error_message: str
    ) -> MarketPriceResponse:
        """
        If live request fails, return cached prices if any exist, with an explicit warning.
        Never fabricate mock or zero prices.
        """
        cached_available: dict[str, float] = {}
        oldest_ts: Optional[float] = None

        for cid in coin_ids:
            if cid in self._price_cache:
                cached_available[cid] = self._price_cache[cid]["price"]
                ts = self._price_cache[cid]["timestamp"]
                oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)

        if cached_available:
            return MarketPriceResponse(
                prices=cached_available,
                is_cached=True,
                cache_timestamp=oldest_ts,
                error_message=f"{error_message} Showing previously cached prices.",
                is_success=False,
            )

        return MarketPriceResponse(
            prices={},
            is_cached=False,
            cache_timestamp=None,
            error_message=error_message,
            is_success=False,
        )

    def _normalize_historical_prices(self, raw_prices: list[list[Any]]) -> list[dict[str, Any]]:
        """
        Normalize raw CoinGecko [[timestamp_ms, price], ...] into daily records.
        Groups by UTC calendar date ('YYYY-MM-DD') and retains the latest observation as the day's close.
        Ensures ascending chronological ordering.
        """
        if not raw_prices:
            return []

        daily_map: dict[str, dict[str, Any]] = {}
        for item in raw_prices:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            ts_ms, price = item[0], item[1]
            if price is None:
                continue
            try:
                price_val = float(price)
                ts_int = int(ts_ms)
            except (ValueError, TypeError):
                continue

            if price_val < 0:
                continue

            # Convert to UTC calendar date
            dt = datetime.datetime.fromtimestamp(ts_int / 1000.0, tz=datetime.timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

            # Retain the latest timestamp observation for this date
            if date_str not in daily_map or ts_int >= daily_map[date_str]["timestamp"]:
                daily_map[date_str] = {
                    "date": date_str,
                    "timestamp": ts_int,
                    "price_inr": price_val,
                }

        # Chronological sort by date
        sorted_dates = sorted(daily_map.keys())
        return [daily_map[d] for d in sorted_dates]

    def get_historical_market_chart(
        self,
        coin_id: str,
        days: int | str = 30,
    ) -> HistoricalDataResponse:
        """
        Fetch historical price observations for a coin over a specified number of days.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin').
            days: Number of days ('1', '7', '14', '30', '90', '180', '365', 'max' or integer).

        Returns:
            HistoricalDataResponse containing normalized daily records or error details.
        """
        coin_id = coin_id.strip().lower()
        if not coin_id:
            return HistoricalDataResponse(
                records=[],
                error_message="Cryptocurrency ID cannot be empty.",
                is_success=False,
            )

        now = time.time()
        cache_key = f"chart_{coin_id}_{days}_{self.currency}"

        # Check in-memory cache
        if cache_key in self._historical_cache:
            entry = self._historical_cache[cache_key]
            if now - entry["timestamp"] < self.cache_ttl:
                return HistoricalDataResponse(
                    records=entry["records"],
                    is_cached=True,
                    is_success=True,
                )

        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": self.currency,
            "days": str(days),
            "interval": "daily",
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout,
            )

            if response.status_code == 429:
                return self._fallback_historical(
                    cache_key,
                    "CoinGecko API rate limit reached (HTTP 429). Please wait before requesting historical data.",
                )

            if response.status_code == 404:
                return HistoricalDataResponse(
                    records=[],
                    error_message=f"Cryptocurrency '{coin_id}' was not found on CoinGecko (HTTP 404).",
                    is_success=False,
                )

            if response.status_code != 200:
                return self._fallback_historical(
                    cache_key,
                    f"CoinGecko API returned HTTP {response.status_code} when fetching history.",
                )

            data = response.json()
            raw_prices = data.get("prices", [])
            if not raw_prices:
                return HistoricalDataResponse(
                    records=[],
                    error_message=f"No historical prices returned by CoinGecko for '{coin_id}'.",
                    is_success=False,
                )

            normalized = self._normalize_historical_prices(raw_prices)
            self._historical_cache[cache_key] = {
                "records": normalized,
                "timestamp": now,
            }

            return HistoricalDataResponse(
                records=normalized,
                is_cached=False,
                is_success=True,
            )

        except requests.exceptions.Timeout:
            return self._fallback_historical(
                cache_key,
                f"Historical data request timed out after {self.timeout}s.",
            )
        except requests.exceptions.ConnectionError:
            return self._fallback_historical(
                cache_key,
                "Unable to connect to CoinGecko API. Please check your internet connection.",
            )
        except Exception as exc:
            return self._fallback_historical(
                cache_key,
                f"Unexpected error fetching historical data: {exc}",
            )

    def get_historical_range(
        self,
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int,
    ) -> HistoricalDataResponse:
        """
        Fetch historical price observations across an explicit UNIX timestamp range.

        Args:
            coin_id: CoinGecko coin ID.
            from_timestamp: Start timestamp in seconds.
            to_timestamp: End timestamp in seconds.

        Returns:
            HistoricalDataResponse.
        """
        coin_id = coin_id.strip().lower()
        if not coin_id:
            return HistoricalDataResponse(
                records=[],
                error_message="Cryptocurrency ID cannot be empty.",
                is_success=False,
            )

        if from_timestamp >= to_timestamp:
            return HistoricalDataResponse(
                records=[],
                error_message="Start date must precede end date.",
                is_success=False,
            )

        now = time.time()
        cache_key = f"range_{coin_id}_{from_timestamp}_{to_timestamp}_{self.currency}"

        if cache_key in self._historical_cache:
            entry = self._historical_cache[cache_key]
            if now - entry["timestamp"] < self.cache_ttl:
                return HistoricalDataResponse(
                    records=entry["records"],
                    is_cached=True,
                    is_success=True,
                )

        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": self.currency,
            "from": str(from_timestamp),
            "to": str(to_timestamp),
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout,
            )

            if response.status_code == 429:
                return self._fallback_historical(
                    cache_key,
                    "CoinGecko API rate limit reached (HTTP 429).",
                )

            if response.status_code != 200:
                return self._fallback_historical(
                    cache_key,
                    f"CoinGecko API returned HTTP {response.status_code}.",
                )

            data = response.json()
            raw_prices = data.get("prices", [])
            normalized = self._normalize_historical_prices(raw_prices)
            self._historical_cache[cache_key] = {
                "records": normalized,
                "timestamp": now,
            }

            return HistoricalDataResponse(
                records=normalized,
                is_cached=False,
                is_success=True,
            )

        except requests.exceptions.Timeout:
            return self._fallback_historical(
                cache_key,
                f"Historical range request timed out after {self.timeout}s.",
            )
        except requests.exceptions.ConnectionError:
            return self._fallback_historical(
                cache_key,
                "Unable to connect to CoinGecko API. Please check network.",
            )
        except Exception as exc:
            return self._fallback_historical(
                cache_key,
                f"Error fetching historical range: {exc}",
            )

    def _fallback_historical(
        self, cache_key: str, error_message: str
    ) -> HistoricalDataResponse:
        """Fallback to cached historical data if available, never fabricate prices."""
        if cache_key in self._historical_cache:
            return HistoricalDataResponse(
                records=self._historical_cache[cache_key]["records"],
                is_cached=True,
                error_message=f"{error_message} Showing previously cached records.",
                is_success=False,
            )
        return HistoricalDataResponse(
            records=[],
            is_cached=False,
            error_message=error_message,
            is_success=False,
        )


# Global default instance
market_service = CoinGeckoMarketService()
