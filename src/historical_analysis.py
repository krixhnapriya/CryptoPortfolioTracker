"""Historical price analysis, return calculations, and forecasting data preparation.

Provides pure functions to validate chronological time-series observations,
compute daily returns, extract descriptive statistics, and prepare clean datasets.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


def validate_and_sort_prices(
    records: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate, clean, and chronologically sort historical price observations.

    Ensures:
    1. Chronological ascending order by date.
    2. No duplicate dates (keeps latest observation per calendar day).
    3. Prices are valid non-negative floats.
    4. No random shuffling (strict time-series integrity).

    Args:
        records: List of dicts or DataFrame containing at least 'date' and 'price_inr'.

    Returns:
        Clean DataFrame with columns ['date', 'timestamp', 'price_inr'] sorted chronologically.

    Raises:
        ValueError: If records contain negative prices or invalid structure.
    """
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        if not records:
            return pd.DataFrame(columns=["date", "timestamp", "price_inr"])
        df = pd.DataFrame(records)

    required_cols = {"date", "price_inr"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Historical records missing required columns: {missing}")

    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        df["timestamp"] = 0

    # Clean dates and prices
    df["date"] = df["date"].astype(str).str.strip()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")

    # Drop any NaNs in price
    df = df.dropna(subset=["date", "price_inr"])

    # Check for negative prices
    if (df["price_inr"] < 0).any():
        raise ValueError("Historical prices cannot contain negative values.")

    # Deduplicate by date, keeping the last row
    df = df.drop_duplicates(subset=["date"], keep="last")

    # Sort strictly ascending by date (Chronological order)
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

    return df[["date", "timestamp", "price_inr"]]


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns from chronological price series.

    Formula:
        Daily Return (%) = ((Price_t - Price_{t-1}) / Price_{t-1}) * 100

    Guards:
    - If Price_{t-1} is 0 or NaN, return is set to NaN to avoid division by zero.
    - First observation has no predecessor, so return is NaN.

    Args:
        df: DataFrame containing at least 'date' and 'price_inr' in chronological order.

    Returns:
        DataFrame with added column 'daily_return_pct'.
    """
    clean_df = validate_and_sort_prices(df)
    if clean_df.empty:
        clean_df["daily_return_pct"] = pd.Series(dtype=float)
        return clean_df

    # Calculate percentage change
    prices = clean_df["price_inr"].values
    returns = np.full(len(prices), np.nan, dtype=float)

    for i in range(1, len(prices)):
        prev_p = prices[i - 1]
        curr_p = prices[i]
        if prev_p > 0:
            returns[i] = ((curr_p - prev_p) / prev_p) * 100.0
        else:
            returns[i] = np.nan

    clean_df["daily_return_pct"] = returns
    return clean_df


def calculate_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute essential statistical metrics for the historical period.

    Metrics include:
    - Starting price, latest price, lowest price, highest price.
    - Total percentage change over period.
    - Average daily return and standard deviation (volatility).
    - Maximum single-day gain and loss.
    - Record count and date coverage.

    Args:
        df: DataFrame containing historical prices and optionally daily returns.

    Returns:
        Dictionary of summary statistics.
    """
    df_with_returns = calculate_daily_returns(df)
    n = len(df_with_returns)

    if n == 0:
        return {
            "record_count": 0,
            "start_date": None,
            "end_date": None,
            "start_price": 0.0,
            "latest_price": 0.0,
            "min_price": 0.0,
            "max_price": 0.0,
            "total_return_pct": 0.0,
            "avg_daily_return_pct": 0.0,
            "daily_return_std_pct": 0.0,
            "max_daily_gain_pct": 0.0,
            "max_daily_loss_pct": 0.0,
        }

    prices = df_with_returns["price_inr"].values
    start_price = float(prices[0])
    latest_price = float(prices[-1])
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))

    # Total return across entire selected period
    if start_price > 0:
        total_return_pct = ((latest_price - start_price) / start_price) * 100.0
    else:
        total_return_pct = 0.0

    valid_returns = df_with_returns["daily_return_pct"].dropna().values

    if len(valid_returns) > 0:
        avg_daily_return = float(np.mean(valid_returns))
        # Use ddof=1 for sample standard deviation if n > 1
        std_daily_return = float(np.std(valid_returns, ddof=1)) if len(valid_returns) > 1 else 0.0
        max_daily_gain = float(np.max(valid_returns))
        max_daily_loss = float(np.min(valid_returns))
    else:
        avg_daily_return = 0.0
        std_daily_return = 0.0
        max_daily_gain = 0.0
        max_daily_loss = 0.0

    return {
        "record_count": n,
        "start_date": str(df_with_returns["date"].iloc[0]),
        "end_date": str(df_with_returns["date"].iloc[-1]),
        "start_price": start_price,
        "latest_price": latest_price,
        "min_price": min_price,
        "max_price": max_price,
        "total_return_pct": total_return_pct,
        "avg_daily_return_pct": avg_daily_return,
        "daily_return_std_pct": std_daily_return,
        "max_daily_gain_pct": max_daily_gain,
        "max_daily_loss_pct": max_daily_loss,
    }


def prepare_forecasting_dataset(
    df: pd.DataFrame,
    coin_id: str,
) -> pd.DataFrame:
    """
    Format a clean, chronological dataset ready for export or forecasting models (ARIMA / LSTM).

    Guarantees:
    - Pure chronological ordering without data shuffling (prevents look-ahead leakage).
    - Clear, standard column names: ['Date', 'Coin_ID', 'Price_INR', 'Daily_Return_Pct'].
    - Continuous index.

    Args:
        df: Raw or processed historical prices.
        coin_id: CoinGecko coin identifier.

    Returns:
        Structured pandas DataFrame.
    """
    df_with_returns = calculate_daily_returns(df)
    if df_with_returns.empty:
        return pd.DataFrame(
            columns=["Date", "Coin_ID", "Price_INR", "Daily_Return_Pct"]
        )

    export_df = pd.DataFrame(
        {
            "Date": df_with_returns["date"],
            "Coin_ID": coin_id.strip().lower(),
            "Price_INR": df_with_returns["price_inr"],
            "Daily_Return_Pct": df_with_returns["daily_return_pct"],
        }
    )

    return export_df
