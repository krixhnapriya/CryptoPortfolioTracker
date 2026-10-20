"""Unit tests for historical data validation, returns, and time-series preparation."""

import math
import numpy as np
import pandas as pd
import pytest
from src.historical_analysis import (
    validate_and_sort_prices,
    calculate_daily_returns,
    calculate_summary_statistics,
    prepare_forecasting_dataset,
)


class TestValidateAndSortPrices:
    """Tests for historical price validation and chronological sorting."""

    def test_chronological_sorting(self):
        # Unsorted inputs
        records = [
            {"date": "2026-01-03", "timestamp": 300, "price_inr": 3000.0},
            {"date": "2026-01-01", "timestamp": 100, "price_inr": 1000.0},
            {"date": "2026-01-02", "timestamp": 200, "price_inr": 2000.0},
        ]
        df = validate_and_sort_prices(records)
        assert len(df) == 3
        assert list(df["date"]) == ["2026-01-01", "2026-01-02", "2026-01-03"]
        assert list(df["price_inr"]) == [1000.0, 2000.0, 3000.0]

    def test_duplicate_dates_deduplication(self):
        records = [
            {"date": "2026-01-01", "timestamp": 100, "price_inr": 1000.0},
            {"date": "2026-01-01", "timestamp": 150, "price_inr": 1050.0},  # duplicate date
        ]
        df = validate_and_sort_prices(records)
        assert len(df) == 1
        assert df["date"].iloc[0] == "2026-01-01"
        assert df["price_inr"].iloc[0] == 1050.0

    def test_negative_price_raises_value_error(self):
        records = [
            {"date": "2026-01-01", "timestamp": 100, "price_inr": -50.0},
        ]
        with pytest.raises(ValueError, match="cannot contain negative values"):
            validate_and_sort_prices(records)

    def test_missing_required_columns_raises(self):
        records = [{"wrong_column": 123}]
        with pytest.raises(ValueError, match="missing required columns"):
            validate_and_sort_prices(records)

    def test_empty_records(self):
        df = validate_and_sort_prices([])
        assert df.empty
        assert "date" in df.columns
        assert "price_inr" in df.columns


class TestCalculateDailyReturns:
    """Tests for daily percentage return calculation."""

    def test_daily_returns_normal_series(self):
        records = [
            {"date": "2026-01-01", "price_inr": 100.0},
            {"date": "2026-01-02", "price_inr": 110.0},  # +10%
            {"date": "2026-01-03", "price_inr": 99.0},   # -10% from 110
        ]
        df = calculate_daily_returns(records)
        assert len(df) == 3
        # First return must be NaN
        assert pd.isna(df["daily_return_pct"].iloc[0])
        # Second return: (110 - 100)/100 * 100 = 10.0%
        assert df["daily_return_pct"].iloc[1] == pytest.approx(10.0)
        # Third return: (99 - 110)/110 * 100 = -10.0%
        assert df["daily_return_pct"].iloc[2] == pytest.approx(-10.0)

    def test_zero_predecessor_price_guard(self):
        records = [
            {"date": "2026-01-01", "price_inr": 0.0},
            {"date": "2026-01-02", "price_inr": 50.0},
        ]
        # Should not raise ZeroDivisionError
        df = calculate_daily_returns(records)
        assert len(df) == 2
        assert pd.isna(df["daily_return_pct"].iloc[1])

    def test_empty_dataframe_returns(self):
        df = calculate_daily_returns([])
        assert df.empty
        assert "daily_return_pct" in df.columns

    def test_daily_returns_single_observation(self):
        records = [{"date": "2026-01-01", "price_inr": 100.0}]
        df = calculate_daily_returns(records)
        assert len(df) == 1
        assert pd.isna(df["daily_return_pct"].iloc[0])

    def test_clean_returns_chart_filtering_behavior(self):
        # Scenario 1: Only 1 observation -> chart dataframe should be empty after filtering
        records_single = [{"date": "2026-01-01", "price_inr": 100.0}]
        df_single = calculate_daily_returns(records_single)
        chart_df_single = df_single.dropna(subset=["daily_return_pct"]).copy()
        chart_df_single = chart_df_single[np.isfinite(chart_df_single["daily_return_pct"])]
        assert chart_df_single.empty

        # Scenario 2: Multiple observations with positive, zero, and negative returns
        records_multi = [
            {"date": "2026-01-01", "price_inr": 100.0},
            {"date": "2026-01-02", "price_inr": 120.0},  # +20%
            {"date": "2026-01-03", "price_inr": 120.0},  # 0%
            {"date": "2026-01-04", "price_inr": 90.0},   # -25%
        ]
        df_multi = calculate_daily_returns(records_multi)
        chart_df_multi = df_multi.dropna(subset=["daily_return_pct"]).copy()
        chart_df_multi = chart_df_multi[np.isfinite(chart_df_multi["daily_return_pct"])]
        assert len(chart_df_multi) == 3
        assert list(chart_df_multi["daily_return_pct"]) == [20.0, 0.0, -25.0]


class TestSummaryStatistics:
    """Tests for descriptive statistics and performance metrics."""

    def test_summary_statistics_normal(self):
        records = [
            {"date": "2026-01-01", "price_inr": 100.0},
            {"date": "2026-01-02", "price_inr": 150.0},  # +50%
            {"date": "2026-01-03", "price_inr": 120.0},  # -20%
        ]
        stats = calculate_summary_statistics(records)
        assert stats["record_count"] == 3
        assert stats["start_date"] == "2026-01-01"
        assert stats["end_date"] == "2026-01-03"
        assert stats["start_price"] == 100.0
        assert stats["latest_price"] == 120.0
        assert stats["min_price"] == 100.0
        assert stats["max_price"] == 150.0
        # Total return: (120 - 100)/100 = +20%
        assert stats["total_return_pct"] == pytest.approx(20.0)
        # Daily returns: [50.0, -20.0] -> avg = 15.0
        assert stats["avg_daily_return_pct"] == pytest.approx(15.0)
        assert stats["max_daily_gain_pct"] == pytest.approx(50.0)
        assert stats["max_daily_loss_pct"] == pytest.approx(-20.0)
        assert stats["daily_return_std_pct"] > 0

    def test_summary_statistics_single_observation(self):
        records = [{"date": "2026-01-01", "price_inr": 500.0}]
        stats = calculate_summary_statistics(records)
        assert stats["record_count"] == 1
        assert stats["start_price"] == 500.0
        assert stats["latest_price"] == 500.0
        assert stats["total_return_pct"] == 0.0
        assert stats["avg_daily_return_pct"] == 0.0
        assert stats["daily_return_std_pct"] == 0.0

    def test_summary_statistics_empty(self):
        stats = calculate_summary_statistics([])
        assert stats["record_count"] == 0
        assert stats["start_price"] == 0.0
        assert stats["total_return_pct"] == 0.0


class TestPrepareForecastingDataset:
    """Tests for clean chronological time-series preparation without data leakage."""

    def test_forecasting_dataset_structure_and_order(self):
        # Shuffled input
        records = [
            {"date": "2026-01-04", "price_inr": 400.0},
            {"date": "2026-01-01", "price_inr": 100.0},
            {"date": "2026-01-03", "price_inr": 300.0},
            {"date": "2026-01-02", "price_inr": 200.0},
        ]
        export_df = prepare_forecasting_dataset(records, coin_id="bitcoin")

        assert len(export_df) == 4
        assert list(export_df.columns) == ["Date", "Coin_ID", "Price_INR", "Daily_Return_Pct"]
        # Must be strictly chronological
        assert list(export_df["Date"]) == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
        assert (export_df["Coin_ID"] == "bitcoin").all()
        assert list(export_df["Price_INR"]) == [100.0, 200.0, 300.0, 400.0]

    def test_forecasting_dataset_empty(self):
        export_df = prepare_forecasting_dataset([], coin_id="ethereum")
        assert export_df.empty
        assert list(export_df.columns) == ["Date", "Coin_ID", "Price_INR", "Daily_Return_Pct"]
