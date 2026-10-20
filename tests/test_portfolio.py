"""Unit tests for portfolio calculations and table generation."""

import math
import pandas as pd
import pytest
from src.portfolio import (
    calculate_holding_value,
    calculate_total_portfolio_value,
    calculate_portfolio_weights,
    calculate_allocation_percentages,
    generate_portfolio_summary_table,
)


class TestHoldingValueCalculation:
    """Tests for individual holding value calculation."""

    def test_holding_value_normal(self):
        # 0.5 BTC * 5,000,000 INR = 2,500,000 INR
        assert calculate_holding_value(0.5, 5_000_000.0) == 2_500_000.0

    def test_holding_value_zero_quantity(self):
        assert calculate_holding_value(0.0, 5_000_000.0) == 0.0

    def test_holding_value_zero_price(self):
        assert calculate_holding_value(10.0, 0.0) == 0.0

    def test_holding_value_negative_quantity_raises(self):
        with pytest.raises(ValueError, match="Quantity cannot be negative"):
            calculate_holding_value(-1.5, 50_000.0)

    def test_holding_value_negative_price_raises(self):
        with pytest.raises(ValueError, match="Price cannot be negative"):
            calculate_holding_value(2.0, -100.0)

    def test_holding_value_non_numeric_raises(self):
        with pytest.raises(TypeError):
            calculate_holding_value("not_a_number", 100.0)


class TestTotalPortfolioValue:
    """Tests for total portfolio value calculation."""

    def test_total_portfolio_value_normal(self):
        values = [1000.0, 2500.50, 500.25]
        assert calculate_total_portfolio_value(values) == pytest.approx(4000.75)

    def test_total_portfolio_value_empty(self):
        assert calculate_total_portfolio_value([]) == 0.0

    def test_total_portfolio_value_zeros(self):
        assert calculate_total_portfolio_value([0.0, 0.0, 0.0]) == 0.0

    def test_total_portfolio_value_negative_raises(self):
        with pytest.raises(ValueError, match="Holding value cannot be negative"):
            calculate_total_portfolio_value([100.0, -50.0])


class TestPortfolioWeights:
    """Tests for portfolio weights and allocation percentages."""

    def test_weights_sum_to_one(self):
        values = [2500.0, 5000.0, 2500.0]
        weights = calculate_portfolio_weights(values)
        assert len(weights) == 3
        assert weights[0] == pytest.approx(0.25)
        assert weights[1] == pytest.approx(0.50)
        assert weights[2] == pytest.approx(0.25)
        assert math.isclose(sum(weights), 1.0, rel_tol=1e-5)

    def test_allocation_percentages_sum_to_100(self):
        values = [123.45, 678.90, 456.78]
        weights = calculate_portfolio_weights(values)
        percentages = calculate_allocation_percentages(weights)
        assert math.isclose(sum(percentages), 100.0, abs_tol=0.01)

    def test_weights_empty_portfolio(self):
        assert calculate_portfolio_weights([]) == []

    def test_weights_zero_total_value(self):
        # When all holding values are zero, weights should be 0.0 without division by zero
        weights = calculate_portfolio_weights([0.0, 0.0])
        assert weights == [0.0, 0.0]

    def test_single_holding_weight(self):
        weights = calculate_portfolio_weights([50000.0])
        assert weights == [1.0]
        percentages = calculate_allocation_percentages(weights)
        assert percentages == [100.0]


class TestPortfolioSummaryTable:
    """Tests for generating portfolio summary table and handling edge cases."""

    def test_summary_empty_portfolio(self):
        df, total_val, warnings = generate_portfolio_summary_table([], {})
        assert df.empty
        assert total_val == 0.0
        assert warnings == []
        assert "Cryptocurrency" in df.columns
        assert "Allocation (%)" in df.columns

    def test_summary_normal_portfolio(self):
        holdings = [
            {"coin_id": "bitcoin", "symbol": "BTC", "name": "Bitcoin", "quantity": 0.1},
            {"coin_id": "ethereum", "symbol": "ETH", "name": "Ethereum", "quantity": 2.0},
        ]
        # Mock prices in INR
        mock_prices = {
            "bitcoin": 6_000_000.0,
            "ethereum": 300_000.0,
        }
        df, total_val, warnings = generate_portfolio_summary_table(holdings, mock_prices)

        # BTC: 0.1 * 6,000,000 = 600,000
        # ETH: 2.0 * 300,000 = 600,000
        # Total = 1,200,000
        assert total_val == 1_200_000.0
        assert len(warnings) == 0
        assert len(df) == 2
        assert df.loc[df["Symbol"] == "BTC", "Holding Value (INR)"].iloc[0] == 600_000.0
        assert df.loc[df["Symbol"] == "BTC", "Allocation (%)"].iloc[0] == pytest.approx(50.0)
        assert df.loc[df["Symbol"] == "ETH", "Allocation (%)"].iloc[0] == pytest.approx(50.0)

    def test_summary_missing_market_price(self):
        holdings = [
            {"coin_id": "bitcoin", "symbol": "BTC", "name": "Bitcoin", "quantity": 0.1},
            {"coin_id": "solana", "symbol": "SOL", "name": "Solana", "quantity": 10.0},
        ]
        # Solana price is missing
        mock_prices = {"bitcoin": 6_000_000.0}

        df, total_val, warnings = generate_portfolio_summary_table(holdings, mock_prices)
        assert len(warnings) == 1
        assert "Price unavailable for Solana (SOL)" in warnings[0]
        # Total is only BTC
        assert total_val == 600_000.0
        sol_row = df.loc[df["Symbol"] == "SOL"].iloc[0]
        assert pd.isna(sol_row["Current Price (INR)"])
        assert sol_row["Holding Value (INR)"] == 0.0
        assert sol_row["Allocation (%)"] == 0.0
