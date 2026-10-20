"""Portfolio mathematics and financial calculation engine.

Handles holding values, total portfolio valuation, and allocation weights.
Pure functions designed for testability and zero-side-effect calculations.
"""

from __future__ import annotations

import math
from typing import Any
import pandas as pd


def calculate_holding_value(quantity: float, price_inr: float) -> float:
    """
    Calculate holding value in INR for an individual cryptocurrency.

    Formula: Value = Quantity * Price (INR)

    Args:
        quantity: The quantity of cryptocurrency held (must be non-negative).
        price_inr: The current price in INR (must be non-negative).

    Returns:
        Total holding value in INR as a float.

    Raises:
        ValueError: If quantity or price is negative.
        TypeError: If quantity or price cannot be converted to float.
    """
    try:
        qty = float(quantity)
        price = float(price_inr)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Quantity and price must be numeric values: {err}") from err

    if qty < 0:
        raise ValueError(f"Quantity cannot be negative. Received: {qty}")
    if price < 0:
        raise ValueError(f"Price cannot be negative. Received: {price}")

    return qty * price


def calculate_total_portfolio_value(holding_values: list[float]) -> float:
    """
    Calculate the total value of all holdings in the portfolio.

    Formula: Total = sum(Holding Values)

    Args:
        holding_values: List of individual holding values in INR.

    Returns:
        Sum of holding values in INR. Returns 0.0 for empty portfolio.

    Raises:
        ValueError: If any holding value is negative.
    """
    if not holding_values:
        return 0.0

    total = 0.0
    for val in holding_values:
        try:
            v = float(val)
        except (TypeError, ValueError) as err:
            raise TypeError(f"All holding values must be numeric: {err}") from err

        if v < 0:
            raise ValueError(f"Holding value cannot be negative. Received: {v}")
        total += v

    return total


def calculate_portfolio_weights(holding_values: list[float]) -> list[float]:
    """
    Calculate relative portfolio weight (0.0 to 1.0) for each holding.

    Formula: Weight_i = Value_i / Total Value

    If the total portfolio value is 0 (or empty), each weight is 0.0.
    When total value > 0, weights sum to 1.0 (within floating point precision).

    Args:
        holding_values: List of individual holding values in INR.

    Returns:
        List of weights as floats between 0.0 and 1.0.
    """
    if not holding_values:
        return []

    total_value = calculate_total_portfolio_value(holding_values)

    if total_value <= 0.0:
        return [0.0 for _ in holding_values]

    weights = [val / total_value for val in holding_values]

    # Verify sanity: weights must sum to approximately 1.0
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        # Normalize slightly to handle precision drift
        sum_w = sum(weights)
        weights = [w / sum_w for w in weights]

    return weights


def calculate_allocation_percentages(weights: list[float]) -> list[float]:
    """
    Convert decimal portfolio weights to percentage values (0.0% to 100.0%).

    Formula: Percentage_i = Weight_i * 100
    """
    return [round(w * 100.0, 4) for w in weights]


def generate_portfolio_summary_table(
    holdings: list[dict[str, Any]],
    current_prices: dict[str, float],
) -> tuple[pd.DataFrame, float, list[str]]:
    """
    Combine user holdings and live market prices into a structured summary DataFrame.

    Args:
        holdings: List of dicts with keys: 'coin_id', 'symbol', 'name', 'quantity'.
        current_prices: Mapping of coin_id to current INR price.

    Returns:
        Tuple containing:
            - DataFrame with formatted columns.
            - Total portfolio value in INR (float).
            - List of warning messages (e.g. coins with missing prices).
    """
    if not holdings:
        empty_df = pd.DataFrame(
            columns=[
                "Cryptocurrency",
                "Symbol",
                "Quantity",
                "Current Price (INR)",
                "Holding Value (INR)",
                "Allocation (%)",
            ]
        )
        return empty_df, 0.0, []

    rows = []
    holding_values = []
    warnings = []

    for item in holdings:
        coin_id = item["coin_id"]
        name = item.get("name", coin_id.capitalize())
        symbol = item.get("symbol", "").upper()
        quantity = float(item["quantity"])

        price = current_prices.get(coin_id)
        if price is None:
            warnings.append(f"Price unavailable for {name} ({symbol}). Valuation omitted.")
            holding_val = 0.0
            price_val = None
        else:
            price_val = float(price)
            holding_val = calculate_holding_value(quantity, price_val)

        holding_values.append(holding_val)
        rows.append(
            {
                "coin_id": coin_id,
                "Cryptocurrency": name,
                "Symbol": symbol,
                "Quantity": quantity,
                "Current Price (INR)": price_val,
                "Holding Value (INR)": holding_val,
            }
        )

    total_portfolio_value = calculate_total_portfolio_value(holding_values)
    weights = calculate_portfolio_weights(holding_values)
    percentages = calculate_allocation_percentages(weights)

    for i, row in enumerate(rows):
        row["Allocation (%)"] = percentages[i]

    df = pd.DataFrame(rows)
    return df, total_portfolio_value, warnings
