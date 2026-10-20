"""Cryptocurrency price forecasting module.

Implements:
1. Chronological time-series preparation and train/test splitting (no data leakage).
2. ARIMA forecasting using statsmodels with AIC-guided parameter selection.
3. Sequential deep learning models (LSTM and Bi-LSTM) using TensorFlow/Keras.
4. Recursive multi-step forecasting with documented error accumulation constraints.
5. Standard evaluation metrics: MAE, RMSE, MAPE.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Standardized container for all forecasting model outputs."""

    model_name: str
    is_success: bool = True
    error_message: Optional[str] = None
    params_info: str = ""
    train_count: int = 0
    test_count: int = 0
    train_date_range: tuple[str, str] = ("", "")
    test_dates: list[str] = field(default_factory=list)
    test_actual: list[float] = field(default_factory=list)
    test_predicted: list[float] = field(default_factory=list)
    future_dates: list[str] = field(default_factory=list)
    future_forecast: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # 'mae', 'rmse', 'mape'
    explanation: str = ""


def is_tensorflow_available() -> tuple[bool, str]:
    """Check whether TensorFlow/Keras is installed and importable."""
    try:
        import tensorflow as tf  # noqa: F401
        return True, "TensorFlow is installed and ready."
    except ImportError as err:
        return False, f"TensorFlow is not installed ({err})."
    except Exception as err:
        return False, f"TensorFlow failed to initialize: {err}"


# ============================================================================
# 1. DATA PREPARATION & CHRONOLOGICAL SPLITTING
# ============================================================================


def prepare_time_series_data(records_or_df: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """
    Validate, sort chronologically, deduplicate, and clean historical price observations.

    Guarantees:
    - Strictly ascending chronological order by date.
    - No duplicate dates (keeps latest observation per day).
    - Excludes negative or missing prices.
    - Zero random shuffling (prevents look-ahead bias).

    Returns:
        Clean DataFrame with columns ['date', 'price_inr'].
    """
    if isinstance(records_or_df, pd.DataFrame):
        df = records_or_df.copy()
    else:
        if not records_or_df:
            return pd.DataFrame(columns=["date", "price_inr"])
        df = pd.DataFrame(records_or_df)

    if "date" not in df.columns or "price_inr" not in df.columns:
        raise ValueError("Data must contain 'date' and 'price_inr' columns.")

    df["date"] = df["date"].astype(str).str.strip()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df = df.dropna(subset=["date", "price_inr"])
    df = df[df["price_inr"] >= 0]
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    return df[["date", "price_inr"]]


def chronological_train_test_split(
    df: pd.DataFrame,
    test_horizon: int = 7,
    min_train_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series into chronological train and test partitions.

    Ensures:
    - Train data strictly precedes test data in time.
    - No data leakage between partitions.

    Args:
        df: Chronologically sorted price DataFrame.
        test_horizon: Number of observations reserved for held-out evaluation.
        min_train_size: Minimum required observations in the training set.

    Returns:
        (train_df, test_df)

    Raises:
        ValueError: If total observations are insufficient for split.
    """
    clean_df = prepare_time_series_data(df)
    n = len(clean_df)

    if n < min_train_size + 3:
        raise ValueError(
            f"Insufficient historical data: {n} observations available. "
            f"At least {min_train_size + 3} are required for train/test evaluation."
        )

    # Test size is bounded by horizon, but capped at 25% of total dataset
    test_count = max(3, min(int(test_horizon), int(n * 0.25)))
    train_count = n - test_count

    if train_count < min_train_size:
        raise ValueError(
            f"Training set has only {train_count} records (minimum required: {min_train_size})."
        )

    train_df = clean_df.iloc[:train_count].copy().reset_index(drop=True)
    test_df = clean_df.iloc[train_count:].copy().reset_index(drop=True)
    return train_df, test_df


def generate_future_dates(last_date_str: str, steps: int) -> list[str]:
    """Generate a list of consecutive future calendar dates starting after last_date_str."""
    try:
        last_dt = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except ValueError:
        last_dt = datetime.date.today()

    return [
        (last_dt + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, steps + 1)
    ]


# ============================================================================
# 2. EVALUATION METRICS
# ============================================================================


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error (MAE): Average magnitude of errors in INR."""
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if len(act) == 0:
        return 0.0
    return float(np.mean(np.abs(act - pred)))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE): Square root of mean squared errors."""
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if len(act) == 0:
        return 0.0
    return float(np.sqrt(np.mean((act - pred) ** 2)))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE) in %.

    Safely handles zero actual values by filtering or guarding against division by zero.
    """
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if len(act) == 0:
        return 0.0

    valid_mask = act > 0
    if not np.any(valid_mask):
        return 0.0

    pct_errors = np.abs((act[valid_mask] - pred[valid_mask]) / act[valid_mask])
    return float(np.mean(pct_errors) * 100.0)


def evaluate_forecast(actual: list[float] | np.ndarray, predicted: list[float] | np.ndarray) -> dict[str, float]:
    """Compute dictionary containing MAE, RMSE, and MAPE."""
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    return {
        "mae": round(calculate_mae(act, pred), 2),
        "rmse": round(calculate_rmse(act, pred), 2),
        "mape": round(calculate_mape(act, pred), 2),
    }


# ============================================================================
# 3. STATISTICAL MODEL: ARIMA
# ============================================================================


def train_predict_arima(
    df: pd.DataFrame,
    horizon: int = 7,
    test_horizon: int = 7,
) -> ForecastResult:
    """
    Train an ARIMA model with AIC parameter selection and generate future forecasts.

    Steps:
    1. Chronological train/test split.
    2. Transparent parameter search over candidate (p, d, q) orders using AIC.
    3. Evaluate best model on held-out test partition.
    4. Fit chosen order on the complete historical series to project future horizon.

    Args:
        df: Historical price DataFrame.
        horizon: Future forecast steps (7, 14, 30).
        test_horizon: Number of held-out test points for evaluation.

    Returns:
        ForecastResult container with metrics, test comparison, and future predictions.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return ForecastResult(
            model_name="ARIMA",
            is_success=False,
            error_message="The 'statsmodels' library is not installed. Please install it via pip.",
        )

    clean_df = prepare_time_series_data(df)
    n = len(clean_df)

    if n < 15:
        return ForecastResult(
            model_name="ARIMA",
            is_success=False,
            error_message=(
                f"ARIMA requires at least 15 daily historical price observations. "
                f"Currently, only {n} records are available. Please fetch a longer timeframe."
            ),
        )

    try:
        train_df, test_df = chronological_train_test_split(
            clean_df, test_horizon=test_horizon, min_train_size=10
        )
    except ValueError as err:
        return ForecastResult(
            model_name="ARIMA",
            is_success=False,
            error_message=str(err),
        )

    train_prices = train_df["price_inr"].values.astype(float)
    test_prices = test_df["price_inr"].values.astype(float)
    full_prices = clean_df["price_inr"].values.astype(float)
    test_steps = len(test_prices)

    # Transparent parameter search over parsimonious (p, d, q) candidates
    candidate_orders = [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 0),
        (1, 1, 2),
        (1, 0, 0),
    ]

    best_aic = float("inf")
    best_order = (1, 1, 1)
    best_test_model = None

    for order in candidate_orders:
        try:
            model = ARIMA(train_prices, order=order)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
                best_test_model = fitted
        except Exception:
            continue

    if best_test_model is None:
        try:
            best_order = (1, 1, 0)
            best_test_model = ARIMA(train_prices, order=best_order).fit()
        except Exception as exc:
            return ForecastResult(
                model_name="ARIMA",
                is_success=False,
                error_message=f"ARIMA model failed to converge on training data: {exc}",
            )

    # Forecast held-out test period
    try:
        test_pred_raw = best_test_model.forecast(steps=test_steps)
        test_pred = [max(0.0, float(p)) for p in test_pred_raw]
    except Exception as exc:
        return ForecastResult(
            model_name="ARIMA",
            is_success=False,
            error_message=f"ARIMA test evaluation forecast failed: {exc}",
        )

    metrics = evaluate_forecast(test_prices, test_pred)

    # Fit chosen order on the full dataset for future horizon forecasting
    try:
        full_model = ARIMA(full_prices, order=best_order).fit()
        future_pred_raw = full_model.forecast(steps=horizon)
        future_pred = [max(0.0, float(p)) for p in future_pred_raw]
    except Exception as exc:
        return ForecastResult(
            model_name="ARIMA",
            is_success=False,
            error_message=f"ARIMA future forecasting failed: {exc}",
        )

    future_dates = generate_future_dates(str(clean_df["date"].iloc[-1]), horizon)

    explanation = (
        f"ARIMA{best_order} was selected by minimizing the Akaike Information Criterion (AIC={best_aic:.1f}). "
        f"The autoregressive component (p={best_order[0]}) captures correlation with past prices, "
        f"differencing (d={best_order[1]}) stabilizes price trends, and the moving average component "
        f"(q={best_order[2]}) models transient shocks."
    )

    return ForecastResult(
        model_name="ARIMA",
        is_success=True,
        params_info=f"ARIMA{best_order} (AIC={best_aic:.1f})",
        train_count=len(train_df),
        test_count=len(test_df),
        train_date_range=(str(train_df["date"].iloc[0]), str(train_df["date"].iloc[-1])),
        test_dates=list(test_df["date"]),
        test_actual=[float(x) for x in test_prices],
        test_predicted=test_pred,
        future_dates=future_dates,
        future_forecast=future_pred,
        metrics=metrics,
        explanation=explanation,
    )


# ============================================================================
# 4. DEEP LEARNING: LSTM & Bi-LSTM
# ============================================================================


def _create_sliding_windows(series: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1D time series into sliding input windows (X) and target outputs (y)."""
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


def train_predict_deep_learning(
    df: pd.DataFrame,
    horizon: int = 7,
    test_horizon: int = 7,
    is_bidirectional: bool = False,
    epochs: int = 25,
    random_seed: int = 42,
) -> ForecastResult:
    """
    Train an LSTM or Bidirectional LSTM neural network for sequential price forecasting.

    Steps:
    1. Chronological train/test split.
    2. Fit MinMaxScaler strictly on training data (no lookahead data leakage).
    3. Construct sliding window sequences of past observations.
    4. Train sequential LSTM / Bi-LSTM network with fixed random seeds for repeatability.
    5. Evaluate on held-out test data.
    6. Generate multi-step future forecasts using recursive prediction.

    Args:
        df: Historical price DataFrame.
        horizon: Number of future days to forecast (7, 14, 30).
        test_horizon: Evaluation test period length.
        is_bidirectional: True for Bi-LSTM, False for standard LSTM.
        epochs: Training iterations.
        random_seed: Random seed for reproducibility.

    Returns:
        ForecastResult container.
    """
    model_name = "Bi-LSTM" if is_bidirectional else "LSTM"

    # Check TensorFlow availability
    tf_ok, tf_msg = is_tensorflow_available()
    if not tf_ok:
        return ForecastResult(
            model_name=model_name,
            is_success=False,
            error_message=(
                f"TensorFlow is required for {model_name} forecasting but is not available in the current environment. "
                f"Details: {tf_msg}. You can run statistical forecasts using the ARIMA model."
            ),
        )

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    # Enforce reproducibility
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    clean_df = prepare_time_series_data(df)
    n = len(clean_df)

    if n < 25:
        return ForecastResult(
            model_name=model_name,
            is_success=False,
            error_message=(
                f"{model_name} requires at least 25 daily observations to construct sequences and train layers. "
                f"Currently, only {n} records exist. Please fetch more historical data."
            ),
        )

    try:
        train_df, test_df = chronological_train_test_split(
            clean_df, test_horizon=test_horizon, min_train_size=15
        )
    except ValueError as err:
        return ForecastResult(
            model_name=model_name,
            is_success=False,
            error_message=str(err),
        )

    train_prices = train_df["price_inr"].values.astype(float).reshape(-1, 1)
    test_prices = test_df["price_inr"].values.astype(float).reshape(-1, 1)
    full_prices = clean_df["price_inr"].values.astype(float).reshape(-1, 1)
    test_steps = len(test_prices)

    # Fit scaler ONLY on training data (Strict prevention of data leakage)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_scaled = scaler.fit_transform(train_prices).flatten()
    full_scaled = scaler.transform(full_prices).flatten()

    # Determine window size (W)
    window_size = min(7, max(3, len(train_scaled) // 4))

    X_train, y_train = _create_sliding_windows(train_scaled, window_size)

    if len(X_train) < 5:
        return ForecastResult(
            model_name=model_name,
            is_success=False,
            error_message="Insufficient sequence windows created from training partition.",
        )

    # Reshape X for recurrent layers: [samples, time_steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Construct Keras sequential architecture
    try:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(window_size, 1)))

        if is_bidirectional:
            model.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(32, activation="tanh")
                )
            )
        else:
            model.add(tf.keras.layers.LSTM(32, activation="tanh"))

        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=0)
    except Exception as exc:
        return ForecastResult(
            model_name=model_name,
            is_success=False,
            error_message=f"Neural network training failed: {exc}",
        )

    # Evaluate on held-out test partition using recursive multi-step forecasting
    test_pred_scaled = []
    # Seed window from the very end of training set
    current_window = train_scaled[-window_size:].copy().tolist()

    for _ in range(test_steps):
        inp = np.array(current_window).reshape((1, window_size, 1))
        pred_val = float(model.predict(inp, verbose=0)[0, 0])
        test_pred_scaled.append(pred_val)
        current_window.append(pred_val)
        current_window.pop(0)

    test_pred_arr = np.array(test_pred_scaled).reshape(-1, 1)
    test_pred_inr = scaler.inverse_transform(test_pred_arr).flatten()
    test_pred = [max(0.0, float(p)) for p in test_pred_inr]

    metrics = evaluate_forecast(test_prices.flatten(), test_pred)

    # Future forecasting: seed window from the very end of the FULL dataset
    future_pred_scaled = []
    current_future_window = full_scaled[-window_size:].copy().tolist()

    for _ in range(horizon):
        inp = np.array(current_future_window).reshape((1, window_size, 1))
        pred_val = float(model.predict(inp, verbose=0)[0, 0])
        future_pred_scaled.append(pred_val)
        current_future_window.append(pred_val)
        current_future_window.pop(0)

    future_pred_arr = np.array(future_pred_scaled).reshape(-1, 1)
    future_pred_inr = scaler.inverse_transform(future_pred_arr).flatten()
    future_pred = [max(0.0, float(p)) for p in future_pred_inr]

    future_dates = generate_future_dates(str(clean_df["date"].iloc[-1]), horizon)

    if is_bidirectional:
        explanation = (
            f"Bidirectional LSTM processes the historical price sequence forward and backward "
            f"across a {window_size}-day sliding window. This captures both antecedent price momentum and "
            f"reversal patterns before projecting {horizon} days forward using recursive multi-step estimation."
        )
    else:
        explanation = (
            f"Unidirectional LSTM uses memory cells and forget gates over a {window_size}-day sliding window "
            f"to capture temporal dependencies in daily price momentum. Multi-step forecasts are generated "
            f"recursively for the next {horizon} days."
        )

    return ForecastResult(
        model_name=model_name,
        is_success=True,
        params_info=f"{model_name} (Window={window_size}, Epochs={epochs})",
        train_count=len(train_df),
        test_count=len(test_df),
        train_date_range=(str(train_df["date"].iloc[0]), str(train_df["date"].iloc[-1])),
        test_dates=list(test_df["date"]),
        test_actual=[float(x) for x in test_prices.flatten()],
        test_predicted=test_pred,
        future_dates=future_dates,
        future_forecast=future_pred,
        metrics=metrics,
        explanation=explanation,
    )


# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================


def run_model_comparison(
    df: pd.DataFrame,
    horizon: int = 7,
    test_horizon: int = 7,
) -> tuple[dict[str, ForecastResult], pd.DataFrame]:
    """
    Run ARIMA, LSTM, and Bi-LSTM on the exact same chronological train/test split.

    Returns:
        results_map: dict of {model_name: ForecastResult}
        comparison_df: DataFrame summarizing evaluation metrics and model ranking
    """
    results: dict[str, ForecastResult] = {}

    # 1. ARIMA
    arima_res = train_predict_arima(df, horizon=horizon, test_horizon=test_horizon)
    results["ARIMA"] = arima_res

    # 2. LSTM
    lstm_res = train_predict_deep_learning(
        df, horizon=horizon, test_horizon=test_horizon, is_bidirectional=False
    )
    results["LSTM"] = lstm_res

    # 3. Bi-LSTM
    bilstm_res = train_predict_deep_learning(
        df, horizon=horizon, test_horizon=test_horizon, is_bidirectional=True
    )
    results["Bi-LSTM"] = bilstm_res

    # Construct comparison summary table
    rows = []
    for mname, res in results.items():
        if res.is_success:
            rows.append(
                {
                    "Model": mname,
                    "Parameters": res.params_info,
                    "MAE (INR)": res.metrics.get("mae", 0.0),
                    "RMSE (INR)": res.metrics.get("rmse", 0.0),
                    "MAPE (%)": res.metrics.get("mape", 0.0),
                    "Status": "Evaluated Successfully",
                }
            )
        else:
            rows.append(
                {
                    "Model": mname,
                    "Parameters": "N/A",
                    "MAE (INR)": np.nan,
                    "RMSE (INR)": np.nan,
                    "MAPE (%)": np.nan,
                    "Status": f"Unavailable: {res.error_message}",
                }
            )

    comp_df = pd.DataFrame(rows)
    # Sort successful models by lowest RMSE
    if not comp_df.empty and comp_df["RMSE (INR)"].notna().any():
        comp_df = comp_df.sort_values(by="RMSE (INR)", ascending=True).reset_index(drop=True)

    return results, comp_df
