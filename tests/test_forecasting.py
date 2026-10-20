"""Unit tests for forecasting module: data preparation, metrics, ARIMA, and LSTM."""

import numpy as np
import pandas as pd
import pytest
from src.forecasting import (
    prepare_time_series_data,
    chronological_train_test_split,
    generate_future_dates,
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    evaluate_forecast,
    train_predict_arima,
    train_predict_deep_learning,
    run_model_comparison,
    is_tensorflow_available,
)


@pytest.fixture
def sample_price_series():
    """Generate 40 days of synthetic daily cryptocurrency prices for testing."""
    dates = pd.date_range(start="2026-01-01", periods=40, freq="D").strftime("%Y-%m-%d")
    # Upward trending prices with mild oscillation
    base_price = 50000.0
    prices = [base_price + (i * 150.0) + (100.0 if i % 2 == 0 else -100.0) for i in range(40)]
    return pd.DataFrame({"date": dates, "price_inr": prices})


class TestDataPreparationAndSplitting:
    """Tests for time-series cleaning and chronological train/test splitting."""

    def test_prepare_time_series_sorting_and_deduplication(self):
        records = [
            {"date": "2026-01-03", "price_inr": 300.0},
            {"date": "2026-01-01", "price_inr": 100.0},
            {"date": "2026-01-02", "price_inr": 200.0},
            {"date": "2026-01-02", "price_inr": 205.0},  # duplicate date
        ]
        df = prepare_time_series_data(records)
        assert len(df) == 3
        assert list(df["date"]) == ["2026-01-01", "2026-01-02", "2026-01-03"]
        assert df["price_inr"].iloc[1] == 205.0  # keeps latest duplicate

    def test_prepare_time_series_empty(self):
        df = prepare_time_series_data([])
        assert df.empty
        assert list(df.columns) == ["date", "price_inr"]

    def test_prepare_time_series_missing_columns_raises(self):
        with pytest.raises(ValueError, match="must contain 'date' and 'price_inr'"):
            prepare_time_series_data([{"foo": 1}])

    def test_chronological_train_test_split_integrity(self, sample_price_series):
        train_df, test_df = chronological_train_test_split(
            sample_price_series, test_horizon=7, min_train_size=10
        )
        assert len(train_df) + len(test_df) == len(sample_price_series)
        assert len(test_df) == 7
        # Strict chronological ordering: train dates must strictly precede test dates
        assert train_df["date"].max() < test_df["date"].min()

    def test_chronological_split_insufficient_data_raises(self):
        short_df = pd.DataFrame(
            {"date": ["2026-01-01", "2026-01-02"], "price_inr": [10.0, 11.0]}
        )
        with pytest.raises(ValueError, match="Insufficient historical data"):
            chronological_train_test_split(short_df, test_horizon=7, min_train_size=10)

    def test_generate_future_dates(self):
        future = generate_future_dates("2026-01-10", 3)
        assert future == ["2026-01-11", "2026-01-12", "2026-01-13"]


class TestEvaluationMetrics:
    """Tests for MAE, RMSE, and MAPE calculations."""

    def test_exact_predictions_zero_error(self):
        act = np.array([100.0, 200.0, 300.0])
        pred = np.array([100.0, 200.0, 300.0])
        assert calculate_mae(act, pred) == 0.0
        assert calculate_rmse(act, pred) == 0.0
        assert calculate_mape(act, pred) == 0.0

    def test_metric_computations(self):
        act = np.array([100.0, 200.0])
        pred = np.array([110.0, 190.0])
        # Errors: |100-110|=10, |200-190|=10 -> MAE = 10.0
        assert calculate_mae(act, pred) == pytest.approx(10.0)
        # RMSE: sqrt((100+100)/2) = 10.0
        assert calculate_rmse(act, pred) == pytest.approx(10.0)
        # MAPE: ((10/100 + 10/200) / 2) * 100 = (0.10 + 0.05)/2 * 100 = 7.5%
        assert calculate_mape(act, pred) == pytest.approx(7.5)

    def test_mape_safe_division_by_zero(self):
        act = np.array([0.0, 100.0])
        pred = np.array([10.0, 110.0])
        # Point with actual=0 is guarded, only actual=100 is computed
        mape = calculate_mape(act, pred)
        assert mape == pytest.approx(10.0)

    def test_evaluate_forecast_dict(self):
        act = [500.0, 600.0]
        pred = [510.0, 590.0]
        res = evaluate_forecast(act, pred)
        assert "mae" in res
        assert "rmse" in res
        assert "mape" in res
        assert res["mae"] == 10.0


class TestARIMAForecasting:
    """Tests for statsmodels ARIMA forecasting."""

    def test_arima_successful_forecast(self, sample_price_series):
        res = train_predict_arima(sample_price_series, horizon=7, test_horizon=7)
        assert res.is_success is True
        assert res.model_name == "ARIMA"
        assert len(res.future_forecast) == 7
        assert len(res.future_dates) == 7
        assert len(res.test_actual) == 7
        assert len(res.test_predicted) == 7
        assert res.metrics["mae"] >= 0.0
        assert res.metrics["rmse"] >= 0.0
        assert res.metrics["mape"] >= 0.0
        assert "ARIMA" in res.params_info

    def test_arima_insufficient_data(self):
        short_df = pd.DataFrame(
            {"date": ["2026-01-01", "2026-01-02", "2026-01-03"], "price_inr": [10.0, 11.0, 12.0]}
        )
        res = train_predict_arima(short_df, horizon=7)
        assert res.is_success is False
        assert "at least 15" in res.error_message


class TestDeepLearningForecasting:
    """Tests for LSTM and Bi-LSTM models."""

    def test_lstm_insufficient_data(self):
        short_df = pd.DataFrame(
            {"date": ["2026-01-01", "2026-01-02"], "price_inr": [10.0, 12.0]}
        )
        res = train_predict_deep_learning(short_df, horizon=7, is_bidirectional=False)
        assert res.is_success is False
        assert "at least 25" in res.error_message.lower() or "tensorflow" in res.error_message.lower()

    def test_lstm_execution_if_tensorflow_available(self, sample_price_series):
        tf_ok, _ = is_tensorflow_available()
        if not tf_ok:
            pytest.skip("TensorFlow is not installed in this environment.")

        # Test standard LSTM
        res_lstm = train_predict_deep_learning(
            sample_price_series, horizon=7, test_horizon=7, is_bidirectional=False, epochs=5
        )
        assert res_lstm.is_success is True
        assert res_lstm.model_name == "LSTM"
        assert len(res_lstm.future_forecast) == 7
        assert len(res_lstm.test_predicted) == len(res_lstm.test_actual)

        # Test Bi-LSTM
        res_bilstm = train_predict_deep_learning(
            sample_price_series, horizon=7, test_horizon=7, is_bidirectional=True, epochs=5
        )
        assert res_bilstm.is_success is True
        assert res_bilstm.model_name == "Bi-LSTM"
        assert len(res_bilstm.future_forecast) == 7


class TestModelComparison:
    """Tests for multi-model comparison table and metrics."""

    def test_run_model_comparison(self, sample_price_series):
        results_map, comp_df = run_model_comparison(
            sample_price_series, horizon=7, test_horizon=7
        )
        assert "ARIMA" in results_map
        assert "LSTM" in results_map
        assert "Bi-LSTM" in results_map
        assert isinstance(comp_df, pd.DataFrame)
        assert list(comp_df.columns) == [
            "Model", "Parameters", "MAE (INR)", "RMSE (INR)", "MAPE (%)", "Status"
        ]
        # ARIMA should always succeed with 40 data points
        assert results_map["ARIMA"].is_success is True
