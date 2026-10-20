"""Compatibility tests for Streamlit components and parameters."""

import inspect
import streamlit as st


def test_streamlit_component_signatures():
    """Inspect parameters of Streamlit components."""
    df_params = inspect.signature(st.dataframe).parameters
    plotly_params = inspect.signature(st.plotly_chart).parameters
    button_params = inspect.signature(st.button).parameters
    download_params = inspect.signature(st.download_button).parameters
    form_btn_params = inspect.signature(st.form_submit_button).parameters
    
    assert "width" in df_params
    # Check what parameters are present
    print(f"\nst.dataframe has width: {'width' in df_params}")
    print(f"st.plotly_chart has width: {'width' in plotly_params}")
    print(f"st.button has width: {'width' in button_params}")
    print(f"st.download_button has width: {'width' in download_params}")
    print(f"st.form_submit_button has width: {'width' in form_btn_params}")


def test_database_imports_in_app():
    """Verify all symbols imported by src.app from src.database actually exist and can be imported."""
    from src.database import (
        get_holdings,
        add_or_update_holding,
        delete_holding,
        clear_all_holdings,
        save_historical_prices,
        get_historical_prices,
        get_stored_date_range,
        delete_historical_prices,
    )
    assert callable(get_holdings)
    assert callable(save_historical_prices)
    assert callable(get_historical_prices)
    assert callable(get_stored_date_range)


def test_no_deprecated_use_container_width_in_app():
    """Verify that src/app.py does not contain deprecated use_container_width arguments."""
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "src" / "app.py"
    content = app_path.read_text(encoding="utf-8")
    assert "use_container_width" not in content


def test_no_undefined_clean_returns_df_in_app():
    """Verify that the typo clean_returns_df is not used in src/app.py."""
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "src" / "app.py"
    content = app_path.read_text(encoding="utf-8")
    assert "clean_returns_df" not in content
    assert "clean_returns_chart_df" in content


def test_forecasting_dashboard_function_and_navigation_in_app():
    """Verify that render_forecasting_dashboard exists in src.app and is wired into navigation."""
    import src.app as app
    assert hasattr(app, "render_forecasting_dashboard")
    assert callable(app.render_forecasting_dashboard)
    
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "src" / "app.py"
    content = app_path.read_text(encoding="utf-8")
    assert "Price Forecasting (Module 3)" in content
    assert "train_predict_arima" in content
    assert "train_predict_deep_learning" in content
    assert "run_model_comparison" in content
