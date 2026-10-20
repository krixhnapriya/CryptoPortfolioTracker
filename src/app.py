"""Streamlit Dashboard for Cryptocurrency Portfolio Tracker & Decision-Support System.

Supports:
- Module 1: Real-time portfolio tracking, INR valuation, and asset allocation.
- Module 2: Historical market analysis, daily returns, trend visualizations, and forecasting data prep.
"""

from __future__ import annotations

import datetime
import importlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import src.database
importlib.reload(src.database)

from src.config import SUPPORTED_COINS, DEFAULT_CURRENCY
from src.database import (
    get_holdings,
    add_or_update_holding,
    delete_holding,
    clear_all_holdings,
    save_historical_prices,
    get_historical_prices,
    get_stored_date_range,
)
from src.market_data import market_service
from src.portfolio import generate_portfolio_summary_table
from src.historical_analysis import (
    calculate_daily_returns,
    calculate_summary_statistics,
    prepare_forecasting_dataset,
)
from src.forecasting import (
    train_predict_arima,
    train_predict_deep_learning,
    run_model_comparison,
    ForecastResult,
)


def format_inr(value: float | None) -> str:
    """Format float into INR currency representation (₹)."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"₹{value:,.2f}"


def format_percentage(value: float | None) -> str:
    """Format float as percentage string."""
    if value is None or pd.isna(value):
        return "0.00%"
    return f"{value:+.2f}%" if value != 0 else "0.00%"


def init_app():
    """Configure Streamlit page layout and title."""
    st.set_page_config(
        page_title="Crypto Decision-Support System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ============================================================================
# MODULE 1: PORTFOLIO TRACKING
# ============================================================================


def render_portfolio_sidebar():
    """Render sidebar holding management controls for Module 1."""
    st.sidebar.header("💼 Manage Holdings")

    coin_options = {
        f"{info['name']} ({info['symbol']})": cid
        for cid, info in SUPPORTED_COINS.items()
    }
    coin_labels = list(coin_options.keys())

    with st.sidebar.form(key="add_holding_form", clear_on_submit=False):
        st.subheader("Add or Update Coin")
        selected_label = st.selectbox(
            "Select Cryptocurrency",
            options=coin_labels,
            index=0,
            help="Choose from supported cryptocurrencies tracked via CoinGecko.",
        )
        selected_coin_id = coin_options[selected_label]
        coin_info = SUPPORTED_COINS[selected_coin_id]

        quantity_input = st.number_input(
            "Quantity Held",
            min_value=0.00000001,
            max_value=1_000_000_000.0,
            value=1.0,
            step=0.1,
            format="%.6f",
            help="Specify positive units held. Existing coins will update quantity.",
        )

        submit_btn = st.form_submit_button("Save Holding", width="stretch")

        if submit_btn:
            if quantity_input <= 0:
                st.sidebar.error("Quantity must be greater than zero.")
            else:
                try:
                    add_or_update_holding(
                        coin_id=selected_coin_id,
                        symbol=coin_info["symbol"],
                        name=coin_info["name"],
                        quantity=quantity_input,
                    )
                    st.sidebar.success(
                        f"Saved {quantity_input:g} {coin_info['symbol']} successfully!"
                    )
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Error saving holding: {exc}")

    # Remove Holdings Section
    existing_holdings = get_holdings()
    if existing_holdings:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Remove Holdings")
        delete_options = {
            f"{h['name']} ({h['symbol']}) - {h['quantity']:g}": h["coin_id"]
            for h in existing_holdings
        }
        coin_to_delete = st.sidebar.selectbox(
            "Select Coin to Remove",
            options=list(delete_options.keys()),
        )
        if st.sidebar.button("Delete Selected Coin", width="stretch"):
            cid = delete_options[coin_to_delete]
            delete_holding(cid)
            st.sidebar.warning(f"Removed {coin_to_delete} from portfolio.")
            st.rerun()

        if st.sidebar.button("Clear Entire Portfolio", type="secondary", width="stretch"):
            clear_all_holdings()
            st.sidebar.warning("Cleared all holdings.")
            st.rerun()


def render_portfolio_dashboard():
    """Main dashboard rendering function for Module 1."""
    st.title("📊 Cryptocurrency Portfolio Tracker")
    st.caption(
        "Module 1: Real-Time Portfolio Tracking, INR Valuation & Asset Allocation Breakdown"
    )

    holdings = get_holdings()
    coin_ids = [h["coin_id"] for h in holdings]

    # Fetch live prices
    price_response = market_service.get_prices(coin_ids)

    # API Status Banners
    if price_response.error_message:
        if price_response.is_cached:
            st.warning(f"⚠️ {price_response.error_message}")
        else:
            st.error(f"❌ Market Data Error: {price_response.error_message}")
    elif price_response.is_cached and price_response.cache_timestamp:
        cache_time_str = datetime.datetime.fromtimestamp(
            price_response.cache_timestamp
        ).strftime("%H:%M:%S")
        st.info(f"ℹ️ Displaying cached prices from {cache_time_str} (refreshes automatically).")

    df, total_portfolio_value, warnings = generate_portfolio_summary_table(
        holdings=holdings,
        current_prices=price_response.prices,
    )

    for warn in warnings:
        st.warning(f"⚠️ {warn}")

    # Prominent Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Total Portfolio Value",
            value=format_inr(total_portfolio_value),
        )
    with m2:
        st.metric(
            label="Holdings Count",
            value=f"{len(holdings)} assets",
        )
    with m3:
        if not df.empty and total_portfolio_value > 0:
            top_asset_row = df.loc[df["Holding Value (INR)"].idxmax()]
            top_asset_str = f"{top_asset_row['Symbol']} ({top_asset_row['Allocation (%)']:.1f}%)"
        else:
            top_asset_str = "None"
        st.metric(label="Largest Asset Allocation", value=top_asset_str)
    with m4:
        st.metric(
            label="Base Currency",
            value=DEFAULT_CURRENCY.upper(),
        )

    st.markdown("---")

    if df.empty:
        st.info(
            "👋 Your portfolio is currently empty. Use the sidebar to select a cryptocurrency and enter the quantity held to begin tracking."
        )
        return

    col_table, col_chart = st.columns([1.2, 1.0])

    with col_table:
        st.subheader("📋 Portfolio Holdings Breakdown")
        display_df = df.copy()
        display_df["Quantity"] = display_df["Quantity"].apply(lambda q: f"{q:g}")
        display_df["Current Price (INR)"] = display_df["Current Price (INR)"].apply(format_inr)
        display_df["Holding Value (INR)"] = display_df["Holding Value (INR)"].apply(format_inr)
        display_df["Allocation (%)"] = display_df["Allocation (%)"].apply(lambda a: f"{a:.2f}%")

        cols_to_show = [
            "Cryptocurrency",
            "Symbol",
            "Quantity",
            "Current Price (INR)",
            "Holding Value (INR)",
            "Allocation (%)",
        ]
        st.dataframe(
            display_df[cols_to_show],
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "📌 **Academic Note**: Portfolio weight represents asset allocation share = "
            "`(Holding Value / Total Value) × 100%`. It reflects exposure concentration, "
            "not profit or loss."
        )

    with col_chart:
        st.subheader("🥧 Asset Allocation")
        if total_portfolio_value > 0:
            fig = px.pie(
                df,
                names="Cryptocurrency",
                values="Holding Value (INR)",
                hole=0.45,
                hover_data=["Symbol", "Allocation (%)"],
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Value: ₹%{value:,.2f}<br>Weight: %{percent:.2%}<extra></extra>",
            )
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("All holdings have 0 valuation. Chart will appear when prices are available.")

    col_btn, _ = st.columns([0.25, 0.75])
    with col_btn:
        if st.button("🔄 Force Refresh Prices", width="stretch"):
            market_service._price_cache.clear()
            st.rerun()


# ============================================================================
# MODULE 2: HISTORICAL MARKET ANALYSIS
# ============================================================================


def render_historical_dashboard():
    """Main dashboard rendering function for Module 2: Historical Market Analysis."""
    st.title("📈 Historical Cryptocurrency Data & Trend Analysis")
    st.caption(
        "Module 2: Historical Price Collection, Daily Returns, Volatility Metrics, and Clean Time-Series Preparation"
    )

    # Sidebar parameters for Historical Analysis
    st.sidebar.header("⚙️ Historical Controls")

    coin_options = {
        f"{info['name']} ({info['symbol']})": cid
        for cid, info in SUPPORTED_COINS.items()
    }
    coin_labels = list(coin_options.keys())

    selected_label = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=coin_labels,
        index=0,
        help="Select any supported cryptocurrency to analyze historical prices.",
    )
    selected_coin_id = coin_options[selected_label]
    coin_info = SUPPORTED_COINS[selected_coin_id]

    timeframe_map = {
        "30 Days (Daily)": 30,
        "7 Days (Daily)": 7,
        "90 Days (Daily)": 90,
        "180 Days (Daily)": 180,
        "1 Year (365 Days)": 365,
        "Custom Date Range": "custom",
    }
    selected_timeframe_label = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(timeframe_map.keys()),
        index=0,
        help="Select predefined timeframe or custom date range.",
    )
    timeframe_val = timeframe_map[selected_timeframe_label]

    custom_start_date = None
    custom_end_date = None
    if timeframe_val == "custom":
        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=30)
        custom_start_date = st.sidebar.date_input("Start Date", value=default_start)
        custom_end_date = st.sidebar.date_input("End Date", value=today)

    fetch_button = st.sidebar.button(
        "📥 Fetch / Sync from CoinGecko",
        type="primary",
        width="stretch",
        help="Query CoinGecko API and save daily observations into local SQLite storage.",
    )

    # Check local SQLite database for existing records
    if timeframe_val == "custom" and custom_start_date and custom_end_date:
        stored_records = get_historical_prices(
            selected_coin_id,
            start_date=str(custom_start_date),
            end_date=str(custom_end_date),
        )
    else:
        # For relative presets, calculate approximate start date
        days_int = int(timeframe_val) if isinstance(timeframe_val, int) else 30
        approx_start = (datetime.date.today() - datetime.timedelta(days=days_int)).strftime("%Y-%m-%d")
        stored_records = get_historical_prices(selected_coin_id, start_date=approx_start)

    # If user explicitly clicked fetch OR no local records exist yet
    if fetch_button:
        with st.spinner(f"Fetching historical data for {coin_info['name']} from CoinGecko..."):
            if timeframe_val == "custom" and custom_start_date and custom_end_date:
                from_ts = int(datetime.datetime.combine(custom_start_date, datetime.time.min).timestamp())
                to_ts = int(datetime.datetime.combine(custom_end_date, datetime.time.max).timestamp())
                api_res = market_service.get_historical_range(selected_coin_id, from_ts, to_ts)
            else:
                days_param = timeframe_val if isinstance(timeframe_val, int) else 30
                api_res = market_service.get_historical_market_chart(selected_coin_id, days=days_param)

            if api_res.is_success and api_res.records:
                saved = save_historical_prices(selected_coin_id, api_res.records)
                st.sidebar.success(f"Successfully synced {saved} daily records to SQLite!")
                # Reload stored records
                if timeframe_val == "custom" and custom_start_date and custom_end_date:
                    stored_records = get_historical_prices(
                        selected_coin_id,
                        start_date=str(custom_start_date),
                        end_date=str(custom_end_date),
                    )
                else:
                    days_int = int(timeframe_val) if isinstance(timeframe_val, int) else 30
                    approx_start = (datetime.date.today() - datetime.timedelta(days=days_int)).strftime("%Y-%m-%d")
                    stored_records = get_historical_prices(selected_coin_id, start_date=approx_start)
            else:
                st.sidebar.error(f"API Error: {api_res.error_message}")

    # Inspect total database status for this coin
    min_db_date, max_db_date, total_db_cnt = get_stored_date_range(selected_coin_id)

    # If still no records available, guide the user
    if not stored_records:
        st.info(
            f"ℹ️ No local historical price records found for **{coin_info['name']} ({coin_info['symbol']})** "
            "in the selected timeframe. Click **'📥 Fetch / Sync from CoinGecko'** in the sidebar to download "
            "and store daily observations."
        )
        if total_db_cnt > 0:
            st.caption(
                f"Note: Local database currently holds {total_db_cnt} records for {coin_info['name']} "
                f"spanning from {min_db_date} to {max_db_date}."
            )
        return

    # Process records
    hist_df = pd.DataFrame(stored_records)
    returns_df = calculate_daily_returns(hist_df)
    stats = calculate_summary_statistics(returns_df)

    # 1. Overview and Statistics Cards
    st.subheader(f"📌 {coin_info['name']} ({coin_info['symbol']}) Performance Overview")
    st.caption(
        f"Covered Period: **{stats['start_date']}** to **{stats['end_date']}** "
        f"({stats['record_count']} daily observations in local SQLite database)"
    )

    r1_col1, r1_col2, r1_col3, r1_col4, r1_col5 = st.columns(5)
    with r1_col1:
        st.metric(
            label="Starting Price",
            value=format_inr(stats["start_price"]),
        )
    with r1_col2:
        st.metric(
            label="Latest Price",
            value=format_inr(stats["latest_price"]),
            delta=f"{stats['total_return_pct']:+.2f}%",
        )
    with r1_col3:
        st.metric(
            label="Period High (Peak)",
            value=format_inr(stats["max_price"]),
        )
    with r1_col4:
        st.metric(
            label="Period Low (Trough)",
            value=format_inr(stats["min_price"]),
        )
    with r1_col5:
        st.metric(
            label="Total Period Return",
            value=f"{stats['total_return_pct']:+.2f}%",
        )

    # Return Risk & Descriptive Metrics
    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
    with r2_col1:
        st.metric(
            label="Average Daily Return",
            value=f"{stats['avg_daily_return_pct']:+.3f}%",
            help="Arithmetic mean of daily percentage changes.",
        )
    with r2_col2:
        st.metric(
            label="Daily Volatility (Std Dev)",
            value=f"{stats['daily_return_std_pct']:.3f}%",
            help="Standard deviation of daily returns across the period (higher indicates greater price fluctuations).",
        )
    with r2_col3:
        st.metric(
            label="Best Single Day Gain",
            value=f"{stats['max_daily_gain_pct']:+.2f}%",
            help="Maximum positive single-day price surge.",
        )
    with r2_col4:
        st.metric(
            label="Worst Single Day Drop",
            value=f"{stats['max_daily_loss_pct']:+.2f}%",
            help="Maximum negative single-day price decline.",
        )

    st.markdown("---")

    # 2. Historical Price Line Chart (Plotly)
    st.subheader("📉 Historical Price Trend (INR)")
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=returns_df["date"],
            y=returns_df["price_inr"],
            mode="lines+markers",
            name=f"{coin_info['symbol']} Price (INR)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=4),
            hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (INR, ₹)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
        ),
    )
    st.plotly_chart(fig_price, width="stretch")

    # 3. Daily Returns Distribution / Bar Chart
    st.subheader("📊 Daily Returns Distribution (%)")
    st.caption(
        "Daily percentage change: $R_t = \\frac{P_t - P_{t-1}}{P_{t-1}} \\times 100\\%$. "
        "Positive returns (gains) are in green, negative returns (losses) in red."
    )

    clean_returns_chart_df = returns_df.dropna(subset=["daily_return_pct"]).copy()
    # Filter out any non-finite numbers (e.g. inf / -inf) while preserving valid positive and negative returns
    if not clean_returns_chart_df.empty:
        clean_returns_chart_df = clean_returns_chart_df[
            np.isfinite(clean_returns_chart_df["daily_return_pct"])
        ]

    if not clean_returns_chart_df.empty:
        colors = [
            "#2ca02c" if r >= 0 else "#d62728"
            for r in clean_returns_chart_df["daily_return_pct"]
        ]
        fig_ret = go.Figure()
        fig_ret.add_trace(
            go.Bar(
                x=clean_returns_chart_df["date"],
                y=clean_returns_chart_df["daily_return_pct"],
                marker_color=colors,
                name="Daily Return (%)",
                hovertemplate="<b>Date:</b> %{x}<br><b>Daily Return:</b> %{y:+.2f}%<extra></extra>",
            )
        )
        fig_ret.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        fig_ret.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily Return (%)",
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_ret, width="stretch")
    else:
        st.info(
            "ℹ️ At least two chronological price observations are required to calculate and display daily percentage returns."
        )

    # 4. Data Preparation for Future Forecasting Modules (ARIMA / LSTM)
    st.markdown("---")
    st.subheader("💾 Clean Time-Series Dataset & Export")
    st.caption(
        "This dataset is stored chronologically with strict ordering to prevent data leakage "
        "when building subsequent ARIMA and LSTM forecasting models."
    )

    forecasting_df = prepare_forecasting_dataset(returns_df, selected_coin_id)

    col_preview, col_export = st.columns([1.5, 1.0])
    with col_preview:
        display_export_df = forecasting_df.copy()
        display_export_df["Price_INR"] = display_export_df["Price_INR"].apply(format_inr)
        display_export_df["Daily_Return_Pct"] = display_export_df["Daily_Return_Pct"].apply(
            lambda r: f"{r:+.3f}%" if pd.notna(r) else "N/A"
        )
        st.dataframe(
            display_export_df.tail(10),
            width="stretch",
            hide_index=True,
        )
        st.caption("Showing latest 10 observations in dataset.")

    with col_export:
        csv_data = forecasting_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Export {coin_info['symbol']} Historical CSV",
            data=csv_data,
            file_name=f"{selected_coin_id}_historical_inr_{stats['start_date']}_{stats['end_date']}.csv",
            mime="text/csv",
            width="stretch",
            help="Download clean CSV for machine learning or statistical time-series modeling.",
        )
        st.info(
            "📘 **Academic Design Note**:\n\n"
            "- **Time-Series Integrity**: Observations are sorted in ascending chronological order.\n"
            "- **Leakage Prevention**: Future statistical and deep-learning modules will fit scaling transformations on training periods only.\n"
            "- **Explainability**: Historical volatility and daily returns directly feed the upcoming factor-based portfolio risk module."
        )


# ============================================================================
# MODULE 3: CRYPTOCURRENCY PRICE FORECASTING
# ============================================================================


def render_forecasting_dashboard():
    """Main dashboard rendering function for Module 3: Price Forecasting."""
    st.title("🔮 Cryptocurrency Price Forecasting")
    st.caption(
        "Module 3: Statistical (ARIMA) & Deep Learning (LSTM, Bi-LSTM) Multi-Step Price Projections"
    )

    # 1. Sidebar Configuration Controls
    st.sidebar.header("⚙️ Forecasting Controls")

    # Build coin options with local database status indicators
    coin_options = {}
    for cid, info in SUPPORTED_COINS.items():
        _, _, count = get_stored_date_range(cid)
        status_label = f"({count} days in DB)" if count > 0 else "(no data in DB)"
        coin_options[f"{info['name']} ({info['symbol']}) {status_label}"] = cid

    selected_label = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(coin_options.keys()),
        index=0,
        help="Choose a cryptocurrency. Assets with stored daily observations in SQLite are ready for forecasting.",
    )
    selected_coin_id = coin_options[selected_label]
    coin_info = SUPPORTED_COINS[selected_coin_id]

    # Horizon selection (7, 14, 30 days)
    horizon_map = {
        "7 Days (Short-Term)": 7,
        "14 Days (Medium-Term)": 14,
        "30 Days (Extended)": 30,
    }
    selected_horizon_label = st.sidebar.selectbox(
        "Forecast Horizon",
        options=list(horizon_map.keys()),
        index=0,
        help="Number of future days to predict using recursive multi-step estimation.",
    )
    selected_horizon = horizon_map[selected_horizon_label]

    # Model selection
    model_choices = [
        "ARIMA (Auto-Regressive Integrated Moving Average)",
        "LSTM (Long Short-Term Memory Neural Network)",
        "Bi-LSTM (Bidirectional LSTM Neural Network)",
        "Compare All Models (ARIMA vs LSTM vs Bi-LSTM)",
    ]
    selected_model_choice = st.sidebar.selectbox(
        "Select Forecasting Model",
        options=model_choices,
        index=0,
        help="Select a statistical time-series model, recurrent deep learning network, or full comparative benchmark.",
    )

    train_button = st.sidebar.button(
        "🚀 Train & Forecast",
        type="primary",
        width="stretch",
        help="Train the model on chronological historical data and project forward prices.",
    )

    # 2. Check Local Database Records
    stored_records = get_historical_prices(selected_coin_id)
    min_db_date, max_db_date, total_db_cnt = get_stored_date_range(selected_coin_id)

    # Case A: 0 records in SQLite
    if not stored_records or total_db_cnt == 0:
        st.warning(
            f"⚠️ No historical price records found in SQLite for **{coin_info['name']} ({coin_info['symbol']})**."
        )
        st.info(
            "💡 **How to sync historical data:**\n\n"
            "1. Switch to **'📈 Historical Analysis (Module 2)'** using the sidebar navigation.\n"
            f"2. Select **{coin_info['name']} ({coin_info['symbol']})**.\n"
            "3. Choose a timeframe (e.g., 30 Days, 90 Days, or 1 Year) and click **'📥 Fetch / Sync from CoinGecko'**.\n"
            "4. Return to this page to train your forecasting models on real historical prices!"
        )
        return

    # Case B: Less than 15 records
    if len(stored_records) < 15:
        st.warning(
            f"⚠️ Insufficient historical data: **{coin_info['name']}** currently has only **{len(stored_records)}** daily records in SQLite.\n\n"
            "- **ARIMA** requires at least 15 daily observations to fit autoregressive and moving average parameters.\n"
            "- **LSTM and Bi-LSTM** require at least 25 daily observations to build sliding input sequences and train neural network layers.\n\n"
            "💡 **Next Step**: Navigate to **Module 2 (Historical Analysis)** and click **'📥 Fetch / Sync from CoinGecko'** with 30 Days or more."
        )
        return

    # Display Data Context Overview Cards
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        st.metric(label="Cryptocurrency", value=f"{coin_info['name']} ({coin_info['symbol']})")
    with ov2:
        st.metric(label="Database History", value=f"{total_db_cnt} daily records")
    with ov3:
        st.metric(label="Historical Coverage", value=f"{min_db_date} to {max_db_date}")
    with ov4:
        st.metric(label="Selected Horizon", value=f"{selected_horizon} Days")

    st.markdown("---")

    # Prepare DataFrame
    hist_df = pd.DataFrame(stored_records)

    # Determine test horizon (capped at 25% of dataset, minimum 3, standard 7)
    test_horizon_val = max(3, min(7, int(len(hist_df) * 0.25)))

    # Session state key to retain results during interactive UI events
    cache_key = f"fc_{selected_coin_id}_{selected_model_choice}_{selected_horizon}"

    # Check if user triggered training or cached result exists
    if train_button:
        with st.spinner(f"Training {selected_model_choice} on {total_db_cnt} daily observations..."):
            if selected_model_choice.startswith("ARIMA"):
                res = train_predict_arima(hist_df, horizon=selected_horizon, test_horizon=test_horizon_val)
                st.session_state[cache_key] = ("single", res)
            elif selected_model_choice.startswith("LSTM"):
                res = train_predict_deep_learning(
                    hist_df, horizon=selected_horizon, test_horizon=test_horizon_val, is_bidirectional=False
                )
                st.session_state[cache_key] = ("single", res)
            elif selected_model_choice.startswith("Bi-LSTM"):
                res = train_predict_deep_learning(
                    hist_df, horizon=selected_horizon, test_horizon=test_horizon_val, is_bidirectional=True
                )
                st.session_state[cache_key] = ("single", res)
            else:
                # Compare All Models
                results_map, comp_df = run_model_comparison(
                    hist_df, horizon=selected_horizon, test_horizon=test_horizon_val
                )
                st.session_state[cache_key] = ("compare", (results_map, comp_df))
        st.session_state["active_forecast_key"] = cache_key

    # Retrieve current active result from session state if available
    current_key = st.session_state.get("active_forecast_key")
    if not current_key or current_key != cache_key or cache_key not in st.session_state:
        st.info(
            "👆 Click **'🚀 Train & Forecast'** in the sidebar to run the model and generate price projections."
        )
        return

    mode, payload = st.session_state[cache_key]

    # =========================================================================
    # RENDER SINGLE MODEL RESULTS
    # =========================================================================
    if mode == "single":
        res: ForecastResult = payload

        if not res.is_success:
            st.error(f"❌ Forecasting Failed: {res.error_message}")
            return

        st.subheader(f"📊 {res.model_name} Forecast Results")

        # Row of metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="MAE (Mean Absolute Error)",
                value=format_inr(res.metrics.get("mae")),
                help="Average magnitude of forecasting errors on the held-out test period.",
            )
        with m2:
            st.metric(
                label="RMSE (Root Mean Squared Error)",
                value=format_inr(res.metrics.get("rmse")),
                help="Root mean squared error on held-out test data. Penalizes larger forecasting errors.",
            )
        with m3:
            st.metric(
                label="MAPE (Mean Absolute % Error)",
                value=f"{res.metrics.get('mape', 0.0):.2f}%",
                help="Average percentage error relative to actual prices.",
            )
        with m4:
            st.metric(
                label="Model Configuration",
                value=res.params_info,
                help="Best parameter order (ARIMA) or sliding window configuration (LSTM).",
            )

        # Interactive Plotly Chart
        st.subheader("📈 Price Trajectory & Out-of-Sample Forecast")
        st.caption(
            "Visualizing chronological training prices, held-out test evaluation (Actual vs. Predicted), "
            f"and future {selected_horizon}-day price trajectory."
        )

        fig = go.Figure()

        # Training history line
        train_end_idx = len(hist_df) - len(res.test_dates)
        train_df_subset = hist_df.iloc[:train_end_idx]
        fig.add_trace(
            go.Scatter(
                x=train_df_subset["date"],
                y=train_df_subset["price_inr"],
                mode="lines",
                name="Historical Training Data",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>",
            )
        )

        # Test Actuals
        fig.add_trace(
            go.Scatter(
                x=res.test_dates,
                y=res.test_actual,
                mode="lines+markers",
                name="Held-out Test (Actual Ground Truth)",
                line=dict(color="#2ca02c", width=2.5),
                marker=dict(size=6, symbol="circle"),
                hovertemplate="<b>Date:</b> %{x}<br><b>Actual:</b> ₹%{y:,.2f}<extra></extra>",
            )
        )

        # Test Predicted
        fig.add_trace(
            go.Scatter(
                x=res.test_dates,
                y=res.test_predicted,
                mode="lines+markers",
                name=f"Held-out Test Predicted ({res.model_name})",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                hovertemplate="<b>Date:</b> %{x}<br><b>Predicted:</b> ₹%{y:,.2f}<extra></extra>",
            )
        )

        # Future Forecast (connected seamlessly from latest observation)
        connect_dates = [str(hist_df["date"].iloc[-1])] + res.future_dates
        connect_prices = [float(hist_df["price_inr"].iloc[-1])] + res.future_forecast
        fig.add_trace(
            go.Scatter(
                x=connect_dates,
                y=connect_prices,
                mode="lines+markers",
                name=f"Future Forecast ({selected_horizon} Days)",
                line=dict(color="#9467bd", width=2.5, dash="dot"),
                marker=dict(size=7, symbol="star"),
                hovertemplate="<b>Date:</b> %{x}<br><b>Forecast:</b> ₹%{y:,.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{coin_info['name']} ({coin_info['symbol']}) - {res.model_name} Forecast Overview",
            xaxis_title="Date",
            yaxis_title="Price (INR, ₹)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width="stretch")

        # Two Tabs: Held-out Test Details & Future Projections
        tab_eval, tab_future, tab_notes = st.tabs(
            [
                "📋 Held-Out Test Validation",
                "🔮 Future Projections Table",
                "📘 Academic Methodology & Notes",
            ]
        )

        with tab_eval:
            st.markdown("##### Out-of-Sample Evaluation Table")
            st.caption(
                "These actual prices were withheld during training to measure how well the model predicts unseen price changes."
            )
            eval_rows = []
            for d, act, pred in zip(res.test_dates, res.test_actual, res.test_predicted):
                diff = pred - act
                pct_err = abs(diff / act * 100.0) if act > 0 else 0.0
                eval_rows.append(
                    {
                        "Date": d,
                        "Actual Price (INR)": format_inr(act),
                        "Predicted Price (INR)": format_inr(pred),
                        "Error (INR)": f"{diff:+,.2f}",
                        "Absolute Error (%)": f"{pct_err:.2f}%",
                    }
                )
            eval_df = pd.DataFrame(eval_rows)
            st.dataframe(eval_df, width="stretch", hide_index=True)

        with tab_future:
            st.markdown(f"##### Forward {selected_horizon}-Day Projected Prices")
            latest_price = float(hist_df["price_inr"].iloc[-1])
            future_rows = []
            for d, pred in zip(res.future_dates, res.future_forecast):
                pct_diff = ((pred - latest_price) / latest_price * 100.0) if latest_price > 0 else 0.0
                future_rows.append(
                    {
                        "Forecast Date": d,
                        "Projected Price (INR)": format_inr(pred),
                        "Projected Change from Current (%)": f"{pct_diff:+.2f}%",
                    }
                )
            future_df = pd.DataFrame(future_rows)
            st.dataframe(future_df, width="stretch", hide_index=True)

            # CSV Download
            csv_export = pd.DataFrame(
                {
                    "Coin_ID": selected_coin_id,
                    "Forecast_Date": res.future_dates,
                    "Projected_Price_INR": res.future_forecast,
                }
            ).to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"📥 Download {res.model_name} Forecast CSV",
                data=csv_export,
                file_name=f"{selected_coin_id}_{res.model_name.lower()}_forecast_{selected_horizon}d.csv",
                mime="text/csv",
                width="stretch",
            )

        with tab_notes:
            st.markdown("##### Model Rationale & Academic Design")
            st.info(f"💡 **Model Details**:\n\n{res.explanation}")
            st.markdown(
                "- **Zero Look-Ahead Data Leakage**: The dataset is strictly split chronologically. Preprocessing scalers are fitted exclusively on training records.\n"
                "- **Recursive Multi-Step Constraints**: Multi-step forecasts beyond Day 1 feed previous model outputs into subsequent inputs. This simulates forward momentum but naturally compounds uncertainty over longer horizons.\n"
                "- **Explainable Metrics**: MAE represents the average error magnitude in ₹, RMSE penalizes large volatility spikes, and MAPE normalizes error for cross-asset comparisons."
            )

    # =========================================================================
    # RENDER COMPARE ALL MODELS
    # =========================================================================
    else:
        results_map, comp_df = payload

        st.subheader("🏆 Model Comparison Benchmark")
        st.caption(
            "All three forecasting models were trained and tested on the exact same chronological train/test split."
        )

        # Highlight Best Model
        successful_df = comp_df[comp_df["Status"] == "Evaluated Successfully"].copy()
        if not successful_df.empty and "RMSE (INR)" in successful_df.columns:
            best_row = successful_df.sort_values(by="RMSE (INR)", ascending=True).iloc[0]
            st.success(
                f"🏆 **Top-Performing Model**: **{best_row['Model']}** "
                f"(RMSE: {format_inr(best_row['RMSE (INR)'])}, MAPE: {best_row['MAPE (%)']:.2f}%)"
            )

        # Display Comparison Table
        display_comp = comp_df.copy()
        display_comp["MAE (INR)"] = display_comp["MAE (INR)"].apply(
            lambda v: format_inr(v) if pd.notna(v) else "N/A"
        )
        display_comp["RMSE (INR)"] = display_comp["RMSE (INR)"].apply(
            lambda v: format_inr(v) if pd.notna(v) else "N/A"
        )
        display_comp["MAPE (%)"] = display_comp["MAPE (%)"].apply(
            lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A"
        )
        st.dataframe(display_comp, width="stretch", hide_index=True)

        # Comparative Visualization
        st.subheader("📈 Multi-Model Comparative Projections")
        st.caption(
            f"Comparing {selected_horizon}-day future trajectories across ARIMA, LSTM, and Bi-LSTM."
        )

        fig_comp = go.Figure()

        # Historical data line
        fig_comp.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["price_inr"],
                mode="lines",
                name="Historical Price",
                line=dict(color="#7f7f7f", width=2),
                hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>",
            )
        )

        # Colors for models
        model_colors = {
            "ARIMA": "#1f77b4",
            "LSTM": "#ff7f0e",
            "Bi-LSTM": "#2ca02c",
        }

        last_date = str(hist_df["date"].iloc[-1])
        last_price = float(hist_df["price_inr"].iloc[-1])

        for mname, mres in results_map.items():
            if mres.is_success and mres.future_forecast:
                c_dates = [last_date] + mres.future_dates
                c_prices = [last_price] + mres.future_forecast
                fig_comp.add_trace(
                    go.Scatter(
                        x=c_dates,
                        y=c_prices,
                        mode="lines+markers",
                        name=f"{mname} Forecast",
                        line=dict(color=model_colors.get(mname, "#9467bd"), width=2.5, dash="dash"),
                        marker=dict(size=5),
                        hovertemplate=f"<b>Date:</b> %{{x}}<br><b>{mname}:</b> ₹%{{y:,.2f}}<extra></extra>",
                    )
                )

        fig_comp.update_layout(
            title=f"{coin_info['name']} ({coin_info['symbol']}) - Multi-Model Price Projection ({selected_horizon} Days)",
            xaxis_title="Date",
            yaxis_title="Price (INR, ₹)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_comp, width="stretch")

        # Bar chart comparing errors
        if not successful_df.empty:
            st.subheader("📊 Error Metrics Comparison (Lower is Better)")
            bar_fig = go.Figure()
            bar_fig.add_trace(
                go.Bar(
                    x=successful_df["Model"],
                    y=successful_df["MAE (INR)"],
                    name="MAE (INR)",
                    marker_color="#1f77b4",
                )
            )
            bar_fig.add_trace(
                go.Bar(
                    x=successful_df["Model"],
                    y=successful_df["RMSE (INR)"],
                    name="RMSE (INR)",
                    marker_color="#ff7f0e",
                )
            )
            bar_fig.update_layout(
                barmode="group",
                yaxis_title="Error (INR, ₹)",
                margin=dict(l=40, r=40, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(bar_fig, width="stretch")

        # Individual Model Breakdown Expanders
        st.subheader("🔍 Individual Model Breakdown")
        for mname, mres in results_map.items():
            with st.expander(f"📌 {mname} Details & Configuration", expanded=False):
                if mres.is_success:
                    st.write(f"**Parameters**: {mres.params_info}")
                    st.write(f"**Explanation**: {mres.explanation}")
                    ec1, ec2, ec3 = st.columns(3)
                    with ec1:
                        st.metric("MAE", format_inr(mres.metrics.get("mae")))
                    with ec2:
                        st.metric("RMSE", format_inr(mres.metrics.get("rmse")))
                    with ec3:
                        st.metric("MAPE", f"{mres.metrics.get('mape', 0.0):.2f}%")
                else:
                    st.warning(f"Model unavailable: {mres.error_message}")


# ============================================================================
# MAIN APP ENTRY POINT
# ============================================================================


def main():
    init_app()

    # Primary Navigation
    st.sidebar.title("Navigation")
    active_page = st.sidebar.radio(
        "Select Module",
        options=[
            "💼 Portfolio Tracker (Module 1)",
            "📈 Historical Analysis (Module 2)",
            "🔮 Price Forecasting (Module 3)",
        ],
        index=0,
    )
    st.sidebar.markdown("---")

    if active_page == "💼 Portfolio Tracker (Module 1)":
        render_portfolio_sidebar()
        render_portfolio_dashboard()
    elif active_page == "📈 Historical Analysis (Module 2)":
        render_historical_dashboard()
    else:
        render_forecasting_dashboard()


if __name__ == "__main__":
    main()
