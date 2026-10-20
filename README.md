# Cryptocurrency Portfolio Management & Decision-Support System

A modular B.Tech Computer Science capstone project delivering an explainable decision-support system for cryptocurrency portfolio tracking, historical trend analysis, forecasting, and risk analysis.

---

## Architecture & Modules Overview

### Module 1: Real-Time Portfolio Tracking & Asset Allocation
- **Persistent Local Holdings**: Stored in SQLite (`holdings` table).
- **Live Market Data**: CoinGecko REST API (`/simple/price`) denominated in INR (`₹`).
- **Valuation & Allocation Engine**:
  - Individual holding value: $\text{Value}_i = \text{Quantity}_i \times \text{Price}_{\text{INR}}$
  - Total portfolio value: $\text{Total} = \sum_{i} \text{Value}_i$
  - Portfolio allocation weight: $w_i = \frac{\text{Value}_i}{\text{Total}}$ (guaranteed $\sum w_i = 100\%$ when total $> 0$).
- **Interactive Visualization**: Streamlit dashboard with Plotly donut charts.
- **Academic Distinction**: Explicitly separates portfolio weight (concentration) from profit/loss.

### Module 2: Historical Market Analysis & Price Trend Analytics
- **Historical Price Collection**:
  - Queries CoinGecko daily market chart and range endpoints (`/coins/{id}/market_chart`).
  - Normalizes multiple intraday observations into clean UTC daily close prices.
  - Supports 7-day, 30-day, 90-day, 180-day, 1-year, and custom date ranges.
- **Deduplicated SQLite Storage**:
  - Stored in `historical_prices` table with primary key `(coin_id, date)`.
  - Avoids duplicate records through UPSERT (`INSERT ... ON CONFLICT DO UPDATE`).
  - Reuses local records to minimize external network requests and protect rate limits.
- **Statistical Trend & Return Analysis**:
  - Period metrics: Starting price, latest price, period peak, period trough, total period change.
  - Daily percentage return calculation: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100\%$.
  - Descriptive statistics: Average daily return, daily return standard deviation (volatility), maximum single-day gain, and maximum single-day loss.
- **Interactive Plotly Visualizations**:
  - Interactive price line chart with range slider and unified hover inspection.
  - Daily returns distribution bar chart (color-coded positive/negative).
- **Forecasting Preparation & CSV Export**:
  - Enforces strict chronological order without random shuffling (preventing time-series look-ahead leakage).
  - One-click CSV export button for datasets destined for subsequent ARIMA and LSTM forecasting models.

### Module 3: Cryptocurrency Price Forecasting (ARIMA, LSTM & Bi-LSTM)
- **Time-Series Integrity & Zero Look-Ahead Data Leakage**:
  - Chronological train/test split where training strictly precedes test data in time.
  - MinMax normalization scalers are fitted exclusively on the training partition.
- **Statistical Forecasting (ARIMA)**:
  - Fits Auto-Regressive Integrated Moving Average models via `statsmodels`.
  - Parsimonious grid search over candidate $(p, d, q)$ orders guided by Akaike Information Criterion (AIC).
  - Evaluates out-of-sample predictions on held-out test data and projects future horizons.
- **Sequential Deep Learning (LSTM & Bi-LSTM)**:
  - Formulates temporal sequences using dynamic sliding input windows.
  - Unidirectional LSTM captures memory-gated sequential dependencies.
  - Bidirectional LSTM (Bi-LSTM) processes sequences in both chronological and reverse temporal directions.
  - Multi-step forward forecasting via recursive estimation.
- **Standard Evaluation Metrics**:
  - **MAE** (Mean Absolute Error): Average price error in INR ($\text{₹}$).
  - **RMSE** (Root Mean Squared Error): Square root of mean squared errors, penalizing large forecasting deviations.
  - **MAPE** (Mean Absolute Percentage Error): Percentage error normalized against ground truth, guarded against zero division.
- **Multi-Model Comparative Benchmarking**:
  - Side-by-side performance leaderboard ranking models by test-set RMSE.
  - Comparative Plotly projection overlays and error metric bar charts.

---

## Directory Structure

```text
CryptoPortfolioTracker/
├── .env.example              # Template for environment variables (CoinGecko API key if available)
├── .gitignore                # Protects secrets (.env), database files, and bytecode from git
├── README.md                 # Complete project documentation and running instructions
├── requirements.txt          # Pinned dependencies (streamlit, pandas, plotly, statsmodels, scikit-learn, tensorflow, pytest)
├── data/
│   └── portfolio.db          # Local SQLite database (holdings & historical_prices tables)
├── src/
│   ├── __init__.py           # Python package marker
│   ├── config.py             # Central configuration (supported coins, cache TTL, API endpoints)
│   ├── database.py           # SQLite CRUD operations for holdings and historical prices
│   ├── market_data.py        # CoinGecko client (live prices & historical daily charts with caching)
│   ├── portfolio.py          # Portfolio financial calculations (values, weights, allocation %)
│   ├── historical_analysis.py# Daily returns, summary statistics, and time-series data preparation
│   ├── forecasting.py        # ARIMA, LSTM, Bi-LSTM models, evaluation metrics, and comparison engine
│   └── app.py                # Integrated Streamlit dashboard with 3-module navigation
└── tests/
    ├── __init__.py           # Test package marker
    ├── test_database.py      # SQLite CRUD tests for holdings and historical prices
    ├── test_market_data.py   # CoinGecko client tests with network mocks (live & historical)
    ├── test_portfolio.py     # Pure portfolio math and edge-case unit tests
    ├── test_historical_analysis.py # Time-series validation, returns, and statistical tests
    ├── test_forecasting.py   # ARIMA, LSTM, Bi-LSTM, metrics, and splitting unit tests
    └── test_streamlit_compat.py    # Streamlit component signatures and dashboard wiring tests
```

---

## Setup & Installation Instructions (Windows)

### Prerequisites
- Python 3.12 (check using `python --version`)
- Windows PowerShell or Command Prompt

### Step 1: Open Terminal in the Project Directory
```powershell
cd C:\Users\HP\Desktop\CryptoPortfolioTracker
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Configure Environment (Optional)
The CoinGecko free public tier works directly without an API key. If you have a CoinGecko Demo or Pro API key to raise rate limits:
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and enter your key:
   ```env
   COINGECKO_API_KEY=your_coingecko_api_key_here
   ```

---

## Running the Application

To launch the Streamlit web dashboard:
```powershell
python -m streamlit run src/app.py
```

Open your web browser and navigate to:
```text
http://localhost:8501
```

### Using the Application
1. **Module 1 (Portfolio Tracker)**:
   - Use the sidebar to add cryptocurrencies (e.g., Bitcoin, Ethereum, Solana) and specify positive quantities held.
   - Monitor total portfolio valuation in INR and inspect asset allocation percentages and donut charts.
2. **Module 2 (Historical Analysis)**:
   - Switch to **"📈 Historical Analysis (Module 2)"** using the sidebar navigation.
   - Select a cryptocurrency and choose a timeframe (e.g., 30 Days, 90 Days, 1 Year, or Custom Date Range).
   - Click **"📥 Fetch / Sync from CoinGecko"** to query and persist daily observations in SQLite.
   - Inspect key performance statistics (starting, latest, peak, trough, average return, volatility).
   - Explore the interactive price trend line chart and daily returns bar chart.
   - Click **"📥 Export Historical CSV"** to download the clean, chronologically ordered dataset for forecasting.
3. **Module 3 (Price Forecasting)**:
   - Switch to **"🔮 Price Forecasting (Module 3)"** using the sidebar navigation.
   - Select any cryptocurrency with local daily records stored in SQLite.
   - Choose a forward horizon (7, 14, or 30 days).
   - Choose a model (**ARIMA**, **LSTM**, **Bi-LSTM**, or **Compare All Models**).
   - Click **"🚀 Train & Forecast"**:
     - Inspect performance metric cards: MAE, RMSE, MAPE.
     - Explore the interactive Plotly projection showing training history, held-out test actuals vs. predictions, and the future forecast trajectory.
     - Review the out-of-sample test validation table and the future projections table.
     - Download the forward forecast CSV or compare models side-by-side on the leaderboard.

---

## Running Automated Tests

Run the full automated test suite using `pytest`:
```powershell
python -m pytest tests -v
```

This runs **80 automated unit and integration tests** verifying:
- Live holding values, totals, and weights summing to 100%.
- Division-by-zero guards for empty portfolios and zero prices.
- SQLite persistence, update isolation, and deduplication via primary keys.
- Historical date parsing, chronological ordering, and duplicate date resolution.
- Daily returns formulas: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100\%$ with zero-predecessor guards.
- Descriptive statistics (mean, sample standard deviation, max gain, max loss).
- Network mocks simulating HTTP 200, HTTP 429 rate limits, HTTP 404, and network timeouts.
- Chronological train/test splitting and zero look-ahead data leakage guards.
- Evaluation metrics: MAE, RMSE, MAPE with zero-actual protection.
- ARIMA AIC order grid search, convergence, and out-of-sample forecasting.
- LSTM and Bi-LSTM sliding sequence window construction, training, and recursive forecasting.
- Streamlit component parameter compatibility and 3-module dashboard navigation integrity.

---

## CoinGecko API Restrictions & Academic Design Notes

1. **Rate Limiting**:
   - The public CoinGecko API tier allows approximately 10 to 30 requests per minute.
   - An in-memory cache and local SQLite storage protect the system against repetitive queries and HTTP 429 rate limits.
   - If an API rate limit occurs, previously cached records are displayed alongside an informative warning banner. Hardcoded or fabricated prices are **never** substituted.
2. **Time-Series Integrity & No Data Leakage**:
   - Observations are strictly sorted in ascending chronological order without random shuffling.
   - Chronological train/test partitioning guarantees that test data strictly follows training data in time.
   - Normalization transformations (`MinMaxScaler`) are fit exclusively on the training partition to eliminate look-ahead leakage.
3. **Multi-Step Recursive Forecasting Constraints**:
   - For multi-day projections beyond Day 1, models feed their own prior outputs as subsequent inputs.
   - While effective for capturing near-term momentum, prediction variance naturally compounds over longer horizons (14 and 30 days).
