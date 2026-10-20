"""Configuration settings for Crypto Portfolio Tracker."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")

# API Configuration
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()
REQUEST_TIMEOUT_SECONDS = 10.0
PRICE_CACHE_TTL_SECONDS = 60.0  # Avoid repeated requests to protect rate limits

# Target currency
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "inr").lower()

# Local SQLite Database Path
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "portfolio.db")))

# Curated supported cryptocurrencies with stable CoinGecko IDs
SUPPORTED_COINS: dict[str, dict[str, str]] = {
    "bitcoin": {"name": "Bitcoin", "symbol": "BTC"},
    "ethereum": {"name": "Ethereum", "symbol": "ETH"},
    "solana": {"name": "Solana", "symbol": "SOL"},
    "cardano": {"name": "Cardano", "symbol": "ADA"},
    "ripple": {"name": "XRP", "symbol": "XRP"},
    "binancecoin": {"name": "BNB", "symbol": "BNB"},
    "dogecoin": {"name": "Dogecoin", "symbol": "DOGE"},
    "polkadot": {"name": "Polkadot", "symbol": "DOT"},
    "avalanche-2": {"name": "Avalanche", "symbol": "AVAX"},
    "chainlink": {"name": "Chainlink", "symbol": "LINK"},
    "polygon-ecosystem-token": {"name": "Polygon Ecosystem Token", "symbol": "POL"},
    "shiba-inu": {"name": "Shiba Inu", "symbol": "SHIB"},
    "uniswap": {"name": "Uniswap", "symbol": "UNI"},
    "litecoin": {"name": "Litecoin", "symbol": "LTC"},
    "near": {"name": "NEAR Protocol", "symbol": "NEAR"},
}
