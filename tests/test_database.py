"""Unit tests for SQLite database holdings management."""

import pytest
from src.database import (
    init_db,
    get_holdings,
    add_or_update_holding,
    delete_holding,
    clear_all_holdings,
    save_historical_prices,
    get_historical_prices,
    get_stored_date_range,
    delete_historical_prices,
)


@pytest.fixture
def test_db_path(tmp_path):
    """Provide a temporary database path for isolated unit testing."""
    return tmp_path / "test_portfolio.db"


class TestDatabaseHoldings:
    """Tests for SQLite portfolio holdings CRUD operations."""

    def test_init_and_empty_holdings(self, test_db_path):
        init_db(test_db_path)
        holdings = get_holdings(test_db_path)
        assert holdings == []

    def test_add_holding_and_retrieve(self, test_db_path):
        add_or_update_holding("bitcoin", "BTC", "Bitcoin", 0.75, db_path=test_db_path)
        holdings = get_holdings(test_db_path)
        assert len(holdings) == 1
        assert holdings[0]["coin_id"] == "bitcoin"
        assert holdings[0]["symbol"] == "BTC"
        assert holdings[0]["name"] == "Bitcoin"
        assert holdings[0]["quantity"] == 0.75

    def test_update_existing_holding_quantity(self, test_db_path):
        add_or_update_holding("bitcoin", "BTC", "Bitcoin", 0.5, db_path=test_db_path)
        add_or_update_holding("bitcoin", "BTC", "Bitcoin", 1.25, db_path=test_db_path)
        holdings = get_holdings(test_db_path)
        assert len(holdings) == 1
        assert holdings[0]["quantity"] == 1.25

    def test_delete_holding(self, test_db_path):
        add_or_update_holding("ethereum", "ETH", "Ethereum", 5.0, db_path=test_db_path)
        assert delete_holding("ethereum", db_path=test_db_path) is True
        holdings = get_holdings(test_db_path)
        assert holdings == []

    def test_delete_nonexistent_holding(self, test_db_path):
        assert delete_holding("nonexistent", db_path=test_db_path) is False

    def test_clear_all_holdings(self, test_db_path):
        add_or_update_holding("bitcoin", "BTC", "Bitcoin", 1.0, db_path=test_db_path)
        add_or_update_holding("solana", "SOL", "Solana", 20.0, db_path=test_db_path)
        assert len(get_holdings(test_db_path)) == 2
        clear_all_holdings(test_db_path)
        assert get_holdings(test_db_path) == []

    def test_invalid_negative_or_zero_quantity(self, test_db_path):
        with pytest.raises(ValueError, match="Quantity must be greater than zero"):
            add_or_update_holding("bitcoin", "BTC", "Bitcoin", 0.0, db_path=test_db_path)

        with pytest.raises(ValueError, match="Quantity must be greater than zero"):
            add_or_update_holding("bitcoin", "BTC", "Bitcoin", -2.5, db_path=test_db_path)

    def test_empty_coin_id_raises(self, test_db_path):
        with pytest.raises(ValueError, match="Cryptocurrency ID cannot be empty"):
            add_or_update_holding("", "BTC", "Bitcoin", 1.0, db_path=test_db_path)


class TestHistoricalPricesDatabase:
    """Tests for SQLite historical prices storage and retrieval."""

    def test_save_and_retrieve_historical_prices(self, test_db_path):
        records = [
            {"date": "2026-01-01", "timestamp": 1767225600000, "price_inr": 5000000.0},
            {"date": "2026-01-02", "timestamp": 1767312000000, "price_inr": 5100000.0},
            {"date": "2026-01-03", "timestamp": 1767398400000, "price_inr": 4950000.0},
        ]
        saved_count = save_historical_prices("bitcoin", records, db_path=test_db_path)
        assert saved_count == 3

        retrieved = get_historical_prices("bitcoin", db_path=test_db_path)
        assert len(retrieved) == 3
        assert retrieved[0]["date"] == "2026-01-01"
        assert retrieved[0]["price_inr"] == 5000000.0
        assert retrieved[2]["date"] == "2026-01-03"

    def test_duplicate_date_upsert(self, test_db_path):
        records_1 = [
            {"date": "2026-01-01", "timestamp": 1767225600000, "price_inr": 5000000.0},
        ]
        save_historical_prices("bitcoin", records_1, db_path=test_db_path)

        # Upsert new price for same date
        records_2 = [
            {"date": "2026-01-01", "timestamp": 1767225600000, "price_inr": 5050000.0},
        ]
        save_historical_prices("bitcoin", records_2, db_path=test_db_path)

        retrieved = get_historical_prices("bitcoin", db_path=test_db_path)
        assert len(retrieved) == 1
        assert retrieved[0]["price_inr"] == 5050000.0

    def test_date_range_filtering(self, test_db_path):
        records = [
            {"date": "2026-01-01", "timestamp": 100, "price_inr": 100.0},
            {"date": "2026-01-02", "timestamp": 200, "price_inr": 200.0},
            {"date": "2026-01-03", "timestamp": 300, "price_inr": 300.0},
            {"date": "2026-01-04", "timestamp": 400, "price_inr": 400.0},
        ]
        save_historical_prices("solana", records, db_path=test_db_path)

        filtered = get_historical_prices(
            "solana", start_date="2026-01-02", end_date="2026-01-03", db_path=test_db_path
        )
        assert len(filtered) == 2
        assert filtered[0]["date"] == "2026-01-02"
        assert filtered[1]["date"] == "2026-01-03"

    def test_get_stored_date_range(self, test_db_path):
        # Empty before insert
        min_d, max_d, cnt = get_stored_date_range("ethereum", db_path=test_db_path)
        assert min_d is None
        assert max_d is None
        assert cnt == 0

        records = [
            {"date": "2026-02-10", "timestamp": 1000, "price_inr": 250000.0},
            {"date": "2026-02-20", "timestamp": 2000, "price_inr": 260000.0},
        ]
        save_historical_prices("ethereum", records, db_path=test_db_path)

        min_d, max_d, cnt = get_stored_date_range("ethereum", db_path=test_db_path)
        assert min_d == "2026-02-10"
        assert max_d == "2026-02-20"
        assert cnt == 2

    def test_delete_historical_prices(self, test_db_path):
        records = [{"date": "2026-01-01", "timestamp": 100, "price_inr": 100.0}]
        save_historical_prices("cardano", records, db_path=test_db_path)
        assert len(get_historical_prices("cardano", db_path=test_db_path)) == 1

        deleted = delete_historical_prices("cardano", db_path=test_db_path)
        assert deleted == 1
        assert len(get_historical_prices("cardano", db_path=test_db_path)) == 0
