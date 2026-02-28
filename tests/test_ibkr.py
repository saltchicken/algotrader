from algotrader.external_api.ibkr_api import IBKRTradeClient


def test_ibkr_market_order():
    """Test placing a market order through IBKR."""
    ibkr = IBKRTradeClient()

    try:
        trade = ibkr.place_market_order("AAPL", "BUY", 1)
        print("\n--- Trade Details ---")
        print(trade)

        assert trade is not None
    finally:
        # Putting disconnect in a finally block ensures it runs
        # even if an error occurs during the order
        ibkr.disconnect()


if __name__ == "__main__":
    test_ibkr_market_order()
