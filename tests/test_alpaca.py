from datetime import datetime, timedelta
from algotrader.external_api.alpaca_api import AlpacaDataClient

def test_alpaca_historical_data():
    """Test fetching historical data from Alpaca."""
    alpaca = AlpacaDataClient()

    # Define parameters for our data request
    target_symbol = "AAPL"

    # Let's get data for the last 7 days
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=7)

    try:
        # Fetch the dataframe
        df = alpaca.get_historical_bars(target_symbol, start_dt, end_dt)

        print("\n--- Historical Data ---")
        print(df)
        
        # Simple assertion to ensure data is returned if using pytest
        assert df is not None
        assert not df.empty

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    test_alpaca_historical_data()
