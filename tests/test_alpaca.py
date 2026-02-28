from datetime import datetime, timedelta
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.indicators import add_technical_indicators


def test_alpaca_historical_data_and_features():
    """Test fetching historical data from Alpaca and generating features."""
    alpaca = AlpacaDataClient()
    target_symbol = "AAPL"

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)

    try:
        df = alpaca.get_historical_bars(target_symbol, start_dt, end_dt)

        assert df is not None
        assert not df.empty
        assert "close" in df.columns

        df_featured = add_technical_indicators(df)

        print(df_featured)
        df_featured.tail(50).to_csv(f"{target_symbol}_features.csv")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    test_alpaca_historical_data_and_features()
