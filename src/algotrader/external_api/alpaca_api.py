import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaDataClient:
    def __init__(self):
        load_dotenv()
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = os.getenv("ALPACA_API_SECRET")

        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set in the .env file"
            )

        self.client = StockHistoricalDataClient(
            self.alpaca_api_key, self.alpaca_secret_key
        )

    def get_historical_bars(
        self, symbol: str, start_date: datetime, end_date: datetime
    ):
        """
        Fetches daily historical bars for a given stock symbol.
        """
        print(f"Fetching historical data for {symbol}...")

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )

        bars = self.client.get_stock_bars(request_params)

        # Return the data as a pandas DataFrame for easy viewing/manipulation
        return bars.df
