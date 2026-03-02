import os
import requests
from dotenv import load_dotenv
from algotrader.logger import get_logger

logger = get_logger(__name__)


class PolygonClient:
    def __init__(self):
        """
        Initializes the Polygon API client.
        Requires POLYGON_API_KEY to be set in the .env file.
        """
        load_dotenv()
        self.api_key = os.getenv("POLYGON_API_KEY")

        if not self.api_key:
            raise ValueError(
                "POLYGON_API_KEY must be set in the .env file to use Polygon data."
            )

        self.base_url = "https://api.polygon.io"

    def get_ticker_details(self, symbol: str) -> dict:
        """
        Fetches general company details from Polygon (market cap, employees, etc).
        """
        url = f"{self.base_url}/v3/reference/tickers/{symbol.upper()}?apiKey={self.api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json().get("results", {})

        logger.error(
            f"Failed to fetch Polygon ticker details for {symbol}: {response.text}"
        )
        return {}

    def get_historical_financials(self, symbol: str, limit: int = 1) -> list:
        """
        Fetches point-in-time historical financial statements (Income, Balance Sheet, Cash Flow).
        Useful for preventing look-ahead bias in ML models.

        Args:
            symbol (str): The stock ticker (e.g., 'AAPL')
            limit (int): Number of historical financial reports to return.
        """
        url = f"{self.base_url}/vX/reference/financials?ticker={symbol.upper()}&timeframe=quarterly&limit={limit}&apiKey={self.api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json().get("results", [])

        logger.error(
            f"Failed to fetch Polygon financials for {symbol}: {response.text}"
        )
        return []
