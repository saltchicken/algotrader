import time
import finviz
from finviz.screener import Screener
from algotrader.logger import get_logger

logger = get_logger(__name__)


class FinvizClient:
    def __init__(self, delay_seconds: float = 3.0):
        """
        Initializes the Finviz API client using the mariostoev/finviz library.
        Includes a rate limiter wrapper to prevent IP bans.

        Args:
            delay_seconds (float): Minimum seconds to wait between requests.
        """
        self.delay_seconds = delay_seconds
        self._last_request_time = 0.0

    def _enforce_rate_limit(self):
        """Blocks execution until the required delay time has passed since the last request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            sleep_time = self.delay_seconds - elapsed
            logger.debug(f"Finviz rate limit active. Pausing for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def get_stock_fundamentals(self, symbol: str) -> dict:
        """
        Fetches the complete fundamental data table (P/E, Debt/Eq, ROE, etc.) for a ticker.
        """
        self._enforce_rate_limit()
        logger.info(f"Fetching fundamental data for {symbol} using finviz library...")

        try:
            # The finviz library returns a dictionary of 90+ data points automatically
            data = finviz.get_stock(symbol.upper())
            logger.info(
                f"Successfully fetched {len(data)} fundamental metrics for {symbol}."
            )
            return data

        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return {}

    def get_screener_tickers(self, filters: list) -> list:
        """
        Navigates Finviz screener pages and extracts all matching tickers.

        Args:
            filters (list): List of Finviz filter codes.
                            (e.g., ['cap_midover', 'fa_pe_u20', 'fa_debteq_u1'])
        """
        self._enforce_rate_limit()
        logger.info(f"Running Finviz screener with filters: {filters}...")

        try:
            # The Screener object automatically handles multi-page results under the hood
            stock_list = Screener(filters=filters)

            # The library returns a list of dictionaries; we just extract the Ticker symbols
            tickers = [stock["Ticker"] for stock in stock_list]

            logger.info(
                f"Screener execution complete. Found {len(tickers)} total matching tickers."
            )
            return tickers

        except Exception as e:
            logger.error(f"Error executing Finviz screener: {e}")
            return []
