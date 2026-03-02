import os
import copy
import time
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

    def _make_request(self, url: str, max_retries: int = 2) -> requests.Response:
        """
        Internal helper to make requests with automatic retry on 429 (Rate Limit).
        Polygon's free tier limits users to 5 requests per minute.
        """
        wait_time = 60

        for attempt in range(max_retries):
            response = requests.get(url)

            if response.status_code == 429:
                logger.warning(
                    f"Polygon rate limit hit. Retrying in {wait_time}s "
                    f"(Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                return response

        # Return the last response if all retries are exhausted
        return response

    def get_ticker_details(self, symbol: str) -> dict:
        """
        Fetches general company details from Polygon (market cap, employees, etc).
        """
        url = f"{self.base_url}/v3/reference/tickers/{symbol.upper()}?apiKey={self.api_key}"
        response = self._make_request(url)

        if response.status_code == 200:
            return response.json().get("results", {})

        logger.error(
            f"Failed to fetch Polygon ticker details for {symbol}: {response.text}"
        )
        return {}

    def get_historical_financials(self, symbol: str, limit: int = 1) -> list:
        """
        Fetches point-in-time historical quarterly financials.
        Automatically detects missing quarters (e.g., dropped Q4s) and forward-fills
        them using the previous quarter's data to maintain strict ML time-series alignment.
        """
        # Fetch extra records internally (limit * 2) so we have historical buffer data to fill gaps
        url = f"{self.base_url}/vX/reference/financials?ticker={symbol.upper()}&timeframe=quarterly&limit={limit * 2}&apiKey={self.api_key}"
        response = self._make_request(url)

        if response.status_code != 200:
            logger.error(
                f"Failed to fetch Polygon financials for {symbol}: {response.text}"
            )
            return []

        raw_results = response.json().get("results", [])
        if not raw_results:
            return []

        continuous_results = []

        # Iterate from Newest to Oldest to construct a perfect timeline
        for i in range(len(raw_results)):
            continuous_results.append(raw_results[i])

            # Stop if we've successfully gathered the exact amount the user requested
            if len(continuous_results) >= limit:
                break

            # Compare current (newer) report with the next (older) report in the list
            if i + 1 < len(raw_results):
                curr_rep = raw_results[i]
                next_rep = raw_results[i + 1]

                curr_q_str = curr_rep.get("fiscal_period", "")
                curr_y = curr_rep.get("fiscal_year")
                next_q_str = next_rep.get("fiscal_period", "")
                next_y = next_rep.get("fiscal_year")

                # Ensure we have valid quarterly string formats before doing math
                if not (
                    curr_q_str.startswith("Q")
                    and next_q_str.startswith("Q")
                    and curr_y
                    and next_y
                ):
                    continue

                curr_q = int(curr_q_str.replace("Q", ""))

                # Calculate what the next OLDER quarter logically should be
                expected_older_q = curr_q - 1
                expected_older_y = int(curr_y)

                if expected_older_q == 0:
                    expected_older_q = 4
                    expected_older_y -= 1

                next_q = int(next_q_str.replace("Q", ""))

                # Loop to forward-fill gaps (e.g., jump from 2025 Q1 to 2024 Q3 implies missing 2024 Q4)
                while (
                    expected_older_q != next_q or expected_older_y != int(next_y)
                ) and len(continuous_results) < limit:
                    logger.warning(
                        f"Missing SEC filing detected for {symbol}: {expected_older_y} Q{expected_older_q}. "
                        f"Forward-filling from {int(next_y)} Q{next_q}."
                    )

                    # Deep copy the older available quarter and carry its values FORWARD into the missing gap
                    imputed_rep = copy.deepcopy(next_rep)
                    imputed_rep["fiscal_period"] = f"Q{expected_older_q}"
                    imputed_rep["fiscal_year"] = expected_older_y

                    continuous_results.append(imputed_rep)

                    # Decrement expectation again in case multiple quarters in a row are missing
                    expected_older_q -= 1
                    if expected_older_q == 0:
                        expected_older_q = 4
                        expected_older_y -= 1

        return continuous_results[:limit]

    def get_historical_news(self, symbol: str, limit: int = 10) -> list:
        """
        Fetches historical news articles for a given ticker.
        Useful for sentiment analysis and NLP-based machine learning models.
        """
        url = f"{self.base_url}/v2/reference/news?ticker={symbol.upper()}&limit={limit}&apiKey={self.api_key}"
        response = self._make_request(url)

        if response.status_code == 200:
            return response.json().get("results", [])

        logger.error(f"Failed to fetch historical news for {symbol}: {response.text}")
        return []

    def get_historical_dividends(self, symbol: str, limit: int = 100) -> list:
        """
        Fetches historical cash dividend distributions for a given ticker.
        Returns a list containing dividend values, ex-dividend dates, and payment dates.
        """
        url = f"{self.base_url}/v3/reference/dividends?ticker={symbol.upper()}&limit={limit}&apiKey={self.api_key}"
        response = self._make_request(url)

        if response.status_code == 200:
            return response.json().get("results", [])

        logger.error(
            f"Failed to fetch historical dividends for {symbol}: {response.text}"
        )
        return []
