import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

from algotrader.logger import get_logger

logger = get_logger(__name__)


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
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        max_retries: int = 5,
    ) -> pd.DataFrame:
        """
        Fetches daily historical bars for a given stock symbol.
        Includes an Exponential Backoff strategy to handle Alpaca's API rate limits
        (429 Too Many Requests) without crashing the broader application.
        """
        logger.info(
            f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}..."
        )

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )

        base_wait = 2  # The initial wait time in seconds

        for attempt in range(max_retries):
            try:
                bars = self.client.get_stock_bars(request_params)

                if bars.df.empty:
                    logger.warning(
                        f"No market data found for {symbol} in the requested timeframe."
                    )
                    return None

                logger.info(f"Successfully fetched {len(bars.df)} bars for {symbol}")
                return bars.df

            except APIError as e:
                # 429 is the standard HTTP code for Rate Limit Exceeded
                if e.status_code == 429:
                    wait_time = base_wait * (2**attempt)
                    logger.warning(
                        f"Rate limit (429) hit for {symbol}. "
                        f"Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    # If it's a 403 Forbidden or other error, do not retry. Just log it.
                    logger.error(f"Alpaca APIError fetching {symbol}: {e}")
                    break

            except Exception as e:
                # Fallback catch in case the underlying requests library throws a raw connection error
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = base_wait * (2**attempt)
                    logger.warning(
                        f"Rate limit hit for {symbol}. "
                        f"Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error fetching {symbol}: {e}")
                    break

        # If the loop finishes without returning, it means all retries were exhausted
        logger.error(
            f"Failed to fetch data for {symbol} after {max_retries} attempts. Skipping asset."
        )
        return None
