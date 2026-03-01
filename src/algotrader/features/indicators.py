import pandas as pd
import numpy as np
from algotrader.logger import get_logger

logger = get_logger(__name__)


def add_technical_indicators(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Takes a dataframe with a 'close' column and calculates technical indicators.
    Returns a new dataframe with the added feature columns.
    """
    logger.debug("Calculating technical indicators...")
    
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()

    # Price Returns
    data["returns"] = data["close"].pct_change()
    data["direction"] = np.where(data["returns"] > 0, 1, -1)

    # Simple Moving Averages
    data["sma50"] = data["close"].rolling(window=50).mean()
    data["sma200"] = data["close"].rolling(window=200).mean()

    # MACD (Moving Average Convergence Divergence)
    # adjust=False is standard for financial exponential moving averages
    ema_fast = data["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow_period, adjust=False).mean()

    data["macd"] = ema_fast - ema_slow
    data["macd_signal"] = data["macd"].ewm(span=signal_period, adjust=False).mean()

    neutral_threshold = 0.1
    macd_diff = data["macd"] - data["macd_signal"]
    data["macd_trading_signal"] = np.where(
        macd_diff > neutral_threshold,
        1,
        np.where(macd_diff < -neutral_threshold, -1, 0),
    )

    # Bollinger Bands
    data["bb_middle"] = data["close"].rolling(window=20).mean()
    bb_std = data["close"].rolling(window=20).std()
    data["bb_upper"] = data["bb_middle"] + (bb_std * 2)
    data["bb_lower"] = data["bb_middle"] - (bb_std * 2)
    # Feature: Distance from closing price to the lower band (useful for mean-reversion)
    data["bb_lower_dist"] = (data["close"] - data["bb_lower"]) / data["close"]

    # RSI 14-day
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi_14"] = 100 - (100 / (1 + rs))

    ###
    # --- Programmatic Machine Learning Features ---
    ###

    # Set defaults if none provided
    lag_columns = ["returns"]
    lags = [1, 2, 5]

    new_lag_features = {}

    for col in lag_columns:
        if col in data.columns:
            for lag in lags:
                col_name = f"{col}_lag_{lag}"
                new_lag_features[col_name] = data[col].shift(lag)
        else:
            logger.warning(f"Column '{col}' not found. Skipping lags for this feature.")

    data = data.assign(**new_lag_features)

    ###
    # Target Variables
    ###
    data["target_next_day_return"] = data["returns"].shift(-1)
    data["target_direction"] = np.where(data["target_next_day_return"] > 0, 1, -1)

    logger.debug("Successfully added technical indicators.")
    return data
