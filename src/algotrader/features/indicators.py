import pandas as pd
import numpy as np


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
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()

    # Price Returns
    data["returns"] = data["close"].pct_change()

    # Simple Moving Averages
    data["sma50"] = data["close"].rolling(window=50).mean()
    data["sma200"] = data["close"].rolling(window=200).mean()

    # MACD (Moving Average Convergence Divergence)
    # adjust=False is standard for financial exponential moving averages
    ema_fast = data["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow_period, adjust=False).mean()

    data["macd"] = ema_fast - ema_slow
    data["signal"] = data["macd"].ewm(span=signal_period, adjust=False).mean()

    # Trading Signals (1 for Buy, 0 for neutral/sell)
    data["buy_signal"] = np.where(data["macd"] > data["signal"], 1, 0)
    data["sell_signal"] = np.where(data["macd"] < data["signal"], 1, 0)

    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    # Feature: Distance from closing price to the lower band (useful for mean-reversion)
    data['bb_lower_dist'] = (data['close'] - data['bb_lower']) / data['close']

    # RSI 14-day
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))

    ###
    # --- Machine Learning Features ---
    ###

    # Lagged returns (How did the stock do yesterday, and the day before?)
    data['return_lag_1'] = data['returns'].shift(1)
    data['return_lag_2'] = data['returns'].shift(2)
    data['return_lag_5'] = data['returns'].shift(5) # 1 week ago

    # TARGET VARIABLE: Next day's return (for training ML models)
    # Note: This will be NaN for the very last row (today), which is exactly what you 
    # want when predicting tomorrow's movement in production!
    data['target_next_day_return'] = data['returns'].shift(-1)
    data['target_direction'] = np.where(data['target_next_day_return'] > 0, 1, 0)

    return data
