import pandas as pd
import numpy as np
from algotrader.logger import get_logger

logger = get_logger(__name__)


def build_bracket_targets(
    df: pd.DataFrame,
    horizon: int = 21,
    take_profit: float = 0.10,
    stop_loss: float = -0.05,
) -> pd.DataFrame:
    """
    Builds target variables by simulating a bracket order.
    For each day, simulates buying at 'close' and setting a take profit and stop loss.
    Checks the subsequent 'horizon' days to see which is hit first.

    Outcomes:
      1: Take Profit hit first
     -1: Stop Loss hit first
      0: Neither hit (expired at horizon)

    Args:
        df: DataFrame containing Alpaca market data ('close', 'high', 'low').
        horizon: Number of trading days to hold the position.
        take_profit: Take profit percentage (e.g., 0.10 for +10%).
        stop_loss: Stop loss percentage (e.g., -0.05 for -5%).

    Returns:
        DataFrame with new bracket target columns appended.
    """
    df = df.copy()

    if "close" not in df.columns:
        logger.error(
            "Dataframe is missing 'close' column required for bracket calculation."
        )
        return df

    # Use NumPy arrays for massive speedup when scanning look-ahead windows
    closes = df["close"].values
    highs = df["high"].values if "high" in df.columns else closes
    lows = df["low"].values if "low" in df.columns else closes

    n = len(df)
    outcomes = np.full(n, np.nan)
    returns = np.full(n, np.nan)
    durations = np.full(n, np.nan)

    for i in range(n - horizon):
        entry_price = closes[i]
        
        # Skip invalid prices
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        tp_price = entry_price * (1 + take_profit)
        sl_price = entry_price * (1 + stop_loss)

        # Look ahead window: tomorrow up to horizon days out
        window_highs = highs[i + 1 : i + 1 + horizon]
        window_lows = lows[i + 1 : i + 1 + horizon]

        tp_hits = window_highs >= tp_price
        sl_hits = window_lows <= sl_price

        # Find the first index where conditions are met.
        # np.argmax returns the first True. If no True exists, we default to the horizon length.
        tp_idx = np.argmax(tp_hits) if np.any(tp_hits) else horizon
        sl_idx = np.argmax(sl_hits) if np.any(sl_hits) else horizon

        if tp_idx == horizon and sl_idx == horizon:
            # Neither hit within the horizon. Trade closes at the end of the window.
            exit_price = closes[i + horizon]
            outcomes[i] = 0
            returns[i] = (exit_price - entry_price) / entry_price
            durations[i] = horizon
            
        elif sl_idx <= tp_idx:
            # Stop loss hit first, or both hit on the very same day.
            # We conservatively assume the SL triggered first on a volatile day.
            outcomes[i] = -1
            returns[i] = stop_loss
            durations[i] = sl_idx + 1
            
        else:
            # Take profit hit first!
            outcomes[i] = 1
            returns[i] = take_profit
            durations[i] = tp_idx + 1

    # Append our features back into the dataframe
    df[f"bracket_outcome_{horizon}d"] = outcomes
    df[f"bracket_return_{horizon}d"] = returns
    df[f"bracket_duration_{horizon}d"] = durations

    return df
