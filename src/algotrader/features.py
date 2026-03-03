import pandas as pd
from algotrader.logger import get_logger

logger = get_logger(__name__)


def build_targets(
    df: pd.DataFrame, horizons: list = [21], thresholds: list = [0.05, 0.10, 0.20]
) -> pd.DataFrame:
    """
    Builds target variables for future stock returns.

    Args:
        df: DataFrame containing Alpaca market data with a 'close' column.
        horizons: List of integers representing future trading days (e.g., 21 days ~ 1 month).
        thresholds: List of float thresholds for target returns (e.g., 0.05 = 5%).

    Returns:
        DataFrame with new target columns appended.
    """
    df = df.copy()

    if "close" not in df.columns:
        logger.error("Dataframe is missing 'close' column required for target calculation.")
        return df

    for h in horizons:
        # 1. Shift the close price backwards to align future prices with today's row
        future_col = f"close_future_{h}d"
        df[future_col] = df["close"].shift(-h)

        # 2. Calculate the future percentage return
        return_col = f"return_future_{h}d"
        df[return_col] = (df[future_col] - df["close"]) / df["close"]

        # 3. Create binary classification targets for each threshold
        for t in thresholds:
            target_col = f"target_{h}d_{int(t*100)}pct"
            
            # 1 if return >= threshold, else 0
            df[target_col] = (df[return_col] >= t).astype(float)
            
            # Re-apply NaNs to the end of the dataset so we don't treat missing future data as a "0" (False)
            df.loc[df[return_col].isna(), target_col] = pd.NA

    # Optional: Drop the intermediate future_close column if we only care about the return/targets
    df.drop(columns=[f"close_future_{h}d" for h in horizons], inplace=True)

    return df
