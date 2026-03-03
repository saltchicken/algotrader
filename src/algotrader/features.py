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

    # Use 'high' for the rolling max if it exists to catch intraday target hits, otherwise fallback to 'close'
    ref_col = "high" if "high" in df.columns else "close"

    for h in horizons:
        # 1. Use a forward rolling window to find the maximum price over the next 'h' days
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=h)
        future_max = df[ref_col].shift(-1).rolling(window=indexer, min_periods=1).max()

        # 2. Calculate the maximum percentage return achieved at any point during the window
        return_col = f"max_return_future_{h}d"
        df[return_col] = (future_max - df["close"]) / df["close"]

        # 3. Create binary classification targets for each threshold
        valid_horizon_mask = df["close"].shift(-h).notna()

        for t in thresholds:
            target_col = f"target_{h}d_{int(t*100)}pct"
            
            # 1 if max return >= threshold, else 0
            df[target_col] = (df[return_col] >= t).astype(float)
            
            # Re-apply NaNs to the end of the dataset so we don't treat missing future data as a "0" (False)
            df.loc[~valid_horizon_mask, target_col] = pd.NA

    return df
