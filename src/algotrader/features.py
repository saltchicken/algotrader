import pandas as pd
from algotrader.logger import get_logger

logger = get_logger(__name__)


def build_targets(
    df: pd.DataFrame,
    horizons: list = [21],
    thresholds: list = [0.05, 0.10, 0.20],
    loss_thresholds: list = [-0.05, -0.10, -0.20],
) -> pd.DataFrame:
    """
    Builds target variables for future stock returns.

    Args:
        df: DataFrame containing Alpaca market data with a 'close' column.
        horizons: List of integers representing future trading days (e.g., 21 days ~ 1 month).
        thresholds: List of float thresholds for positive target returns (e.g., 0.05 = 5%).
        loss_thresholds: List of float thresholds for negative target returns (e.g., -0.05 = -5%).

    Returns:
        DataFrame with new target columns appended.
    """
    df = df.copy()

    if "close" not in df.columns:
        logger.error(
            "Dataframe is missing 'close' column required for target calculation."
        )
        return df

    # Use 'high' for rolling max and 'low' for rolling min to catch intraday target hits
    high_col = "high" if "high" in df.columns else "close"
    low_col = "low" if "low" in df.columns else "close"

    for h in horizons:
        # 1. Use a forward rolling window to find the max and min prices over the next 'h' days
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=h)

        future_max = df[high_col].shift(-1).rolling(window=indexer, min_periods=1).max()
        future_min = df[low_col].shift(-1).rolling(window=indexer, min_periods=1).min()

        # 2. Calculate the max positive return and max drawdown achieved during the window
        max_return_col = f"max_return_future_{h}d"
        min_return_col = f"min_return_future_{h}d"

        df[max_return_col] = (future_max - df["close"]) / df["close"]
        df[min_return_col] = (future_min - df["close"]) / df["close"]

        # 3. Create binary classification targets for each threshold
        valid_horizon_mask = df["close"].shift(-h).notna()

        for t in thresholds:
            target_col = f"target_{h}d_{int(t*100)}pct"

            # 1 if max return >= threshold, else 0
            df[target_col] = (df[max_return_col] >= t).astype(float)

            # Re-apply NaNs to the end of the dataset
            df.loc[~valid_horizon_mask, target_col] = pd.NA

        for t in loss_thresholds:
            # 1 if min return <= negative threshold, else 0
            # E.g., target_63d_loss_5pct
            target_col = f"target_{h}d_loss_{abs(int(t*100))}pct"

            df[target_col] = (df[min_return_col] <= t).astype(float)

            # Re-apply NaNs to the end of the dataset
            df.loc[~valid_horizon_mask, target_col] = pd.NA

    return df
