import os
import json
import pandas as pd
import numpy as np
from algotrader.logger import get_logger

logger = get_logger(__name__)


def _estimate_filing_date(report: dict) -> pd.Timestamp:
    """
    Estimates a missing SEC filing date based on standard SEC deadlines:
    ~45 days after quarter end for Q1-Q3.
    ~90 days after year end for Q4.
    """
    filing_date = report.get("filing_date")
    if filing_date:
        return pd.to_datetime(filing_date)

    end_date = report.get("end_date")
    period = report.get("fiscal_period", "")
    year = report.get("fiscal_year")

    # Standard SEC filing deadlines
    days_to_file = 90 if period == "Q4" else 45

    if end_date:
        return pd.to_datetime(end_date) + pd.Timedelta(days=days_to_file)

    if year and period:
        # Fallback if both filing_date and end_date are completely missing
        month_map = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}
        month = month_map.get(period, 3)
        # Create a timestamp for the 1st of the month, then jump to the end of the month
        estimated_end = pd.Timestamp(
            year=int(year), month=month, day=1
        ) + pd.offsets.MonthEnd(1)
        return estimated_end + pd.Timedelta(days=days_to_file)

    return pd.NaT


def merge_financials(
    alpaca_df: pd.DataFrame, symbol: str, polygon_dir: str
) -> pd.DataFrame:
    """
    Loads Polygon quarterly financials and merges them with Alpaca daily pricing data.
    Uses merge_asof on the filing_date to prevent look-ahead bias.
    """
    fin_path = os.path.join(polygon_dir, symbol, "financials.json")

    # If no financials exist, return the original dataframe
    if not os.path.exists(fin_path):
        logger.warning(
            f"No financials.json found for {symbol}. Skipping financial merge."
        )
        return alpaca_df

    with open(fin_path, "r") as f:
        raw_data = json.load(f)

    if not raw_data:
        return alpaca_df

    # 1. Flatten the nested JSON structure into a tabular list
    records = []
    for report in raw_data:
        est_filing_date = _estimate_filing_date(report)
        if pd.isna(est_filing_date):
            continue  # Skip if we completely fail to deduce a timeframe

        record = {
            "filing_date": est_filing_date,
            "fiscal_period": report.get("fiscal_period"),
            "fiscal_year": report.get("fiscal_year"),
        }

        # Safely extract financials block
        fin = report.get("financials", {})

        # Income Statement
        inc = fin.get("income_statement", {})
        record["revenue"] = inc.get("revenues", {}).get("value", np.nan)
        record["net_income"] = inc.get("net_income_loss", {}).get("value", np.nan)
        record["operating_expenses"] = inc.get("operating_expenses", {}).get(
            "value", np.nan
        )

        # Balance Sheet
        bs = fin.get("balance_sheet", {})
        record["assets"] = bs.get("assets", {}).get("value", np.nan)
        record["liabilities"] = bs.get("liabilities", {}).get("value", np.nan)
        record["equity"] = bs.get("equity", {}).get("value", np.nan)

        # Cash Flow
        cf = fin.get("cash_flow_statement", {})
        record["net_cash_flow"] = cf.get("net_cash_flow", {}).get("value", np.nan)

        records.append(record)

    fin_df = pd.DataFrame(records)

    # 2. Sort financial data by filing date (required for merge_asof)
    fin_df = fin_df.sort_values("filing_date").dropna(subset=["filing_date"])

    # 3. Prepare Alpaca dataframe
    # Ensure index is a proper datetime index and strictly sorted
    alpaca_df = alpaca_df.copy()
    if not isinstance(alpaca_df.index, pd.DatetimeIndex):
        alpaca_df.index = pd.to_datetime(alpaca_df.index)
    alpaca_df = alpaca_df.sort_index()

    # 4. Point-in-time merge
    # This matches the Alpaca row's index (timestamp) with the most recent filing_date
    # strictly BEFORE or ON the trading day. This prevents lookahead bias and inherently
    # forward-fills missing quarters.
    merged_df = pd.merge_asof(
        alpaca_df, fin_df, left_index=True, right_on="filing_date", direction="backward"
    )

    # Clean up the output index
    merged_df.index = alpaca_df.index

    return merged_df


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
