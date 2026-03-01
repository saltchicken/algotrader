import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Centralize feature columns so train.py and trade.py always stay perfectly synced
FEATURE_COLUMNS = [
    "returns",
    "volatility",
    "rsi",
    "dist_sma20",
]


def get_features(df):
    """Calculates indicators used for both training and trading."""
    data = df.copy()

    # Alpaca multi-index cleanup if necessary
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index(level=0, drop=True)

    # Standardize column names to lowercase
    data.columns = [c.lower() for c in data.columns]

    # Feature Engineering
    data["returns"] = data["close"].pct_change()
    data["volatility"] = data["returns"].rolling(window=20).std()

    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))

    # Add Moving Average Distance
    data["sma_20"] = data["close"].rolling(window=20).mean()
    data["dist_sma20"] = (data["close"] - data["sma_20"]) / data["sma_20"]

    data["volume_change"] = data["volume"].pct_change()

    # 2. MACD (Measures trend acceleration/deceleration)
    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    # 3. Bollinger Band Width (Measures volatility squeezes)
    rolling_std = data["close"].rolling(window=20).std()
    data["bb_width"] = (rolling_std * 4) / data["sma_20"]

    # 4. Normalized Average True Range (Volatility proxy)
    # Using High - Low as a simplified daily True Range, normalized by close price
    true_range = data["high"] - data["low"]
    data["atr_norm"] = true_range.rolling(window=14).mean() / data["close"]

    # Drop rows with NaN from rolling calculations
    data = data.dropna()
    return data


def apply_triple_barrier(df, profit_pct=0.10, loss_pct=0.05, horizon=20):
    """Labels targets using the Triple Barrier Method."""
    targets = np.zeros(len(df))
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(len(df) - horizon):
        entry = closes[i]
        upper = entry * (1 + profit_pct)
        lower = entry * (1 - loss_pct)

        for j in range(1, horizon + 1):
            idx = i + j
            if lows[idx] <= lower:
                targets[i] = 0
                break
            elif highs[idx] >= upper:
                targets[i] = 1
                break

    df_res = df.copy()
    df_res["target"] = targets
    return df_res.iloc[:-horizon]


def prepare_and_scale_data(df, is_training=True, scaler=None):
    """Calculates indicators and scales the features."""
    data = get_features(df)
    X_raw = data[FEATURE_COLUMNS]

    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        return data, X_scaled, scaler
    else:
        X_scaled = scaler.transform(X_raw)
        return data, X_scaled


def create_lstm_sequences(scaled_data, target_labels, sequence_length=10):
    """Slices 2D data into 3D tensors."""
    X_seq, y_seq = [], []
    for i in range(len(scaled_data) - sequence_length):
        X_seq.append(scaled_data[i : i + sequence_length])
        y_seq.append(target_labels.iloc[i + sequence_length])

    X_tensor = torch.FloatTensor(np.array(X_seq))
    y_tensor = torch.FloatTensor(np.array(y_seq).reshape(-1, 1))
    return X_tensor, y_tensor
