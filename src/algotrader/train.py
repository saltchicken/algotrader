import os
import joblib
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from algotrader.logger import get_logger
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.prep import (
    apply_triple_barrier,
    get_features,
    FEATURE_COLUMNS,
    create_lstm_sequences,
)
from algotrader.models.lstm import LSTMTradingNet, train_lstm_with_early_stopping

logger = get_logger(__name__)
SEQ_LENGTH = 10


def get_sp500_tickers():
    """Scrapes the current S&P 500 tickers from Wikipedia."""
    logger.info("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
    tickers = table["Symbol"].tolist()
    # Clean up tickers for Alpaca compatibility (e.g., BRK.B -> BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def setup_parser(subparsers):
    parser = subparsers.add_parser("train", help="Train the LSTM on historical data")
    parser.add_argument(
        "--symbol", type=str, help="Stock symbol to train on (e.g., AAPL)"
    )
    parser.add_argument(
        "--sp500", action="store_true", help="Train on the entire S&P 500 universe"
    )

    # Default to 3 years ago to get enough data for deep learning
    default_start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument("--start-date", type=str, default=default_start)
    parser.add_argument("--end-date", type=str, default=default_end)
    parser.set_defaults(func=handle_train)


def handle_train(args):
    if not args.symbol and not args.sp500:
        logger.error("Must provide either --symbol AAPL or use the --sp500 flag.")
        return

    symbols = get_sp500_tickers() if args.sp500 else [args.symbol]
    model_name = "sp500" if args.sp500 else args.symbol

    logger.info(
        f"Starting LSTM training for {len(symbols)} symbol(s) from {args.start_date} to {args.end_date}"
    )

    # 1. Initialize Clients and Storage
    alpaca = AlpacaDataClient()
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    global_X_train_seqs = []
    global_y_train_seqs = []
    global_X_val_seqs = []
    global_y_val_seqs = []

    global_scaler = StandardScaler()
    all_train_features = []
    stock_datasets = {}

    # 2. Pass 1: Fetch all data, extract unified features, apply barriers
    for sym in symbols:
        try:
            df = alpaca.get_historical_bars(sym, start_dt, end_dt)
            if df.empty or len(df) < 50:
                continue

            # 1. Get features FIRST (this properly handles DropNA from rolling windows)
            df_features = get_features(df)

            # 2. Apply triple barrier on the cleaned feature dataset
            df_labeled = apply_triple_barrier(
                df_features, profit_pct=0.10, loss_pct=0.05, horizon=20
            )

            # 3. Safely map targets to the centralized feature definitions
            X_raw = df_labeled[FEATURE_COLUMNS]
            y = df_labeled["target"]

            # Chronological split index
            split_idx = int(len(X_raw) * 0.8)
            all_train_features.append(X_raw.iloc[:split_idx])

            stock_datasets[sym] = (X_raw, y, split_idx)
            logger.info(f"Successfully processed raw features for {sym}")
        except Exception as e:
            logger.warning(f"Failed to process {sym}: {e}")

    if not stock_datasets:
        logger.error("No data fetched. Exiting.")
        return

    # 3. Fit the Global Scaler on ALL training data combined
    logger.info("Fitting global market scaler...")
    global_train_df = pd.concat(all_train_features)
    global_scaler.fit(global_train_df)

    # 4. Pass 2: Scale each stock, create sequences, and append to global tensors
    for sym, (X_raw, y, split_idx) in stock_datasets.items():
        X_scaled = global_scaler.transform(X_raw)

        X_train_scaled, X_val_scaled = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_t, y_train_t = create_lstm_sequences(
            X_train_scaled, y_train, SEQ_LENGTH
        )
        X_val_t, y_val_t = create_lstm_sequences(X_val_scaled, y_val, SEQ_LENGTH)

        if len(X_train_t) > 0:
            global_X_train_seqs.append(X_train_t)
            global_y_train_seqs.append(y_train_t)
        if len(X_val_t) > 0:
            global_X_val_seqs.append(X_val_t)
            global_y_val_seqs.append(y_val_t)

    # 5. Concatenate massive 3D PyTorch Tensors
    final_X_train = torch.cat(global_X_train_seqs, dim=0)
    final_y_train = torch.cat(global_y_train_seqs, dim=0)
    final_X_val = torch.cat(global_X_val_seqs, dim=0)
    final_y_val = torch.cat(global_y_val_seqs, dim=0)

    logger.info(f"Massive Global Tensor Created: {final_X_train.shape}")

    # 6. Train the Model on the entire S&P 500
    input_features = final_X_train.shape[2]
    model = LSTMTradingNet(input_size=input_features)
    trained_model = train_lstm_with_early_stopping(
        model, final_X_train, final_y_train, final_X_val, final_y_val
    )

    # 7. Save Global Model and Global Scaler
    os.makedirs("src/algotrader/models/saved", exist_ok=True)
    torch.save(
        trained_model.state_dict(), f"src/algotrader/models/saved/{model_name}_lstm.pt"
    )
    joblib.dump(
        global_scaler, f"src/algotrader/models/saved/{model_name}_scaler.joblib"
    )

    logger.info(f"Training complete. Universal model '{model_name}_lstm.pt' saved.")
