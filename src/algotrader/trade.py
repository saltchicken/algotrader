import os
import torch
import joblib
from datetime import datetime, timedelta
from algotrader.logger import get_logger
from algotrader.external_api.ibkr_api import IBKRTradeClient
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.prep import prepare_and_scale_data
from algotrader.models.lstm import LSTMTradingNet

logger = get_logger(__name__)
SEQ_LENGTH = 10


def setup_parser(subparsers):
    parser = subparsers.add_parser("trade", help="Run the live/paper trading bot")
    parser.add_argument(
        "--symbol", type=str, required=True, help="Stock symbol to trade (e.g., AAPL)"
    )
    parser.add_argument(
        "--quantity", type=int, default=1, help="Quantity of shares to trade"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in LIVE trading mode (connects to port 7496)",
    )
    parser.add_argument(
        "--use-global-model",
        action="store_true",
        help="Use the sp500 generalized model instead of a ticker-specific one",
    )
    parser.set_defaults(func=handle_trade)


def handle_trade(args):
    logger.info(f"Starting LSTM trading bot for symbol: {args.symbol}")

    # 1. Load Model and Scaler (Switching dynamically based on flag)
    model_prefix = "sp500" if args.use_global_model else args.symbol
    model_path = f"src/algotrader/models/saved/{model_prefix}_lstm.pt"
    scaler_path = f"src/algotrader/models/saved/{model_prefix}_scaler.joblib"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(
            f"Saved model '{model_prefix}_lstm.pt' not found. Run 'train' first."
        )
    scaler = joblib.load(scaler_path)

    # 2. Fetch Recent Data
    # Fetch extra days to calculate rolling indicators properly before slicing the 10-day sequence
    alpaca = AlpacaDataClient()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=60)
    df = alpaca.get_historical_bars(args.symbol, start_dt, end_dt)

    # 3. Prepare Data using the loaded scaler
    df_featured, X_scaled = prepare_and_scale_data(df, is_training=False, scaler=scaler)

    # 4. Extract the most recent 10-day sequence
    recent_sequence = X_scaled[-SEQ_LENGTH:]
    X_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)  # Add batch dimension

    # 5. Run Inference
    # Model needs same input_features size as trained (2 in this case: returns, volatility)
    model = LSTMTradingNet(input_size=X_tensor.shape[2])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        raw_output = model(X_tensor)
        probability = torch.sigmoid(raw_output).item()

    logger.info(f"LSTM Probability of 10% move before 5% stop: {probability:.4f}")

    # 6. Trade Execution Logic
    if probability > 0.85:
        logger.info("SNIPER THRESHOLD MET. Initiating Bracket Order.")
        current_price = df_featured["close"].iloc[-1]

        # Calculate Barriers based on your training parameters
        take_profit = round(current_price * 1.10, 2)
        stop_loss = round(current_price * 0.95, 2)

        port = 7496 if args.live else 7497
        ibkr = IBKRTradeClient(port=port)

        try:
            trade = ibkr.place_bracket_order(
                args.symbol, "BUY", args.quantity, take_profit, stop_loss
            )
            logger.info(
                f"Successfully placed bracket order. Current Price: {current_price}"
            )
        finally:
            ibkr.disconnect()
    else:
        logger.info("Signal below 0.85 threshold. Sitting in cash today.")
