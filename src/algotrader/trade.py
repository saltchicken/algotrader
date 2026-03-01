import os
import torch
import joblib
from datetime import datetime, timedelta
from algotrader.logger import get_logger
from algotrader.external_api.ibkr_api import IBKRTradeClient
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.prep import prepare_and_scale_data, FEATURE_COLUMNS
from algotrader.models.lstm import LSTMTradingNet
from algotrader.external_api.tickers import get_sp500_tickers

logger = get_logger(__name__)
SEQ_LENGTH = 10


def setup_parser(subparsers):
    parser = subparsers.add_parser("trade", help="Run the live/paper trading bot")
    parser.add_argument(
        "--symbol", type=str, help="Stock symbol to trade (e.g., AAPL)"
    )
    parser.add_argument(
        "--sp500", action="store_true", help="Scan and trade the entire S&P 500"
    )
    parser.add_argument(
        "--quantity", type=int, default=1, help="Quantity of shares to trade"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85, help="Probability threshold for trade entry"
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
    if not args.symbol and not args.sp500:
        logger.error("Must provide either --symbol AAPL or use the --sp500 flag.")
        return

    symbols = get_sp500_tickers() if args.sp500 else [args.symbol]

    logger.info(f"Starting LSTM trading bot for {len(symbols)} symbol(s)...")

    # 1. Load Model and Scaler (Switching dynamically based on flag)
    model_prefix = "sp500" if (args.use_global_model or args.sp500) else args.symbol
    model_path = f"src/algotrader/models/saved/{model_prefix}_lstm.pt"
    scaler_path = f"src/algotrader/models/saved/{model_prefix}_scaler.joblib"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(
            f"Saved model '{model_prefix}_lstm.pt' not found. Run 'train' first."
        )
        return
    
    scaler = joblib.load(scaler_path)

    # Pre-load model once
    model = LSTMTradingNet(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Fetch Recent Data
    alpaca = AlpacaDataClient()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=60)

    signals = []

    for sym in symbols:
        try:
            df = alpaca.get_historical_bars(sym, start_dt, end_dt)
            
            if df.empty:
                continue

            # 3. Prepare Data using the loaded scaler
            df_featured, X_scaled = prepare_and_scale_data(df, is_training=False, scaler=scaler)

            if len(X_scaled) < SEQ_LENGTH:
                continue

            # 4. Extract the most recent 10-day sequence
            recent_sequence = X_scaled[-SEQ_LENGTH:]
            X_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)  # Add batch dimension

            # 5. Run Inference
            with torch.no_grad():
                raw_output = model(X_tensor)
                probability = torch.sigmoid(raw_output).item()

            if probability >= args.threshold:
                current_price = df_featured["close"].iloc[-1]
                logger.info(f"[{sym}] Signal: {probability:.4f} >= {args.threshold} - ADDING TO TRADE LIST")
                signals.append((sym, probability, current_price))
                
        except Exception as e:
            logger.warning(f"Error processing {sym}: {e}")

    # 6. Trade Execution Logic
    if signals:
        # Sort by highest probability first to execute the best setups
        signals.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(signals)} actionable trade signals. Connecting to IBKR...")

        port = 7496 if args.live else 7497
        ibkr = IBKRTradeClient(port=port)

        try:
            for sym, prob, current_price in signals:
                # Calculate Barriers based on your training parameters
                take_profit = round(current_price * 1.10, 2)
                stop_loss = round(current_price * 0.95, 2)

                trade = ibkr.place_bracket_order(
                    sym, "BUY", args.quantity, take_profit, stop_loss
                )
                logger.info(
                    f"Successfully placed bracket order for {sym} (Prob: {prob:.4f}). Current Price: {current_price}"
                )
        finally:
            ibkr.disconnect()
    else:
        logger.info(f"No signals met the {args.threshold} threshold today. Sitting in cash.")
