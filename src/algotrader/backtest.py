import os
import random
import joblib
import torch
import pandas as pd
from datetime import datetime, timedelta

from algotrader.logger import get_logger
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.prep import get_features, FEATURE_COLUMNS
from algotrader.models.lstm import LSTMTradingNet
from algotrader.train import get_sp500_tickers

logger = get_logger(__name__)
SEQ_LENGTH = 10


def setup_parser(subparsers):
    parser = subparsers.add_parser("backtest", help="Backtest model on a random subset of S&P 500")
    
    # Default to a 1-year backtest ending today
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    parser.add_argument("--num-stocks", type=int, default=10, help="Number of random stocks to backtest")
    parser.add_argument("--start-date", type=str, default=default_start, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=default_end, help="End date (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Probability threshold for trade entry")
    parser.add_argument("--profit-pct", type=float, default=0.10, help="Take profit percentage (e.g., 0.10 for 10%)")
    parser.add_argument("--loss-pct", type=float, default=0.05, help="Stop loss percentage (e.g., 0.05 for 5%)")
    parser.add_argument("--horizon", type=int, default=20, help="Max holding period in days")
    parser.add_argument("--model-prefix", type=str, default="sp500", help="Prefix of the saved model (default: sp500)")
    
    parser.set_defaults(func=handle_backtest)


def handle_backtest(args):
    logger.info(f"Initializing Backtest Engine...")

    # 1. Load the Model and Scaler
    model_path = f"src/algotrader/models/saved/{args.model_prefix}_lstm.pt"
    scaler_path = f"src/algotrader/models/saved/{args.model_prefix}_scaler.joblib"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(f"Required model files not found for '{args.model_prefix}'. Run 'train' first.")
        return

    scaler = joblib.load(scaler_path)
    
    # Initialize model dynamically based on our feature columns
    model = LSTMTradingNet(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Setup Data Sources and Targets
    alpaca = AlpacaDataClient()
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    all_tickers = get_sp500_tickers()
    selected_tickers = random.sample(all_tickers, min(args.num_stocks, len(all_tickers)))
    
    logger.info(f"Backtesting on {len(selected_tickers)} random stocks: {', '.join(selected_tickers)}")

    trades = []

    # 3. Simulate Trades over Historical Data
    for sym in selected_tickers:
        logger.info(f"Running simulation for {sym}...")
        
        # We fetch extra days to cover the rolling windows in get_features()
        data_start_dt = start_dt - timedelta(days=60)
        df = alpaca.get_historical_bars(sym, data_start_dt, end_dt)
        
        if df.empty:
            logger.warning(f"No data available for {sym}. Skipping.")
            continue

        # Extract features and scale them
        df_features = get_features(df)
        if len(df_features) < SEQ_LENGTH + args.horizon:
            logger.warning(f"Not enough clean data for {sym} after feature engineering. Skipping.")
            continue

        X_scaled = scaler.transform(df_features[FEATURE_COLUMNS])

        in_trade = False
        entry_price = 0.0
        entry_date = None
        days_in_trade = 0

        # Step through the dataframe day-by-day to simulate real-time forward execution
        for i in range(SEQ_LENGTH - 1, len(df_features)):
            current_row = df_features.iloc[i]
            current_date = str(current_row.name).split(' ')[0]

            # A. Manage Active Trade
            if in_trade:
                days_in_trade += 1
                
                high = current_row['high']
                low = current_row['low']
                close = current_row['close']

                tp_price = entry_price * (1 + args.profit_pct)
                sl_price = entry_price * (1 - args.loss_pct)

                # Conservative backtesting: If both hit, assume SL hit first.
                if low <= sl_price:
                    profit_pct = (sl_price - entry_price) / entry_price
                    trades.append({'symbol': sym, 'entry_date': entry_date, 'exit_date': current_date, 'entry': entry_price, 'exit': sl_price, 'pnl': profit_pct, 'type': 'Stop Loss', 'days': days_in_trade})
                    in_trade = False
                elif high >= tp_price:
                    profit_pct = (tp_price - entry_price) / entry_price
                    trades.append({'symbol': sym, 'entry_date': entry_date, 'exit_date': current_date, 'entry': entry_price, 'exit': tp_price, 'pnl': profit_pct, 'type': 'Take Profit', 'days': days_in_trade})
                    in_trade = False
                elif days_in_trade >= args.horizon:
                    profit_pct = (close - entry_price) / entry_price
                    trades.append({'symbol': sym, 'entry_date': entry_date, 'exit_date': current_date, 'entry': entry_price, 'exit': close, 'pnl': profit_pct, 'type': 'Time Horizon', 'days': days_in_trade})
                    in_trade = False

            # B. Look for New Entries (If we aren't already holding the stock)
            if not in_trade and i < len(df_features) - 1:
                # Grab the exact sequence the model would have seen on this date
                seq = X_scaled[i - SEQ_LENGTH + 1 : i + 1]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)

                with torch.no_grad():
                    prob = torch.sigmoid(model(seq_tensor)).item()

                if prob >= args.threshold:
                    # Execute at the close price of the signal day
                    in_trade = True
                    entry_price = current_row['close']
                    entry_date = current_date
                    days_in_trade = 0

    # 4. Results Aggregation and Reporting
    if not trades:
        logger.info("No trades met the probability threshold during this backtest period.")
        return

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    avg_pnl = trades_df['pnl'].mean() * 100
    
    # Calculate compounded cumulative PNL assuming 100% reallocation per trade (serial execution)
    cumulative_pnl = ((1 + trades_df['pnl']).prod() - 1) * 100

    logger.info("\n" + "="*40)
    logger.info("          BACKTEST RESULTS")
    logger.info("="*40)
    logger.info(f"Total Trades Taken : {total_trades}")
    logger.info(f"Overall Win Rate   : {win_rate:.2f}%")
    logger.info(f"Avg PNL Per Trade  : {avg_pnl:.2f}%")
    logger.info(f"Compounded Return  : {cumulative_pnl:.2f}%")
    logger.info("="*40)

    # Break down how the trades actually resolved
    breakdown = trades_df['type'].value_counts()
    logger.info("Trade Resolutions:")
    for exit_type, count in breakdown.items():
        logger.info(f"  - {exit_type}: {count} trades")

    logger.info("\nRun complete. (Note: Excludes slippage/commissions)")
