import os
import random
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from algotrader.logger import get_logger
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.prep import get_features, FEATURE_COLUMNS
from algotrader.models.lstm import LSTMTradingNet
from algotrader.external_api.tickers import get_sp500_tickers

logger = get_logger(__name__)
SEQ_LENGTH = 10


def setup_parser(subparsers):
    parser = subparsers.add_parser("backtest", help="Backtest model using a realistic portfolio simulation")
    
    # Default to a 1-year backtest ending today
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    parser.add_argument("--num-stocks", type=int, default=10, help="Number of random stocks to backtest")
    parser.add_argument("--symbol", type=str, default=None, help="Specific stock symbol to backtest (e.g., AAPL). Overrides --num-stocks")
    parser.add_argument("--start-date", type=str, default=default_start, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=default_end, help="End date (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Probability threshold for trade entry")
    parser.add_argument("--profit-pct", type=float, default=0.10, help="Take profit percentage")
    parser.add_argument("--loss-pct", type=float, default=0.05, help="Stop loss percentage")
    parser.add_argument("--horizon", type=int, default=20, help="Max holding period in days")
    parser.add_argument("--model-prefix", type=str, default="sp500", help="Prefix of the saved model")
    
    # Portfolio Configuration
    parser.add_argument("--initial-capital", type=float, default=50000.0, help="Starting portfolio cash balance")
    parser.add_argument("--position-size", type=float, default=0.10, help="Pct of current equity to risk per trade (e.g. 0.10 = 10%)")
    
    parser.set_defaults(func=handle_backtest)


def handle_backtest(args):
    logger.info(f"Initializing Realistic Portfolio Simulation...")
    logger.info(f"Initial Capital: ${args.initial_capital:,.2f} | Position Sizing: {args.position_size*100}% per trade")

    # 1. Load the Model and Scaler
    model_path = f"src/algotrader/models/saved/{args.model_prefix}_lstm.pt"
    scaler_path = f"src/algotrader/models/saved/{args.model_prefix}_scaler.joblib"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(f"Required model files not found for '{args.model_prefix}'. Run 'train' first.")
        return

    scaler = joblib.load(scaler_path)
    
    model = LSTMTradingNet(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Setup Data Sources
    alpaca = AlpacaDataClient()
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.symbol:
        selected_tickers = [args.symbol.upper()]
        logger.info(f"Fetching data for specific stock: {selected_tickers[0]}...")
    else:
        all_tickers = get_sp500_tickers()
        selected_tickers = random.sample(all_tickers, min(args.num_stocks, len(all_tickers)))
        logger.info(f"Fetching data for {len(selected_tickers)} random stocks...")
    
    market_data = {}
    all_dates = set()

    # 3. Phase 1: Pre-compute Signals for all stocks
    for sym in selected_tickers:
        # Fetch extra days to cover rolling windows
        data_start_dt = start_dt - timedelta(days=60)
        df = alpaca.get_historical_bars(sym, data_start_dt, end_dt)
        
        if df.empty:
            continue

        df_features = get_features(df)
        if len(df_features) < SEQ_LENGTH:
            continue
            
        # Clean index to string YYYY-MM-DD for reliable chronological sorting
        df_features['date_str'] = df_features.index.astype(str).str.split(' ').str[0]
        df_features.set_index('date_str', inplace=True)
        
        X_scaled = scaler.transform(df_features[FEATURE_COLUMNS])
        
        df_features['signal'] = False
        
        # Batch inference (Much faster than row-by-row prediction)
        seqs = []
        valid_indices = []
        
        for i in range(SEQ_LENGTH - 1, len(df_features)):
            seqs.append(X_scaled[i - SEQ_LENGTH + 1 : i + 1])
            valid_indices.append(df_features.index[i])
            
        if seqs:
            seq_tensor = torch.FloatTensor(np.array(seqs))
            with torch.no_grad():
                # .flatten() safely converts to 1D array regardless of batch size
                probs = torch.sigmoid(model(seq_tensor)).numpy().flatten()
                
            for idx, prob in zip(valid_indices, probs):
                if prob >= args.threshold:
                    df_features.loc[idx, 'signal'] = True
                    
        market_data[sym] = df_features
        all_dates.update(df_features.index.tolist())

    if not market_data:
        logger.error("No valid market data retrieved. Exiting.")
        return

    # Sort all unique dates to walk forward in time
    sorted_dates = sorted(list(all_dates))

    # 4. Phase 2: Chronological Portfolio Simulation
    cash = args.initial_capital
    equity = args.initial_capital
    positions = {}
    trade_history = []

    logger.info("Executing chronological trade simulation...")

    for current_date in sorted_dates:
        # A. Update Active Positions (Check Exits)
        for sym in list(positions.keys()):
            if current_date not in market_data[sym].index:
                continue # Stock might be halted or data missing for this day
                
            row = market_data[sym].loc[current_date]
            pos = positions[sym]
            pos['days'] += 1
            
            high, low, close = row['high'], row['low'], row['close']
            entry_price = pos['entry_price']

            # Check conservative Stop Loss first
            if low <= pos['sl']:
                exit_price = pos['sl']
                exit_type = 'Stop Loss'
            elif high >= pos['tp']:
                exit_price = pos['tp']
                exit_type = 'Take Profit'
            elif pos['days'] >= args.horizon:
                exit_price = close
                exit_type = 'Time Horizon'
            else:
                continue # Trade remains active

            # Execute the Exit
            cash += pos['shares'] * exit_price
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_dollar = pos['shares'] * (exit_price - entry_price)
            
            trade_history.append({
                'symbol': sym, 'entry_date': pos['entry_date'], 'exit_date': current_date,
                'shares': pos['shares'], 'entry': entry_price, 'exit': exit_price,
                'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar, 'type': exit_type, 'days': pos['days']
            })
            del positions[sym]

        # B. Calculate Current Equity (To size new trades properly)
        current_portfolio_value = cash
        for sym, pos in positions.items():
            if current_date in market_data[sym].index:
                current_portfolio_value += pos['shares'] * market_data[sym].loc[current_date, 'close']
            else:
                current_portfolio_value += pos['shares'] * pos['entry_price'] # Fallback
        equity = current_portfolio_value

        # C. Look for New Entries
        for sym in market_data.keys():
            # Don't buy if we already hold it
            if sym not in positions and current_date in market_data[sym].index:
                row = market_data[sym].loc[current_date]
                
                if row['signal']:
                    trade_allocation = equity * args.position_size
                    close_price = row['close']
                    
                    # Do we have enough cash left? (Capital Constraint)
                    if cash >= trade_allocation:
                        shares = int(trade_allocation // close_price)
                        
                        if shares > 0:
                            cost = shares * close_price
                            cash -= cost
                            positions[sym] = {
                                'shares': shares,
                                'entry_price': close_price,
                                'entry_date': current_date,
                                'tp': close_price * (1 + args.profit_pct),
                                'sl': close_price * (1 - args.loss_pct),
                                'days': 0
                            }

    # 5. Phase 3: Mark-to-Market (Close out any remaining open trades at the end)
    for sym, pos in positions.items():
        last_row = market_data[sym].iloc[-1]
        exit_price = last_row['close']
        cash += pos['shares'] * exit_price
        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        
        trade_history.append({
            'symbol': sym, 'entry_date': pos['entry_date'], 'exit_date': 'Mark-to-Market',
            'shares': pos['shares'], 'entry': pos['entry_price'], 'exit': exit_price,
            'pnl_pct': pnl_pct, 'pnl_dollar': pos['shares'] * (exit_price - pos['entry_price']),
            'type': 'End of Backtest', 'days': pos['days']
        })
    
    final_equity = cash
    
    # 6. Results Aggregation and Reporting
    if not trade_history:
        logger.info("No trades were executed during this period.")
        return

    trades_df = pd.DataFrame(trade_history)
    total_trades = len(trades_df)
    win_rate = (trades_df['pnl_dollar'] > 0).mean() * 100
    avg_pnl_pct = trades_df['pnl_pct'].mean() * 100
    avg_pnl_dollar = trades_df['pnl_dollar'].mean()
    
    # Calculate True Portfolio Return
    total_return_pct = ((final_equity - args.initial_capital) / args.initial_capital) * 100

    logger.info("\n" + "="*50)
    logger.info("             PORTFOLIO BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Initial Capital    : ${args.initial_capital:,.2f}")
    logger.info(f"Final Equity       : ${final_equity:,.2f}")
    logger.info(f"Net Profit         : ${final_equity - args.initial_capital:,.2f}")
    logger.info(f"Total Return       : {total_return_pct:.2f}%")
    logger.info("-" * 50)
    logger.info(f"Total Trades Taken : {total_trades}")
    logger.info(f"Overall Win Rate   : {win_rate:.2f}%")
    logger.info(f"Avg PNL Per Trade  : {avg_pnl_pct:.2f}% (${avg_pnl_dollar:,.2f})")
    logger.info("="*50)

    breakdown = trades_df['type'].value_counts()
    logger.info("Trade Resolutions:")
    for exit_type, count in breakdown.items():
        logger.info(f"  - {exit_type}: {count} trades")

    logger.info("\nRun complete. (Note: Excludes slippage/commissions)")
