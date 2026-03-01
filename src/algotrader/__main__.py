import argparse
import logging
from datetime import datetime, timedelta

from algotrader.train import handle_train
from algotrader.trade import handle_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Algotrader: Machine learning trading bot using Alpaca and IBKR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands to run",
        required=True,
    )

    # --- 'train' command ---
    train_parser = subparsers.add_parser(
        "train", help="Train the ML model on historical data"
    )
    train_parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol to train on (e.g., AAPL)",
    )

    # Default to 1 year ago for start date
    default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    train_parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help="Start date for training data (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help="End date for training data (YYYY-MM-DD)",
    )
    train_parser.set_defaults(func=handle_train)

    # --- 'trade' command ---
    trade_parser = subparsers.add_parser("trade", help="Run the live/paper trading bot")
    trade_parser.add_argument(
        "--symbol", type=str, required=True, help="Stock symbol to trade (e.g., AAPL)"
    )
    trade_parser.add_argument(
        "--quantity", type=int, default=1, help="Base quantity of shares to trade"
    )
    trade_parser.add_argument(
        "--live",
        action="store_true",
        help="Run in LIVE trading mode (connects to port 7496)",
    )
    trade_parser.set_defaults(func=handle_trade)

    args = parser.parse_args()

    # Execute the associated function based on the command provided
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
