import logging

logger = logging.getLogger(__name__)


def setup_parser(subparsers):
    """Sets up the argparse subparser for the 'trade' command."""
    parser = subparsers.add_parser("trade", help="Run the live/paper trading bot")

    parser.add_argument(
        "--symbol", type=str, required=True, help="Stock symbol to trade (e.g., AAPL)"
    )
    parser.add_argument(
        "--quantity", type=int, default=1, help="Base quantity of shares to trade"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in LIVE trading mode (connects to port 7496)",
    )

    parser.set_defaults(func=handle_trade)


def handle_trade(args):
    """Handler for the 'trade' command."""
    logger.info(f"Starting trading bot for symbol: {args.symbol}")
    logger.info(f"Mode: {'LIVE' if args.live else 'PAPER'}")

    # TODO: Load pre-trained model from models/ directory
    # TODO: Initialize IBKRTradeClient
    # TODO: Connect to IBKR, monitor real-time data, and execute trades based on model predictions

    logger.info("Trading bot stopped.")
