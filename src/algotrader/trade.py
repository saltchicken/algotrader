from algotrader.logger import get_logger
from algotrader.external_api.ibkr_api import IBKRTradeClient

logger = get_logger(__name__)


def setup_parser(subparsers):
    """Sets up the argparse subparser for the 'trade' command."""
    parser = subparsers.add_parser("trade", help="Run the live/paper trading bot")

    parser.add_argument(
        "--symbol", type=str, required=True, help="Stock symbol to trade (e.g., AAPL)"
    )
    parser.add_argument(
        "--quantity", type=int, default=1, help="Base quantity of shares to trade"
    )

    parser.set_defaults(func=handle_trade)


def handle_trade(args):
    """Handler for the 'trade' command."""
    logger.info(f"Starting trading bot for symbol: {args.symbol}")
    logger.warning("Trade logic is not implemented yet.")

    ibkr = IBKRTradeClient()
    try:
        trade = ibkr.place_market_order(args.symbol, "BUY", args.quantity)
        logger.info("--- Trade Details ---")
        logger.info(f"\n{trade}")

    finally:
        ibkr.disconnect()

    logger.info("Trading bot stopped.")
