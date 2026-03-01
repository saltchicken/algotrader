from algotrader.logger import setup_logging, get_logger

from algotrader.external_api.finviz_api import FinvizClient

setup_logging()
logger = get_logger(__name__)


def setup_parser(subparsers):
    """Sets up the argparse subparser for the research command."""
    parser = subparsers.add_parser("research", help="Research using the Finviz API client")
    parser.add_argument(
        "--symbol", type=str, default="AAPL", help="Stock symbol to fetch (default: AAPL)"
    )
    parser.set_defaults(func=handle_research)


def handle_research(args):
    """Handler for the 'test-finviz' command."""
    logger.info("Starting Finviz API Test...")
    client = FinvizClient()
    
    # Test 1: Fetch Fundamentals
    logger.info(f"--- Test 1: Fetching Fundamentals for {args.symbol} ---")
    fundamentals = client.get_stock_fundamentals(args.symbol)
    if fundamentals:
        logger.info(f"P/E Ratio: {fundamentals.get('P/E')}")
        logger.info(f"Market Cap: {fundamentals.get('Market Cap')}")
        logger.info(f"ROE: {fundamentals.get('ROE')}")
    else:
        logger.warning("Failed to fetch fundamentals.")

    # # Test 2: Screener
    # logger.info("--- Test 2: Running Screener ---")
    # # Test filters: Mid/Large Cap, P/E < 20, Debt/Eq < 1
    # test_filters = ['cap_midover', 'fa_pe_u20', 'fa_debteq_u1'] 
    # logger.info(f"Using filters: {test_filters}")
    # tickers = client.get_screener_tickers(test_filters)
    #
    # if tickers:
    #     logger.info(f"Screener found {len(tickers)} tickers.")
    #     logger.info(f"First 10 matches: {tickers[:10]}")
    # else:
    #     logger.warning("Screener returned no results.")


