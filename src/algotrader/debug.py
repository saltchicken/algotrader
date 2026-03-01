from datetime import datetime, timedelta
from algotrader.logger import get_logger
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.indicators import add_technical_indicators

logger = get_logger(__name__)


def setup_parser(subparsers):
    """Sets up the argparse subparser for the 'debug' command."""
    parser = subparsers.add_parser("debug", help="Debug and development functionality")

    parser.set_defaults(func=handle_debug)


def handle_debug(args):
    """Handler for the 'debug' command."""
    alpaca = AlpacaDataClient()
    target_symbol = "AAPL"

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)

    logger.info(f"Starting debug run for {target_symbol}")
    df = alpaca.get_historical_bars(target_symbol, start_dt, end_dt)
    df_featured = add_technical_indicators(df)

    # Use logger instead of print
    logger.info(f"\n{df_featured.tail(10)}")
    
    output_file = f"{target_symbol}_features.csv"
    df_featured.tail(50).to_csv(output_file)
    logger.info(f"Saved recent 50 features to {output_file}")
