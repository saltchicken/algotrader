import logging

logger = logging.getLogger(__name__)


from datetime import datetime, timedelta
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.indicators import add_technical_indicators


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

    df = alpaca.get_historical_bars(target_symbol, start_dt, end_dt)
    df_featured = add_technical_indicators(df)

    print(df_featured)
    df_featured.tail(50).to_csv(f"{target_symbol}_features.csv")
