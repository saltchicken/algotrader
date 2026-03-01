from datetime import datetime, timedelta
from algotrader.logger import get_logger
from algotrader.external_api.wikipedia_scraper import get_sp500_symbols

from algotrader.external_api.alpaca_api import AlpacaDataClient


logger = get_logger(__name__)


def setup_parser(subparsers):
    """Sets up the argparse subparser for the 'train' command."""
    parser = subparsers.add_parser(
        "train", help="Train the ML model on historical data"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Name of input that should be used for training",
    )

    # Default to 1 year ago for start date
    default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help="End date for training data (YYYY-MM-DD)",
    )

    parser.set_defaults(func=handle_train)


def handle_train(args):
    """Handler for the 'train' command."""
    logger.info(f"Starting training process for input: {args.input}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")

    if args.input == "sp500":
        symbols = get_sp500_symbols()
    elif args.input == "test":
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    else:
        raise ValueError(f"Unknown input: {args.input}")

    model_name = args.input

    logger.info(
        f"Starting LSTM training for {len(symbols)} symbol(s) from {args.start_date} to {args.end_date}"
    )

    alpaca = AlpacaDataClient()
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    for sym in symbols:
        df = alpaca.get_historical_bars(sym, start_dt, end_dt)
        print(df)


    # TODO: Fetch historical data using args.symbol, args.start_date, args.end_date
    # TODO: Apply feature engineering from algotrader.features.indicators
    # TODO: Train scikit-learn model and save it via joblib to the models/ directory

    logger.info("Training complete. Model saved.")
