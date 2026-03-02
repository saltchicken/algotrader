import os
from datetime import datetime, timedelta
from algotrader.logger import get_logger
from algotrader.external_api.wikipedia_scraper import get_sp500_symbols
from algotrader.external_api.alpaca_api import AlpacaDataClient

logger = get_logger(__name__)


def setup_parser(subparsers):
    """Sets up the argparse subparser for the 'ingest' command."""
    parser = subparsers.add_parser(
        "ingest", help="Download historical data and build local dataset"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input source to ingest (e.g., 'sp500', 'test', or comma-separated symbols like 'AAPL,MSFT')",
    )

    # Default to 1 year ago for start date
    default_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help="Start date for data ingestion (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help="End date for data ingestion (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Local directory to save the ingested dataset",
    )

    parser.set_defaults(func=handle_ingest)


def handle_ingest(args):
    """Handler for the 'ingest' command."""
    logger.info(f"Starting data ingestion process for input: {args.input}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.warning("Ingest logic is not implemented yet to handle survivorship bias")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input.lower() == "sp500":
        symbols = get_sp500_symbols()
    elif args.input.lower() == "test":
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    else:
        symbols = [s.strip().upper() for s in args.input.split(",")]

    logger.info(
        f"Preparing to ingest data for {len(symbols)} symbol(s) into '{args.output_dir}/'"
    )

    alpaca = AlpacaDataClient()
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    for sym in symbols:
        try:
            df = alpaca.get_historical_bars(sym, start_dt, end_dt)
            if df is not None and not df.empty:
                output_path = os.path.join(args.output_dir, f"{sym}.csv")
                df.to_csv(output_path)
                logger.info(f"Successfully saved {sym} -> {output_path}")
            else:
                logger.warning(f"No valid data returned for {sym}. Skipping.")
        except Exception as e:
            logger.error(f"Failed to ingest data for {sym}: {e}")

    logger.info("Data ingestion complete. Local dataset is ready.")
