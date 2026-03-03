import os
import json
from datetime import datetime, timedelta
from algotrader.logger import get_logger
from algotrader.external_api.wikipedia_scraper import get_sp500_symbols
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.external_api.polygon_api import PolygonClient

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

    parser.add_argument(
        "--source",
        type=str,
        choices=["alpaca", "polygon", "all"],
        default="alpaca",
        help="Which data source to ingest from (alpaca, polygon, or all)",
    )

    # Default to 1 year ago for start date
    default_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help="Start date for data ingestion (YYYY-MM-DD) - primarily for Alpaca bars",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help="End date for data ingestion (YYYY-MM-DD) - primarily for Alpaca bars",
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
    logger.info(f"Date range (if applicable): {args.start_date} to {args.end_date}")
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

    # ---------------------------------------------------------
    # 1. Alpaca Ingestion (Pricing Bars)
    # ---------------------------------------------------------
    if args.source in ["alpaca", "all"]:
        logger.info("=== Starting Alpaca Data Ingestion ===")
        alpaca = AlpacaDataClient()
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

        alpaca_dir = os.path.join(args.output_dir, "alpaca")
        os.makedirs(alpaca_dir, exist_ok=True)

        for sym in symbols:
            try:
                df = alpaca.get_historical_bars(sym, start_dt, end_dt)
                if df is not None and not df.empty:
                    output_path = os.path.join(alpaca_dir, f"{sym}.csv")
                    df.to_csv(output_path)
                    logger.info(
                        f"Successfully saved {sym} market data -> {output_path}"
                    )
                else:
                    logger.warning(f"No valid data returned for {sym}. Skipping.")
            except Exception as e:
                logger.error(f"Failed to ingest Alpaca data for {sym}: {e}")

    # ---------------------------------------------------------
    # 2. Polygon Ingestion (Fundamentals, News, Dividends)
    # ---------------------------------------------------------
    if args.source in ["polygon", "all"]:
        logger.info("=== Starting Polygon Data Ingestion ===")
        logger.info(
            "NOTE: This may take a long time due to Polygon's free tier rate limits (5 req/min)."
        )

        poly_client = PolygonClient()
        poly_dir = os.path.join(args.output_dir, "polygon")
        os.makedirs(poly_dir, exist_ok=True)

        for sym in symbols:
            try:
                logger.info(f"Fetching Polygon data for {sym}...")

                # Fetching 20 quarters (~5 years) of historical financials
                details = poly_client.get_ticker_details(sym)
                financials = poly_client.get_historical_financials(sym, limit=20)
                news = poly_client.get_historical_news(sym, limit=50)
                dividends = poly_client.get_historical_dividends(sym, limit=100)

                # Create a subfolder for each ticker
                sym_dir = os.path.join(poly_dir, sym)
                os.makedirs(sym_dir, exist_ok=True)

                # Save as JSON
                with open(os.path.join(sym_dir, "details.json"), "w") as f:
                    json.dump(details, f, indent=4)
                with open(os.path.join(sym_dir, "financials.json"), "w") as f:
                    json.dump(financials, f, indent=4)
                with open(os.path.join(sym_dir, "news.json"), "w") as f:
                    json.dump(news, f, indent=4)
                with open(os.path.join(sym_dir, "dividends.json"), "w") as f:
                    json.dump(dividends, f, indent=4)

                logger.info(f"Successfully saved Polygon data for {sym} -> {sym_dir}/")
            except Exception as e:
                logger.error(f"Failed to ingest Polygon data for {sym}: {e}")

    logger.info("Data ingestion complete. Local dataset is ready.")
