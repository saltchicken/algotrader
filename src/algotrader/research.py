from algotrader.logger import setup_logging, get_logger
from algotrader.external_api.finviz_api import FinvizClient
from algotrader.external_api.polygon_api import PolygonClient

setup_logging()
logger = get_logger(__name__)


def setup_parser(subparsers):
    """Sets up the argparse subparser for the research command."""
    parser = subparsers.add_parser(
        "research", help="Research using Finviz and Polygon APIs"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Stock symbol to fetch (default: AAPL)",
    )
    parser.set_defaults(func=handle_research)


def handle_research(args):
    """Handler for the 'research' command."""
    logger.info("Starting Research API Tests...")

    # --- Test 1: Fetch Current Fundamentals (Finviz) ---
    logger.info(
        f"\n--- Test 1: Fetching Current Fundamentals for {args.symbol} (Finviz) ---"
    )
    finviz_client = FinvizClient()
    fundamentals = finviz_client.get_stock_fundamentals(args.symbol)

    if fundamentals:
        logger.info("=== Company Profile ===")
        logger.info(f"Company:  {fundamentals.get('Company')}")
        logger.info(
            f"Sector:   {fundamentals.get('Sector')} | Industry: {fundamentals.get('Industry')}"
        )
        logger.info(f"Country:  {fundamentals.get('Country')}")

        logger.info("=== Valuation Metrics ===")
        logger.info(f"Market Cap:  {fundamentals.get('Market Cap')}")
        logger.info(
            f"P/E Ratio:   {fundamentals.get('P/E')} | Forward P/E: {fundamentals.get('Forward P/E')}"
        )
        logger.info(f"PEG Ratio:   {fundamentals.get('PEG')}")
        logger.info(
            f"Price/Book:  {fundamentals.get('P/B')} | Price/Sales: {fundamentals.get('P/S')}"
        )

        logger.info("=== Profitability & Growth ===")
        logger.info(
            f"ROE:           {fundamentals.get('ROE')} | ROA: {fundamentals.get('ROA')}"
        )
        logger.info(f"Profit Margin: {fundamentals.get('Profit Margin')}")
        logger.info(
            f"EPS Q/Q:       {fundamentals.get('EPS Q/Q')} | Sales Q/Q: {fundamentals.get('Sales Q/Q')}"
        )

        logger.info("=== Financial Health ===")
        logger.info(f"Debt/Equity:   {fundamentals.get('Debt/Eq')}")
        logger.info(f"Current Ratio: {fundamentals.get('Current Ratio')}")

        logger.info("=== Market Sentiment & Technicals ===")
        logger.info(f"Beta:        {fundamentals.get('Beta')}")
        logger.info(f"Short Float: {fundamentals.get('Short Float')}")
        logger.info(f"Volatility:  {fundamentals.get('Volatility')}")
        logger.info(f"RSI (14):    {fundamentals.get('RSI (14)')}")
        logger.info(f"Rel Volume:  {fundamentals.get('Rel Volume')}")
    else:
        logger.warning("Failed to fetch Finviz fundamentals.")

    # --- Test 2: Fetch Historical Point-in-Time Data (Polygon) ---
    logger.info(
        f"\n--- Test 2: Fetching Historical Data for {args.symbol} (Polygon) ---"
    )
    try:
        poly_client = PolygonClient()

        # 2a. Fetch Ticker Details
        details = poly_client.get_ticker_details(args.symbol)
        if details:
            logger.info("=== Polygon Ticker Details ===")
            logger.info(
                f"Name: {details.get('name')} | Market Cap: {details.get('market_cap')}"
            )
            logger.info(
                f"Employees: {details.get('total_employees')} | Homepage: {details.get('homepage_url')}"
            )

        # 2b. Fetch Point-in-Time Financials (Past 2 Years / 8 Quarters)
        financials = poly_client.get_historical_financials(args.symbol, limit=8)
        if financials:
            logger.info("=== Polygon Quarterly Financials (Past 2 Years) ===")
            for report in financials:
                period = report.get("fiscal_period")
                year = report.get("fiscal_year")

                # Drill down into the raw financial statement structure
                income_stmt = report.get("financials", {}).get("income_statement", {})
                revenue = income_stmt.get("revenues", {}).get("value")
                net_income = income_stmt.get("net_income_loss", {}).get("value")

                rev_str = f"${revenue:,.2f}" if revenue else "N/A"
                ni_str = f"${net_income:,.2f}" if net_income else "N/A"

                logger.info(
                    f"{year} {period} | Rev: {rev_str:>16} | Net Income: {ni_str:>16}"
                )
        else:
            logger.warning("No financial data found via Polygon.")

        # 2c. Fetch Historical News
        logger.info(
            f"\n--- Test 3: Fetching Historical News for {args.symbol} (Polygon) ---"
        )
        news = poly_client.get_historical_news(args.symbol, limit=50)
        if news:
            logger.info("=== Recent News Headlines ===")
            for article in news:
                # Extract just the YYYY-MM-DD from the timestamp
                pub_date = article.get("published_utc", "")[:10]
                title = article.get("title", "No Title")
                publisher = article.get("publisher", {}).get("name", "Unknown")
                logger.info(f"[{pub_date}] {publisher}: {title}")
        else:
            logger.warning("No historical news found via Polygon.")

        # 2d. Fetch Dividends
        logger.info(
            f"\n--- Test 4: Fetching Dividends for {args.symbol} (Polygon) ---"
        )
        dividends = poly_client.get_historical_dividends(args.symbol)
        if dividends:
            logger.info("=== Recent Dividend Payments ===")
            for div in dividends:
                ex_date = div.get("ex_dividend_date", "")
                record_date = div.get("record_date", "")
                payment_date = div.get("pay_date", "")
                amount = div.get("cash_amount", 0.0)
                logger.info(
                    f"Ex Date: {ex_date} | Record Date: {record_date} | Payment Date: {payment_date} | Amount: ${amount:,.2f}"
                )
        else:
            logger.warning("No dividend data found via Polygon.")




    except ValueError as e:
        logger.warning(f"Skipping Polygon tests: {e}")

    # # Test 5: Screener (Finviz)
    # logger.info("\n--- Test 4: Running Screener (Finviz) ---")
    # # Test filters: Mid/Large Cap, P/E < 20, Debt/Eq < 1
    # test_filters = ['cap_midover', 'fa_pe_u20', 'fa_debteq_u1']
    # logger.info(f"Using filters: {test_filters}")
    # tickers = finviz_client.get_screener_tickers(test_filters)
    #
    # if tickers:
    #     logger.info(f"Screener found {len(tickers)} tickers.")
    #     logger.info(f"First 10 matches: {tickers[:10]}")
    # else:
    #     logger.warning("Screener returned no results.")
