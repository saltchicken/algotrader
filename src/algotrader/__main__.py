import argparse
import logging

from algotrader.logger import setup_logging, get_logger

from algotrader.train import setup_parser as setup_train_parser
from algotrader.trade import setup_parser as setup_trade_parser
from algotrader.debug import setup_parser as setup_debug_parser

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Algotrader: Machine learning trading bot using Alpaca and IBKR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands to run",
        required=True,
    )

    setup_train_parser(subparsers)
    setup_trade_parser(subparsers)
    setup_debug_parser(subparsers)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
