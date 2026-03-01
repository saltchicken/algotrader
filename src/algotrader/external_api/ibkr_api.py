from ib_async import IB, Stock, MarketOrder
from algotrader.logger import get_logger

logger = get_logger(__name__)


class IBKRTradeClient:
    """
    Extracted client for handling Interactive Brokers API connections
    and trade execution using ib_async.
    """

    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.ib = IB()

        # Connect to Trader Workstation (TWS) or IB Gateway
        # Port 7497 is the default for Paper Trading. Use 7496 for Live.
        logger.info(f"Attempting to connect to IBKR on {host}:{port}...")
        self.ib.connect(host, port, clientId=client_id)
        logger.info("IBKR Connected and ready to trade!")

        # or manual nextValidId polling. ib_async handles this internally.

    def place_market_order(self, symbol: str, action: str, quantity: float):
        """
        Executes a market order for a given stock symbol.
        """
        contract = Stock(symbol.upper(), "SMART", "USD")
        order = MarketOrder(action.upper(), quantity)

        logger.info(f"Placing {action} market order for {quantity} shares of {symbol}")

        trade = self.ib.placeOrder(contract, order)

        # Wait a moment for the network to sync the initial state before returning
        self.ib.sleep(1)

        return trade

    def disconnect(self):
        """Gracefully close the API connection."""
        logger.info("Disconnecting from IBKR...")
        self.ib.disconnect()
