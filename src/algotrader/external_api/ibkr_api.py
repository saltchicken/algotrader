import os
from dotenv import load_dotenv
from ib_async import IB, Stock, MarketOrder, LimitOrder, StopOrder
from algotrader.logger import get_logger

logger = get_logger(__name__)


class IBKRTradeClient:
    def __init__(self):
        load_dotenv()
        host = os.getenv("IBKR_HOST")
        port = int(os.getenv("IBKR_PORT"))
        client_id = int(os.getenv("IBKR_CLIENT_ID"))

        self.ib = IB()
        logger.info(f"Attempting to connect to IBKR on {host}:{port}...")
        self.ib.connect(host, port, clientId=client_id)
        logger.info("IBKR Connected and ready to trade!")

    def get_cash_balance(self) -> float:
        """
        Retrieves the available funds (cash) for trading.
        """
        logger.info("Fetching account summary to determine cash balance...")

        # accountSummary() fetches account details across all accounts
        summary = self.ib.accountSummary()

        available_funds = None
        total_cash = None

        for item in summary:
            # 'AvailableFunds' is generally the most accurate representation of buying power
            if item.tag == "AvailableFunds" and item.currency == "USD":
                available_funds = float(item.value)
            # 'TotalCashValue' represents settled cash
            elif item.tag == "TotalCashValue" and item.currency == "USD":
                total_cash = float(item.value)

        if available_funds is not None:
            logger.info(f"Current USD Available Funds: ${available_funds:,.2f}")
            return available_funds
        elif total_cash is not None:
            logger.info(f"Current USD Total Cash Value: ${total_cash:,.2f}")
            return total_cash
        else:
            logger.warning("Could not find USD cash balance in account summary.")
            return 0.0

    def place_market_order(self, symbol: str, action: str, quantity: float):
        contract = Stock(symbol.upper(), "SMART", "USD")
        order = MarketOrder(action.upper(), quantity)
        logger.info(f"Placing {action} market order for {quantity} shares of {symbol}")
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        return trade

    def place_bracket_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
    ):
        """Places a Triple Barrier Bracket Order (Entry, Take Profit, Stop Loss)."""
        contract = Stock(symbol.upper(), "SMART", "USD")
        self.ib.qualifyContracts(contract)

        # Parent Entry Order
        parent = MarketOrder(action.upper(), quantity)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False  # Do not transmit yet, wait for children

        # Child Take Profit
        take_profit = LimitOrder(
            "SELL" if action.upper() == "BUY" else "BUY", quantity, take_profit_price
        )
        take_profit.orderId = self.ib.client.getReqId()
        take_profit.parentId = parent.orderId
        take_profit.transmit = False

        # Child Stop Loss
        stop_loss = StopOrder(
            "SELL" if action.upper() == "BUY" else "BUY", quantity, stop_loss_price
        )
        stop_loss.orderId = self.ib.client.getReqId()
        stop_loss.parentId = parent.orderId
        stop_loss.transmit = True  # Transmitting the last child sends the whole bracket

        logger.info(
            f"Placing Bracket Order for {symbol}: Entry Market, TP {take_profit_price}, SL {stop_loss_price}"
        )

        entry_trade = self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, take_profit)
        self.ib.placeOrder(contract, stop_loss)

        self.ib.sleep(1)
        return entry_trade

    def disconnect(self):
        logger.info("Disconnecting from IBKR...")
        self.ib.disconnect()
