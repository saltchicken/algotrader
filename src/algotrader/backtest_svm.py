import os
from datetime import datetime, timedelta
import joblib
import pandas as pd

from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import PandasDataBacktesting
from lumibot.entities import Asset, Data

# Import your feature engineering module
from algotrader.features.indicators import add_technical_indicators
from algotrader.external_api.alpaca_api import AlpacaDataClient

class SVMStrategy(Strategy):
    def initialize(self):
        """
        Initialization runs once when the strategy starts.
        We set up our asset and load the machine learning model.
        """
        self.symbol = "MSFT"
        self.sleeptime = "1D" # Run once a day
        
        # 1. Load the trained SVM model and the StandardScaler trained on S&P 500
        model_path = f"saved_models/sp500_universe_svm_model.joblib"
        scaler_path = f"saved_models/sp500_universe_scaler.joblib"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Model or scaler not found for {self.symbol}. "
                "Please run train_svm.py first to generate them!"
            )
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # 2. Define the exact features the model was trained on
        self.feature_columns = [
            'macd_trading_signal',
            'bb_lower_dist', 'rsi_14', 'returns_lag_1', 'returns_lag_2', 'returns_lag_5'
        ]

    def on_trading_iteration(self):
        """
        This method runs every trading day during the backtest.
        """
        # 1. Get enough historical data to calculate our longest indicator (e.g., SMA200 requires 200 days)
        historical_data = self.get_historical_prices(self.symbol, 250, "day")
        df = historical_data.df
        
        if df.empty or len(df) < 200:
            return # Not enough data to calculate features yet

        # 2. Add technical indicators using our existing function
        df_features = add_technical_indicators(df)
        
        # 3. Extract the very last row (the most recent completed day)
        latest_data = df_features.iloc[-1:]
        
        # Skip if there are NaNs (can happen at the very beginning of the dataset)
        if latest_data[self.feature_columns].isnull().values.any():
            return
        
        # 4. Extract and scale the features exactly like we did in training
        X = latest_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # 5. Make a prediction using the SVM
        # prediction is either 1 (Up) or -1 (Down)
        prediction = self.model.predict(X_scaled)[0]
        
        # 6. Execute trades based on the prediction
        current_position = self.get_position(self.symbol)
        
        # Determine our current state: Long (>0), Short (<0), or Flat (0)
        position_qty = float(current_position.quantity) if current_position else 0.0
        
        if prediction == 1:
            # Bullish signal: We want to be Long
            if position_qty < 0:
                self.sell_all()  # Cover the existing short position first
                
            if position_qty <= 0:
                last_price = self.get_last_price(self.symbol)
                # Use 95% of total portfolio value instead of just cash to handle flipping from short to long,
                # and to leave a 5% cash buffer for margin requirements.
                quantity = int((self.portfolio_value * 0.95) // last_price)
                
                if quantity > 0:
                    order = self.create_order(self.symbol, quantity, "buy")
                    self.submit_order(order)
                    
        elif prediction == -1:
            # Bearish signal: We want to be Short
            if position_qty > 0:
                self.sell_all()  # Liquidate the existing long position first
                
            if position_qty >= 0:
                last_price = self.get_last_price(self.symbol)
                # Calculate how many shares to short using the 95% portfolio rule
                quantity = int((self.portfolio_value * 0.95) // last_price)
                
                if quantity > 0:
                    order = self.create_order(self.symbol, quantity, "sell")
                    self.submit_order(order)

if __name__ == "__main__":
    # Define the backtesting period
    # Run the backtest over the last 3 years up to today
    backtest_end = datetime.now()
    backtest_start = backtest_end - timedelta(days=365 * 3)
    
    print("Fetching historical data from Alpaca...")
    alpaca_client = AlpacaDataClient()
    
    # CRITICAL: We need to fetch data starting BEFORE our backtest so our SMA200 
    # has enough historical data to calculate on the very first day of the backtest!
    data_start = backtest_start - timedelta(days=300) 
    df = alpaca_client.get_historical_bars("MSFT", data_start, backtest_end)
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("MSFT", level="symbol")
    
    # Ensure columns are properly formatted for Lumibot (lowercase open, high, low, close, volume)
    df = df.rename(columns=str.lower)
    
    # Set up the Pandas data feed for Lumibot
    asset = Asset(symbol="MSFT", asset_type="stock")
    data_object = Data(asset=asset, df=df, timestep="day")
    pandas_data = {asset: data_object}
    
    print(f"Starting Lumibot backtest for SVMStrategy from {backtest_start.date()} to {backtest_end.date()}...")
    
    # Run the backtest using our custom Alpaca Pandas data
    backtest_results = SVMStrategy.backtest(
        PandasDataBacktesting,
        backtest_start,
        backtest_end,
        benchmark_asset="MSFT",
        pandas_data=pandas_data,
        show_plot=True # Will open a browser window with a tear sheet
    )
