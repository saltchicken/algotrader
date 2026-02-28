from .external_api.alpaca_api import AlpacaDataClient
from datetime import datetime, timedelta


def main():
    # Instantiate the extracted client
    alpaca = AlpacaDataClient()
    
    # Define parameters for our data request
    target_symbol = "AAPL"
    
    # Let's get data for the last 7 days
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=7)
    
    try:
        # Fetch the dataframe
        df = alpaca.get_historical_bars(target_symbol, start_dt, end_dt)
        
        print("\n--- Historical Data ---")
        print(df)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
