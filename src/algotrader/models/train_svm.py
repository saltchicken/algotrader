import os
import time
import random
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Import your existing clients and feature engineering
from algotrader.external_api.alpaca_api import AlpacaDataClient
from algotrader.features.indicators import add_technical_indicators

def get_sp500_symbols() -> List[str]:
    """Fetches the current S&P 500 symbols dynamically from Wikipedia."""
    print("Fetching S&P 500 symbols from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Add User-Agent to bypass Wikipedia's 403 Forbidden bot protection
    storage_options = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    tables = pd.read_html(url, storage_options=storage_options)
    
    df = tables[0]
    symbols = df['Symbol'].tolist()
    
    # Clean symbols: Alpaca uses '-' instead of '.' for symbols like BRK.B -> BRK-B
    symbols = [sym.replace('.', '-') for sym in symbols]
    return symbols

def prepare_data(symbol: str, start_date: datetime, end_date: datetime):
    """Fetches data and prepares stationary features and targets for ML for a single symbol."""
    client = AlpacaDataClient()
    # If a single string is passed, Alpaca might still return a MultiIndex
    df_raw = client.get_historical_bars(symbol, start_date, end_date)
    
    # Handle multi-index if Alpaca returns it (symbol, timestamp)
    if isinstance(df_raw.index, pd.MultiIndex):
        try:
            df_raw = df_raw.xs(symbol, level='symbol')
        except KeyError:
            pass # fallback if it's structured differently
            
    # Add technical indicators
    df = add_technical_indicators(df_raw)
    df = df.dropna()
    
    feature_columns = [
        'macd_trading_signal',
        'bb_lower_dist', 'rsi_14', 'returns_lag_1', 'returns_lag_2', 'returns_lag_5'
    ]
    target_column = 'target_direction'
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y, df

def train_universal_svm(symbols: List[str], model_name: str = "sp500_universe"):
    """Train a single generalized model on pooled data from the entire S&P 500."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    print(f"\n--- Preparing Universal Model using {len(symbols)} symbols ---")
    
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    
    # We MUST process each symbol individually to prevent technical indicators 
    # (like moving averages) from bleeding across different stocks
    for i, symbol in enumerate(symbols):
        try:
            X, y, _ = prepare_data(symbol, start_date, end_date)
            
            # Perform the 80/20 chronological split per symbol
            split_idx = int(len(X) * 0.8)
            X_train_list.append(X.iloc[:split_idx])
            X_test_list.append(X.iloc[split_idx:])
            y_train_list.append(y.iloc[:split_idx])
            y_test_list.append(y.iloc[split_idx:])
            
            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(symbols)} symbols...")
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            
        # Delay to respect Alpaca's free tier limit of 200 requests/minute
        time.sleep(0.35)
            
    # Combine all individual train/test sets into massive pooled datasets
    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)
    
    print(f"Total Pooled Training Samples: {len(X_train)}, Testing: {len(X_test)}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("Training universal SVM Classifier (this might take a moment on 500 stocks)...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Universal Model Evaluation ---")
    print(f"Accuracy on unseen pooled test data: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, f"saved_models/{model_name}_svm_model.joblib")
    joblib.dump(scaler, f"saved_models/{model_name}_scaler.joblib")
    print(f"Saved Universal Model to /saved_models/{model_name}_svm_model.joblib")

if __name__ == "__main__":
    target_symbols = get_sp500_symbols()
    print(f"Successfully loaded {len(target_symbols)} symbols.")
    
    # Select a random subset of 50 stocks for faster testing
    target_symbols = random.sample(target_symbols, min(50, len(target_symbols)))
    print(f"Training on a random subset of {len(target_symbols)} symbols to speed up training...")
    
    # Train one giant model spanning the selected symbols. No individual models are generated.
    train_universal_svm(target_symbols, model_name="sp500_universe")
