import pandas as pd
import ta
import numpy as np
import joblib
import os
from datetime import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

# Load real-time price, fundamental, and sentiment data
def load_data(pair):
    price_path = f'data/{pair}_price_5min.csv'
    fundamental_path = f'data/{pair}_fundamental_5min.csv'
    sentiment_path = f'data/{pair}_sentiment_5min.csv'
    if not all(os.path.exists(path) for path in [price_path, fundamental_path, sentiment_path]):
        print(f"Missing data files for {pair}")
        return None
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    fundamental_df = pd.read_csv(fundamental_path, index_col=0, parse_dates=True)
    sentiment_df = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
    df = price_df.join(fundamental_df).join(sentiment_df).dropna()
    return df

# Calculate technical indicators for signal generation
def add_technical_indicators(df):
    df = df.copy()
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

# Predict limit order price based on historical movement
def predict_limit_price(df, signal, pair):
    last_close = df['Close'].iloc[-1]
    avg_movement = df['Close'].pct_change().mean() * 10000
    pip_value = 0.0001 if pair != 'USDJPY' else 0.01
    if signal == 1:
        limit_price = last_close + avg_movement * pip_value
    else:
        limit_price = last_close - avg_movement * pip_value
    return round(limit_price, 5)

# Apply stop-loss and take-profit for risk management
def apply_risk_management(df, pair, signal):
    df = df.copy()
    pip_value = 0.0001 if pair != 'USDJPY' else 0.01
    stop_loss_pips = 10
    risk_reward_ratio = 2
    df['Stop_Loss'] = np.where(signal == 1,
                               df['Close'] - stop_loss_pips * pip_value,
                               df['Close'] + stop_loss_pips * pip_value)
    df['Take_Profit'] = np.where(signal == 1,
                                 df['Close'] + (stop_loss_pips * risk_reward_ratio) * pip_value,
                                 df['Close'] - (stop_loss_pips * risk_reward_ratio) * pip_value)
    return df

# Generate buy/sell signal using trained model
def generate_signal(pair):
    df = load_data(pair)
    if df is None:
        return
    df = add_technical_indicators(df)
    df = df.dropna()
    if len(df) < 10:
        print(f"Not enough data for {pair}")
        return
    features = ['SMA_10', 'SMA_20', 'RSI', 'BB_High', 'BB_Low',
                'interest_rate', 'unemployment', 'inflation', 'bond_yield',
                'jobless_claims', 'non_farm_payrolls', 'gdp_growth',
                'consumer_confidence', 'retail_sales', 'sentiment']
    X = df[features]
    model_path = f'models/{pair}_model.pkl'
    if not os.path.exists(model_path):
        print(f"No model found for {pair}")
        return
    model = joblib.load(model_path)
    df['Predicted_Signal'] = model.predict(X)
    df['Confidence'] = model.predict_proba(X)[:, 1]
    latest_signal = df['Predicted_Signal'].iloc[-1]
    latest_confidence = df['Confidence'].iloc[-1]
    if latest_confidence > 0.7:
        limit_price = predict_limit_price(df, latest_signal, pair)
        df = apply_risk_management(df, pair, latest_signal)
        signal_type = "Buy" if latest_signal == 1 else "Sell"
        with open('signals.txt', 'a') as f:
            f.write(f"[{datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{pair}: {signal_type} Limit at {limit_price:.5f}, "
                    f"Confidence: {latest_confidence:.2f}, "
                    f"Stop Loss: {df['Stop_Loss'].iloc[-1]:.5f}, "
                    f"Take Profit: {df['Take_Profit'].iloc[-1]:.5f}\n")
        print(f"[{datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Signal for {pair}: {signal_type} Limit at {limit_price:.5f}, "
              f"Confidence: {latest_confidence:.2f}, "
              f"Stop Loss: {df['Stop_Loss'].iloc[-1]:.5f}, "
              f"Take Profit: {df['Take_Profit'].iloc[-1]:.5f}")
    else:
        print(f"[{datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S')}] "
              f"No high-confidence signal for {pair} (Confidence: {latest_confidence:.2f})")

# Generate signals for all currency pairs
def main():
    pairs = ['XAUUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    for pair in pairs:
        print(f"\nGenerating signal for {pair}...")
        generate_signal(pair)

if __name__ == "__main__":
    main()