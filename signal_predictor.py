import pandas as pd
import ta
import numpy as np
import joblib
import os
from datetime import datetime
import pytz
import warnings

warnings.filterwarnings("ignore")


def load_data(pair):
    """Load real-time price, fundamental, and sentiment data"""
    price_path = (
        f"data/{pair}_price_5min.csv"
        if pair == "XAUUSD"
        else f"data/{pair}_price_90min.csv"
    )
    fundamental_path = (
        f"data/{pair}_fundamental_5min.txt"
        if pair == "XAUUSD"
        else f"data/{pair}_fundamental_90min.txt"
    )
    sentiment_path = (
        f"data/{pair}_sentiment_5min.txt"
        if pair == "XAUUSD"
        else f"data/{pair}_sentiment_90min.txt"
    )
    if not all(
        os.path.exists(path) for path in [price_path, fundamental_path, sentiment_path]
    ):
        print(f"Missing data files for {pair}")
        return None
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)

    required_cols = ["1. open", "2. high", "3. low", "4. close", "5. volume", "vix"]
    price_df = price_df[[col for col in required_cols if col in price_df.columns]]

    with open(fundamental_path, "r") as f:
        fundamental_data = eval(f.read())
        for key, value in fundamental_data.items():
            price_df[key] = value

    with open(sentiment_path, "r") as f:
        sentiment = float(f.read())
        price_df["sentiment"] = sentiment

    return price_df


def add_technical_indicators(df):
    """Calculate technical indicators for signal generation"""
    df = df.copy()
    df["SMA_10"] = ta.trend.sma_indicator(df["4. close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["4. close"], window=20)
    df["EMA_10"] = ta.trend.ema_indicator(df["4. close"], window=10)
    df["EMA_20"] = ta.trend.ema_indicator(df["4. close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["4. close"], window=14)
    df["Williams_%R"] = ta.momentum.williams_r(
        df["2. high"], df["3. low"], df["4. close"], lbp=14
    )
    bb = ta.volatility.BollingerBands(df["4. close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    macd = ta.trend.MACD(df["4. close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    atr = ta.volatility.AverageTrueRange(
        high=df["2. high"], low=df["3. low"], close=df["4. close"], window=14
    )
    df["ATR"] = atr.average_true_range()
    stoch = ta.momentum.StochasticOscillator(
        high=df["2. high"],
        low=df["3. low"],
        close=df["4. close"],
        window=14,
        smooth_window=3,
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    adx = ta.trend.ADXIndicator(
        high=df["2. high"], low=df["3. low"], close=df["4. close"], window=14
    )
    df["ADX"] = adx.adx()
    df["OBV"] = ta.volume.on_balance_volume(df["4. close"], df["5. volume"])
    cci = ta.trend.CCIIndicator(
        high=df["2. high"], low=df["3. low"], close=df["4. close"], window=20
    )
    df["CCI"] = cci.cci()

    # Rolling window features - matching training script exactly
    for col in ["4. close", "RSI", "MACD", "ATR", "vix"]:
        df[f"{col}_roll_mean5"] = df[col].rolling(window=5).mean()
        df[f"{col}_roll_std5"] = df[col].rolling(window=5).std()
        df[f"{col}_roll_min5"] = df[col].rolling(window=5).min()
        df[f"{col}_roll_max5"] = df[col].rolling(window=5).max()

    # Lagged features - matching training script exactly
    for col in ["SMA_10", "RSI", "MACD"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    return df


# Predict limit order price based on historical movement
def predict_limit_price(df, signal, pair):
    last_close = df["4. close"].iloc[-1]
    avg_movement = df["4. close"].pct_change().mean() * 10000
    pip_value = 0.0001 if pair != "USDJPY" else 0.01
    if signal == 1:
        limit_price = last_close + avg_movement * pip_value
    else:
        limit_price = last_close - avg_movement * pip_value
    return round(limit_price, 5)


# Apply stop-loss and take-profit for risk management
def apply_risk_management(df, pair, signal):
    df = df.copy()
    pip_value = 0.0001 if pair != "USDJPY" else 0.01
    stop_loss_pips = 10
    risk_reward_ratio = 2
    df["Stop_Loss"] = np.where(
        signal == 1,
        df["4. close"] - stop_loss_pips * pip_value,
        df["4. close"] + stop_loss_pips * pip_value,
    )
    df["Take_Profit"] = np.where(
        signal == 1,
        df["4. close"] + (stop_loss_pips * risk_reward_ratio) * pip_value,
        df["4. close"] - (stop_loss_pips * risk_reward_ratio) * pip_value,
    )
    return df


# Generate buy/sell signal using trained model
def generate_signal(pair):
    df = load_data(pair)
    if df is None:
        return
    df = add_technical_indicators(df)
    df = df.dropna()
    if len(df) < 20:
        print(f"Not enough data for {pair}")
        return

    # Use the exact same features as training script
    features = [
        "SMA_10",
        "SMA_20",
        "EMA_10",
        "EMA_20",
        "RSI",
        "Williams_%R",
        "BB_High",
        "BB_Low",
        "MACD",
        "MACD_Signal",
        "ATR",
        "Stoch_K",
        "Stoch_D",
        "ADX",
        "OBV",
        "CCI",
        "vix",
        "4. close_roll_mean5",
        "4. close_roll_std5",
        "4. close_roll_min5",
        "4. close_roll_max5",
        "RSI_roll_mean5",
        "RSI_roll_std5",
        "RSI_roll_min5",
        "RSI_roll_max5",
        "MACD_roll_mean5",
        "MACD_roll_std5",
        "MACD_roll_min5",
        "MACD_roll_max5",
        "ATR_roll_mean5",
        "ATR_roll_std5",
        "ATR_roll_min5",
        "ATR_roll_max5",
        "vix_roll_mean5",
        "vix_roll_std5",
        "vix_roll_min5",
        "vix_roll_max5",
        "SMA_10_lag1",
        "SMA_10_lag2",
        "RSI_lag1",
        "RSI_lag2",
        "MACD_lag1",
        "MACD_lag2",
    ]

    # Check if all features are available
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Missing features for {pair}: {missing_features}")
        return

    X = df[features].dropna()
    if len(X) == 0:
        print(f"No valid data after feature calculation for {pair}")
        return
    model_path = f"models/{pair}_model.pkl"
    if not os.path.exists(model_path):
        print(f"No model found for {pair}")
        return
    model = joblib.load(model_path)
    df["Predicted_Signal"] = model.predict(X)
    df["Confidence"] = model.predict_proba(X)[:, 1]
    latest_signal = df["Predicted_Signal"].iloc[-1]
    latest_confidence = df["Confidence"].iloc[-1]
    if latest_confidence > 0.7:
        limit_price = predict_limit_price(df, latest_signal, pair)
        df = apply_risk_management(df, pair, latest_signal)
        signal_type = "Buy" if latest_signal == 1 else "Sell"
        with open("signals.txt", "a") as f:
            f.write(
                f"[{datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{pair}: {signal_type} Limit at {limit_price:.5f}, "
                f"Confidence: {latest_confidence:.2f}, "
                f"Stop Loss: {df['Stop_Loss'].iloc[-1]:.5f}, "
                f"Take Profit: {df['Take_Profit'].iloc[-1]:.5f}\n"
            )
        print(
            f"[{datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Signal for {pair}: {signal_type} Limit at {limit_price:.5f}, "
            f"Confidence: {latest_confidence:.2f}, "
            f"Stop Loss: {df['Stop_Loss'].iloc[-1]:.5f}, "
            f"Take Profit: {df['Take_Profit'].iloc[-1]:.5f}"
        )
    else:
        print(
            f"[{datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S')}] "
            f"No high-confidence signal for {pair} (Confidence: {latest_confidence:.2f})"
        )


# Generate signals for all currency pairs
def main():
    pairs = ["XAUUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    for pair in pairs:
        print(f"\nGenerating signal for {pair}...")
        generate_signal(pair)


if __name__ == "__main__":
    main()
