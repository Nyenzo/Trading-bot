import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os


# Add technical indicators and rolling features
def add_technical_indicators(df):
    required_cols = ["2. high", "3. low", "4. close", "5. volume"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns {required_cols} in DataFrame")
        return df
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
    # Rolling window features
    for col in ["4. close", "RSI", "MACD", "ATR", "vix"]:
        df[f"{col}_roll_mean5"] = df[col].rolling(window=5).mean()
        df[f"{col}_roll_std5"] = df[col].rolling(window=5).std()
        df[f"{col}_roll_min5"] = df[col].rolling(window=5).min()
        df[f"{col}_roll_max5"] = df[col].rolling(window=5).max()
    # Lagged features
    for col in ["SMA_10", "RSI", "MACD"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
    return df


# Prepare dataset with 2-day lookahead and thresholded labels
def prepare_dataset(df, pair, lookahead=2, threshold=0.005):
    df = add_technical_indicators(df)
    df = df.dropna()
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
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features {missing_features} for {pair}. Skipping...")
        return None, None, None, None
    X = df[features]
    # 2-day lookahead, thresholded label
    future_close = df["4. close"].shift(-lookahead)
    pct_change = (future_close - df["4. close"]) / df["4. close"]
    y = np.where(
        pct_change > threshold, 1, np.where(pct_change < -threshold, 0, np.nan)
    )
    # Remove ambiguous/no-move samples
    mask = ~np.isnan(y)
    X = X[mask]
    y = pd.Series(y[mask], index=X.index)
    return X, y, df, features


# Ensemble prediction (majority vote)
def ensemble_predict(models, Xs):
    preds = []
    for model, X in zip(models, Xs):
        preds.append(model.predict(X))
    preds = np.array(preds)
    # Majority vote
    return np.round(np.mean(preds, axis=0)).astype(int)


# Train and evaluate the models for a given pair
def train_and_evaluate(pair):
    print(f"\nTraining and evaluating for {pair}...")
    file_suffix = "" if pair == "XAUUSD" else "X"
    data_file = f"historical_data/{pair}{file_suffix}_daily.csv"
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(
            f"Error: Historical data file for {pair} not found at '{data_file}'. Skipping..."
        )
        return
    if len(df) < 100:
        print(
            f"Error: Insufficient data for {pair} ({len(df)} rows, need at least 100). Skipping..."
        )
        return
    X, y, df, features = prepare_dataset(df, pair)
    if X is None or y is None or len(y) < 100:
        print(
            f"Failed to prepare dataset for {pair} or not enough samples after filtering."
        )
        return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Balance classes
    sample_weight = compute_sample_weight("balanced", y)
    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    fold = 1
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sw_train = sample_weight[train_idx]
        # Train all models
        xgb = XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )
        lgb = LGBMClassifier(random_state=42)
        cat = CatBoostClassifier(verbose=0, random_state=42)
        xgb.fit(X_train, y_train, sample_weight=sw_train)
        lgb.fit(X_train, y_train, sample_weight=sw_train)
        cat.fit(X_train, y_train, sample_weight=sw_train)
        # Ensemble prediction
        y_pred = ensemble_predict([xgb, lgb, cat], [X_test] * 3)
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        print(
            f"Fold {fold}: Accuracy={metrics['accuracy'][-1]:.2f}, Precision={metrics['precision'][-1]:.2f}, Recall={metrics['recall'][-1]:.2f}, F1={metrics['f1'][-1]:.2f}"
        )
        fold += 1
    print(f"\nMean Accuracy: {np.mean(metrics['accuracy']):.2f}")
    print(f"Mean Precision: {np.mean(metrics['precision']):.2f}")
    print(f"Mean Recall: {np.mean(metrics['recall']):.2f}")
    print(f"Mean F1: {np.mean(metrics['f1']):.2f}\n")
    # Fit on all data and save ensemble (save XGB as main model)
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_scaled, y, sample_weight=sample_weight)
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, f"models/{pair}_model.pkl")


# Main execution
def main():
    pairs = ["XAUUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    for pair in pairs:
        train_and_evaluate(pair)


if __name__ == "__main__":
    main()
