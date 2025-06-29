import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os

# Add technical indicators to the dataset
def add_technical_indicators(df):
    required_cols = ['2. high', '3. low', '4. close', '5. volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns {required_cols} in DataFrame")
        return df
    df['SMA_10'] = ta.trend.sma_indicator(df['4. close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['4. close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['4. close'], window=14)
    bb = ta.volatility.BollingerBands(df['4. close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    macd = ta.trend.MACD(df['4. close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    atr = ta.volatility.AverageTrueRange(high=df['2. high'], low=df['3. low'], close=df['4. close'], window=14)
    df['ATR'] = atr.average_true_range()
    stoch = ta.momentum.StochasticOscillator(high=df['2. high'], low=df['3. low'], close=df['4. close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    adx = ta.trend.ADXIndicator(high=df['2. high'], low=df['3. low'], close=df['4. close'], window=14)
    df['ADX'] = adx.adx()
    df['OBV'] = ta.volume.on_balance_volume(df['4. close'], df['5. volume'])
    cci = ta.trend.CCIIndicator(high=df['2. high'], low=df['3. low'], close=df['4. close'], window=20)
    df['CCI'] = cci.cci()
    return df

# Load and prepare dataset for training
def prepare_dataset(df, pair):
    df = add_technical_indicators(df)
    df = df.dropna()
    features = ['SMA_10', 'SMA_20', 'RSI', 'BB_High', 'BB_Low', 'MACD', 'MACD_Signal', 'ATR',
                'Stoch_K', 'Stoch_D', 'ADX', 'OBV', 'CCI',
                'unemployment_rate', 'nonfarm_payrolls', '10-year_treasury_rate', 'vix']
    if 'sentiment' in df.columns and df['sentiment'].nunique() > 1:
        features.append('sentiment')
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features {missing_features} for {pair}. Skipping...")
        return None, None, None, None
    X = df[features]
    y = (df['4. close'].shift(-1) > df['4. close']).astype(int)
    return X, y, df, features

# Train and evaluate the model for a given pair
def train_and_evaluate(pair):
    print(f"Training model for {pair}...")
    file_suffix = '' if pair == 'XAUUSD' else 'X'
    data_file = f'historical_data/{pair}{file_suffix}_daily.csv'
    sentiment_file = f'historical_data/{pair}{file_suffix}_sentiment.txt'
    
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Historical data file for {pair} not found at '{data_file}'. Skipping...")
        return
    
    if len(df) < 100:
        print(f"Error: Insufficient data for {pair} ({len(df)} rows, need at least 100). Skipping...")
        return
    
    df['sentiment'] = 0.0
    if os.path.exists(sentiment_file):
        with open(sentiment_file, 'r') as f:
            df['sentiment'] = float(f.read())

    X, y, df, features = prepare_dataset(df, pair)
    if X is None or y is None:
        print(f"Failed to prepare dataset for {pair}")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y.dropna(), test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    model = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {pair}: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {pair}: {accuracy:.2f}")
    
    feature_importances = pd.Series(best_model.feature_importances_, index=features)
    print(f"Feature importances for {pair}:\n{feature_importances.sort_values(ascending=False)}\n")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/{pair}_model.pkl')

# Main execution
def main():
    pairs = ['XAUUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    for pair in pairs:
        train_and_evaluate(pair)

if __name__ == "__main__":
    main()