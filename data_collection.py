import os
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from fredapi import Fred
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import schedule
import time
import warnings
import dotenv
import requests

warnings.filterwarnings('ignore')

# Load environment variables from .env file
dotenv.load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

if not all([ALPHA_VANTAGE_API_KEY, FRED_API_KEY, NEWS_API_KEY]):
    raise ValueError('Missing API keys in .env file')

# Initialize Alpha Vantage, FRED, and VADER sentiment analyzer
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
fred = Fred(api_key=FRED_API_KEY)
analyzer = SentimentIntensityAnalyzer()

# Forex pairs and XAUUSD
pairs = ['GBPUSD', 'USDJPY', 'AUDUSD']
xauusd_ticker = 'XAUUSD=X'

# Fetch historical price data for a currency pair
def fetch_price_data(pair, start_date="2019-06-01", end_date="2025-06-06", is_xauusd=False):
    try:
        if is_xauusd:
            ticker = yf.Ticker(xauusd_ticker)
            data = ticker.history(start=start_date, end=end_date, interval="5m")
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
                'Open': '1. open', 'High': '2. high', 'Low': '3. low', 'Close': '4. close', 'Volume': '5. volume'
            })
        else:
            # Use TIME_SERIES_INTRADAY for forex with 90-minute interval (closest to original design)
            data, meta_data = ts.get_intraday(symbol=f"{pair}=X", interval='90min', outputsize='compact')
            data = data.rename(columns={
                '1. open': '1. open', '2. high': '2. high', '3. low': '3. low', '4. close': '4. close', '5. volume': '5. volume'
            })
        return data
    except Exception as e:
        print(f"Error fetching price data for {pair}: {e}")
        return None

# Fetch sentiment data from News API
def fetch_sentiment_data(pair):
    url = f"https://newsapi.org/v2/everything?q={pair}&apiKey={NEWS_API_KEY}"
    response = requests.get(url, timeout=5)
    data = response.json()
    if data['status'] == 'ok':
        articles = data['articles'][:5]  # Limit to 5 articles
        sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles if article['title']]
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return 0

# Fetch fundamental data from FRED
def fetch_fundamental_data():
    indicators = {
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'DGS10': '10-Year Treasury Rate'
    }
    data = {}
    for ind, desc in indicators.items():
        try:
            series = fred.get_series(ind)
            data[desc] = series.iloc[-1]
        except Exception as e:
            print(f"Error fetching {desc}: {e}")
    return data

# Schedule data collection during trading hours (6 AM - 6 PM EAT)
def job():
    current_time = datetime.now().hour
    if 6 <= current_time < 18:  # 6 AM - 6 PM EAT
        for pair in pairs:
            price_data = fetch_price_data(pair)
            if price_data is not None:
                price_data.to_csv(f'data/{pair}_price_90min.csv')
                sentiment = fetch_sentiment_data(pair)
                with open(f'data/{pair}_sentiment_30min.txt', 'w') as f:
                    f.write(str(sentiment))
            fundamental_data = fetch_fundamental_data()
            with open('data/fundamental_data.txt', 'w') as f:
                f.write(str(fundamental_data))

        # XAUUSD every 5 minutes
        xauusd_data = fetch_price_data(xauusd_ticker, is_xauusd=True)
        if xauusd_data is not None:
            xauusd_data.to_csv('data/XAUUSD_price_5min.csv')

        # Trigger signal predictor (assumed external script)
        import subprocess
        subprocess.run(['python', 'signal_predictor.py'])

# Schedule jobs
schedule.every(90).minutes.do(job)  # Forex pairs every 90 minutes
schedule.every(5).minutes.do(lambda: job() if datetime.now().minute % 5 == 0 else None)  # XAUUSD every 5 minutes

while True:
    schedule.run_pending()
    time.sleep(60)