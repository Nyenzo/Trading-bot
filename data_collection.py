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
import pytz

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
vix_ticker = '^VIX'

# Fetch price data for a currency pair
def fetch_price_data(pair, interval='90min', is_xauusd=False):
    try:
        if is_xauusd:
            ticker = yf.Ticker(xauusd_ticker)
            data = ticker.history(period='1d', interval='5m')
            data = data.rename(columns={
                'Open': '1. open', 'High': '2. high', 'Low': '3. low', 'Close': '4. close', 'Volume': '5. volume'
            })
        else:
            data, _ = ts.get_intraday(symbol=f"{pair}=X", interval=interval, outputsize='compact')
        data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        print(f"Error fetching price data for {pair}: {e}")
        return None

# Fetch VIX data
def fetch_vix_data(interval='5m'):
    try:
        ticker = yf.Ticker(vix_ticker)
        data = ticker.history(period='1d', interval=interval)
        data.index = data.index.tz_localize(None)
        return data[['Close']].rename(columns={'Close': 'vix'})
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return None

# Fetch sentiment data from News API
def fetch_sentiment_data(pair):
    url = f"https://newsapi.org/v2/everything?q={pair}&apiKey={NEWS_API_KEY}"
    response = requests.get(url, timeout=5)
    data = response.json()
    if data['status'] == 'ok':
        articles = data['articles'][:5]
        sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles if article['title']]
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return 0

# Fetch fundamental data from FRED
def fetch_fundamental_data():
    indicators = {
        'UNRATE': 'unemployment_rate',
        'PAYEMS': 'nonfarm_payrolls',
        'DGS10': '10-year_treasury_rate'
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
    current_time = datetime.now(pytz.timezone('Africa/Nairobi')).hour
    if 6 <= current_time < 18:
        vix_data = fetch_vix_data()
        for pair in pairs:
            price_data = fetch_price_data(pair)
            if price_data is not None:
                if vix_data is not None:
                    price_data = price_data.join(vix_data, how='left')
                    price_data['vix'] = price_data['vix'].ffill()
                price_data.to_csv(f'data/{pair}_price_90min.csv')
                sentiment = fetch_sentiment_data(pair)
                with open(f'data/{pair}_sentiment_90min.txt', 'w') as f:
                    f.write(str(sentiment))
            fundamental_data = fetch_fundamental_data()
            with open(f'data/{pair}_fundamental_90min.txt', 'w') as f:
                f.write(str(fundamental_data))

        xauusd_data = fetch_price_data(xauusd_ticker, interval='5m', is_xauusd=True)
        if xauusd_data is not None:
            if vix_data is not None:
                xauusd_data = xauusd_data.join(vix_data, how='left')
                xauusd_data['vix'] = xauusd_data['vix'].ffill()
            xauusd_data.to_csv('data/XAUUSD_price_5min.csv')
            sentiment = fetch_sentiment_data('gold')
            with open('data/XAUUSD_sentiment_5min.txt', 'w') as f:
                f.write(str(sentiment))

        fundamental_data = fetch_fundamental_data()
        with open('data/fundamental_data.txt', 'w') as f:
            f.write(str(fundamental_data))

        import subprocess
        subprocess.run(['python', 'signal_predictor.py'])

# Schedule jobs
schedule.every(90).minutes.do(job)
schedule.every(5).minutes.do(lambda: job() if datetime.now(pytz.timezone('Africa/Nairobi')).minute % 5 == 0 else None)

while True:
    schedule.run_pending()
    time.sleep(60)