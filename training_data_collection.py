import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import warnings
import requests
import dotenv

# Initialize warning settings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
dotenv.load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

if not all([ALPHA_VANTAGE_API_KEY, FRED_API_KEY, NEWS_API_KEY]):
    raise ValueError('Missing API keys in .env file')

# Define start and end dates for historical data
start_date = "2019-06-01"
end_date = "2025-06-06"

# Create historical_data directory if it doesn't exist
os.makedirs('historical_data', exist_ok=True)

# Initialize FRED and VADER sentiment analyzer
fred = Fred(api_key=FRED_API_KEY)
analyzer = SentimentIntensityAnalyzer()

# Forex pairs and XAUUSD
pairs = ['GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']  # yfinance forex symbol format
xauusd_ticker = 'GC=F'

# Fetch historical price data for a currency pair
def fetch_historical_price_data(pair, start_date=start_date, end_date=end_date):
    try:
        ticker = yf.Ticker(pair)
        data = ticker.history(start=start_date, end=end_date, interval="1d")
        if data.empty:
            raise ValueError("No data returned")
        # Convert timezone-aware index to naive
        data.index = data.index.tz_localize(None)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
            'Open': '1. open', 'High': '2. high', 'Low': '3. low', 'Close': '4. close', 'Volume': '5. volume'
        })
        return data
    except Exception as e:
        print(f"Error fetching historical price data for {pair}: {e}")
        return None

# Fetch historical sentiment data (limited by News API free tier to ~30 days)
def fetch_historical_sentiment_data(asset):
    url = f"https://newsapi.org/v2/everything?q={asset}&from={end_date}&to={end_date}&apiKey={NEWS_API_KEY}"
    response = requests.get(url, timeout=5)
    data = response.json()
    if data['status'] == 'ok':
        articles = data['articles'][:5]
        sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles if article['title']]
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return 0

# Fetch historical fundamental data from FRED
def fetch_historical_fundamental_data(start_date=start_date, end_date=end_date):
    indicators = {
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'DGS10': '10-Year Treasury Rate'
    }
    data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D', tz=None))  # Naive index
    for ind, desc in indicators.items():
        try:
            series = fred.get_series(ind, observation_start=start_date, observation_end=end_date)
            # Convert to naive index immediately after fetch
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            data[desc.lower().replace(' ', '_')] = series.reindex(data.index, method='ffill').fillna(method='bfill')
        except Exception as e:
            print(f"Error fetching historical {desc}: {e}")
    return data

# Main execution
if __name__ == "__main__":
    for pair in pairs:
        price_data = fetch_historical_price_data(pair)
        if price_data is not None:
            fundamental_data = fetch_historical_fundamental_data()
            combined_data = price_data.join(fundamental_data, how='left')
            combined_data.to_csv(f'historical_data/{pair.replace("=", "")}_daily.csv')
            sentiment = fetch_historical_sentiment_data(pair.replace("=X", ""))
            with open(f'historical_data/{pair.replace("=", "")}_sentiment.txt', 'w') as f:
                f.write(str(sentiment))

    xauusd_data = fetch_historical_price_data(xauusd_ticker)
    if xauusd_data is not None:
        fundamental_data = fetch_historical_fundamental_data()
        combined_data = xauusd_data.join(fundamental_data, how='left')
        combined_data.to_csv('historical_data/XAUUSD_daily.csv')
        sentiment = fetch_historical_sentiment_data("gold")  # Sentiment for XAUUSD
        with open('historical_data/XAUUSD_sentiment.txt', 'w') as f:
            f.write(str(sentiment))

    fundamental_data = fetch_historical_fundamental_data()
    with open('historical_data/fundamental_data.txt', 'w') as f:
        f.write(str(fundamental_data.iloc[-1].to_dict()))