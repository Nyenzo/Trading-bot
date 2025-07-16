import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

# Load Alpha Vantage API key
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# List of pairs (yfinance symbols and output names)
pairs = ['GC=F', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
output_names = ['XAUUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

# Only last 2 years for yfinance intraday
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=729)
interval = '1h'

os.makedirs('historical_data', exist_ok=True)

for pair, out_name in zip(pairs, output_names):
    print(f"Downloading {out_name} 1-hour data with yfinance...")
    ticker = yf.Ticker(pair)
    try:
        data = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
    except Exception as e:
        print(f"yfinance error for {out_name}: {e}")
        data = pd.DataFrame()
    if not data.empty:
        data.index = data.index.tz_localize(None)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
            'Open': '1. open', 'High': '2. high', 'Low': '3. low', 'Close': '4. close', 'Volume': '5. volume'
        })
        data.to_csv(f'historical_data/{out_name}_hourly.csv')
        print(f"Saved historical_data/{out_name}_hourly.csv ({len(data)} rows)")
        continue
    # Fallback to Alpha Vantage for forex pairs (not for gold)
    if out_name != 'XAUUSD' and ALPHA_VANTAGE_API_KEY:
        print(f"Trying Alpha Vantage for {out_name}...")
        url = f'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={out_name[:3]}&to_symbol={out_name[3:]}&interval=60min&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
        r = requests.get(url)
        if r.status_code == 200:
            json_data = r.json()
            if 'Time Series FX (60min)' in json_data:
                df = pd.DataFrame.from_dict(json_data['Time Series FX (60min)'], orient='index')
                df = df.rename(columns={
                    '1. open': '1. open',
                    '2. high': '2. high',
                    '3. low': '3. low',
                    '4. close': '4. close'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df['5. volume'] = 0  # Alpha Vantage does not provide volume
                df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
                df = df.astype(float)
                df.to_csv(f'historical_data/{out_name}_hourly.csv')
                print(f"Saved historical_data/{out_name}_hourly.csv from Alpha Vantage ({len(df)} rows)")
            else:
                print(f"Alpha Vantage returned no data for {out_name}.")
        else:
            print(f"Alpha Vantage request failed for {out_name}.")
    else:
        print(f"No data for {out_name}.") 