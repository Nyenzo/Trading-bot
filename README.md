Forex Trading Signal Generator
Overview
This project generates buy/sell limit order signals for forex pairs (XAUUSD, GBPUSD, USDJPY, AUDUSD) using a Random Forest Classifier trained on historical and real-time data. It combines technical indicators (SMA, RSI, Bollinger Bands), fundamental data from the FRED API (e.g., interest rates, jobless claims, non-farm payrolls), and sentiment analysis from the News API. The system collects semi-real-time data (5-minute intervals for XAUUSD, 90-minute for forex pairs) and supports training on 6 years of historical data (June 2019 to June 2025).
Features

Data Collection: Fetches price data (Alpha Vantage, yfinance), fundamentals (FRED), and news sentiment (News API).
Training: Trains models on historical data with technical, fundamental, and sentiment features.
Signal Generation: Produces high-confidence (>70%) limit orders with stop-loss and take-profit.
Real-Time: Updates data during trading hours (6 AM–6 PM EAT) and generates signals automatically.
Modular Design: Separates data collection, training, and signal prediction for flexibility.

Prerequisites

Python 3.8+
API Keys:
Alpha Vantage (free tier, 25 calls/day)
FRED (free tier)
News API (free tier, 100 calls/day)


Libraries:pip install pandas yfinance alpha-vantage ta scikit-learn numpy schedule joblib python-dotenv fredapi vaderSentiment requests



Project Structure
forex_signal_generator/
├── data/                    # Real-time price, fundamental, sentiment data
├── historical_data/         # 6-year historical data for training
├── models/                  # Trained Random Forest models
├── .env                     # API keys
├── data_collection.py       # Real-time data collection
├── data_collection_training.py  # Historical data collection
├── train_evaluate.py        # Model training and evaluation
├── signal_predictor.py      # Signal generation
├── README.md                # Project documentation

Setup

Clone the Repository:
git clone <repository_url>
cd forex_signal_generator


Create .env File:
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FRED_API_KEY=your_fred_key
NEWS_API_KEY=your_newsapi_key


Install Dependencies:
pip install -r requirements.txt

Or directly:
pip install pandas yfinance alpha-vantage ta scikit-learn numpy schedule joblib python-dotenv fredapi vaderSentiment requests


Create Folders:
mkdir data historical_data models



Usage

Collect Historical Data (one-time, ~30 minutes):
python data_collection_training.py


Fetches 6 years (2019-06-01 to 2025-06-06) of price, fundamental, and sentiment data.
Saves to historical_data/{pair}_daily.csv.
Note: News API historical data is limited to ~30 days (free tier); consider paid plans for full coverage.


Train Models (after historical data collection):
python train_evaluate.py


Trains Random Forest models using historical data.
Saves models to models/{pair}_model.pkl.
Retrain weekly or after major market events.


Run Real-Time Data Collection (during 6 AM–6 PM EAT):
python data_collection.py


Fetches:
Price data: Every 90 minutes for GBPUSD, USDJPY, AUDUSD (Alpha Vantage); every 5 minutes for XAUUSD (yfinance).
Fundamental data: Daily at 6 AM (FRED).
Sentiment data: Every 30 minutes (News API).


Saves to data/{pair}_{type}_5min.csv.
Automatically triggers signal_predictor.py after price/sentiment fetches.


Generate Signals (manual or via data_collection.py):
python signal_predictor.py


Generates buy/sell limit signals with >70% confidence.
Example output:[2025-06-06 18:56:00] Signal for XAUUSD: Buy Limit at 1800.52000, Confidence: 0.75, Stop Loss: 1800.42000, Take Profit: 1800.62000





API Rate Limits

Alpha Vantage: 24 calls/day (8 per forex pair).
yfinance: ~144 calls/day for XAUUSD (within ~2,000/hour).
FRED: ~28 calls/day (9 indicators x 4 pairs, no strict limit).
News API: 24 calls/day (every 30 minutes, within 100/day).

Limitations

FRED Data: Non-US indicators (UK, Japan, Australia) use US proxies due to limited FRED series, potentially reducing accuracy for GBPUSD, USDJPY, AUDUSD.
News API: Free tier limits historical data to ~30 days; full 6-year sentiment requires a paid plan.
Real-Time Frequency: Forex pairs update every 90 minutes due to Alpha Vantage limits; XAUUSD updates every 5 minutes.
No Live Trading: Signals are for analysis; integrate with a broker (e.g., OANDA demo) for execution.

Future Improvements

Add backtesting to evaluate signal performance.
Integrate with a broker API for automated trading.
Use paid News API for full historical sentiment data.
Explore alternative fundamental data sources (e.g., Quandl) for non-US economies.

Troubleshooting

API Errors: Verify API keys in .env and check rate limits.
Missing Data: Ensure data_collection_training.py ran successfully before training.
No Signals: Confirm models/{pair}_model.pkl exists and data/{pair}_*.csv files are updated.
Contact: For issues, open a GitHub issue or contact the maintainer.

License
MIT License. See LICENSE file for details.
Acknowledgments

Built with Python, Alpha Vantage, FRED, News API, and open-source libraries.
Inspired by algorithmic trading research and forex market analysis.
