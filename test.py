import yfinance as yf
data = yf.Ticker('GC=F').history(start="2019-06-01", end="2025-06-06", interval="1d")
print(data.head())