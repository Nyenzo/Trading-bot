# Trading Bot

A sophisticated algorithmic trading system that combines technical analysis with machine learning to predict market movements and execute trades in forex and gold markets.

## 🚀 Features

- Real-time market data collection for multiple currency pairs and gold (XAUUSD)
- Advanced technical analysis using various indicators
- Machine learning-based signal prediction using Random Forest
- Automated trading during EAT (East Africa Time) market hours
- Economic indicators integration for enhanced decision making
- Comprehensive backtesting and model evaluation

## 📊 Supported Trading Pairs

- XAUUSD (Gold)
- GBPUSD (British Pound/US Dollar)
- USDJPY (US Dollar/Japanese Yen)
- AUDUSD (Australian Dollar/US Dollar)

## 🛠️ Technical Stack

- **Python 3.x**
- **Key Libraries**:
  - pandas: Data manipulation and analysis
  - scikit-learn: Machine learning implementation
  - yfinance: Market data fetching
  - alpha_vantage: Forex data API
  - ta: Technical analysis indicators
  - schedule: Automated task scheduling

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Trading-bot.git
cd Trading-bot
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn yfinance alpha_vantage ta schedule python-dotenv
```

3. Create a `.env` file in the root directory and add your Alpha Vantage API key:
```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## 📈 Usage

1. **Data Collection**:
```bash
python data_collection.py
```
This will start collecting market data during trading hours (6 AM - 6 PM EAT).

2. **Training Data Preparation**:
```bash
python training_data_collection.py
```

3. **Model Training and Evaluation**:
```bash
python train_evaluate.py
```

4. **Signal Prediction**:
```bash
python signal_predictor.py
```

## 📁 Project Structure

- `data_collection.py`: Real-time market data collection
- `training_data_collection.py`: Historical data preparation for training
- `train_evaluate.py`: Model training and performance evaluation
- `signal_predictor.py`: Real-time trading signal generation

## 📊 Model Performance

The trading bot uses a Random Forest Classifier trained on various technical indicators and economic factors. Model accuracy is evaluated for each trading pair separately, with regular retraining to maintain performance.

## ⚠️ Disclaimer

This trading bot is for educational and research purposes only. Trading financial instruments carries significant risks. Always perform your own analysis and consider your financial circumstances before making any investment decisions.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions and feedback, please open an issue in the repository.