# 🚀 Trading Bot Execution Guide

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Internet connection for market data
- Minimum 4GB RAM recommended

### Installation
1. Download the appropriate package for your operating system
2. Extract the archive to your desired directory
3. Navigate to the extracted folder

### Running the Trading Bot

#### Command Line Usage
```bash
# Demo mode (safe testing)
./TradingBot trade --demo --episodes 5

# Full trading mode (requires API keys)
./TradingBot trade --episodes 10

# Dashboard mode
./TradingBot dashboard
```

#### Windows Users
Use `TradingBot.exe` instead of `./TradingBot`

#### Available Commands
- `trade` - Run trading sessions
- `dashboard` - Launch web dashboard
- `train` - Train ML models
- `signals` - Generate trading signals

#### Command Options
- `--demo` - Run in safe demo mode
- `--episodes N` - Number of trading episodes (default: 10)
- `--pairs "XAUUSD,GBPUSD"` - Specify trading pairs

### API Configuration (Optional)
Create a `.env` file in the same directory:
```
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

## Dashboard Access
When running in dashboard mode, access the web interface at:
http://localhost:8501

## Safety & Risk Disclaimer
⚠️ **IMPORTANT**: This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Never risk money you cannot afford to lose.

## Support
- 📖 Documentation: Check README.md
- 🐛 Issues: Report on GitHub
- 💬 Community: GitHub Discussions

## System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Stable internet connection
- **OS**: Windows 10+, Linux (Ubuntu 20.04+), macOS 10.15+

Happy Trading! 📈