# VPS Production Deployment

Recommended production split:

- VPS runs `live_trader.py` in paper or live mode.
- GitHub Actions trains candidate models and uploads artifacts.
- A model is promoted only after evaluation gates pass.

## Setup

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin tradingbot
sudo mkdir -p /opt/trading-bot
sudo chown tradingbot:tradingbot /opt/trading-bot
sudo -u tradingbot git clone <repo-url> /opt/trading-bot
cd /opt/trading-bot
sudo -u tradingbot python3.11 -m venv .venv
sudo -u tradingbot .venv/bin/pip install -r requirements.txt
sudo -u tradingbot cp .env.production.example .env
sudo -u tradingbot nano .env
```

Start continuous mode:

```bash
sudo cp deploy/trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-bot.service
sudo journalctl -u trading-bot.service -f
```

Or hourly timer mode:

```bash
sudo cp deploy/trading-bot-once.service /etc/systemd/system/
sudo cp deploy/trading-bot.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-bot.timer
```

## Safety

Paper mode is default. Real broker execution requires:

```env
TRADING_MODE=live
BROKER=oanda
ENABLE_LIVE_TRADING=true
```

Create the kill-switch file to stop order placement:

```bash
touch /opt/trading-bot/STOP_TRADING
```

Remove it to resume:

```bash
rm /opt/trading-bot/STOP_TRADING
```
