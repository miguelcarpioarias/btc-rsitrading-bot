# BTC RSI Trading Bot

Advanced cryptocurrency trading bot using RSI (Relative Strength Index) with multi-indicator confirmation, risk management, and persistent data storage.

## Features

- 🤖 **Automated Trading** - Executes trades based on technical indicators
- 📊 **Technical Analysis** - RSI, MACD, ATR, SMA with multi-indicator confirmation
- 💰 **Risk Management** - Stop-loss, position sizing, volatility filtering
- 💾 **Data Persistence** - PostgreSQL database for trades, orders, and performance metrics
- 📈 **Real-time Dashboard** - Interactive Plotly charts with live trading data
- 🎛️ **Configurable** - All strategy parameters adjustable via environment variables
- ☁️ **Render Ready** - Optimized for deployment on Render with PostgreSQL

## Strategy

### Buy Signal (RSI ≤ 30)
1. RSI drops to oversold level (≤ 30)
2. MACD histogram is positive (bullish momentum)
3. Price is above SMA50 (uptrend confirmation)
4. Volatility is below threshold (< 5% ATR)

### Sell Signal (RSI ≥ 70)
1. RSI rises to overbought level (≥ 70)
2. MACD histogram is negative (bearish momentum)
3. Price is below SMA50 (downtrend confirmation)

### Risk Management
- **Stop-Loss**: 3% below entry price (configurable)
- **Position Sizing**: Risk 1% of account per trade (configurable)
- **Volatility Filter**: Only trades when volatility is reasonable

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/miguelcarpioarias/btc-rsitrading-bot.git
cd btc-rsitrading-bot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. Run database initialization (requires PostgreSQL):
```bash
python init_db.py
```

6. Start the application:
```bash
python app.py
```

The dashboard will be available at `http://localhost:10000`

## Deployment on Render

### Prerequisites
- Render account (paid tier recommended for 24/7 trading)
- Alpaca API keys
- GitHub repository pushed

### Deployment Steps

1. **Connect your GitHub repository to Render**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repo

2. **Create PostgreSQL Database**
   - In Render dashboard: New → PostgreSQL
   - Name: `trading-bot-db`
   - Instance type: Standard
   - Region: Oregon (or your preference)

3. **Create Web Service**
   - Build command: `pip install -r requirements.txt`
   - Start command: `python init_db.py && gunicorn app:server --workers 2 --worker-class sync --timeout 120`
   - Environment variables:
     - `ALPACA_API_KEY`: Your Alpaca API key
     - `ALPACA_SECRET_KEY`: Your Alpaca secret key
     - `DATABASE_URL`: Auto-populated from PostgreSQL service
     - `TRADING_ENABLED`: `true`
     - `LOG_LEVEL`: `INFO`

4. **Create Background Worker** (for 24/7 trading on paid tier)
   - New → Background Worker
   - Repository: Same as web service
   - Start command: `python -u -c "from app import scheduler; import time; print('Trading bot scheduler started'); [time.sleep(1) for _ in iter(int, 1)]"`
   - Same environment variables as web service

5. **Deploy**
   - Push to GitHub or click "Deploy" in Render dashboard
   - Check logs to verify database initialization
   - Access dashboard at `https://<your-service-name>.onrender.com`

### Or Use render.yaml

Push the `render.yaml` file and Render will automatically configure everything:

```bash
git push origin main
```

Render will read `render.yaml` and create services, databases, and environment configurations.

## Configuration

All strategy parameters can be configured via environment variables:

```env
# Technical Indicators
RSI_WINDOW=14              # RSI period
RSI_OVERSOLD=30            # Buy signal threshold
RSI_OVERBOUGHT=70          # Sell signal threshold
SMA_WINDOW=50              # Simple Moving Average period
ATR_WINDOW=14              # Average True Range period
VOLATILITY_THRESHOLD=5.0   # Max volatility % to trade

# Risk Management
RISK_PERCENT_PER_TRADE=1.0 # % of account to risk per trade
STOP_LOSS_PERCENT=3.0      # Stop-loss % below entry

# Trading Control
TRADING_ENABLED=true       # Enable/disable automated trading
TRADING_INTERVAL_MINUTES=1 # How often to check for signals

# Application
LOG_LEVEL=INFO             # Logging level: DEBUG, INFO, WARNING, ERROR
DEBUG=false                # Enable debug mode
```

## Database Schema

### trades
- `id`: Primary key
- `symbol`: Trading pair (e.g., BTC/USD)
- `entry_price`: Buy price
- `exit_price`: Sell price
- `entry_time`: When trade opened
- `exit_time`: When trade closed
- `quantity`: Amount of crypto
- `return_percent`: Return %
- `profit_loss`: Absolute P/L
- `stop_loss_price`: Stop-loss price
- `rsi_at_entry`: RSI value at entry
- `status`: 'open', 'closed', 'stopped_out'

### orders
- `id`: Primary key
- `alpaca_order_id`: Order ID from Alpaca API
- `symbol`: Trading pair
- `side`: BUY or SELL
- `quantity`: Order quantity
- `filled_qty`: Actually filled quantity
- `event`: Order event type
- `timestamp`: Order time

### account_balances
- `id`: Primary key
- `timestamp`: When snapshot was taken
- `cash`: Available cash
- `portfolio_value`: Total account value
- `buying_power`: Available buying power

### performance_metrics
- `id`: Primary key
- `metric_date`: Date of metrics
- `total_trades`: Number of trades
- `winning_trades`: Trades with positive return
- `win_rate`: % of winning trades
- `average_return`: Average return per trade
- `cumulative_return`: Total cumulative return
- `max_drawdown`: Largest peak-to-trough decline
- `sharpe_ratio`: Risk-adjusted return metric
- `profit_factor`: Gross profit / gross loss

## API Integration

### Alpaca Trade API
- Paper trading enabled by default (`ALPACA_PAPER=true`)
- Requires API key and secret key
- Supports market and limit orders

### Bitstamp API
- Used for historical OHLC data
- No authentication required (public endpoint)
- 1-hour candles for technical analysis

## Monitoring

### View Logs
```bash
# Local
tail -f app.log

# Render
# View in Render dashboard → Services → Logs
```

### Database Access
```bash
# Connect to PostgreSQL on Render
psql postgresql://user:password@host:5432/trading_bot_db
```

### Dashboard
- Open `https://<your-service-name>.onrender.com`
- View real-time price charts
- Monitor RSI and trading signals
- Track account performance

## Troubleshooting

### Trading Not Executing
1. Check `TRADING_ENABLED=true` in environment
2. Verify Alpaca API keys are correct
3. Ensure paper account has sufficient cash
4. Check logs for error messages

### Database Connection Failed
1. Verify `DATABASE_URL` is set correctly
2. On Render: Restart the database instance
3. Run `python init_db.py` manually
4. Check database credentials

### App Sleeping on Render Free Tier
- Free tier apps spin down after 15 minutes of inactivity
- **Upgrade to paid tier for 24/7 trading**
- Recommended: Standard plan ($7-15/month)

## Risk Disclaimer

**This bot trades REAL money on your Alpaca account. Use at your own risk.**

- Start with paper trading first
- Test with small amounts
- Monitor bot activity regularly
- Understand the risks of algorithmic trading
- Be aware of market volatility and gaps

## License

MIT License - see LICENSE file for details

## Support

For issues and feature requests, please open a GitHub issue or contact the maintainer.
