# Render Deployment Guide

## Quick Start

### 1. Prepare Your Repository

Ensure these files are in your repository:
- `app.py` - Main Flask/Dash application
- `database.py` - SQLAlchemy models
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template
- `Procfile` - Process configuration
- `render.yaml` - Render service configuration (optional)
- `init_db.py` - Database initialization script
- `runtime.txt` - Python version

### 2. Set Up on Render

#### Option A: Manual Setup (Recommended for first-time users)

1. **Create PostgreSQL Database**
   - Dashboard → New → PostgreSQL
   - Name: `trading-bot-db`
   - Plan: Standard
   - Region: Oregon
   - Create

2. **Create Web Service**
   - Dashboard → New → Web Service
   - Connect your GitHub repository
   - Name: `btc-trading-bot`
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python init_db.py && gunicorn app:server --workers 2 --worker-class sync --timeout 120`
   - Plan: Standard (paid)

3. **Set Environment Variables** (in Web Service settings)
   - `ALPACA_API_KEY`: Your API key from alpaca.markets
   - `ALPACA_SECRET_KEY`: Your API secret
   - `ALPACA_PAPER`: `true` (or `false` for live trading)
   - `DATABASE_URL`: (Auto-linked from PostgreSQL)
   - `TRADING_ENABLED`: `true`
   - `LOG_LEVEL`: `INFO`

4. **Create Background Worker** (for 24/7 trading)
   - Dashboard → New → Background Worker
   - Same repository
   - Name: `btc-trading-bot-scheduler`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python -u -c "from app import scheduler; import time; print('Trading scheduler started'); [time.sleep(1) for _ in iter(int, 1)]"`
   - Same environment variables as Web Service
   - Plan: Standard (paid)

#### Option B: Automatic Setup with render.yaml

1. Push your repository with `render.yaml`:
   ```bash
   git push origin main
   ```

2. Dashboard → New → Blueprint → Select repository
3. Render will automatically create:
   - Web service
   - Background worker
   - PostgreSQL database
   - Environment variables

### 3. Verify Deployment

1. Check Web Service Logs
   - Should see: "Database initialized successfully"
   - Should see: "Scheduler started"

2. Access Dashboard
   - Visit: `https://<your-service-name>.onrender.com`
   - Should display trading bot dashboard

3. Check Database
   - In PostgreSQL service, confirm tables exist:
   - `trades`, `orders`, `account_balances`, `performance_metrics`

### 4. Monitor Trading

```bash
# View real-time logs
Render Dashboard → Services → btc-trading-bot → Logs

# Connect to database
psql postgresql://user:password@your-host:5432/trading_bot_db
SELECT COUNT(*) FROM trades;
SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;
```

## Troubleshooting

### Issue: "DATABASE_URL not set"

**Solution:**
1. Verify PostgreSQL service is running
2. In Web Service settings, link the PostgreSQL service
3. Re-deploy

### Issue: "ImportError: No module named 'sqlalchemy'"

**Solution:**
1. Ensure `requirements.txt` has `sqlalchemy>=2.0.0`
2. Build Command must be: `pip install -r requirements.txt`
3. Re-deploy

### Issue: Trading not executing (web only, no worker)

**Solution:**
1. Create a Background Worker service
2. Start Command: `python -u -c "from app import scheduler; import time; print('Scheduler started'); [time.sleep(1) for _ in iter(int, 1)]"`
3. This keeps the scheduler running 24/7

### Issue: App crashes on startup

**Solution:**
1. Check logs for specific error
2. Verify all environment variables are set
3. Ensure Alpaca API credentials are correct
4. Run `init_db.py` manually to verify DB connection

```bash
# SSH into Render instance
psql $DATABASE_URL -c "SELECT 1"  # Test DB connection
```

### Issue: Trades executing but not saving to DB

**Solution:**
1. Verify DATABASE_URL environment variable
2. Check `trades` table permissions
3. Look for SQL errors in logs

```bash
psql $DATABASE_URL -c "SELECT * FROM trades;"
```

## Scaling Considerations

### Free Tier (Not recommended for trading)
- App spins down after 15 minutes of inactivity
- Trading will not execute during sleep
- Good for testing only

### Standard Tier ($7-15/month)
- Recommended for active trading
- Keeps app and worker running 24/7
- Includes PostgreSQL database

### Professional Tier ($25+/month)
- High-performance instances
- More background workers if needed
- Better for high-frequency trading

## Cost Breakdown

| Service | Free | Standard | Professional |
|---------|------|----------|--------------|
| Web Service | $0 | $7/mo | $25/mo |
| Worker | N/A | $7/mo | $25/mo |
| PostgreSQL | $0 | $15/mo | $80/mo |
| **Total** | $0 | **$29/mo** | **$130/mo** |

## Best Practices

1. **Use Paper Trading First**
   - Set `ALPACA_PAPER=true`
   - Verify bot logic before live trading

2. **Monitor Regularly**
   - Check logs daily
   - Review trades in database
   - Track performance metrics

3. **Backup Your Data**
   - Periodically export trades from database
   - Keep API keys secure

4. **Update Strategy Parameters**
   - Test new RSI thresholds carefully
   - Adjust stop-loss based on volatility
   - Monitor Sharpe ratio and drawdown

5. **Use Alerts**
   - Set up error notifications
   - Monitor system resources
   - Check database size regularly

## Integration with GitHub Actions (Optional)

Create `.github/workflows/deploy.yml` for auto-deployment on push:

```yaml
name: Deploy to Render
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Trigger Render deployment
        run: |
          curl -X POST https://api.render.com/deploy/srv-XXXXX?key=${{ secrets.RENDER_DEPLOY_KEY }}
```

## Support

For Render-specific issues:
- Render Documentation: https://render.com/docs
- Render Support: https://render.com/support

For bot-specific issues:
- GitHub Issues: https://github.com/miguelcarpioarias/btc-rsitrading-bot/issues
