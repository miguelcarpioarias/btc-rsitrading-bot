# ✅ Pre-Deployment Checklist

Use this checklist before deploying to Render.

## Local Setup

- [ ] Clone repository
- [ ] Create Python virtual environment
- [ ] Run `pip install -r requirements.txt`
- [ ] Copy `.env.example` to `.env`
- [ ] Edit `.env` with Alpaca API credentials:
  - `ALPACA_API_KEY=your_key`
  - `ALPACA_SECRET_KEY=your_secret`
- [ ] Set `ALPACA_PAPER=true` (or `false` for live trading)
- [ ] Set `DATABASE_URL` if testing with PostgreSQL locally

## Local Testing

- [ ] PostgreSQL installed and running (optional but recommended)
- [ ] Run `python init_db.py` successfully
- [ ] Run `python app.py` without errors
- [ ] Dashboard loads at `http://localhost:10000`
- [ ] Price charts display data
- [ ] RSI and MACD indicators visible
- [ ] Performance chart shows (or "No trades executed" message)
- [ ] Can see Buy/Sell buttons in UI

## Configuration Review

- [ ] Strategy parameters reviewed in `.env`:
  - [ ] RSI thresholds (30/70) appropriate
  - [ ] SMA window (50) correct
  - [ ] Stop-loss % (3%) acceptable
  - [ ] Risk per trade (1%) suitable
- [ ] Timeframe appropriate (1-minute default)
- [ ] Volatility threshold set (5% default)
- [ ] Paper trading enabled for testing

## Security Check

- [ ] API keys NOT visible in `app.py`
- [ ] API keys only in `.env`
- [ ] `.env` is gitignored (check `.gitignore`)
- [ ] `.env` not committed to repository
- [ ] `.env.example` has placeholder values only
- [ ] No API keys in logs or error messages
- [ ] No API keys in documentation files

## Git/GitHub Check

- [ ] Repository pushed to GitHub
- [ ] All new files committed:
  - [ ] `database.py`
  - [ ] `init_db.py`
  - [ ] `render.yaml`
  - [ ] `DEPLOYMENT.md`
  - [ ] `RENDER_READY.md`
  - [ ] `QUICKSTART.md`
  - [ ] `MIGRATION.md`
  - [ ] `IMPLEMENTATION_COMPLETE.md`
  - [ ] `.env.example`
- [ ] `.env` NOT committed (should be in .gitignore)
- [ ] `requirements.txt` includes all dependencies
- [ ] `Procfile` updated with release/web/worker
- [ ] `README.md` updated with deployment instructions

## Render Setup - Database

- [ ] PostgreSQL service created on Render
- [ ] Database name: `trading_bot_db`
- [ ] Database region selected (Oregon or other)
- [ ] Database credentials available
- [ ] Can connect to database (test connection)
- [ ] Empty database ready for tables

## Render Setup - Web Service

- [ ] Web service created
- [ ] GitHub repository connected
- [ ] Build command: `pip install -r requirements.txt`
- [ ] Start command: `python init_db.py && gunicorn app:server --workers 2 --worker-class sync --timeout 120`
- [ ] Python version: 3.12
- [ ] Plan: Standard (paid tier)
- [ ] Region selected

## Render Setup - Environment Variables

Set these in Web Service settings:
- [ ] `ALPACA_API_KEY` = Your Alpaca API key
- [ ] `ALPACA_SECRET_KEY` = Your Alpaca secret
- [ ] `ALPACA_PAPER` = true (for testing) or false (for live)
- [ ] `DATABASE_URL` = Auto-linked from PostgreSQL service
- [ ] `TRADING_ENABLED` = true
- [ ] `LOG_LEVEL` = INFO
- [ ] `DEBUG` = false
- [ ] Any custom strategy parameters (optional)

## Render Setup - Worker Service

- [ ] Background worker created
- [ ] Same GitHub repository connected
- [ ] Build command: `pip install -r requirements.txt`
- [ ] Start command: `python -u -c "from app import scheduler; import time; print('Trading scheduler started'); [time.sleep(1) for _ in iter(int, 1)]"`
- [ ] Same environment variables as Web Service
- [ ] Plan: Standard (paid tier)

## Deployment

- [ ] Double-checked all settings
- [ ] Confirmed paper trading is ON (`ALPACA_PAPER=true`)
- [ ] Push to GitHub or click Deploy
- [ ] Wait for services to start

## Post-Deployment Verification

### Web Service
- [ ] Deployment completed successfully
- [ ] Logs show "Database initialized successfully"
- [ ] No error messages in logs
- [ ] Service URL generated

### Worker Service
- [ ] Deployment completed successfully
- [ ] Logs show "Trading scheduler started"
- [ ] No error messages in logs
- [ ] Service running

### Database
- [ ] Tables created successfully
- [ ] Can query tables from CLI

### Application
- [ ] Dashboard accessible at service URL
- [ ] Charts load with real data
- [ ] Price updating in real-time
- [ ] RSI and indicators visible
- [ ] No JavaScript errors in browser console

### Trading
- [ ] Check logs for "BUY SIGNAL" or "SELL SIGNAL"
- [ ] Query database: `SELECT * FROM trades;`
- [ ] Query database: `SELECT * FROM orders;`
- [ ] Confirm trades are being logged

## Production Checklist

### Before Live Trading
- [ ] Paper trading tested and working
- [ ] All technical indicators working correctly
- [ ] Database persisting trades properly
- [ ] Logs monitoring set up
- [ ] Alert system ready (if applicable)
- [ ] Stop-loss mechanism tested
- [ ] Position sizing appropriate
- [ ] Risk management confirmed

### Going Live
- [ ] Set `ALPACA_PAPER=false` in environment
- [ ] Restart services to apply changes
- [ ] Monitor closely first hour
- [ ] Review first few trades
- [ ] Check profit/loss calculation
- [ ] Verify stop-loss executions

### Ongoing Monitoring
- [ ] Check logs daily
- [ ] Review trades in database weekly
- [ ] Monitor performance metrics
- [ ] Adjust parameters as needed
- [ ] Backup database monthly
- [ ] Review Sharpe ratio and drawdown

## Troubleshooting Issues

### If deployment fails:
- [ ] Check build logs for errors
- [ ] Verify all environment variables set
- [ ] Ensure requirements.txt is complete
- [ ] Check Procfile format
- [ ] Verify database is running

### If dashboard doesn't load:
- [ ] Check web service logs
- [ ] Verify PORT environment variable (should be 10000)
- [ ] Check database connection in logs
- [ ] Verify API credentials in logs (should see "authenticated")

### If trading not executing:
- [ ] Check `TRADING_ENABLED=true`
- [ ] Verify worker service is running
- [ ] Check worker logs for errors
- [ ] Verify Alpaca account has cash
- [ ] Check RSI values in logs

### If database errors:
- [ ] Verify `DATABASE_URL` is set
- [ ] Ensure PostgreSQL service is running
- [ ] Check database user permissions
- [ ] Verify tables were created (`init_db.py` ran)

## Success Indicators

✅ Dashboard loads without errors  
✅ Charts show real-time price data  
✅ Technical indicators calculating correctly  
✅ Trades appearing in database  
✅ Orders logged in orders table  
✅ Account balance snapshots saved  
✅ Performance metrics calculated  
✅ Logs show normal operation  

## Support Resources

- **Documentation**: `README.md`, `DEPLOYMENT.md`
- **Quick Help**: `QUICKSTART.md`
- **Issues**: GitHub repository issues
- **Troubleshooting**: `DEPLOYMENT.md` troubleshooting section
- **Render Help**: https://render.com/support

---

## Final Review

Before clicking "Deploy" on Render:

1. ✅ All security checks passed
2. ✅ All configuration correct
3. ✅ All environment variables set
4. ✅ Database service running
5. ✅ Tests passed locally
6. ✅ Paper trading enabled
7. ✅ Repository committed and pushed

**You're ready to deploy!** 🚀

Once deployed:
- Monitor logs closely for first 24 hours
- Verify trades are executing and saving
- Review performance metrics
- Adjust parameters if needed
- Scale up gradually if satisfied

Happy trading! 📈
