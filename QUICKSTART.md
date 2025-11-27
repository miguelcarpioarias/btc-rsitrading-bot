# Quick Reference - Render Deployment

## Files to Review

1. **`.env.example`** - Copy this to `.env` locally and fill in values
2. **`requirements.txt`** - Updated with all needed packages
3. **`app.py`** - Updated to use environment variables and database
4. **`database.py`** - New SQLAlchemy models
5. **`init_db.py`** - Run this to create database tables
6. **`Procfile`** - Deployment process configuration
7. **`render.yaml`** - Automated deployment configuration
8. **`README.md`** - Complete documentation
9. **`DEPLOYMENT.md`** - Step-by-step deployment guide
10. **`RENDER_READY.md`** - This implementation summary

## Quick Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Initialize database (requires PostgreSQL running)
python init_db.py

# Start the app
python app.py
```

### Database Queries
```bash
# Connect to PostgreSQL
psql $DATABASE_URL

# View trades
SELECT * FROM trades ORDER BY entry_time DESC;

# View orders
SELECT * FROM orders ORDER BY created_at DESC;

# View account balance history
SELECT * FROM account_balances ORDER BY timestamp DESC;

# Calculate performance
SELECT 
    COUNT(*) as total_trades,
    COUNT(CASE WHEN return_percent > 0 THEN 1 END) as wins,
    AVG(return_percent) as avg_return,
    SUM(profit_loss) as total_pnl
FROM trades WHERE status = 'closed';
```

## Environment Variables (Set in Render Dashboard)

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
DATABASE_URL=(auto-linked from PostgreSQL)
TRADING_ENABLED=true
LOG_LEVEL=INFO
```

## Render Services to Create

| Service | Type | Start Command |
|---------|------|--------------|
| Web | Web Service | `python init_db.py && gunicorn app:server --workers 2` |
| Scheduler | Background Worker | `python -u -c "from app import scheduler; import time; [time.sleep(1) for _ in iter(int, 1)]"` |
| Database | PostgreSQL | (Auto-configured) |

## Troubleshooting Checklist

- [ ] API keys set in environment variables
- [ ] PostgreSQL database created and running
- [ ] `DATABASE_URL` environment variable set
- [ ] `init_db.py` ran successfully
- [ ] Tables exist in PostgreSQL
- [ ] Web service logs show "Database initialized successfully"
- [ ] Worker service logs show "Trading scheduler started"
- [ ] Dashboard accessible at service URL
- [ ] Can see real-time price data on charts

## Key Features Implemented

✅ Security: API keys in environment variables  
✅ Database: Full trade/order persistence  
✅ Configuration: All parameters adjustable  
✅ 24/7 Trading: Background worker service  
✅ Logging: Structured error handling  
✅ Documentation: Comprehensive guides  
✅ Deployment: render.yaml for one-click setup  

## Testing Locally

```bash
# 1. Set up local PostgreSQL
# macOS: brew install postgresql
# Windows: Download PostgreSQL installer
# Linux: apt-get install postgresql

# 2. Create local database
createdb trading_bot_db

# 3. Update .env
DATABASE_URL=postgresql://localhost:5432/trading_bot_db

# 4. Initialize database
python init_db.py

# 5. Run the app
python app.py

# 6. Open browser
# Visit: http://localhost:10000
```

## Production Checklist Before Going Live

- [ ] Tested with paper trading (`ALPACA_PAPER=true`)
- [ ] Reviewed strategy parameters in `.env`
- [ ] Set appropriate stop-loss percentage
- [ ] Set appropriate position sizing
- [ ] Verified database backups available
- [ ] Tested Render logs monitoring
- [ ] Confirmed web service and worker both running
- [ ] Confirmed database tables have data
- [ ] Small position size tested
- [ ] Monitoring plan in place

## Support

- Render Docs: https://render.com/docs
- GitHub Issues: Create issue in repository
- Check logs: Render Dashboard → Services → Logs
