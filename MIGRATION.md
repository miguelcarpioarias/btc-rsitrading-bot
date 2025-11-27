# Migration Guide: From Basic to Production-Ready

This guide explains what changed and how to migrate your existing setup.

## What Changed

### Before (Basic Version)
- API keys hardcoded in `app.py` 😱
- In-memory data (lost on restart)
- No database
- Limited configuration options
- Single process (no background workers)
- No production deployment support

### After (Production-Ready)
- API keys in `.env` environment variables ✅
- Full PostgreSQL database ✅
- Data persists across restarts ✅
- All parameters configurable ✅
- Web + Worker processes ✅
- Render deployment ready ✅

## Migration Steps

### Step 1: Backup Your Data (If Any)

```bash
# If you have any existing trades or configuration
git commit -am "Backup before migration"
```

### Step 2: Update Your Repository

```bash
# Pull or merge the latest changes
git pull origin main

# Or if working locally:
git add .
git commit -m "Migration to production-ready setup"
```

### Step 3: Set Up Environment Variables

```bash
# Copy the example
cp .env.example .env

# Edit .env with your credentials
# Replace:
# - ALPACA_API_KEY=your_key_here
# - ALPACA_SECRET_KEY=your_secret_here
```

### Step 4: Install New Dependencies

```bash
# Update Python packages
pip install -r requirements.txt

# New packages added:
# - sqlalchemy>=2.0.0 (ORM)
# - psycopg2-binary>=2.9.0 (PostgreSQL driver)
# - python-dotenv>=1.0.0 (Environment variables)
# - alembic>=1.12.0 (Migrations)
```

### Step 5: Set Up Local Database (Optional but Recommended)

```bash
# For macOS
brew install postgresql
brew services start postgresql

# For Linux
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# For Windows
# Download and install from: https://www.postgresql.org/download/windows/

# Create database
createdb trading_bot_db

# Update .env
DATABASE_URL=postgresql://localhost/trading_bot_db
```

### Step 6: Initialize Database

```bash
python init_db.py

# You should see:
# "Database initialized successfully"
```

### Step 7: Test Locally

```bash
python app.py

# Check:
# - Dashboard loads at http://localhost:10000
# - No errors in console
# - Can see price charts
```

### Step 8: Deploy to Render

Follow the instructions in `DEPLOYMENT.md` to set up:
1. Create PostgreSQL database
2. Create Web Service
3. Create Background Worker
4. Set environment variables
5. Deploy

## Key Differences

### Configuration

**Before:**
```python
API_KEY = "PK93LZQTSB35L3CL60V5"  # Hardcoded!
api_secret = "HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0"  # Visible!
```

**After:**
```python
API_KEY = os.getenv('ALPACA_API_KEY')  # From .env
API_SECRET = os.getenv('ALPACA_SECRET_KEY')  # From environment
```

### Data Persistence

**Before:**
```python
trade_updates_list = []  # Lost on restart!
```

**After:**
```python
# Saves to PostgreSQL database
db_order = Order(
    symbol=order.symbol,
    quantity=float(order.qty),
    ...
)
session.add(db_order)
session.commit()  # Persisted!
```

### Configuration Parameters

**Before:**
```python
# Hardcoded in code
RSI_WINDOW = 14
VOLATILITY_THRESHOLD = 5.0
```

**After:**
```python
# Configurable in .env
RSI_WINDOW = int(os.getenv('RSI_WINDOW', 14))
VOLATILITY_THRESHOLD = float(os.getenv('VOLATILITY_THRESHOLD', 5.0))
```

### Deployment

**Before:**
```
# Procfile
web: gunicorn app:server
# Trading bot stops when app sleeps (free tier)
```

**After:**
```
# Procfile
release: python init_db.py
web: gunicorn app:server --workers 2
worker: python -c "from app import scheduler; ..."
# Web and worker run independently 24/7
```

## Database Schema

New tables created automatically:

```sql
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    entry_price FLOAT,
    exit_price FLOAT,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    quantity FLOAT,
    return_percent FLOAT,
    profit_loss FLOAT,
    stop_loss_price FLOAT,
    rsi_at_entry FLOAT,
    status VARCHAR(20),
    ...
);

CREATE TABLE orders (...);
CREATE TABLE account_balances (...);
CREATE TABLE performance_metrics (...);
```

## Backward Compatibility

### Old Data

If you want to migrate old data from the in-memory lists:
```python
# You'll need to manually import if you kept CSV/JSON exports
# The new system starts fresh
```

### Old Configuration

Old hardcoded settings are now environment variables:
```bash
# Old: Hardcoded in code
# New: In .env file
```

## Breaking Changes

1. **API Keys**: No longer works with hardcoded keys - must use `.env`
2. **Database**: Requires PostgreSQL (or SQLite for local dev)
3. **Data**: Not backward compatible - starts with empty database
4. **Process Management**: Now requires worker process for trading

## Troubleshooting Migration

### "ImportError: No module named 'sqlalchemy'"
```bash
pip install -r requirements.txt
```

### "DATABASE_URL not set"
```bash
# Check .env file has DATABASE_URL
cat .env | grep DATABASE_URL

# Or set manually
export DATABASE_URL=postgresql://localhost/trading_bot_db
```

### "Connection refused"
```bash
# Make sure PostgreSQL is running
psql postgresql://localhost/trading_bot_db -c "SELECT 1;"

# If on Render, check database service is running
```

### "ModuleNotFoundError: No module named 'database'"
```bash
# Make sure database.py is in project root
ls -la database.py

# And .env is configured
cat .env | grep DATABASE_URL
```

## Verification Checklist

After migration, verify:

- [ ] All environment variables in `.env`
- [ ] Database initialized (`python init_db.py` succeeds)
- [ ] App starts without errors (`python app.py`)
- [ ] Dashboard loads (`http://localhost:10000`)
- [ ] Trades are logged to database (check PostgreSQL)
- [ ] Orders are saved (check orders table)
- [ ] Account balance snapshots captured

## Performance Notes

### Local Development
- SQLite works but slower
- PostgreSQL recommended even locally
- Connection pooling enabled

### Production (Render)
- PostgreSQL scales well
- Background worker offloads trading
- Web service handles dashboard requests
- Database handles persistence

## Rollback Instructions

If you need to go back to the old version:

```bash
# Revert to previous commit
git revert HEAD

# Or reset to specific commit
git reset --hard <commit_hash>
```

## Next Steps

1. ✅ Update repository
2. ✅ Configure `.env`
3. ✅ Test locally
4. ✅ Deploy to Render
5. ✅ Monitor in production
6. ✅ Celebrate! 🎉

## Support

- Issues: GitHub Issues
- Docs: `DEPLOYMENT.md` and `README.md`
- Questions: Create GitHub Discussion
