"""Database models for trading bot"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Enum, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import enum

Base = declarative_base()

class OrderEvent(str, enum.Enum):
    """Order event types"""
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    PENDING = "pending"
    CANCELED = "canceled"
    EXPIRED = "expired"
    FAILED = "failed"

class Trade(Base):
    """Model for storing trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time = Column(DateTime)
    quantity = Column(Float, nullable=False)
    return_percent = Column(Float)  # (exit_price - entry_price) / entry_price * 100
    profit_loss = Column(Float)  # Absolute profit/loss in USD
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    status = Column(String(20), default='open')  # 'open', 'closed', 'stopped_out'
    rsi_at_entry = Column(Float)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Trade {self.symbol} {self.entry_time} {self.status}>"

class Order(Base):
    """Model for storing order events"""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    alpaca_order_id = Column(String(100), unique=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    filled_qty = Column(Float, default=0)
    filled_avg_price = Column(Float)
    event = Column(String(50), nullable=False)  # filled, partially_filled, pending, etc.
    status = Column(String(20), default='pending')
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<Order {self.alpaca_order_id} {self.symbol} {self.side}>"

class PerformanceMetric(Base):
    """Model for storing daily/hourly performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)  # Percentage
    average_return = Column(Float, default=0)  # Average return per trade
    cumulative_return = Column(Float, default=0)  # Total cumulative return
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    total_profit_loss = Column(Float, default=0)
    account_equity = Column(Float)  # Account value at end of period
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<PerformanceMetric {self.metric_date} WR:{self.win_rate}%>"

class AccountBalance(Base):
    """Model for tracking account balance over time"""
    __tablename__ = 'account_balances'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    cash = Column(Float, nullable=False)
    portfolio_value = Column(Float, nullable=False)
    buying_power = Column(Float)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<AccountBalance {self.timestamp} ${self.portfolio_value}>"

# Database connection
def get_database_url():
    """Get database URL from environment or use local SQLite"""
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        # Handle Render's postgres:// (deprecated) to postgresql://
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
    return db_url

def init_db():
    """Initialize database connection and create tables"""
    database_url = get_database_url()
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(
        database_url,
        echo=os.getenv('DEBUG', 'false').lower() == 'true',
        pool_size=10,
        max_overflow=20
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    return engine

def get_session():
    """Get database session"""
    database_url = get_database_url()
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(
        database_url,
        pool_size=10,
        max_overflow=20
    )
    
    Session = sessionmaker(bind=engine)
    return Session()

# For development/testing with SQLite
def init_dev_db(db_path='trading_bot.db'):
    """Initialize development database with SQLite"""
    db_url = f'sqlite:///{db_path}'
    engine = create_engine(db_url, echo=True)
    Base.metadata.create_all(engine)
    return engine
