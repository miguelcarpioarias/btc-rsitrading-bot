import os
import threading
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Load environment variables
load_dotenv()

# Alpaca clients (v1 SDK)
try:
    # Try v2 SDK first (newer)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    SDK_VERSION = 2
except ImportError:
    # Fall back to v1 SDK
    import alpaca_trade_api as tradeapi
    SDK_VERSION = 1
    # v1 SDK uses string constants for enums
    class OrderSide:
        BUY = 'buy'
        SELL = 'sell'
    class OrderType:
        MARKET = 'market'
        LIMIT = 'limit'
    class TimeInForce:
        DAY = 'day'
        GTC = 'gtc'
        OPENPAREN = 'opg'

# Database
from database import Trade, Order, PerformanceMetric, AccountBalance, init_db, get_session

# --- Configuration from Environment ---
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
DATABASE_URL = os.getenv('DATABASE_URL')

# Validate required environment variables
if not API_KEY or not API_SECRET:
    raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment variables")

# Initialize Alpaca client based on SDK version
if SDK_VERSION == 2:
    trade_client = TradingClient(API_KEY, API_SECRET, paper=ALPACA_PAPER)
else:
    # v1 SDK
    base_url = 'https://paper-api.alpaca.markets' if ALPACA_PAPER else 'https://api.alpaca.markets'
    trade_client = tradeapi.REST(API_KEY, API_SECRET, base_url=base_url)

# Symbols
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
ALPACA_SYMBOL = "BTC/USD"

# --- Strategy Parameters from Environment ---
RSI_WINDOW = int(os.getenv('RSI_WINDOW', 14))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', 30))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', 70))
SMA_WINDOW = int(os.getenv('SMA_WINDOW', 50))
ATR_WINDOW = int(os.getenv('ATR_WINDOW', 14))
VOLATILITY_THRESHOLD = float(os.getenv('VOLATILITY_THRESHOLD', 5.0))
RISK_PERCENT_PER_TRADE = float(os.getenv('RISK_PERCENT_PER_TRADE', 1.0))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 3.0))
TRADING_ENABLED = os.getenv('TRADING_ENABLED', 'true').lower() == 'true'
TRADING_INTERVAL_MINUTES = int(os.getenv('TRADING_INTERVAL_MINUTES', 1))

# Set up logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize database
try:
    if DATABASE_URL:
        engine = init_db()
        logger.info("Database initialized successfully")
    else:
        logger.warning("DATABASE_URL not set - using in-memory storage (not recommended for production)")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

# --- Streaming Order Updates ---
trade_updates_list = []

async def trade_updates_handler(update):
    try:
        event = update.event
        order = update.order
        trade_update = {
            'event': event,
            'symbol': order.symbol,
            'filled_qty': order.filled_qty,
            'filled_avg_price': order.filled_avg_price,
            'timestamp': update.timestamp
        }
        trade_updates_list.append(trade_update)
        
        # Save to database if DATABASE_URL is set
        if DATABASE_URL:
            try:
                session = get_session()
                db_order = Order(
                    alpaca_order_id=order.id,
                    symbol=order.symbol,
                    side=str(order.side),
                    quantity=float(order.qty),
                    filled_qty=float(order.filled_qty),
                    filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                    event=event,
                    status='filled' if float(order.filled_qty) > 0 else 'pending'
                )
                session.add(db_order)
                session.commit()
                session.close()
            except Exception as e:
                logger.error(f"Failed to save order to database: {e}")
    except Exception as e:
        logger.error(f"Stream handler error: {e}")

def start_trade_stream():
    if SDK_VERSION == 2:
        stream = TradingStream(API_KEY, API_SECRET, paper=True)
        stream.subscribe_trade_updates(trade_updates_handler)
        stream.run()
    else:
        logger.warning("Trade streaming not available with v1 SDK - this is normal for local testing")

threading.Thread(target=start_trade_stream, daemon=True).start()

# --- Technical Indicators & Fetching Candles ---
def compute_rsi(series, window=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, window=14):
    """Average True Range for volatility measurement"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD for momentum confirmation"""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_sma(series, window=50):
    """Simple Moving Average for trend filter"""
    return series.rolling(window).mean()

def fetch_bitstamp_candles(limit=1000, step=60):
    params = {'step': step, 'limit': limit}
    resp = requests.get(BITSTAMP_URL, params=params)
    data = resp.json().get('data', {}).get('ohlc', [])
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df.set_index('timestamp')

# --- Improved RSI Trading Strategy with Risk Management ---
active_trades = {}  # Track open positions for stop-loss management

def calculate_position_size(account_balance, risk_percent=None):
    """
    Kelly Criterion adapted approach: Risk configurable percentage of account per trade.
    Conservative compared to full Kelly to avoid drawdowns.
    """
    if risk_percent is None:
        risk_percent = RISK_PERCENT_PER_TRADE
    return (account_balance * risk_percent) / 100

def save_trade_to_db(trade_data):
    """Save trade to database"""
    if not DATABASE_URL:
        return
    
    try:
        session = get_session()
        trade = Trade(
            symbol=trade_data['symbol'],
            entry_price=trade_data['entry_price'],
            exit_price=trade_data.get('exit_price'),
            entry_time=trade_data['entry_time'],
            exit_time=trade_data.get('exit_time'),
            quantity=trade_data['quantity'],
            return_percent=trade_data.get('return_percent'),
            profit_loss=trade_data.get('profit_loss'),
            stop_loss_price=trade_data.get('stop_loss_price'),
            status=trade_data.get('status', 'open'),
            rsi_at_entry=trade_data.get('rsi_at_entry')
        )
        session.add(trade)
        session.commit()
        session.close()
        logger.info(f"Trade saved to database: {trade}")
    except Exception as e:
        logger.error(f"Failed to save trade to database: {e}")

def save_account_balance_to_db(cash, portfolio_value, buying_power=None):
    """Save account balance snapshot to database"""
    if not DATABASE_URL:
        return
    
    try:
        session = get_session()
        balance = AccountBalance(
            cash=cash,
            portfolio_value=portfolio_value,
            buying_power=buying_power
        )
        session.add(balance)
        session.commit()
        session.close()
    except Exception as e:
        logger.error(f"Failed to save account balance to database: {e}")

def rsi_trading_job():
    if not TRADING_ENABLED:
        logger.info("Trading is disabled")
        return
    
    try:
        df = fetch_bitstamp_candles(limit=1000, step=60)
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        
        # Compute technical indicators
        df['RSI'] = compute_rsi(df['close'], window=RSI_WINDOW)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['close'])
        df['ATR'] = compute_atr(df['high'], df['low'], df['close'], window=ATR_WINDOW)
        df['SMA50'] = compute_sma(df['close'], window=SMA_WINDOW)
        
        last_rsi = df['RSI'].iloc[-1]
        last_macd_hist = df['MACD_Hist'].iloc[-1]
        last_atr = df['ATR'].iloc[-1]
        price = df['close'].iloc[-1]
        sma50 = df['SMA50'].iloc[-1]
        volatility_pct = (last_atr / price) * 100
        
        account = trade_client.get_account()
        usd_avail = float(account.cash)
        account_balance = float(account.portfolio_value)
        
        # Save account balance snapshot
        save_account_balance_to_db(usd_avail, account_balance, float(account.buying_power))
        
        logger.info(f"Price: ${price:.2f} | RSI: {last_rsi:.2f} | ATR: {last_atr:.2f} ({volatility_pct:.2f}%) | SMA50: ${sma50:.2f}")
        
        # --- BUY SIGNAL CONDITIONS ---
        buy_signal = (
            last_rsi <= RSI_OVERSOLD and 
            last_macd_hist > 0 and 
            price > sma50 and 
            volatility_pct < VOLATILITY_THRESHOLD
        )
        
        if buy_signal and usd_avail >= 5:
            logger.info(f"🟢 BUY SIGNAL: RSI={last_rsi:.2f}, MACD={last_macd_hist:.4f}, Price${price:.2f} > SMA50${sma50:.2f}")
            
            target_usd = min(calculate_position_size(account_balance, RISK_PERCENT_PER_TRADE), usd_avail)
            buy_qty = round(target_usd / price - 1e-8, 8)
            
            if buy_qty > 0:
                mo = MarketOrderRequest(
                    symbol=ALPACA_SYMBOL,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    qty=buy_qty
                )
                resp = trade_client.submit_order(order_data=mo)
                
                stop_loss_price = price * (1 - STOP_LOSS_PERCENT / 100)
                active_trades[ALPACA_SYMBOL] = {
                    'entry_price': price,
                    'entry_qty': buy_qty,
                    'stop_loss': stop_loss_price,
                    'timestamp': datetime.now(ZoneInfo('America/New_York')),
                    'rsi_at_entry': last_rsi
                }
                
                # Save trade to database
                save_trade_to_db({
                    'symbol': ALPACA_SYMBOL,
                    'entry_price': price,
                    'entry_time': datetime.now(ZoneInfo('America/New_York')),
                    'quantity': buy_qty,
                    'stop_loss_price': stop_loss_price,
                    'status': 'open',
                    'rsi_at_entry': last_rsi
                })
                
                logger.info(f"BUY executed: +{buy_qty} BTC @ ${price:.2f} | Stop-Loss: ${stop_loss_price:.2f} | Risk: ${target_usd:.2f}")
        
        # --- SELL SIGNAL CONDITIONS ---
        sell_signal = (
            last_rsi >= RSI_OVERBOUGHT and 
            last_macd_hist < 0 and 
            price < sma50
        )
        
        if sell_signal:
            positions = trade_client.get_all_positions()
            clean_symbol = ALPACA_SYMBOL.replace("/", "")
            btc_pos = next((p for p in positions if p.symbol == clean_symbol), None)
            
            if btc_pos:
                logger.info(f"🔴 SELL SIGNAL: RSI={last_rsi:.2f}, MACD={last_macd_hist:.4f}, Price${price:.2f} < SMA50${sma50:.2f}")
                sell_qty = round(float(btc_pos.qty) - 1e-8, 8)
                
                mo = MarketOrderRequest(
                    symbol=ALPACA_SYMBOL,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    qty=sell_qty
                )
                resp = trade_client.submit_order(order_data=mo)
                logger.info(f"SELL executed: -{sell_qty} BTC @ ${price:.2f}")
                
                if ALPACA_SYMBOL in active_trades:
                    del active_trades[ALPACA_SYMBOL]
        
        # --- STOP-LOSS CHECK ---
        if ALPACA_SYMBOL in active_trades:
            trade = active_trades[ALPACA_SYMBOL]
            if price <= trade['stop_loss']:
                positions = trade_client.get_all_positions()
                clean_symbol = ALPACA_SYMBOL.replace("/", "")
                btc_pos = next((p for p in positions if p.symbol == clean_symbol), None)
                
                if btc_pos:
                    loss_pct = ((price - trade['entry_price']) / trade['entry_price']) * 100
                    logger.warning(f"🛑 STOP-LOSS TRIGGERED: Price ${price:.2f} <= Stop ${trade['stop_loss']:.2f} | Loss: {loss_pct:.2f}%")
                    
                    sell_qty = round(float(btc_pos.qty) - 1e-8, 8)
                    mo = MarketOrderRequest(
                        symbol=ALPACA_SYMBOL,
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC,
                        qty=sell_qty
                    )
                    resp = trade_client.submit_order(order_data=mo)
                    logger.info(f"Stop-loss SELL executed: -{sell_qty} BTC @ ${price:.2f}")
                    
                    del active_trades[ALPACA_SYMBOL]
        
        else:
            logger.debug("No trade signal (insufficient confluence or high volatility)")

    except Exception as e:
        logger.error(f"RSI trading job error: {e}", exc_info=True)

# Schedule RSI job with configurable interval
scheduler = BackgroundScheduler(timezone='US/Eastern')
scheduler.add_job(rsi_trading_job, 'interval', minutes=TRADING_INTERVAL_MINUTES)
scheduler.start()

logger.info(f"Scheduler started: RSI trading job every {TRADING_INTERVAL_MINUTES} minute(s)")

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Modern color scheme
brand_colors = {
    'background': '#0f1419',
    'text': '#e0e6ed',
    'accent': '#1f77b4',
    'success': '#00ff41',
    'danger': '#ff4444'
}

app.layout = dbc.Container([
    html.Div(
        children=[
            html.H1(
                '₿ Crypto Trading Dashboard',
                style={
                    'color': brand_colors['text'],
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'fontSize': '2.5rem',
                    'fontWeight': 'bold',
                    'letterSpacing': '2px'
                }
            ),
            html.P(
                'Advanced RSI Strategy with Multi-Indicator Confirmation',
                style={
                    'color': '#888888',
                    'textAlign': 'center',
                    'marginBottom': '30px',
                    'fontSize': '1.1rem'
                }
            )
        ],
        style={'marginBottom': '40px'}
    ),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Quick Trade', style={'color': brand_colors['text'], 'marginBottom': '15px'}),
                    html.Label('BTC Quantity', style={'color': brand_colors['text'], 'fontWeight': 'bold'}),
                    dcc.Input(
                        id='btc-qty',
                        type='number',
                        value=0.5,
                        step=0.1,
                        style={
                            'width': '100%',
                            'padding': '8px',
                            'marginBottom': '15px',
                            'backgroundColor': '#16213e',
                            'color': brand_colors['text'],
                            'border': '1px solid #404040',
                            'borderRadius': '4px'
                        }
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button(
                            '🔷 BUY', id='buy-btc', color='success',
                            className='w-100',
                            style={'backgroundColor': brand_colors['success'], 'color': '#000', 'fontWeight': 'bold'}
                        )),
                        dbc.Col(dbc.Button(
                            '🔶 SELL', id='sell-btc', color='danger',
                            className='w-100',
                            style={'backgroundColor': brand_colors['danger'], 'color': '#fff', 'fontWeight': 'bold'}
                        ))
                    ], className='g-2'),
                    html.Div(
                        id='order-status',
                        style={
                            'color': brand_colors['text'],
                            'marginTop': '15px',
                            'fontSize': '0.9rem',
                            'textAlign': 'center'
                        }
                    )
                ])
            ], style={'backgroundColor': '#16213e', 'borderColor': '#404040'})
        ], width=12, lg=3, className='mb-4'),
        dbc.Col(dcc.Graph(id='price-chart'), width=12, lg=9, className='mb-4')
    ], className='g-4'),

    dbc.Row(dcc.Graph(id='rsi-chart'), className='mt-2 g-4'),

    dbc.Row(dcc.Graph(id='performance-chart'), className='mt-2 g-4'),

    dcc.Interval(id='interval', interval=30*1000, n_intervals=0),

    dbc.Row([
        dbc.Col([
            html.H5('Positions', style={'color': brand_colors['text'], 'marginBottom': '15px'}),
            dash_table.DataTable(
                id='positions-table',
                page_size=10,
                style_header={
                    'backgroundColor': '#16213e',
                    'color': brand_colors['text'],
                    'fontWeight': 'bold',
                    'border': '1px solid #404040'
                },
                style_cell={
                    'backgroundColor': '#0f1419',
                    'color': brand_colors['text'],
                    'border': '1px solid #404040',
                    'padding': '12px',
                    'textAlign': 'center'
                },
                style_data={'height': '35px'}
            )
        ], className='mt-4')
    ], className='g-4'),

    html.H5('Order Stream', style={'color': brand_colors['text'], 'marginTop': '30px', 'marginBottom': '15px'}),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id='orders-table',
            page_size=10,
            style_header={
                'backgroundColor': '#16213e',
                'color': brand_colors['text'],
                'fontWeight': 'bold',
                'border': '1px solid #404040'
            },
            style_cell={
                'backgroundColor': '#0f1419',
                'color': brand_colors['text'],
                'border': '1px solid #404040',
                'padding': '12px',
                'textAlign': 'center'
            },
            style_data={'height': '35px'}
        ))
    ], className='g-4 mb-5')

], fluid=True, style={'backgroundColor': brand_colors['background'], 'padding': '30px', 'minHeight': '100vh'})

# --- Callbacks ---
@app.callback(
    Output('price-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_price(n):
    df = fetch_bitstamp_candles(limit=1000, step=60)
    display_idx = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')
    
    # Add SMA50 for reference
    df['SMA50'] = compute_sma(df['close'], window=50)
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=display_idx,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='BTC/USD',
        increasing_line_color='#00ff41',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='#00cc33',
        decreasing_fillcolor='#cc0000',
        hovertext=[f"O: ${o:.2f}<br>H: ${h:.2f}<br>L: ${l:.2f}<br>C: ${c:.2f}" 
                  for o, h, l, c in zip(df['open'], df['high'], df['low'], df['close'])],
        hoverinfo='x+text'
    ))
    
    # SMA50 line
    fig.add_trace(go.Scatter(
        x=display_idx,
        y=df['SMA50'],
        mode='lines',
        name='SMA50',
        line=dict(color='#FFA500', width=2, dash='dash'),
        hovertemplate='<b>SMA50</b><br>%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='<b>Bitcoin Price (BTC/USD)</b>', font=dict(size=18)),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#eaeaea', family='Arial, sans-serif', size=11),
        xaxis=dict(
            title='<b>Time (ET)</b>',
            tickformat='%H:%M<br>%b %d',
            showgrid=True,
            gridwidth=1,
            gridcolor='#404040',
            showline=True,
            linewidth=1,
            linecolor='#404040'
        ),
        yaxis=dict(
            title='<b>Price (USD)</b>',
            showgrid=True,
            gridwidth=1,
            gridcolor='#404040',
            showline=True,
            linewidth=1,
            linecolor='#404040'
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', bordercolor='#404040', borderwidth=1),
        margin=dict(l=60, r=60, t=60, b=60),
        height=500
    )
    return fig

@app.callback(
    Output('rsi-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_rsi_chart(n):
    df = fetch_bitstamp_candles(limit=1000, step=60)
    df['RSI'] = compute_rsi(df['close'], window=14)
    display_idx = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')
    
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(go.Scatter(
        x=display_idx,
        y=df['RSI'],
        mode='lines',
        name='RSI(14)',
        line=dict(color='#1f77b4', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='<b>RSI</b><br>%{y:.2f}<extra></extra>'
    ))
    
    # Overbought zone (RSI >= 70)
    fig.add_hrect(y0=70, y1=100, fillcolor='#ff4444', opacity=0.1, 
                  layer='below', line_width=0, name='Overbought')
    fig.add_hline(y=70, line_dash='dash', line_color='#ff4444', line_width=1, 
                  annotation_text='Overbought (70)', annotation_position='right')
    
    # Oversold zone (RSI <= 30)
    fig.add_hrect(y0=0, y1=30, fillcolor='#00ff41', opacity=0.1,
                  layer='below', line_width=0, name='Oversold')
    fig.add_hline(y=30, line_dash='dash', line_color='#00ff41', line_width=1,
                  annotation_text='Oversold (30)', annotation_position='right')
    
    # Middle line (neutral)
    fig.add_hline(y=50, line_dash='dot', line_color='#888888', line_width=1, opacity=0.5)
    
    fig.update_layout(
        title=dict(text='<b>RSI(14) - Relative Strength Index</b>', font=dict(size=18)),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#eaeaea', family='Arial, sans-serif', size=11),
        xaxis=dict(
            title='<b>Time (ET)</b>',
            tickformat='%H:%M<br>%b %d',
            showgrid=True,
            gridwidth=1,
            gridcolor='#404040',
            showline=True,
            linewidth=1,
            linecolor='#404040'
        ),
        yaxis=dict(
            title='<b>RSI Value</b>',
            range=[0, 100],
            showgrid=True,
            gridwidth=1,
            gridcolor='#404040',
            showline=True,
            linewidth=1,
            linecolor='#404040'
        ),
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', bordercolor='#404040', borderwidth=1),
        margin=dict(l=60, r=60, t=60, b=60),
        height=400,
        showlegend=False
    )
    return fig

@app.callback(
    Output('order-status', 'children'),
    Input('buy-btc', 'n_clicks'),
    Input('sell-btc', 'n_clicks'),
    State('btc-qty', 'value')
)
def execute_manual_order(buy, sell, qty):
    ctx = callback_context.triggered_id
    if not ctx:
        return ''
    side = OrderSide.BUY if ctx == 'buy-btc' else OrderSide.SELL
    positions = trade_client.get_all_positions()
    available = float(positions[0].qty) if positions else 0.0
    order_qty = min(qty, available) if side == OrderSide.SELL else qty
    mo = MarketOrderRequest(
        symbol=ALPACA_SYMBOL,
        side=side,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=order_qty
    )
    try:
        resp = trade_client.submit_order(order_data=mo)
        return f"✅ Order {resp.id} submitted (filled_qty={resp.filled_qty})"
    except Exception as e:
        return f"❌ Order failed: {e}"

@app.callback(
    Output('positions-table', 'data'),
    Output('positions-table', 'columns'),
    Input('interval', 'n_intervals')
)
def update_positions(n):
    rows = []
    try:
        for p in trade_client.get_all_positions() or []:
            if p.symbol == ALPACA_SYMBOL.replace('/', ''):
                rows.append({
                    'Symbol': p.symbol,
                    'Qty': p.qty,
                    'Unrealized P/L': p.unrealized_pl,
                    'Market Value': p.market_value
                })
    except Exception:
        rows = []
    if not rows:
        rows = [{'Symbol': 'None', 'Qty': 0, 'Unrealized P/L': 0, 'Market Value': 0}]
    cols = [{'name': c, 'id': c} for c in rows[0].keys()]
    return rows, cols

@app.callback(
    Output('orders-table', 'data'),
    Output('orders-table', 'columns'),
    Input('interval', 'n_intervals')
)
def update_orders(n):
    rows = trade_updates_list[-20:] or [{
        'event': 'None', 'symbol': '', 'filled_qty': 0, 'filled_avg_price': 0, 'timestamp': ''
    }]
    cols = [{'name': k, 'id': k} for k in rows[0].keys()]
    return rows, cols

# ← Enhanced callback for performance metrics with Sharpe Ratio & Drawdown
@app.callback(
    Output('performance-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_performance(n):
    df = fetch_bitstamp_candles(limit=1000, step=60)
    df['RSI'] = compute_rsi(df['close'], window=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['close'])
    df['SMA50'] = compute_sma(df['close'], window=50)

    trades = []
    in_position = False
    buy_price = buy_time = None

    for ts, row in df.iterrows():
        price = row['close']
        rsi = row['RSI']
        macd_hist = row['MACD_Hist']
        sma50 = row['SMA50']
        
        # Improved entry logic with multi-indicator confirmation
        if not in_position and rsi <= 30 and macd_hist > 0 and price > sma50:
            in_position = True
            buy_price = price
            buy_time = ts
        
        # Improved exit logic with multi-indicator confirmation
        elif in_position and rsi >= 70 and macd_hist < 0 and price < sma50:
            ret = (price - buy_price) / buy_price * 100
            trades.append({
                'buy_time': buy_time,
                'sell_time': ts,
                'return': ret,
                'entry_price': buy_price,
                'exit_price': price
            })
            in_position = False

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trades executed", showarrow=False)
        return fig

    # Calculate performance metrics
    trades_df['cumulative'] = trades_df['return'].cumsum()
    
    # Win rate
    wins = int((trades_df['return'] > 0).sum())
    losses = int((trades_df['return'] <= 0).sum())
    total_trades = len(trades_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Average return per trade
    avg_return = trades_df['return'].mean()
    
    # Sharpe Ratio (assuming 252 trading days, risk-free rate ≈ 0)
    returns = trades_df['return'].values
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252) if len(returns) > 1 else 0
    
    # Maximum Drawdown
    cumulative = trades_df['cumulative'].values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max)
    max_drawdown = drawdown.min()
    
    # Profit Factor (gross profit / gross loss)
    gross_profit = trades_df[trades_df['return'] > 0]['return'].sum()
    gross_loss = abs(trades_df[trades_df['return'] <= 0]['return'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=trades_df['sell_time'],
        y=trades_df['return'],
        marker_color=['#00ff41' if r>0 else '#ff4444' for r in trades_df['return']],
        marker_line=dict(color=['#00cc33' if r>0 else '#cc0000' for r in trades_df['return']], width=1.5),
        name='Trade Return (%)',
        hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>',
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=trades_df['sell_time'],
        y=trades_df['cumulative'],
        mode='lines+markers',
        name='Cumulative P&L (%)',
        yaxis='y2',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4', line=dict(color='#0d4a8e', width=2)),
        hovertemplate='<b>%{x}</b><br>Cumulative P&L: %{y:.2f}%<extra></extra>',
        fillcolor='rgba(31, 119, 180, 0.1)',
        fill='tozeroy'
    ))

    # Title with key metrics
    title_text = (
        f"<b>Improved RSI Strategy Performance</b><br>"
        f"<sub>Trades: {total_trades} | Win Rate: {win_rate:.1f}% ({wins}W/{losses}L) | "
        f"Avg Return: {avg_return:.2f}% | Sharpe Ratio: {sharpe_ratio:.2f} | "
        f"Max Drawdown: {max_drawdown:.2f}% | Profit Factor: {profit_factor:.2f}</sub>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#eaeaea', family='Arial, sans-serif', size=11),
        xaxis=dict(
            title='<b>Exit Time</b>',
            showgrid=True,
            gridwidth=1,
            gridcolor='#404040',
            showline=True,
            linewidth=1,
            linecolor='#404040'
        ),
        yaxis=dict(
            title='<b>Return per Trade (%)</b>',
            showgrid=True,
            gridwidth=1,
            gridcolor='#404040',
            showline=True,
            linewidth=1,
            linecolor='#404040'
        ),
        yaxis2=dict(
            title='<b>Cumulative P&L (%)</b>',
            overlaying='y',
            side='right',
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='#404040'
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', bordercolor='#404040', borderwidth=1),
        margin=dict(l=70, r=70, t=100, b=70),
        height=550,
        hovermode='x unified',
        dragmode='zoom'
    )

    return fig

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
