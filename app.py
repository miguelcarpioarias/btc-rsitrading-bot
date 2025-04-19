import os
import threading
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import requests
import numpy as np

# Alpaca clients
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, AssetClass
from alpaca.trading.requests import MarketOrderRequest

# ===== Configuration =====
API_KEY = os.getenv('ALPACA_API_KEY_ID', "PK93LZQTSB35L3CL60V5")
API_SECRET = os.getenv('ALPACA_API_SECRET_KEY', "HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0")
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Symbols and endpoints
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
ALPACA_SYMBOL = 'BTC/USD'  # Alpaca trading symbol
EXCHANGE = 'FTXU'  # Required for crypto trading

# Trading parameters
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
TRADE_QTY = 0.25  # Fixed BTC quantity per trade

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== Trading Stream Setup =====
trade_updates_list = []

async def trade_updates_handler(update):
    """Handle real-time trade updates"""
    try:
        trade_updates_list.append({
            'event': update.event,
            'symbol': update.order.symbol,
            'filled_qty': update.order.filled_qty,
            'filled_avg_price': update.order.filled_avg_price,
            'timestamp': update.timestamp.isoformat()
        })
    except Exception as e:
        logger.error(f"Trade update error: {str(e)}")

def start_trade_stream():
    """Initialize and maintain trading websocket connection"""
    while True:
        try:
            stream = TradingStream(API_KEY, API_SECRET, paper=True)
            stream.subscribe_trade_updates(trade_updates_handler)
            stream.run()
        except Exception as e:
            logger.error(f"Stream connection error: {str(e)}")
            time.sleep(5)

# ===== Core Trading Logic =====
def compute_rsi(series, window=14):
    """Calculate RSI with proper edge case handling"""
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def fetch_bitstamp_candles(limit=1000, step=60):
    """Fetch OHLC data from Bitstamp with error handling"""
    try:
        params = {'step': step, 'limit': limit}
        resp = requests.get(BITSTAMP_URL, params=params, timeout=10)
        resp.raise_for_status()
        
        data = resp.json().get('data', {}).get('ohlc', [])
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        return df.set_index('timestamp').iloc[::-1]  # Reverse to chronological order
    except Exception as e:
        logger.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()

def execute_trade(side: OrderSide):
    """Execute market order with proper crypto parameters"""
    try:
        mo = MarketOrderRequest(
            symbol=ALPACA_SYMBOL,
            qty=TRADE_QTY,
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            asset_class=AssetClass.CRYPTO,
            exchange=EXCHANGE
        )
        order = trade_client.submit_order(order_data=mo)
        logger.info(f"Submitted {side} order: {order.id}")
        return order
    except Exception as e:
        logger.error(f"Order failed: {str(e)}")
        return None

def get_crypto_position():
    """Get current BTC position with proper error handling"""
    try:
        positions = trade_client.get_all_positions()
        for p in positions:
            if p.symbol == ALPACA_SYMBOL and float(p.qty) > 0:
                return float(p.qty)
        return 0.0
    except Exception as e:
        logger.error(f"Position check error: {str(e)}")
        return 0.0

def rsi_trading_job():
    """Main trading logic executed periodically"""
    try:
        # 1. Fetch and prepare data
        df = fetch_bitstamp_candles(limit=1000, step=60)
        if df.empty:
            return
            
        df = df.tz_localize('UTC').tz_convert('America/New_York')
        df['RSI'] = compute_rsi(df['close'])
        last_rsi = df['RSI'].iloc[-1]
        
        # 2. Get current position
        current_position = get_crypto_position()
        logger.info(f"RSI: {last_rsi:.2f} | Position: {current_position:.4f} BTC")

        # 3. Trading logic
        if last_rsi <= RSI_OVERSOLD and current_position == 0:
            logger.info("RSI <=30 - BUY SIGNAL")
            execute_trade(OrderSide.BUY)
            
        elif last_rsi >= RSI_OVERBOUGHT and current_position >= TRADE_QTY:
            logger.info("RSI >=70 - SELL SIGNAL")
            execute_trade(OrderSide.SELL)

    except Exception as e:
        logger.error(f"Trading job error: {str(e)}")

# ===== Dashboard Setup =====
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
brand_colors = {'background': '#2C3E50', 'text': '#ECF0F1'}

app.layout = dbc.Container([
    html.H2('BTC/USD RSI Trading Bot', style={'color': brand_colors['text']}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Live Metrics"),
                dbc.CardBody([
                    html.Div(id='live-rsi', style={'color': brand_colors['text']}),
                    html.Div(id='live-position', style={'color': brand_colors['text']}),
                    html.Div(id='order-status', style={'color': brand_colors['text']})
                ])
            ], className='mb-3'),
            dcc.Interval(id='metrics-interval', interval=10*1000)
        ], width=4),
        
        dbc.Col(dcc.Graph(id='price-chart'), width=8)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='rsi-chart'), width=12)
    ], className='mt-4'),
    
    dcc.Interval(id='chart-interval', interval=60*1000),
    
    dbc.Row([
        dbc.Col([
            html.H4('Recent Trades', className='mt-4'),
            dash_table.DataTable(
                id='trades-table',
                page_size=5,
                style_header={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']},
                style_cell={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']}
            )
        ], width=12)
    ])
], fluid=True, style={'backgroundColor': brand_colors['background'], 'padding': '20px'})

# ===== Dashboard Callbacks =====
@app.callback(
    [Output('price-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('trades-table', 'data')],
    [Input('chart-interval', 'n_intervals')]
)
def update_charts(n):
    """Update price and RSI charts"""
    df = fetch_bitstamp_candles(limit=100, step=60)
    
    # Price chart
    price_fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USD'
        )
    ])
    price_fig.update_layout(
        title='Live BTC/USD Price',
        template='plotly_dark',
        showlegend=False
    )
    
    # RSI chart
    rsi_fig = go.Figure()
    if not df.empty:
        df['RSI'] = compute_rsi(df['close'])
        rsi_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            line={'color': '#FF6D00'},
            name='RSI(14)'
        ))
    rsi_fig.update_layout(
        title='RSI Indicator',
        yaxis_range=[0, 100],
        template='plotly_dark',
        showlegend=False
    )
    
    # Trade history
    trades = pd.DataFrame(trade_updates_list[-10:])
    
    return price_fig, rsi_fig, trades.to_dict('records')

@app.callback(
    [Output('live-rsi', 'children'),
     Output('live-position', 'children')],
    [Input('metrics-interval', 'n_intervals')]
)
def update_metrics(n):
    """Update real-time metrics"""
    df = fetch_bitstamp_candles(limit=1, step=60)
    rsi = compute_rsi(df['close']).iloc[-1] if not df.empty else 0
    position = get_crypto_position()
    
    return [
        f"Current RSI(14): {rsi:.2f}",
        f"BTC Position: {position:.4f}"
    ]

# ===== Scheduler Initialization =====
if __name__ == '__main__':
    # Start trade stream in background
    threading.Thread(target=start_trade_stream, daemon=True).start()
    
    # Schedule trading job every minute
    scheduler = BackgroundScheduler(timezone='America/New_York')
    scheduler.add_job(rsi_trading_job, 'interval', minutes=1)
    scheduler.start()
    
    # Start Dash app
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=False)
