import os
import threading
import time
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler

# Alpaca clients
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Configuration ---
# Hard-coded Alpaca paper credentials
API_KEY    = "PK93LZQTSB35L3CL60V5"
API_SECRET = "HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0"
# Initialize trading client
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Initialize data client for crypto bars
data_client = CryptoHistoricalDataClient()
SYMBOL = 'BTC/USD'
SYMBOL = 'BTC/USD'

# --- Streaming Order Updates ---
trade_updates_list = []
async def trade_updates_handler(data):
    event = data.get('event')
    order = data.get('order', {})
    trade_updates_list.append({
        'event': event,
        'symbol': order.get('symbol'),
        'filled_qty': order.get('filled_qty'),
        'filled_avg_price': order.get('filled_avg_price'),
        'timestamp': data.get('timestamp')
    })

def start_trade_stream():
    stream = TradingStream(API_KEY, API_SECRET, paper=True)
    stream.subscribe_trade_updates(trade_updates_handler)
    stream.run()
threading.Thread(target=start_trade_stream, daemon=True).start()

# --- RSI Computation & Trading Job ---
def compute_rsi(series, window=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def rsi_trading_job():
    try:
        now_et = datetime.now(ZoneInfo('America/New_York'))
        req = CryptoBarsRequest(
            symbol_or_symbols=[SYMBOL],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=now_et - timedelta(days=45),
            limit=45
        )
        df = data_client.get_crypto_bars(req).df.reset_index()
        df['close'] = df['close'].astype(float)
        df['rsi'] = compute_rsi(df['close'], window=14)
        last_rsi = df['rsi'].iloc[-1]
        positions = trade_client.get_all_positions()
        flat = not any(p.symbol.replace('/','')==SYMBOL.replace('/','') and float(p.qty)!=0 for p in positions)
        if last_rsi < 30 and flat:
            entry = df['close'].iloc[-1]
            tp = round(entry * 1.02, 2)
            sl = round(entry * 0.95, 2)
            mo = MarketOrderRequest(
                symbol=SYMBOL, side=OrderSide.BUY, type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC, qty=0.001,
                take_profit={'limit_price': tp}, stop_loss={'stop_price': sl}
            )
            trade_client.submit_order(order_data=mo)
        elif last_rsi > 70 and not flat:
            qty = sum(float(p.qty) for p in positions if p.symbol.replace('/','')==SYMBOL.replace('/',''))
            if qty > 0:
                mo = MarketOrderRequest(
                    symbol=SYMBOL, side=OrderSide.SELL, type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC, qty=qty
                )
                trade_client.submit_order(order_data=mo)
    except Exception as e:
        print(f"RSI trading job error: {e}")

scheduler = BackgroundScheduler(timezone='US/Eastern')
scheduler.add_job(rsi_trading_job, 'interval', minutes=1)
scheduler.start()

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
brand_colors = {'background':'#2C3E50', 'text':'#ECF0F1'}

app.layout = dbc.Container([
    html.H2('Crypto Trading Dashboard (BTC/USD)', style={'color':brand_colors['text']}),
    dbc.Row([
        dbc.Col([
            html.Label('BTC Quantity', style={'color':brand_colors['text']}),
            dcc.Input(id='btc-qty', type='number', value=0.001, step=0.001), html.Br(), html.Br(),
            dbc.Button('Buy BTC', id='buy-btc', color='success', className='me-2'),
            dbc.Button('Sell BTC', id='sell-btc', color='danger'), html.Br(), html.Br(),
            html.Div(id='order-status', style={'color':brand_colors['text']})
        ], width=3),
        dbc.Col(dcc.Graph(id='price-chart'), width=9)
    ]),
    dcc.Interval(id='interval', interval=30*1000, n_intervals=0),
    dbc.Row(dbc.Col(dash_table.DataTable(
        id='positions-table', page_size=10,
        style_header={'backgroundColor':brand_colors['background'],'color':brand_colors['text']},
        style_cell={'backgroundColor':brand_colors['background'],'color':brand_colors['text']}
    )), className='mt-4'),
    html.H4('Order Stream History', style={'color':brand_colors['text'],'marginTop':'20px'}),
    dbc.Row(dbc.Col(dash_table.DataTable(
        id='orders-table', page_size=10,
        style_header={'backgroundColor':brand_colors['background'],'color':brand_colors['text']},
        style_cell={'backgroundColor':brand_colors['background'],'color':brand_colors['text']}
    )), className='mt-2')
], fluid=True, style={'backgroundColor':brand_colors['background'],'padding':'20px'})

# --- Callbacks ---
@app.callback(
    Output('positions-table','data'), Output('positions-table','columns'),
    Input('interval','n_intervals')
)
def update_positions(n):
    # Safely fetch open positions
    positions = []
    try:
        positions = trade_client.get_all_positions() or []
    except:
        positions = []

    rows = []
    for p in positions:
        if p.symbol == SYMBOL:
            rows.append({
                'Symbol': p.symbol,
                'Qty': p.qty,
                'Unrealized P/L': p.unrealized_pl,
                'Market Value': p.market_value
            })
    # Fallback if no positions
    if not rows:
        rows = [{'Symbol': 'None', 'Qty': 0, 'Unrealized P/L': 0, 'Market Value': 0}]
    # Build columns list
    columns = [{'name': col, 'id': col} for col in rows[0].keys()]
    return rows, columns

@app.callback(
    Output('orders-table','data'), Output('orders-table','columns'),
    Input('interval','n_intervals')
)
def update_orders(n):
    data = trade_updates_list[-20:]
    rows = data or [{'event':'None','symbol':'','filled_qty':0,'filled_avg_price':0,'timestamp':''}]
    cols=[{'name':k,'id':k} for k in rows[0].keys()]
    return rows, cols

if __name__=='__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
