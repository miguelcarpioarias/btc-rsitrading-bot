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

# Alpaca clients
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# --- Configuration ---
API_KEY    = os.getenv('ALPACA_KEY') or os.getenv('ALPACA_API_KEY') or "PK93LZQTSB35L3CL60V5"
API_SECRET = os.getenv('ALPACA_SECRET') or os.getenv('ALPACA_SECRET_KEY') or "HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0"
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Symbols
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
ALPACA_SYMBOL = "BTC/USD"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# --- Streaming Order Updates ---
trade_updates_list = []

async def trade_updates_handler(update):
    try:
        event = update.event
        order = update.order
        trade_updates_list.append({
            'event': event,
            'symbol': order.symbol,
            'filled_qty': order.filled_qty,
            'filled_avg_price': order.filled_avg_price,
            'timestamp': update.timestamp
        })
    except Exception as e:
        logging.error(f"Stream handler error: {e}")

def start_trade_stream():
    stream = TradingStream(API_KEY, API_SECRET, paper=True)
    stream.subscribe_trade_updates(trade_updates_handler)
    stream.run()

threading.Thread(target=start_trade_stream, daemon=True).start()

# --- RSI Computation & Fetching Candles ---
def compute_rsi(series, window=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def fetch_bitstamp_candles(limit=1000, step=60):
    params = {'step': step, 'limit': limit}
    resp = requests.get(BITSTAMP_URL, params=params)
    data = resp.json().get('data', {}).get('ohlc', [])
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df.set_index('timestamp')

# --- RSI Trading Job ---
# --- Modified RSI Trading Job ---
def rsi_trading_job():
    try:
        # [Keep existing data fetching and RSI calculation unchanged]
        df = fetch_bitstamp_candles(limit=1000, step=60)
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        df['RSI'] = compute_rsi(df['close'], window=14)
        last_rsi = df['RSI'].iloc[-1]
        logging.info(f"Latest RSI: {last_rsi:.2f}")
        
        # [Key Change 1: Remove position check for buying]
        # Check if we have an existing position
        account = trade_client.get_account()
        usd_avail = float(account.cash)
        
        # [Key Change 1: Remove position check for buying]
        if last_rsi <= 30:  # Buy regardless of existing position
            logging.info(f"RSI ≤30 → Attempting BUY (stacking)")
            price = df['close'].iloc[-1]
            
            # Buy $100 worth each time (or whatever's available)
            target_usd = min(100, usd_avail)
            buy_qty = round(target_usd / price - 1e-8, 8)
            
            if buy_qty >= 0 and target_usd >= 5:  # Alpaca's $5 minimum
                mo = MarketOrderRequest(
                    symbol=ALPACA_SYMBOL,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    qty=buy_qty
                )
                resp = trade_client.submit_order(order_data=mo)
                logging.info(f"BUY executed: +{buy_qty} BTC (${target_usd:.2f})")
            else:
                logging.warning("Insufficient funds for BUY")

        # [Key Change 2: Sell ENTIRE position when RSI ≥70]
        elif last_rsi >= 70:
            logging.info(f"RSI ≥70 → Attempting SELL (liquidating)")
            positions = trade_client.get_all_positions()
            clean_symbol = ALPACA_SYMBOL.replace("/", "")
            btc_pos = next((p for p in positions if p.symbol == clean_symbol), None)
            
            if btc_pos:  # Sell only if position exists
                sell_qty = round(float(btc_pos.qty) - 1e-8, 8)
                logging.info(f"RSI ≥70 → Selling ENTIRE position: {sell_qty} BTC")
                
                mo = MarketOrderRequest(
                    symbol=ALPACA_SYMBOL,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    qty=sell_qty
                )
                resp = trade_client.submit_order(order_data=mo)
                logging.info(f"SELL executed: -{sell_qty} BTC")

        else:
            logging.info("No trade signal")

    except Exception as e:
        logging.error(f"RSI trading job error: {e}")


# Schedule RSI job every minute
scheduler = BackgroundScheduler(timezone='US/Eastern')
scheduler.add_job(rsi_trading_job, 'interval', minutes=1)
scheduler.start()

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
brand_colors = {'background': '#2C3E50', 'text': '#ECF0F1'}

app.layout = dbc.Container([
    html.H2('Crypto Dashboard (BTC/USD)', style={'color': brand_colors['text']}),
    dbc.Row([
        dbc.Col([
            html.Label('BTC Qty', style={'color': brand_colors['text']}),
            dcc.Input(id='btc-qty', type='number', value=0.5, step=0.1),
            html.Br(), html.Br(),
            dbc.Button('Buy', id='buy-btc', color='success', className='me-2'),
            dbc.Button('Sell', id='sell-btc', color='danger'),
            html.Br(), html.Br(),
            html.Div(id='order-status', style={'color': brand_colors['text']})
        ], width=3),
        dbc.Col(dcc.Graph(id='price-chart'), width=9)
    ]),
    dbc.Row(dcc.Graph(id='rsi-chart'), className='mt-4'),
    dcc.Interval(id='interval', interval=30*1000, n_intervals=0),
    dbc.Row(dbc.Col(dash_table.DataTable(
        id='positions-table', page_size=10,
        style_header={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']},
        style_cell={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']}
    )), className='mt-4'),
    html.H4('Order Stream', style={'color': brand_colors['text'], 'marginTop': '20px'}),
    dbc.Row(dbc.Col(dash_table.DataTable(
        id='orders-table', page_size=10,
        style_header={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']},
        style_cell={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']}
    )), className='mt-2')
], fluid=True, style={'backgroundColor': brand_colors['background'], 'padding': '20px'})

# --- Callbacks ---
@app.callback(
    Output('price-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_price(n):
    df = fetch_bitstamp_candles(limit=1000, step=60)
    # build a display-only index in NY time
    display_idx = pd.to_datetime(df.index)
    display_idx = display_idx.tz_localize('UTC').tz_convert('America/New_York')
    
    fig = go.Figure(data=[
        go.Candlestick(
            x=display_idx,
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )
    ])
    fig.update_layout(
        paper_bgcolor=brand_colors['background'],
        plot_bgcolor=brand_colors['background'],
        font_color=brand_colors['text'],
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            title='Time (ET)',
            tickformat='%H:%M\n%b %d'
        )
    )
    return fig


@app.callback(
    Output('rsi-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_rsi_chart(n):
    df = fetch_bitstamp_candles(limit=1000, step=60)
    df['RSI'] = compute_rsi(df['close'], window=14)

    # build a display-only index in NY time
    display_idx = pd.to_datetime(df.index)
    display_idx = display_idx.tz_localize('UTC').tz_convert('America/New_York')
    
    fig = go.Figure(data=[
        go.Scatter(
            x=display_idx,
            y=df['RSI'],
            mode='lines',
            name='RSI'
        )
    ])
    fig.update_layout(
        paper_bgcolor=brand_colors['background'],
        plot_bgcolor=brand_colors['background'],
        font_color=brand_colors['text'],
        yaxis=dict(range=[0,100]),
        xaxis=dict(
            title='Time (ET)',
            tickformat='%H:%M\n%b %d'
        )
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
