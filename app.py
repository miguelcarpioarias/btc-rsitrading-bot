import os
import threading
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta

# Alpaca clients
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Configuration ---
API_KEY    = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
if not (API_KEY and API_SECRET):
    raise RuntimeError('Set ALPACA_API_KEY and ALPACA_SECRET_KEY in environment')
# Initialize trading and data clients
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client  = CryptoHistoricalDataClient()
# Streaming setup
trade_updates_list = []
SYMBOL = 'BTC/USD'

# Streaming callback
async def trade_updates_handler(data):
    # Extract relevant fields
    event = data['event']
    order = data.get('order', {})
    trade_updates_list.append({
        'event': event,
        'symbol': order.get('symbol'),
        'filled_qty': order.get('filled_qty'),
        'filled_avg_price': order.get('filled_avg_price'),
        'timestamp': data.get('timestamp')
    })

# Start streaming in background thread
def start_trade_stream():
    stream = TradingStream(API_KEY, API_SECRET, paper=True)
    stream.subscribe_trade_updates(trade_updates_handler)
    stream.run()
threading.Thread(target=start_trade_stream, daemon=True).start()

# --- App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
brand_colors = {'background':'#2C3E50','text':'#ECF0F1'}

app.layout = dbc.Container([
    html.H2('Crypto Trading Dashboard (BTC/USD)', style={'color':brand_colors['text']}),
    dbc.Row([
        dbc.Col([
            html.Label('BTC Qty', style={'color':brand_colors['text']}),
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
    Output('price-chart','figure'),
    Input('interval','n_intervals')
)
def update_price(n):
    now = datetime.utcnow()
    req = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=now - timedelta(hours=1), limit=60
    )
    df = data_client.get_crypto_bars(req).df.reset_index()
    fig = go.Figure(data=[
        go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=SYMBOL)
    ])
    fig.update_layout(paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'], xaxis_rangeslider_visible=False)
    return fig

@app.callback(
    Output('order-status','children'),
    Input('buy-btc','n_clicks'), Input('sell-btc','n_clicks'), State('btc-qty','value')
)
def execute_order(buy, sell, qty):
    ctx = callback_context.triggered_id
    if not ctx: return ''
    side = OrderSide.BUY if ctx=='buy-btc' else OrderSide.SELL
    mo_req = MarketOrderRequest(symbol=SYMBOL, side=side, type=OrderType.MARKET, time_in_force=TimeInForce.GTC, qty=qty)
    try:
        resp = trade_client.submit_order(order_data=mo_req)
        return f"✅ Order {resp.id} submitted (filled_qty={resp.filled_qty})"
    except Exception as e:
        return f"❌ Order failed: {str(e)}"

@app.callback(
    Output('positions-table','data'), Output('positions-table','columns'),
    Input('interval','n_intervals')
)
def update_positions(n):
    positions = trade_client.get_all_positions()
    rows=[]
    for p in positions:
        if p.symbol==SYMBOL:
            rows.append({'Symbol':p.symbol,'Qty':p.qty,'Unrealized P/L':p.unrealized_pl,'Market Value':p.market_value})
    if not rows: rows=[{'Symbol':'None','Qty':0,'Unrealized P/L':0,'Market Value':0}]
    cols=[{'name':c,'id':c} for c in rows[0].keys()]
    return rows,cols

@app.callback(
    Output('orders-table','data'), Output('orders-table','columns'),
    Input('interval','n_intervals')
)
def update_orders(n):
    # Show last 20 updates
    data = trade_updates_list[-20:]
    # Format if needed
    rows=data.copy()
    if not rows:
        rows=[{'event':'None','symbol':'','filled_qty':0,'filled_avg_price':0,'timestamp':''}]
    cols=[{'name':k,'id':k} for k in rows[0].keys()]
    return rows,cols

# --- Run Server ---
if __name__=='__main__':
    port=int(os.environ.get('PORT',10000))
    app.run(host='0.0.0.0', port=port, debug=False)
