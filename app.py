import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Configuration ---
# --- Configuration ---
# Hard-coded Alpaca paper credentials
API_KEY    = "PK93LZQTSB35L3CL60V5"
API_SECRET = "HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0"

# Initialize trading client
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

SYMBOL = 'BTC/USD'
SYMBOL = 'BTC/USD'

# --- App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Branding
brand_colors = {
    'background': '#2C3E50',
    'text': '#ECF0F1',
    'primary': '#18BC9C',
    'accent': '#E74C3C'
}

app.layout = dbc.Container([
    html.H2('Crypto Trading Dashboard (BTC/USD)', style={'color': brand_colors['text']}),
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
    dbc.Row(dbc.Col(dash.dash_table.DataTable(
        id='positions-table',
        style_header={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']},
        style_cell={'backgroundColor': brand_colors['background'], 'color': brand_colors['text']}
    )), className='mt-4')
], fluid=True, style={'backgroundColor': brand_colors['background'], 'padding':'20px'})

# --- Callbacks ---
@app.callback(
    Output('positions-table','data'),
    Output('positions-table','columns'),
    Input('interval','n_intervals')
)
def update_positions(n):
    # Fetch open positions
    positions = trade_client.get_all_positions()
    rows = []
    for p in positions:
        # Normalize symbol for crypto (BTCUSD vs BTC/USD)
        sym = p.symbol.replace("/", "")
        if sym == SYMBOL.replace("/", ""):
            rows.append({
                'Symbol': p.symbol,
                'Qty': p.qty,
                'Unrealized P/L': p.unrealized_pl,
                'Market Value': p.market_value
            })
    # If no positions for BTC, show placeholder
    if not rows:
        rows = [{'Symbol': 'None', 'Qty': 0, 'Unrealized P/L': 0, 'Market Value': 0}]
    columns = [{"name": c, "id": c} for c in rows[0].keys()]
    return rows, columns

# --- Run Server ---
if __name__=='__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
