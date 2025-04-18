import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta

# OANDA REST & Market Data
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.contrib.requests import TradeCandleRequest
from oandapyV20.endpoints.positions import OpenPositions

# === BRAND PALETTE ===
brand_colors = {
    'primary': '#2C3E50',      # Dark Blue
    'secondary': '#18BC9C',    # Teal
    'accent': '#E74C3C',       # Red
    'background': '#2C3E50',   # Dark background
    'text': '#ECF0F1'          # Light text
}

# === CONFIGURATION ===
OANDA_TOKEN = '52be1c517d0004bdcc1cc4749066aace-d079344f7066573742457a3545b5a3e3'
OANDA_ACCOUNT_ID = '101-001-30590569-001'
sendgrid_api_key = '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'
FROM_EMAIL = 'miguelcarpioariasec@gmail.com'
TO_EMAIL = 'miguelcarpioariasec@gmail.com'

# === GLOBAL STATE ===
forex_df = pd.DataFrame(columns=['Price'])
trades_df = pd.DataFrame(columns=['entry_time','entry_price','exit_time','exit_price','side'])
run_settings = {}
trade_thread = None

# === CLIENTS ===
oanda_api = API(access_token=OANDA_TOKEN)

# === HELPER FUNCTIONS ===
def get_candles(count=3):
    """Fetch last `count` 15-minute candles for EUR_USD"""
    req = TradeCandleRequest(instrument="EUR_USD", params={"count": count, "granularity": "M15"})
    resp = oanda_api.request(req)
    candles = resp.get('candles', [])
    df = pd.DataFrame([{
        'Open': float(c['mid']['o']),
        'Close': float(c['mid']['c']),
        'High': float(c['mid']['h']),
        'Low': float(c['mid']['l']),
        'time': c['time']
    } for c in candles])
    df['Price'] = df['Close']
    return df

# === RSI Calculation ===
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === Pattern Signal Generator ===
def pattern_signal(df):
    if len(df) < 2:
        return None
    o, c = df.Open.iloc[-2], df.Close.iloc[-2]
    o1, c1 = df.Open.iloc[-1], df.Close.iloc[-1]
    # bearish engulfing
    if o1 > c1 and o < c and c1 < o and o1 >= c:
        return 'short'
    # bullish engulfing
    if o1 < c1 and o > c and c1 > o and o1 <= c:
        return 'long'
    return None

# === Composite Signal ===
def generate_signal(df, strategy, rsi_thresh):
    if strategy == 'Pattern':
        return pattern_signal(df)
    # Forecast placeholder: short if price falls 3%
    forecast = df.Price.shift(1) * 0.97
    base = forecast < df.Price
    if strategy == 'Forecast + RSI':
        rsi = compute_rsi(df.Price)
        return 'short' if (base & (rsi > rsi_thresh)).iloc[-1] else None
    return 'short' if base.iloc[-1] else None

# === Email Alert ===
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_trade_alert(subject, content):
    sg = SendGridAPIClient(sendgrid_api_key)
    msg = Mail(from_email=FROM_EMAIL, to_emails=TO_EMAIL, subject=subject, plain_text_content=content)
    try:
        sg.send(msg)
    except Exception as e:
        print('Email error:', e)

# === Trade Execution ===
def execute_trade(signal, settings):
    price = forex_df.Price.iloc[-1]
    units = settings['qty']
    tp, sl = settings['tp'], settings['sl']
    prev_range = abs(forex_df.High.iloc[-2] - forex_df.Low.iloc[-2])
    if signal == 'short':
        units = -units
        tp_price = round(price - prev_range * tp, 5)
        sl_price = round(price + prev_range * sl, 5)
    else:
        tp_price = round(price + prev_range * tp, 5)
        sl_price = round(price - prev_range * sl, 5)
    data = MarketOrderRequest(
        instrument='EUR_USD', units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp_price)).data,
        stopLossOnFill=StopLossDetails(price=str(sl_price)).data
    ).data
    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=data)
    resp = oanda_api.request(r)
    trades_df.loc[len(trades_df)] = [datetime.utcnow(), price, None, None, signal]
    send_trade_alert('Trade Executed', f"{signal} {units}@{price} TP:{tp_price} SL:{sl_price}")
    return resp

# === Background Trading Loop ===
def auto_trade_loop(settings):
    global forex_df
    while True:
        df = get_candles()
        forex_df = df
        sig = generate_signal(df, settings['strategy'], settings['rsi_threshold'])
        if sig:
            execute_trade(sig, settings)
        time.sleep(900)

# === DASH APP ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2('EUR/USD Pattern Bot', style={'color': brand_colors['text']}), width=12), align='center'),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('Settings'), dbc.CardBody([
                html.Label('Quantity', style={'color': brand_colors['text']}),
                dcc.Input(id='qty', type='number', value=1000, step=100),
                html.Br(), html.Label('Take Profit %', style={'color': brand_colors['text']}),
                dcc.Slider(id='tp', min=0.5, max=3, step=0.5, value=2, marks={i: str(i) for i in range(1, 4)}),
                html.Br(), html.Label('Stop Loss %', style={'color': brand_colors['text']}),
                dcc.Slider(id='sl', min=0.5, max=3, step=0.5, value=1, marks={i: str(i) for i in range(1, 4)}),
                html.Br(), html.Label('Strategy', style={'color': brand_colors['text']}),
                dcc.RadioItems(id='strategy', options=[
                    {'label': 'Forecast Only', 'value': 'Forecast'},
                    {'label': 'Forecast + RSI', 'value': 'Forecast + RSI'},
                    {'label': 'Pattern', 'value': 'Pattern'}], value='Pattern'),
                html.Br(), html.Label('RSI Threshold', style={'color': brand_colors['text']}),
                dcc.Slider(id='rsi-threshold', min=50, max=90, step=5, value=70),
                html.Br(), dbc.Button('Start Bot', id='start-btn', color='secondary')
            ])
        ]), width=4),
        dbc.Col(dcc.Graph(id='price-chart'), width=8)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='pnl-chart'), width=6), dbc.Col(dcc.Graph(id='drawdown-chart'), width=6)]),
    dbc.Row(dbc.Col(html.Div(id='position-table')), className='mt-4')
], fluid=True, style={'backgroundColor': brand_colors['background']})

@app.callback(
    Output('price-chart', 'figure'), Output('pnl-chart', 'figure'),
    Output('drawdown-chart', 'figure'), Output('position-table', 'children'),
    Input('start-btn', 'n_clicks'), Input('qty', 'value'),
    Input('tp', 'value'), Input('sl', 'value'), Input('strategy', 'value'),
    Input('rsi-threshold', 'value')
)
def update_dash(n, qty, tp, sl, strategy, rsi_threshold):
    global trade_thread, run_settings
    settings = {'qty': int(qty), 'tp': tp / 100, 'sl': sl / 100, 'strategy': strategy, 'rsi_threshold': rsi_threshold}
    run_settings = settings
    if n and trade_thread is None:
        trade_thread = threading.Thread(target=auto_trade_loop, args=(settings,), daemon=True)
        trade_thread.start()

    df = forex_df.copy()
    fig = go.Figure([go.Scatter(x=df.time, y=df.Price, mode='lines', line=dict(color=brand_colors['secondary']))])
    pnl_fig = go.Figure(); dd_fig = go.Figure()
    if not trades_df.empty:
        pnl = []
        for _, row in trades_df.iterrows():
            entry, exit_price = row.entry_price, forex_df.Price.iloc[-1]
            ret = (entry - exit_price) / entry if row.side == 'short' else (exit_price - entry) / entry
            pnl.append(ret)
        cum = np.cumsum(pnl); draw = cum - np.maximum.accumulate(cum)
        pnl_fig.add_trace(go.Scatter(x=trades_df.entry_time, y=cum, mode='lines', line=dict(color=brand_colors['accent'])))
        dd_fig.add_trace(go.Scatter(x=trades_df.entry_time, y=draw, mode='lines', line=dict(color=brand_colors['accent'])))
    pos_data = []
    pos_resp = oanda_api.request(OpenPositions(OANDA_ACCOUNT_ID))
    for p in pos_resp.get('positions', []):
        if p['instrument'] == 'EUR_USD':
            units = float(p['long']['units']) + float(p['short']['units'])
            val = float(p['long']['unrealizedPL']) + float(p['short']['unrealizedPL'])
            pos_data.append({'Instrument': 'EUR_USD', 'Units': units, 'Unrealized P/L': f"${val:.2f}"})
    pos_df = pd.DataFrame(pos_data) if pos_data else pd.DataFrame([{'Instrument': 'None', 'Units': 0, 'Unrealized P/L': '$0'}])
    table = dbc.Table.from_dataframe(pos_df, striped=True, bordered=True, hover=True)
    return fig, pnl_fig, dd_fig, table

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
