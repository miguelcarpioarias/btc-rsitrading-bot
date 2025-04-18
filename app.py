import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
import logging
from datetime import datetime, timedelta

# OANDA REST & Market Data
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.positions import OpenPositions

# Setup logging
logging.basicConfig(level=logging.INFO)

# === BRAND PALETTE ===
brand_colors = {
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'accent': '#E74C3C',
    'background': '#2C3E50',
    'text': '#ECF0F1'
}

# === CONFIGURATION ===
OANDA_TOKEN = '52be1c517d0004bdcc1cc4749066aace-d079344f7066573742457a3545b5a3e3'
OANDA_ACCOUNT_ID = '101-001-30590569-001'
sendgrid_api_key = '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'
FROM_EMAIL = 'miguelcarpioariasec@gmail.com'
TO_EMAIL = 'miguelcarpioariasec@gmail.com'

# Initialize OANDA API client for practice
# Set environment to 'practice' for paper trading
from oandapyV20 import API as OandaAPI

# Instantiate with practice environment
# This ensures requests go to the practice (paper) endpoints
api_client = OandaAPI(access_token=OANDA_TOKEN, environment="practice")

# Global state
forex_df = pd.DataFrame(columns=['time','Open','High','Low','Close','Price'])
trades_df = pd.DataFrame(columns=['entry_time','entry_price','exit_time','exit_price','side'])
run_settings = {}
trade_thread = None

# Helper: fetch latest candles
def get_candles(count=3):
    params = {"count": count, "granularity": "M15"}
    req = InstrumentsCandles(instrument="EUR_USD", params=params)
    resp = oanda_api.request(req)
    candles = resp.get('candles', [])
    df = pd.DataFrame([{ 
        'time': datetime.fromisoformat(c['time'].replace('Z','')),
        'Open': float(c['mid']['o']),
        'High': float(c['mid']['h']),
        'Low': float(c['mid']['l']),
        'Close': float(c['mid']['c'])
    } for c in candles])
    df['Price'] = df['Close']
    return df

# RSI calculation
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Pattern signal
def pattern_signal(df):
    if len(df) < 2: return None
    o, c = df.Open.iloc[-2], df.Close.iloc[-2]
    o1, c1 = df.Open.iloc[-1], df.Close.iloc[-1]
    if o1 > c1 and o < c and c1 < o and o1 >= c:
        return 'short'
    if o1 < c1 and o > c and c1 > o and o1 <= c:
        return 'long'
    return None

# Composite signal generator
def generate_signal(df, strategy, rsi_thresh):
    if strategy == 'Pattern':
        sig = pattern_signal(df)
    else:
        forecast = df.Price.shift(1) * 0.97
        base = forecast < df.Price
        if strategy == 'Forecast + RSI':
            rsi = compute_rsi(df.Price)
            sig = 'short' if (base & (rsi > rsi_thresh)).iloc[-1] else None
        else:
            sig = 'short' if base.iloc[-1] else None
    logging.info(f"Signal generated: {sig} at {datetime.now()}")
    return sig

# Email alert (placeholder)
def send_trade_alert(subject, content): pass

# Trade execution
def execute_trade(signal, settings):
    price = forex_df.Price.iloc[-1]
    units = -settings['qty'] if signal=='short' else settings['qty']
    prev_range = abs(forex_df.High.iloc[-2] - forex_df.Low.iloc[-2])
    tp_price = price - prev_range*settings['tp'] if signal=='short' else price + prev_range*settings['tp']
    sl_price = price + prev_range*settings['sl'] if signal=='short' else price - prev_range*settings['sl']
    data = MarketOrderRequest(
        instrument='EUR_USD', units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(round(tp_price,5))).data,
        stopLossOnFill=StopLossDetails(price=str(round(sl_price,5))).data
    ).data
    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=data)
    oanda_api.request(r)
    trades_df.loc[len(trades_df)] = [datetime.now(), price, None, None, signal]
    logging.info(f"Trade executed: {signal} {units}@{price}")
    send_trade_alert('Trade Executed', f"{signal} {units}@{price} TP:{tp_price} SL:{sl_price}")

# Background loop
def auto_trade_loop(settings):
    global forex_df
    while True:
        df = get_candles()
        forex_df = df
        sig = generate_signal(df, settings['strategy'], settings['rsi_threshold'])
        if sig:
            execute_trade(sig, settings)
        time.sleep(900)

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([
    dcc.Interval(id='interval', interval=5*60*1000, n_intervals=0),
    dbc.Row(dbc.Col(html.H2('EUR/USD Pattern Bot', style={'color':brand_colors['text']}), width=12)),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader('Settings'),
                dbc.CardBody([
                    html.Label('Quantity', style={'color': brand_colors['text']}),
                    dcc.Input(id='qty', type='number', value=1000, step=100),
                    html.Br(),
                    html.Label('Take Profit %', style={'color': brand_colors['text']}),
                    dcc.Slider(id='tp', min=0.5, max=3, step=0.5, value=2),
                    html.Br(),
                    html.Label('Stop Loss %', style={'color': brand_colors['text']}),
                    dcc.Slider(id='sl', min=0.5, max=3, step=0.5, value=1),
                    html.Br(),
                    html.Label('Strategy', style={'color': brand_colors['text']}),
                    dcc.RadioItems(id='strategy', options=[
                        {'label': 'Forecast Only', 'value': 'Forecast'},
                        {'label': 'Forecast + RSI', 'value': 'Forecast + RSI'},
                        {'label': 'Pattern', 'value': 'Pattern'}
                    ], value='Pattern'),
                    html.Br(),
                    html.Label('RSI Threshold', style={'color': brand_colors['text']}),
                    dcc.Slider(id='rsi-threshold', min=50, max=90, step=5, value=70),
                    html.Br(),
                    dbc.Button('Start Bot', id='start-btn', color='secondary')
                ])
            ]), width=4
        ),
        dbc.Col(dcc.Graph(id='price-chart'), width=8)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='pnl-chart'), width=6), dbc.Col(dcc.Graph(id='drawdown-chart'), width=6)]),
    dbc.Row(dbc.Col(html.Div(id='position-table')), className='mt-4')
], fluid=True)

@app.callback(
    Output('price-chart','figure'), Output('pnl-chart','figure'),
    Output('drawdown-chart','figure'), Output('position-table','children'),
    Input('start-btn','n_clicks'), Input('interval','n_intervals'),
    Input('qty','value'), Input('tp','value'), Input('sl','value'),
    Input('strategy','value'), Input('rsi-threshold','value')
)
def update_dash(n_clicks, n_intervals, qty, tp, sl, strategy, rsi_threshold):
    global trade_thread, run_settings
    settings = {'qty':int(qty),'tp':tp/100,'sl':sl/100,'strategy':strategy,'rsi_threshold':rsi_threshold}
    run_settings = settings
    if n_clicks and trade_thread is None:
        trade_thread = threading.Thread(target=auto_trade_loop, args=(settings,), daemon=True)
        trade_thread.start()
    # Build Figures
    df = forex_df.copy()
    price_fig = go.Figure([go.Scatter(x=df.time, y=df.Price, mode='lines', line=dict(color=brand_colors['secondary']))])
    # Trades markers
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            price_fig.add_trace(go.Scatter(x=[t['entry_time']], y=[t['entry_price']], mode='markers', marker=dict(symbol='triangle-down', color=brand_colors['accent'])))
            price_fig.add_trace(go.Scatter(x=[t['exit_time']], y=[t['exit_price']], mode='markers', marker=dict(symbol='triangle-up', color=brand_colors['primary'])))
    price_fig.update_layout(paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])
    # P&L & Drawdown
    pnl_fig = go.Figure(); dd_fig = go.Figure()
    if not trades_df.empty:
        returns = []
        for _, row in trades_df.iterrows():
            if row['exit_price'] is not None:
                ret = (row['entry_price'] - row['exit_price'])/row['entry_price'] if row['side']=='short' else (row['exit_price'] - row['entry_price'])/row['entry_price']
                returns.append(ret)
        cum = np.cumsum(returns); draw = cum - np.maximum.accumulate(cum)
        pnl_fig.add_trace(go.Scatter(x=trades_df['entry_time'], y=cum, mode='lines', line=dict(color=brand_colors['accent'])))
        dd_fig.add_trace(go.Scatter(x=trades_df['entry_time'], y=draw, mode='lines', line=dict(color=brand_colors['accent'])))
    pnl_fig.update_layout(paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'], title='Cumulative P&L')
    dd_fig.update_layout(paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'], title='Drawdown')
    # Positions Table
    pos_resp = oanda_api.request(OpenPositions(accountID=OANDA_ACCOUNT_ID))
    pos_data = []
    for p in pos_resp.get('positions', []):
        if p['instrument']=='EUR_USD':
            units = float(p['long']['units']) + float(p['short']['units'])
            unreal = float(p['long']['unrealizedPL']) + float(p['short']['unrealizedPL'])
            pos_data.append({'Instrument':'EUR_USD','Units':units,'Unrealized P/L':f"${unreal:.2f}"})
    pos_df = pd.DataFrame(pos_data) if pos_data else pd.DataFrame([{'Instrument':'None','Units':0,'Unrealized P/L':'$0'}])
    pos_table = dbc.Table.from_dataframe(pos_df, striped=True, bordered=True, hover=True)
    return price_fig, pnl_fig, dd_fig, pos_table

if __name__=='__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
