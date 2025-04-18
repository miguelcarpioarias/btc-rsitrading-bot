import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
import logging
from datetime import datetime

# OANDA REST & Market Data
from oandapyV20 import API as OandaAPI
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.positions import OpenPositions

# Setup logging
t = logging.getLogger()
t.setLevel(logging.INFO)

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
api_client = OandaAPI(access_token=OANDA_TOKEN, environment='practice')

# State
forex_df = pd.DataFrame(columns=['time','Open','High','Low','Close','Price'])
trades_df = pd.DataFrame(columns=['time','price','side'])
trade_thread = None

# Pattern detection
def pattern_signal(df):
    if len(df) < 2: return None
    o, c = df.Open.iloc[-2], df.Close.iloc[-2]
    o1, c1 = df.Open.iloc[-1], df.Close.iloc[-1]
    if o1 > c1 and o < c and c1 < o and o1 >= c:
        return 'short'
    if o1 < c1 and o > c and c1 > o and o1 <= c:
        return 'long'
    return None

# Fetch candles
def get_candles(n=3):
    req = InstrumentsCandles(instrument="EUR_USD", params={"count":n,"granularity":"M15"})
    resp = api_client.request(req)
    df = pd.DataFrame([{ 
        'time': datetime.fromisoformat(c['time'].replace('Z','')),
        'Open': float(c['mid']['o']),
        'High': float(c['mid']['h']),
        'Low': float(c['mid']['l']),
        'Close': float(c['mid']['c'])
    } for c in resp.get('candles',[])])
    df['Price'] = df['Close']
    return df

# Execute trade
def execute_trade(side, qty, tp_pct, sl_pct):
    df = forex_df
    price = df.Price.iloc[-1]
    prev_range = abs(df.High.iloc[-2] - df.Low.iloc[-2])
    units = -qty if side=='short' else qty
    tp_price = round(price - prev_range*tp_pct,5) if side=='short' else round(price + prev_range*tp_pct,5)
    sl_price = round(price + prev_range*sl_pct,5) if side=='short' else round(price - prev_range*sl_pct,5)
    data = MarketOrderRequest(
        instrument='EUR_USD', units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp_price)).data,
        stopLossOnFill=StopLossDetails(price=str(sl_price)).data
    ).data
    order = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
    api_client.request(order)
    trades_df.loc[len(trades_df)] = [datetime.now(), price, side]
    t.info(f"Executed {side} {units}@{price} TP:{tp_price} SL:{sl_price}")

# Background loop
def auto_trade_loop(settings):
    global forex_df
    while True:
        forex_df = get_candles()
        sig = pattern_signal(forex_df)
        if sig:
            execute_trade(sig, settings['qty'], settings['tp'], settings['sl'])
        time.sleep(900)

# Dash setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2('EUR/USD Candlestick Bot', style={'color':brand_colors['text']}), width=12)),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('Settings'), dbc.CardBody([
                html.Label('Quantity (min 10)', style={'color':brand_colors['text']}),
                dcc.Input(id='qty', type='number', value=10, min=10, step=10),
                html.Br(), html.Label('Take Profit %', style={'color':brand_colors['text']}),
                dcc.Input(id='tp', type='number', value=2, step=1),
                html.Br(), html.Label('Stop Loss %', style={'color':brand_colors['text']}),
                dcc.Input(id='sl', type='number', value=1, step=1),
                html.Br(),
                dbc.Button('Start Auto', id='start-btn', color='primary'), ' ',
                dbc.Button('Force Short', id='force-short', color='danger'), ' ',
                dbc.Button('Force Long', id='force-long', color='success')
            ])
        ]), width=4),
        dbc.Col(dcc.Graph(id='price-chart'), width=8)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='pnl-chart'), width=6), dbc.Col(dcc.Graph(id='drawdown-chart'), width=6)]),
    dbc.Row(dbc.Col(html.Div(id='position-table')), className='mt-4')
], fluid=True, style={'backgroundColor':brand_colors['background']})

# Callbacks
@app.callback(
    Output('price-chart','figure'), Output('pnl-chart','figure'), Output('drawdown-chart','figure'), Output('position-table','children'),
    Input('start-btn','n_clicks'), Input('force-short','n_clicks'), Input('force-long','n_clicks'),
    Input('qty','value'), Input('tp','value'), Input('sl','value'), Input('interval','n_intervals')
)
def update_dash(start, fshort, flong, qty, tp, sl, n):
    global trade_thread
    settings = {'qty':qty,'tp':tp/100,'sl':sl/100}
    # start auto-trader
    if start and trade_thread is None:
        trade_thread = threading.Thread(target=auto_trade_loop, args=(settings,), daemon=True)
        trade_thread.start()
    # force trades
    ctx = dash.callback_context.triggered_id
    if ctx=='force-short': execute_trade('short', qty, tp/100, sl/100)
    if ctx=='force-long': execute_trade('long', qty, tp/100, sl/100)
    # update charts
    df = forex_df.copy()
    price_fig = go.Figure([go.Scatter(x=df.time, y=df.Price, mode='lines', line=dict(color=brand_colors['secondary']))])
    # trades markers
    for _, row in trades_df.iterrows():
        price_fig.add_trace(go.Scatter(x=[row.time], y=[row.price], mode='markers', marker=dict(color=brand_colors['accent'], size=10)))
    price_fig.update_layout(paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])
    # pnl & drawdown
    pnl = np.arange(len(trades_df)) # placeholder
    pnl_fig = go.Figure(); dd_fig = go.Figure()
    # positions
    pos = api_client.request(OpenPositions(accountID=OANDA_ACCOUNT_ID))
    pos_data=[]
    for p in pos.get('positions',[]):
        if p['instrument']=='EUR_USD': units=float(p['long']['units'])+float(p['short']['units']); unreal=float(p['long']['unrealizedPL'])+float(p['short']['unrealizedPL']); pos_data.append({'Instrument':'EUR_USD','Units':units,'Unrealized P/L':f"${unreal:.2f}"})
    table = dbc.Table.from_dataframe(pd.DataFrame(pos_data) if pos_data else pd.DataFrame([{'Instrument':'None','Units':0,'Unrealized P/L':'$0'}]), striped=True, bordered=True, hover=True)
    return price_fig, pnl_fig, dd_fig, table

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)
