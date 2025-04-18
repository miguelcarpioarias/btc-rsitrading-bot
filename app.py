import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback_context
from dash import dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
import pytz

# OANDA REST
from oandapyV20 import API as OandaAPI
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.positions import OpenPositions

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Branding
brand_colors = {
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'accent': '#E74C3C',
    'background': '#2C3E50',
    'text': '#ECF0F1'
}

# Config
OANDA_TOKEN = '52be1c517d0004bdcc1cc4749066aace-d079344f7066573742457a3545b5a3e3'
OANDA_ACCOUNT = '101-001-30590569-001'
api_client = OandaAPI(access_token=OANDA_TOKEN, environment='practice')

# Global state
default_cols = ['time','Open','High','Low','Close','Price']
forex_df = pd.DataFrame(columns=default_cols)
trades_df = pd.DataFrame(columns=['time','price','side','pair'])
run_settings = {}
trade_thread = None

def pattern_signal(df):
    if len(df) < 2: return None
    o, c = df.Open.iloc[-2], df.Close.iloc[-2]
    o1, c1 = df.Open.iloc[-1], df.Close.iloc[-1]
    if o1 > c1 and o < c and c1 < o and o1 >= c:
        return 'short'
    if o1 < c1 and o > c and c1 > o and o1 <= c:
        return 'long'
    return None

def sma_signal(df, fast, slow):
    if len(df) < slow: return None
    sma_f = df.Price.rolling(window=fast).mean()
    sma_s = df.Price.rolling(window=slow).mean()
    if sma_f.iloc[-2] < sma_s.iloc[-2] and sma_f.iloc[-1] > sma_s.iloc[-1]:
        return 'long'
    if sma_f.iloc[-2] > sma_s.iloc[-2] and sma_f.iloc[-1] < sma_s.iloc[-1]:
        return 'short'
    return None

def generate_signal(df, settings):
    sig = pattern_signal(df)
    if sig:
        return sig
    if settings['strategy'] == 'SMA':
        return sma_signal(df, settings['sma_fast'], settings['sma_slow'])
    return None

# Eastern timezone
eastern = pytz.timezone('US/Eastern')

def get_candles(pair, n=50):
    req = InstrumentsCandles(instrument=pair, params={'count': n, 'granularity': 'M15'})
    resp = api_client.request(req).get('candles', [])
    data = []
    for c in resp:
        utc_dt = datetime.fromisoformat(c['time'].replace('Z', '')).replace(tzinfo=pytz.utc)
        est_dt = utc_dt.astimezone(eastern)
        data.append({
            'time': est_dt,
            'Open': float(c['mid']['o']),
            'High': float(c['mid']['h']),
            'Low': float(c['mid']['l']),
            'Close': float(c['mid']['c'])
        })
    df = pd.DataFrame(data)
    df['Price'] = df['Close']
    return df

def execute_trade(side, settings):
    global trades_df
    df = forex_df
    price = df.Price.iloc[-1]
    rng = abs(df.High.iloc[-2] - df.Low.iloc[-2])
    units = -settings['qty'] if side == 'short' else settings['qty']
    tp = round(price + (rng * settings['tp']) * (1 if side == 'long' else -1), 5)
    sl = round(price - (rng * settings['sl']) * (1 if side == 'long' else -1), 5)
    data = MarketOrderRequest(
        instrument=settings['pair'], units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp)).data,
        stopLossOnFill=StopLossDetails(price=str(sl)).data
    ).data
    req = orders.OrderCreate(accountID=OANDA_ACCOUNT, data=data)
    api_client.request(req)
    trades_df = trades_df.append({
        'time': datetime.now().astimezone(eastern),
        'price': price,
        'side': side,
        'pair': settings['pair']
    }, ignore_index=True)
    logging.info(f"Trade {side}@{price} ({settings['pair']}), TP={tp}, SL={sl}")

# Background loop reads run_settings

def auto_trade_loop():
    global forex_df
    while True:
        settings = run_settings
        df = get_candles(settings['pair'])
        forex_df = df
        sig = generate_signal(df, settings)
        if sig:
            execute_trade(sig, settings)
        time.sleep(900)

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([
    dcc.Interval(id='interval', interval=300000, n_intervals=0),
    dbc.Row(dbc.Col(html.H2('Forex Bot (Pattern + SMA)', style={'color': brand_colors['text']}), width=12)),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('Settings'), dbc.CardBody([
                html.Label('Pair', style={'color': brand_colors['text']}),
                dcc.Dropdown(id='pair', options=[
                    {'label': 'EUR/USD', 'value': 'EUR_USD'},
                    {'label': 'AUD/USD', 'value': 'AUD_USD'}
                ], value='EUR_USD'), html.Br(),
                html.Label('Qty (min 10)', style={'color': brand_colors['text']}),
                dcc.Input(id='qty', type='number', value=10, min=10, step=10), html.Br(),
                html.Label('TP %', style={'color': brand_colors['text']}),
                dcc.Input(id='tp', type='number', value=3, step=1), html.Br(),
                html.Label('SL %', style={'color': brand_colors['text']}),
                dcc.Input(id='sl', type='number', value=5, step=1), html.Br(),
                html.Label('Strategy', style={'color': brand_colors['text']}),
                dcc.RadioItems(id='strategy', options=[
                    {'label': 'Pattern Only', 'value': 'Pattern'},
                    {'label': 'SMA Crossover', 'value': 'SMA'}
                ], value='Pattern'), html.Br(),
                html.Label('SMA Fast', style={'color': brand_colors['text']}),
                dcc.Input(id='sma-fast', type='number', value=5, step=1), html.Br(),
                html.Label('SMA Slow', style={'color': brand_colors['text']}),
                dcc.Input(id='sma-slow', type='number', value=20, step=1), html.Br(),
                dbc.Button('Start Auto', id='start-btn', color='primary')
            ])
        ]), width=4),
        dbc.Col(dcc.Graph(id='price-chart'), width=8)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='pnl-chart'), width=6), dbc.Col(dcc.Graph(id='drawdown-chart'), width=6)]),
    dbc.Row(dbc.Col(dash_table.DataTable(id='trades-table', page_size=10)), className='mt-4')
], fluid=True, style={'backgroundColor': brand_colors['background']})

@app.callback(
    Output('price-chart', 'figure'), Output('pnl-chart', 'figure'),
    Output('drawdown-chart', 'figure'), Output('trades-table', 'data'),
    Input('start-btn', 'n_clicks'), Input('interval', 'n_intervals'),
    Input('pair', 'value'), Input('qty', 'value'), Input('tp', 'value'),
    Input('sl', 'value'), Input('strategy', 'value'),
    Input('sma-fast', 'value'), Input('sma-slow', 'value')
)
def update_dash(n_clicks, n_intervals, pair, qty, tp, sl, strat, sf, ss):
    global trade_thread, run_settings
    run_settings = {'pair': pair, 'qty': qty, 'tp': tp/100, 'sl': sl/100,
                    'strategy': strat, 'sma_fast': sf, 'sma_slow': ss}
    if n_clicks and trade_thread is None:
        threading.Thread(target=auto_trade_loop, daemon=True).start()
        trade_thread = True
    df = forex_df.copy()
    # Price chart with candlesticks
    price_fig = go.Figure(data=[
        go.Candlestick(x=df.time, open=df.Open, high=df.High, low=df.Low, close=df.Close, name=pair)
    ])
    if strat == 'SMA':
        price_fig.add_trace(go.Scatter(x=df.time, y=df.Price.rolling(window=sf).mean(), mode='lines', name='SMA Fast', line=dict(color='#FFFF00')))
        price_fig.add_trace(go.Scatter(x=df.time, y=df.Price.rolling(window=ss).mean(), mode='lines', name='SMA Slow', line=dict(color='#FFA500')))
    # Trade markers
    trades = trades_df[trades_df.pair == pair]
    for _, r in trades.iterrows():
        price_fig.add_trace(go.Scatter(x=[r.time], y=[r.price], mode='markers', marker=dict(color=brand_colors['accent'], size=10)))
    price_fig.update_layout(xaxis_title='Time (EST)', paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])
    # P&L & Drawdown
    pnl_fig = go.Figure(); dd_fig = go.Figure()
    for _, trade in trades.iterrows():
        series = (trade.price - forex_df.Price)/trade.price if trade.side=='short' else (forex_df.Price - trade.price)/trade.price
        pnl_fig.add_trace(go.Scatter(x=forex_df.time, y=series, mode='lines', name=str(trade.time)))
        draw = series - series.cummax()
        dd_fig.add_trace(go.Scatter(x=forex_df.time, y=draw, mode='lines', name=str(trade.time)))
    pnl_fig.update_layout(title='Unrealized P&L', paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])
    dd_fig.update_layout(title='Drawdown', paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])
    # Trades table
    data = trades.to_dict('records')
    return price_fig, pnl_fig, dd_fig, data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
