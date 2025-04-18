import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback_context, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
import queue
from datetime import datetime, timedelta
import pytz
import logging

# OANDA REST & STREAMING
from oandapyV20 import API as OandaAPI
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.endpoints.positions import OpenPositions
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingStream

# Logging
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
environment = 'practice'
OANDA_TOKEN   = '52be1c517d0004bdcc1cc4749066aace-d079344f7066573742457a3545b5a3e3'
OANDA_ACCOUNT = '101-001-30590569-001'
api_client    = OandaAPI(access_token=OANDA_TOKEN, environment=environment)

# Global State
price_queue     = queue.Queue()
forex_df_lock   = threading.Lock()
forex_df        = pd.DataFrame(columns=['time','Open','High','Low','Close','Price'])
trades_df       = pd.DataFrame(columns=['time','price','side','pair'])
run_settings    = {}
stream_started  = False

# Eastern timezone
eastern = pytz.timezone('US/Eastern')

# Signal Generators
def pattern_signal(df):
    if len(df) < 2: return None
    o,c = df.Open.iloc[-2], df.Close.iloc[-2]
    o1,c1 = df.Open.iloc[-1], df.Close.iloc[-1]
    if o1>c1 and o<c and c1<o and o1>=c: return 'short'
    if o1<c1 and o>c and c1>o and o1<=c: return 'long'
    return None

def sma_signal(df, fast, slow):
    if len(df) < slow: return None
    sma_f = df.Price.rolling(fast).mean()
    sma_s = df.Price.rolling(slow).mean()
    if sma_f.iloc[-2]<sma_s.iloc[-2] and sma_f.iloc[-1]>sma_s.iloc[-1]: return 'long'
    if sma_f.iloc[-2]>sma_s.iloc[-2] and sma_f.iloc[-1]<sma_s.iloc[-1]: return 'short'
    return None

def generate_signal(df, settings):
    sig = pattern_signal(df)
    if sig:
        return sig
    if settings['strategy']=='SMA':
        return sma_signal(df, settings['sma_fast'], settings['sma_slow'])
    return None

# Streaming handlers
def process_tick(tick):
    if tick.get('type')=='PRICE':
        price_queue.put({
            'time': datetime.fromisoformat(tick['time'].replace('Z','')).replace(tzinfo=pytz.utc).astimezone(eastern),
            'price': float(tick['closeoutBid'])
        })


def streaming_worker():
    global run_settings
    while True:
        try:
            params = { 'instruments': run_settings.get('pair','EUR_USD') }
            stream = PricingStream(accountID=OANDA_ACCOUNT, params=params)
            for tick in api_client.request(stream):
                process_tick(tick)
        except Exception as e:
            logging.error(f"Stream error: {e}")
            time.sleep(5)


def candle_formation_worker():
    global forex_df, trades_df
    current_candle = None
    while True:
        try:
            tick = price_queue.get(timeout=1)
            with forex_df_lock:
                # Update or form candles
                if current_candle is None:
                    current_candle = {
                        'time': tick['time'].replace(second=0, microsecond=0) - timedelta(minutes=tick['time'].minute%15),
                        'Open': tick['price'], 'High': tick['price'],
                        'Low': tick['price'], 'Close': tick['price']
                    }
                else:
                    # If new candle
                    ct = tick['time'].replace(second=0,microsecond=0) - timedelta(minutes=tick['time'].minute%15)
                    if ct > current_candle['time']:
                        row = {**current_candle}
                        row['Price'] = row['Close']
                        forex_df = pd.concat([forex_df, pd.DataFrame([row])], ignore_index=True)
                        sig = generate_signal(forex_df, run_settings)
                        if sig:
                            execute_trade(sig, run_settings)
                        # start next
                        current_candle = {'time':ct,'Open':tick['price'],'High':tick['price'],'Low':tick['price'],'Close':tick['price']}
                    else:
                        current_candle['High'] = max(current_candle['High'], tick['price'])
                        current_candle['Low']  = min(current_candle['Low'], tick['price'])
                        current_candle['Close']= tick['price']
        except queue.Empty:
            continue


def start_streaming():
    global stream_started
    if not stream_started:
        threading.Thread(target=streaming_worker, daemon=True).start()
        threading.Thread(target=candle_formation_worker, daemon=True).start()
        stream_started = True

# Trade execution
def execute_trade(side, settings):
    global trades_df
    with forex_df_lock:
        df = forex_df.copy()
    price = df.Price.iloc[-1]
    rng   = abs(df.High.iloc[-2]-df.Low.iloc[-2])
    units = -settings['qty'] if side=='short' else settings['qty']
    tp    = round(price + rng*settings['tp']*(1 if side=='long' else -1),5)
    sl    = round(price - rng*settings['sl']*(1 if side=='long' else -1),5)
    data  = MarketOrderRequest(
        instrument=settings['pair'], units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp)).data,
        stopLossOnFill=StopLossDetails(price=str(sl)).data
    ).data
    req   = orders.OrderCreate(accountID=OANDA_ACCOUNT, data=data)
    api_client.request(req)
    trades_df = trades_df.append({
        'time': datetime.now().astimezone(eastern), 'price':price,
        'side':side, 'pair':settings['pair']
    }, ignore_index=True)
    logging.info(f"Executed {side}@{price} {settings['pair']}")

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2('Streaming Forex Bot', style={'color':brand_colors['text']}), width=12)),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('Settings'), dbc.CardBody([
                html.Label('Pair', style={'color':brand_colors['text']}),
                dcc.Dropdown(id='pair', options=[{'label':'EUR/USD','value':'EUR_USD'},{'label':'AUD/USD','value':'AUD_USD'}], value='EUR_USD'), html.Br(),
                html.Label('Qty', style={'color':brand_colors['text']}),
                dcc.Input(id='qty', type='number', value=10, min=10), html.Br(),
                html.Label('TP %', style={'color':brand_colors['text']}),
                dcc.Input(id='tp', type='number', value=3), html.Br(),
                html.Label('SL %', style={'color':brand_colors['text']}),
                dcc.Input(id='sl', type='number', value=5), html.Br(),
                html.Label('Strategy', style={'color':brand_colors['text']}),
                dcc.RadioItems(id='strategy', options=[{'label':'Pattern','value':'Pattern'},{'label':'SMA','value':'SMA'}], value='Pattern'), html.Br(),
                html.Label('SMA Fast'), dcc.Input(id='sma-fast', type='number', value=5), html.Br(),
                html.Label('SMA Slow'), dcc.Input(id='sma-slow', type='number', value=20), html.Br(),
                dbc.Button('Start Streaming', id='start-btn', color='primary')
            ])
        ]), width=4),
        dbc.Col(dcc.Graph(id='price-chart'), width=8)
    ]),
    dcc.Interval(id='interval', interval=60000, n_intervals=0),  # update every minute
    dbc.Row([dbc.Col(dcc.Graph(id='pnl-chart'), width=6), dbc.Col(dcc.Graph(id='drawdown-chart'), width=6)]),
    dbc.Row(dbc.Col(dash_table.DataTable(id='trades-table', page_size=10)), className='mt-4')
], fluid=True, style={'backgroundColor':brand_colors['background']})

@app.callback(
    Output('price-chart','figure'), Output('pnl-chart','figure'), Output('drawdown-chart','figure'), Output('trades-table','data'),
    Input('start-btn','n_clicks'), Input('interval','n_intervals'),
    Input('pair','value'), Input('qty','value'), Input('tp','value'), Input('sl','value'),
    Input('strategy','value'), Input('sma-fast','value'), Input('sma-slow','value')
)
def update_dash(n_clicks, n_intervals, pair, qty, tp, sl, strategy, sf, ss):
    global run_settings
    run_settings.update({'pair':pair,'qty':qty,'tp':tp/100,'sl':sl/100,'strategy':strategy,'sma_fast':sf,'sma_slow':ss})
    start_streaming()
    with forex_df_lock:
        df = forex_df.copy()
    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.time, open=df.Open, high=df.High, low=df.Low, close=df.Close)])
    if strategy=='SMA':
        fig.add_trace(go.Scatter(x=df.time, y=df.Price.rolling(sf).mean(), name='SMA Fast', line=dict(color='#FFFF00')))
        fig.add_trace(go.Scatter(x=df.time, y=df.Price.rolling(ss).mean(), name='SMA Slow', line=dict(color='#FFA500')))
    trades = trades_df[trades_df.pair==pair]
    for _,r in trades.iterrows():
        fig.add_trace(go.Scatter(x=[r.time],y=[r.price],mode='markers',marker=dict(color=brand_colors['accent'],size=10)))
    fig.update_layout(xaxis_title='Time (EST)', paper_bgcolor=brand_colors['background'],plot_bgcolor=brand_colors['background'],font_color=brand_colors['text'])
    # P&L and drawdown plots
    pnl = go.Figure(); dd=go.Figure()
    for _,t in trades.iterrows():
        series=(t.price- df.Price)/t.price if t.side=='short' else (df.Price-t.price)/t.price
        pnl.add_trace(go.Scatter(x=df.time,y=series,mode='lines',name=str(t.time)))
        dd.add_trace(go.Scatter(x=df.time,y=series-series.cummax(),mode='lines',name=str(t.time)))
    pnl.update_layout(title='Unrealized P&L',paper_bgcolor=brand_colors['background'],plot_bgcolor=brand_colors['background'],font_color=brand_colors['text'])
    dd.update_layout(title='Drawdown',paper_bgcolor=brand_colors['background'],plot_bgcolor=brand_colors['background'],font_color=brand_colors['text'])
    data = trades.to_dict('records')
    return fig, pnl, dd, data

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=False)
start_streaming()
threading.Thread(target=streaming_worker, daemon=True).start()
threading.Thread(target=candle_formation_worker, daemon=True).start()
app.run_server(debug=True, use_reloader=False)
    # app.run_server(debug=True, use_reloader=False)
    # app.run_server(debug=True, use_reloader=False)
