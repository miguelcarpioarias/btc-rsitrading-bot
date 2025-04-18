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
import yfinance as yf

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
    if sig: return sig
    if settings.get('mode') == 'Live' and settings['strategy']=='SMA':
        return sma_signal(df, settings['sma_fast'], settings['sma_slow'])
    return None

# Historical backtest
def backtest(pair, tp, sl, strategy, sma_fast, sma_slow):
    # Fetch historical 15m data
    hist = yf.download(f"{pair.replace('_','')}=X", period='6mo', interval='15m')
    hist = hist.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close'})
    hist['Price'] = hist['Close']
    hist = hist.reset_index().rename(columns={'index':'time'})
    cash, position = 10000.0, 0
    equity_curve = []
    trades = []
    for i in range(1,len(hist)):
        window = hist.iloc[:i+1]
        sig = None
        if strategy=='Pattern': sig = pattern_signal(window)
        else: sig = sma_signal(window, sma_fast, sma_slow)
        price = hist.Close.iloc[i]
        # simple market entry
        if sig=='long' and position==0:
            entry=price; position=1
            trades.append({'time':hist.time.iloc[i],'price':price,'side':'long','entry':price})
        if sig=='short' and position==0:
            entry=price; position=-1
            trades.append({'time':hist.time.iloc[i],'price':price,'side':'short','entry':price})
        # exit logic
        if position!=0:
            ret = (price-entry)/entry * position
            if ret>=tp or ret<=-sl:
                cash *= (1+ret)
                trades[-1].update({'exit_time':hist.time.iloc[i],'exit':price,'ret':ret,'cash':cash})
                position=0
        eq = cash*(1+((price-entry)/entry*position if position!=0 else 0))
        equity_curve.append({'time':hist.time.iloc[i],'equity':eq})
    return hist, pd.DataFrame(trades), pd.DataFrame(equity_curve)

# Streaming handlers omitted for brevity...
# Trade execution omitted...

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.layout = dbc.Tabs([
    dbc.Tab(label='Live Mode', children=[
        dbc.Container([
            dbc.Row(dbc.Col(html.H2('Streaming Forex Bot', style={'color':brand_colors['text']}), width=12)),
            # live settings & charts same as before...
        ], fluid=True)
    ]),
    dbc.Tab(label='Backtest Mode', children=[
        dbc.Container([
            dbc.Row(dbc.Col(html.H2('Backtest Historical Strategy', style={'color':brand_colors['text']}), width=12)),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader('Backtest Settings'), dbc.CardBody([
                        html.Label('Pair', style={'color':brand_colors['text']}),
                        dcc.Dropdown(id='bt-pair', options=[{'label':'EUR/USD','value':'EUR_USD'},{'label':'AUD/USD','value':'AUD_USD'}], value='EUR_USD'), html.Br(),
                        html.Label('TP %', style={'color':brand_colors['text']}), dcc.Input(id='bt-tp', type='number', value=3), html.Br(),
                        html.Label('SL %', style={'color':brand_colors['text']}), dcc.Input(id='bt-sl', type='number', value=5), html.Br(),
                        html.Label('Strategy', style={'color':brand_colors['text']}),
                        dcc.RadioItems(id='bt-strategy', options=[{'label':'Pattern','value':'Pattern'},{'label':'SMA','value':'SMA'}], value='Pattern'), html.Br(),
                        html.Label('SMA Fast', style={'color':brand_colors['text']}), dcc.Input(id='bt-sf', type='number', value=5), html.Br(),
                        html.Label('SMA Slow', style={'color':brand_colors['text']}), dcc.Input(id='bt-ss', type='number', value=20), html.Br(),
                        dbc.Button('Run Backtest', id='bt-run', color='secondary')
                    ])
                ]), width=4),
                dbc.Col(dcc.Graph(id='bt-price-chart'), width=8)
            ]),
            dbc.Row(dbc.Col(html.Div(id='bt-metrics')), className='mt-4')
        ], fluid=True)
    ])
])

# Live Mode callback omitted...

@app.callback(
    Output('bt-price-chart','figure'),
    Output('bt-metrics','children'),
    Input('bt-run','n_clicks'),
    Input('bt-pair','value'),
    Input('bt-tp','value'),
    Input('bt-sl','value'),
    Input('bt-strategy','value'),
    Input('bt-sf','value'),
    Input('bt-ss','value')
)
def run_backtest(n, pair, tp, sl, strategy, sf, ss):
    if not n:
        return go.Figure(), ''
    hist, trades, eq = backtest(pair, tp/100, sl/100, strategy, sf, ss)
    # Price & trades
    fig = go.Figure(data=[
        go.Candlestick(x=hist.time, open=hist.Open, high=hist.High, low=hist.Low, close=hist.Close)
    ])
    for _, t in trades.iterrows():
        fig.add_trace(go.Scatter(x=[t.time], y=[t.entry], mode='markers', marker=dict(color=brand_colors['accent'], size=10)))
    # Metrics
    total_return = eq.equity.iloc[-1]/10000 -1
    sharpe = (eq.equity.pct_change().mean()/eq.equity.pct_change().std())*np.sqrt(252)
    metrics = html.Ul([
        html.Li(f"Trades: {len(trades)}"),
        html.Li(f"Total Return: {total_return:.2%}"),
        html.Li(f"Sharpe: {sharpe:.2f}"),
        html.Li(f"Max Drawdown: {(eq.equity/eq.equity.cummax()-1).min():.2%}")
    ], style={'color':brand_colors['text']})
    return fig, metrics

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
