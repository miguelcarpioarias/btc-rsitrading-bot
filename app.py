import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
import logging
from datetime import datetime

# OANDA REST
from oandapyV20 import API as OandaAPI
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.positions import OpenPositions

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
OANDA_TOKEN = '52be1c517d0004bdcc1cc4749066aace-d079344f7066573742457a3545b5a3e3'
OANDA_ACCOUNT = '101-001-30590569-001'
api_client = OandaAPI(access_token=OANDA_TOKEN, environment='practice')

# State
forex_df = pd.DataFrame(columns=['time','Open','High','Low','Close','Price'])
trades_df = pd.DataFrame(columns=['time','price','side'])
trade_thread = None

# Pattern signal
def pattern_signal(df):
    if len(df)<2: return None
    o,c = df.Open.iloc[-2], df.Close.iloc[-2]
    o1,c1 = df.Open.iloc[-1], df.Close.iloc[-1]
    if o1>c1 and o<c and c1<o and o1>=c: return 'short'
    if o1<c1 and o>c and c1>o and o1<=c: return 'long'
    return None

# SMA crossover signal
def sma_signal(df, fast, slow):
    if len(df)<slow: return None
    sma_f = df.Price.rolling(window=fast).mean()
    sma_s = df.Price.rolling(window=slow).mean()
    if sma_f.iloc[-2]<sma_s.iloc[-2] and sma_f.iloc[-1]>sma_s.iloc[-1]: return 'long'
    if sma_f.iloc[-2]>sma_s.iloc[-2] and sma_f.iloc[-1]<sma_s.iloc[-1]: return 'short'
    return None

# Generate trade signal
def generate_signal(df, settings):
    strat = settings['strategy']
    if strat=='Pattern': return pattern_signal(df)
    if strat=='SMA': return sma_signal(df, settings['sma_fast'], settings['sma_slow'])
    return None

# Fetch latest candles
def get_candles(n=3):
    req = InstrumentsCandles(instrument='EUR_USD', params={'count':n,'granularity':'M15'})
    resp = api_client.request(req).get('candles',[])
    df = pd.DataFrame([{ 
        'time':datetime.fromisoformat(c['time'].replace('Z','')),
        'Open':float(c['mid']['o']),
        'High':float(c['mid']['h']),
        'Low':float(c['mid']['l']),
        'Close':float(c['mid']['c'])
    } for c in resp])
    df['Price']=df.Close
    return df

# Execute trade
def execute_trade(side, qty, tp_pct, sl_pct):
    df=forex_df
    price=df.Price.iloc[-1]
    rng=abs(df.High.iloc[-2]-df.Low.iloc[-2])
    units = -qty if side=='short' else qty
    tp = round(price + (rng*tp_pct)*(1 if side=='long' else -1),5)
    sl = round(price - (rng*sl_pct)*(1 if side=='long' else -1),5)
    data = MarketOrderRequest(
        instrument='EUR_USD', units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp)).data,
        stopLossOnFill=StopLossDetails(price=str(sl)).data
    ).data
    req=orders.OrderCreate(accountID=OANDA_ACCOUNT, data=data)
    api_client.request(req)
    trades_df.loc[len(trades_df)]=[datetime.now(),price,side]
    logging.info(f"Trade {side}@{price}, TP={tp}, SL={sl}")

# Background loop
def auto_trade_loop(settings):
    global forex_df
    while True:
        forex_df=get_candles(3)
        sig=generate_signal(forex_df,settings)
        if sig: execute_trade(sig,settings['qty'],settings['tp'],settings['sl'])
        time.sleep(900)

# Dash app
app=dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server=app.server

app.layout=dbc.Container([
    dcc.Interval(id='interval', interval=300000, n_intervals=0),
    dbc.Row(dbc.Col(html.H2('EUR/USD Bot',style={'color':brand_colors['text']}),width=12)),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('Settings'),dbc.CardBody([
                html.Label('Qty (min 10)',style={'color':brand_colors['text']}),
                dcc.Input(id='qty',type='number',value=10,min=10,step=10), html.Br(),
                html.Label('TP %',style={'color':brand_colors['text']}),
                dcc.Input(id='tp',type='number',value=3,step=1), html.Br(),
                html.Label('SL %',style={'color':brand_colors['text']}),
                dcc.Input(id='sl',type='number',value=5,step=1), html.Br(),
                html.Label('Strategy',style={'color':brand_colors['text']}),
                dcc.RadioItems(id='strategy',options=[
                    {'label':'Pattern','value':'Pattern'},
                    {'label':'SMA Crossover','value':'SMA'}
                ],value='Pattern'), html.Br(),
                html.Label('SMA Fast',style={'color':brand_colors['text']}),
                dcc.Input(id='sma-fast',type='number',value=5,step=1), html.Br(),
                html.Label('SMA Slow',style={'color':brand_colors['text']}),
                dcc.Input(id='sma-slow',type='number',value=20,step=1), html.Br(),
                dbc.Button('Start Auto',id='start-btn',color='primary')
            ])
        ]),width=4),
        dbc.Col(dcc.Graph(id='price-chart'),width=8)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='pnl-chart'),width=6),dbc.Col(dcc.Graph(id='drawdown-chart'),width=6)]),
    dbc.Row(dbc.Col(html.Div(id='position-table')),className='mt-4')
],fluid=True,style={'backgroundColor':brand_colors['background']})

@app.callback(
    Output('price-chart','figure'),Output('pnl-chart','figure'),
    Output('drawdown-chart','figure'),Output('position-table','children'),
    Input('start-btn','n_clicks'),Input('interval','n_intervals'),
    Input('qty','value'),Input('tp','value'),Input('sl','value'),
    Input('strategy','value'),Input('sma-fast','value'),Input('sma-slow','value')
)
def update_dash(n_clicks,n_intervals,qty,tp,sl,strategy,sf,ss):
    global trade_thread
    settings={'qty':qty,'tp':tp/100,'sl':sl/100,'strategy':strategy,'sma_fast':sf,'sma_slow':ss}
    if n_clicks and trade_thread is None:
        threading.Thread(target=auto_trade_loop,args=(settings,),daemon=True).start(); trade_thread=1
    df=forex_df.copy()
    price_fig=go.Figure([go.Scatter(x=df.time,y=df.Price,mode='lines',line=dict(color=brand_colors['secondary']))])
    if strategy=='SMA':
        price_fig.add_trace(go.Scatter(x=df.time,y=df.Price.rolling(window=sf).mean(),mode='lines',name='SMA Fast',line=dict(color='#FFFF00')))
        price_fig.add_trace(go.Scatter(x=df.time,y=df.Price.rolling(window=ss).mean(),mode='lines',name='SMA Slow',line=dict(color='#FFA500')))
    for _,r in trades_df.iterrows():
        price_fig.add_trace(go.Scatter(x=[r.time],y=[r.price],mode='markers',marker=dict(color=brand_colors['accent'],size=10)))
    price_fig.update_layout(paper_bgcolor=brand_colors['background'],plot_bgcolor=brand_colors['background'],font_color=brand_colors['text'])
    pnl_fig=go.Figure(); dd_fig=go.Figure()
    pos_resp=api_client.request(OpenPositions(accountID=OANDA_ACCOUNT))
    pos_data=[{'Instrument':'EUR_USD','Units':float(p['long']['units'])+float(p['short']['units']),'Unrealized P/L':f"${float(p['long']['unrealizedPL'])+float(p['short']['unrealizedPL']):.2f}"} for p in pos_resp.get('positions',[]) if p['instrument']=='EUR_USD']
    pos_df=pd.DataFrame(pos_data) if pos_data else pd.DataFrame([{'Instrument':'None','Units':0,'Unrealized P/L':'$0'}])
    pos_table=dbc.Table.from_dataframe(pos_df,striped=True,bordered=True,hover=True)
    return price_fig,pnl_fig,dd_fig,pos_table

if __name__=='__main__': app.run(host='0.0.0.0',port=8080,debug=False)
