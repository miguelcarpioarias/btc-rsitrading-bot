import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import sendgrid
from sendgrid.helpers.mail import Mail
import threading
import asyncio
from datetime import datetime, timedelta, timezone
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import TimeFrame

# === BRAND PALETTE ===
brand_colors = {
    'primary': '#2C3E50',      # Dark Blue
    'secondary': '#18BC9C',    # Teal
    'accent': '#E74C3C',       # Red
    'background': '#2C3E50',   # Dark background
    'text': '#ECF0F1'          # Light text
}

# === CONFIGURATION ===
ALPACA_API_KEY = 'PK93LZQTSB35L3CL60V5'
ALPACA_SECRET_KEY = 'HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
SENDGRID_API_KEY = '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'
FROM_EMAIL = 'miguelcarpioariasec@gmail.com'
TO_EMAIL = 'miguelcarpioariasec@gmail.com'

# === GLOBAL STATE ===
crypto_df = pd.DataFrame(columns=['Price'])
trades_df = pd.DataFrame(columns=['entry_time','entry_price','exit_time','exit_price'])
run_settings = {}
stream_thread = None

# === CLIENTS ===
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)

# === RSI CALC ===
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === SIGNAL GENERATOR ===
def generate_signal(prices, strategy='Forecast', rsi_threshold=70):
    forecast = prices.shift(1) * 0.97
    base_signal = (forecast < prices).astype(int)
    if strategy == 'Forecast + RSI':
        rsi = compute_rsi(prices)
        return (base_signal & (rsi > rsi_threshold)).astype(int)
    return base_signal

# === EMAIL ALERT ===
def send_trade_alert(subject, content):
    msg = Mail(from_email=FROM_EMAIL, to_emails=TO_EMAIL,
               subject=subject, plain_text_content=content)
    try:
        sg.send(msg)
    except Exception as e:
        print(f"Email failed: {e}")

# === STREAM HANDLERS ===
async def on_crypto_bars(bar):
    global crypto_df, trades_df, run_settings
    # bar is a single CryptoBar
    ts = bar.Timestamp.replace(tzinfo=None)
    price = bar.Close
    crypto_df.loc[ts] = price
    # compute signal and trade
    sig = generate_signal(crypto_df['Price'], run_settings['strategy'], run_settings['rsi_threshold'])
    if sig.iloc[-1] == 1:
        qty = run_settings['qty']; tp = run_settings['tp']; sl = run_settings['sl']
        # place short trade
        filled_order = api.submit_order(
            symbol='BTC/USD', qty=qty, side='sell', type='market', time_in_force='gtc',
            order_class='bracket', take_profit={'limit_price': round(price*(1-tp),2)},
            stop_loss={'stop_price': round(price*(1+sl),2)}
        )
        send_trade_alert('BTC Short Executed', f"Qty: {qty} at {price}, TP: {tp}, SL: {sl}")

async def on_trade_update(data):
    global trades_df
    o = data['order']
    event = data['event']
    # fill events record exit of short
    if event == 'fill' and o['asset_class']=='crypto':
        # find matching entry or just append
        trades_df.loc[len(trades_df)] = [data['timestamp'], None, data['timestamp'], float(o['filled_avg_price'])]

def start_stream():
    stream = Stream(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL, data_stream='crypto')
    stream.subscribe_crypto_bars(on_crypto_bars, 'BTC/USD')
    stream.subscribe_trade_updates(on_trade_update)
    asyncio.run(stream._run_forever())

# === DASH APP ===
external_styles = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_styles)
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("BTC/USD Short-Selling Bot"), width=12), align='center'),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Settings"), dbc.CardBody([
                html.Label("Quantity (BTC)", style={'color':brand_colors['text']}),
                dcc.Input(id='qty', type='number', value=0.001, step=0.0001),
                html.Br(), html.Label("Take Profit (%)", style={'color':brand_colors['text']}),
                dcc.Slider(id='tp', min=0.01, max=0.1, step=0.01, value=0.05),
                html.Br(), html.Label("Stop Loss (%)", style={'color':brand_colors['text']}),
                dcc.Slider(id='sl', min=0.01, max=0.1, step=0.01, value=0.03),
                html.Br(), html.Label("Strategy", style={'color':brand_colors['text']}),
                dcc.RadioItems(id='strategy', options=[{'label':'Forecast Only','value':'Forecast'},{'label':'Forecast + RSI','value':'Forecast + RSI'}], value='Forecast'),
                html.Br(), html.Label("RSI Threshold (%)", style={'color':brand_colors['text']}),
                dcc.Slider(id='rsi-threshold', min=50, max=90, step=1, value=70),
                html.Br(), dbc.Button("Start Streaming", id='start-btn', color='secondary')
            ])
        ]), width=4),
        dbc.Col(dcc.Graph(id='price-chart', config={'displayModeBar':False}), width=8)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='pnl-chart', config={'displayModeBar':False}), width=6),
        dbc.Col(dcc.Graph(id='drawdown-chart', config={'displayModeBar':False}), width=6)
    ]),
    dbc.Row(dbc.Col([html.H4("Open Position", style={'color':brand_colors['text']}), html.Div(id='position-table')]), className='mt-4')
], fluid=True, style={'backgroundColor':brand_colors['background']})

@app.callback(
    Output('price-chart','figure'), Output('pnl-chart','figure'),
    Output('drawdown-chart','figure'), Output('position-table','children'),
    Input('start-btn','n_clicks'), Input('qty','value'), Input('tp','value'),
    Input('sl','value'), Input('strategy','value'), Input('rsi-threshold','value')
)
def update_all(n, qty, tp, sl, strategy, rsi_threshold):
    global run_settings, stream_thread
    # set run settings
    run_settings = {'qty':qty,'tp':tp,'sl':sl,'strategy':strategy,'rsi_threshold':rsi_threshold}
    # start streaming once
    if n and stream_thread is None:
        stream_thread = threading.Thread(target=start_stream, daemon=True)
        stream_thread.start()
    # Price
    df = crypto_df.copy()
    fig = go.Figure([go.Scatter(x=df.index, y=df['Price'], mode='lines', line=dict(color=brand_colors['secondary']))])
    # trades
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            fig.add_trace(go.Scatter(x=[t['entry_time']], y=[t['entry_price']], mode='markers', marker=dict(symbol='triangle-down', color=brand_colors['accent'])))
            fig.add_trace(go.Scatter(x=[t['exit_time']], y=[t['exit_price']], mode='markers', marker=dict(symbol='triangle-up', color=brand_colors['primary'])))
    fig.update_layout(paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])
    # P&L & drawdown
    pnl_fig=go.Figure(); dd_fig=go.Figure()
    if not trades_df.empty:
        pnl=(trades_df['entry_price']-trades_df['exit_price'])/trades_df['entry_price']
        cum_pnl=pnl.cumsum(); cum_max=cum_pnl.cummax(); dd=cum_pnl-cum_max
        pnl_fig.add_trace(go.Scatter(x=trades_df['exit_time'], y=cum_pnl, mode='lines', line=dict(color=brand_colors['accent'])))
        dd_fig.add_trace(go.Scatter(x=trades_df['exit_time'], y=dd, mode='lines', line=dict(color=brand_colors['accent'])))
    # position table
    try:
        pos=api.get_position('BTC/USD');
        pos_df=pd.DataFrame([{'Symbol':pos.symbol,'Qty':pos.qty,'Market Value':f"${float(pos.market_value):,.2f}",'Unrealized P/L':f"${float(pos.unrealized_pl):,.2f}",'Side':pos.side}])
    except:
        pos_df=pd.DataFrame([{'Symbol':'None','Qty':0,'Market Value':'$0','Unrealized P/L':'$0','Side':'none'}])
    table=dbc.Table.from_dataframe(pos_df,striped=True,bordered=True,hover=True)
    return fig, pnl_fig, dd_fig, table

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
