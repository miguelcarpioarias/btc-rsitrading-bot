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
import time
from datetime import datetime, timedelta, timezone
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
ALPACA_SECRET_KEY = '4O0tgiK1Nua4HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0PK93LZQTSB35L3CL60V5'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets/v2'
SENDGRID_API_KEY = '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'
FROM_EMAIL = 'miguelcarpioariasec@gmail.com'
TO_EMAIL = 'miguelcarpioariasec@gmail.com'

# === CLIENT INITIALIZATION ===
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)

# === DATA FETCH: STREAM VIA ALPACA BARS ===
def get_crypto_data(symbol='BTC/USD', limit_hours=168):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=limit_hours)
    bars = api.get_crypto_bars(symbol, TimeFrame.Hour, start.isoformat(), end.isoformat()).df
    bars = bars.tz_convert(None)
    return bars[['close']].rename(columns={'close': 'Price'})

# === RSI CALCULATION ===
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === SENDGRID EMAIL ALERT ===
def send_trade_alert(subject, content):
    msg = Mail(from_email=FROM_EMAIL, to_emails=TO_EMAIL,
               subject=subject, plain_text_content=content)
    try:
        sg.send(msg)
    except Exception as e:
        print(f"Email failed: {e}")

# === SIGNAL GENERATOR ===
def generate_signal(prices, strategy='Forecast', rsi_threshold=70):
    forecast = prices.shift(1) * 0.97
    base_signal = (forecast < prices).astype(int)
    if strategy == 'Forecast + RSI':
        rsi = compute_rsi(prices)
        rsi_signal = (rsi > rsi_threshold).astype(int)
        return (base_signal & rsi_signal).astype(int)
    return base_signal

# === TRADE EXECUTION ===
def place_short_trade(qty, tp_pct, sl_pct):
    price = float(api.get_latest_trade('BTC/USD').price)
    tp_price = round(price * (1 - tp_pct), 2)
    sl_price = round(price * (1 + sl_pct), 2)
    api.submit_order(
        symbol='BTC/USD', qty=qty, side='sell', type='market', time_in_force='gtc',
        order_class='bracket', take_profit={'limit_price': tp_price}, stop_loss={'stop_price': sl_price}
    )
    send_trade_alert('BTC Short Executed', f"Qty: {qty} at {price}, TP: {tp_price}, SL: {sl_price}")

# === TRADE MARKERS & P&L ===
def get_trade_history(days=7):
    after = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    orders = api.list_orders(status='all', limit=100, direction='desc', after=after)
    trades = []
    entry = None
    for o in sorted(orders, key=lambda x: x.submitted_at):
        if o.symbol == 'BTC/USD' and o.side == 'sell' and o.order_type == 'market':
            entry = {'entry_time': o.submitted_at, 'entry_price': float(o.filled_avg_price)}
        elif entry and o.symbol == 'BTC/USD' and o.side == 'buy':
            entry['exit_time'] = o.filled_at
            entry['exit_price'] = float(o.filled_avg_price)
            trades.append(entry)
            entry = None
    return pd.DataFrame(trades)

# === BACKGROUND AUTO TRADER ===
def auto_trade_loop(qty, tp_pct, sl_pct, strategy, rsi_threshold):
    while True:
        prices = get_crypto_data()['Price']
        sig = generate_signal(prices, strategy, rsi_threshold)
        if sig.iloc[-1] == 1:
            place_short_trade(qty, tp_pct, sl_pct)
            print(f"Trade placed: {datetime.now()}")
        time.sleep(3600)

# === DASH APP SETUP ===
external_styles = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_styles)
server = app.server

# === LAYOUT ===
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("BTC/USD Short-Selling Bot", style={'color': brand_colors['text']}), width=12)], align='center'),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Settings"), dbc.CardBody([
                html.Label("Quantity (BTC)", style={'color': brand_colors['text']}),
                dcc.Input(id='qty', type='number', value=0.001, step=0.0001),
                html.Br(), html.Label("Take Profit (%)", style={'color': brand_colors['text']}),
                dcc.Slider(id='tp', min=0.01, max=0.1, step=0.01, value=0.05, marks={i/100: f"{i}%" for i in range(1, 11)}),
                html.Br(), html.Label("Stop Loss (%)", style={'color': brand_colors['text']}),
                dcc.Slider(id='sl', min=0.01, max=0.1, step=0.01, value=0.03, marks={i/100: f"{i}%" for i in range(1, 11)}),
                html.Br(), html.Label("Strategy", style={'color': brand_colors['text']}),
                dcc.RadioItems(id='strategy', options=[
                    {'label':'Forecast Only','value':'Forecast'}, {'label':'Forecast + RSI','value':'Forecast + RSI'}], value='Forecast'),
                html.Br(), html.Label("RSI Threshold (%)", style={'color': brand_colors['text']}),
                dcc.Slider(id='rsi-threshold', min=50, max=90, step=1, value=70, marks={i: str(i) for i in range(50, 91, 5)}),
                html.Br(), dbc.Button("Start", id='start-btn', color='secondary')
            ])
        ]), width=4),
        dbc.Col(dcc.Graph(id='price-chart', config={'displayModeBar':False}, style={'backgroundColor': brand_colors['background']}), width=8)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='pnl-chart', config={'displayModeBar':False}), width=6),
        dbc.Col(dcc.Graph(id='drawdown-chart', config={'displayModeBar':False}), width=6)
    ]),
    dbc.Row(dbc.Col([html.H4("Open Position", style={'color': brand_colors['text']}), html.Div(id='position-table')]), className='mt-4')
], fluid=True, style={'backgroundColor': brand_colors['background']})

# === CALLBACK ===
@app.callback(
    Output('price-chart','figure'), Output('pnl-chart','figure'),
    Output('drawdown-chart','figure'), Output('position-table','children'),
    Input('start-btn','n_clicks'), Input('qty','value'), Input('tp','value'),
    Input('sl','value'), Input('strategy','value'), Input('rsi-threshold','value')
)
def update_all(n, qty, tp, sl, strategy, rsi_threshold):
    df = get_crypto_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', line=dict(color=brand_colors['secondary']), name='Price'))
    trades = get_trade_history()
    if not trades.empty:
        for _, t in trades.iterrows():
            fig.add_trace(go.Scatter(x=[t['entry_time']], y=[t['entry_price']], mode='markers', marker=dict(symbol='triangle-down', size=10, color=brand_colors['accent']), name='Entry'))
            fig.add_trace(go.Scatter(x=[t['exit_time']], y=[t['exit_price']], mode='markers', marker=dict(symbol='triangle-up', size=10, color=brand_colors['primary']), name='Exit'))
    fig.update_layout(paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'], title="BTC/USD Price with Trades")

    pnl_fig = go.Figure()
    if not trades.empty:
        pnl = ((trades['entry_price'] - trades['exit_price']) / trades['entry_price']).cumsum()
        pnl_fig.add_trace(go.Scatter(x=trades['exit_time'], y=pnl, mode='lines+markers', name='Cumulative P&L', line=dict(color=brand_colors['accent'])))
    pnl_fig.update_layout(title='Cumulative P&L', paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])

    dd_fig = go.Figure()
    if not trades.empty:
        dd = pnl - pnl.cummax()
        dd_fig.add_trace(go.Scatter(x=trades['exit_time'], y=dd, mode='lines', name='Drawdown', line=dict(color=brand_colors['accent'])))
    dd_fig.update_layout(title='Drawdown', paper_bgcolor=brand_colors['background'], plot_bgcolor=brand_colors['background'], font_color=brand_colors['text'])

    try:
        pos = api.get_position('BTC/USD')
        pos_df = pd.DataFrame([{'Symbol': pos.symbol, 'Qty': pos.qty, 'Market Value': f"${float(pos.market_value):,.2f}", 'Unrealized P/L': f"${float(pos.unrealized_pl):,.2f}", 'Side': pos.side}])
    except:
        pos_df = pd.DataFrame([{'Symbol':'None','Qty':0,'Market Value':'$0','Unrealized P/L':'$0','Side':'none'}])
    table = dbc.Table.from_dataframe(pos_df, striped=True, bordered=True, hover=True)

    if n and n == 1:
        threading.Thread(target=auto_trade_loop, args=(qty, tp, sl, strategy, rsi_threshold), daemon=True).start()

    return fig, pnl_fig, dd_fig, table

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
