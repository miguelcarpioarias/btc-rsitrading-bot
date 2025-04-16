
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
import threading
import time
from datetime import datetime

# === Alpaca API Setup ===
ALPACA_API_KEY = 'PKZ02G1UDCK1CFD62BUZ'
ALPACA_SECRET_KEY = '4O0tgiK1Nua4rwe9E5GwbOg4D1h4Ds5nndYuxJSV'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

# === BTC Data Fetch ===
def get_crypto_data(symbol='BTC-USD'):
    df = yf.download(symbol, period='7d', interval='1h')[['Close']]
    df.columns = ['Price']
    return df

# === Signal Generator (simple 3% drop forecast) ===
def generate_signal(prices):
    forecast = prices.shift(1) * 0.97
    signal = (forecast < prices).astype(int)
    return signal

# === Place Live Short Trade on Alpaca ===
def place_short_trade(symbol='BTC/USD', qty=0.001, take_profit_pct=0.05, stop_loss_pct=0.03):
    current_price = float(api.get_latest_trade(symbol).price)
    take_profit_price = round(current_price * (1 - take_profit_pct), 2)
    stop_loss_price = round(current_price * (1 + stop_loss_pct), 2)

    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='gtc',
        order_class='bracket',
        take_profit={'limit_price': take_profit_price},
        stop_loss={'stop_price': stop_loss_price}
    )
    return order

# === Auto Trader (runs every hour) ===
def auto_trade():
    while True:
        try:
            prices = get_crypto_data()['Price']
            signal = generate_signal(prices)
            if signal.iloc[-1] == 1:
                place_short_trade()
                print(f"Trade placed at {datetime.now()}")
        except Exception as e:
            print(f"Error in auto trade: {e}")
        time.sleep(3600)  # Wait 1 hour

# Start auto trader in background
threading.Thread(target=auto_trade, daemon=True).start()

# === Dash Layout ===
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("BTC/USD Short-Selling Bot (Alpaca + Dash)"),
    dcc.Graph(id='price-graph'),
    html.Div(id='last-trade-status')
])

@app.callback(
    Output('price-graph', 'figure'),
    Output('last-trade-status', 'children'),
    Input('price-graph', 'id')
)
def update_graph(_):
    df = get_crypto_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='BTC/USD'))
    return fig, f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
