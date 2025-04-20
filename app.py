import os
import threading
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import requests

# Alpaca clients
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# --- Configuration ---
API_KEY    = os.getenv('ALPACA_KEY') or os.getenv('ALPACA_API_KEY') or "PK93LZQTSB35L3CL60V5"
API_SECRET = os.getenv('ALPACA_SECRET') or os.getenv('ALPACA_SECRET_KEY') or "HDn7c1Mp3JVvgq98dphRDJH1nt3She3pe5Y9bJi0"
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

BITSTAMP_URL  = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
ALPACA_SYMBOL = "BTC/USD"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# --- Stream handler for order updates ---
trade_updates_list = []
async def trade_updates_handler(update):
    try:
        event = update.event
        order = update.order
        trade_updates_list.append({
            'event': event,
            'symbol': order.symbol,
            'filled_qty': order.filled_qty,
            'filled_avg_price': order.filled_avg_price,
            'timestamp': update.timestamp
        })
    except Exception as e:
        logging.error(f"Stream handler error: {e}")

def start_trade_stream():
    stream = TradingStream(API_KEY, API_SECRET, paper=True)
    stream.subscribe_trade_updates(trade_updates_handler)
    stream.run()

threading.Thread(target=start_trade_stream, daemon=True).start()

# --- RSI helpers ---
def compute_rsi(series, window=14):
    delta = series.diff().dropna()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def fetch_bitstamp_candles(limit=1000, step=60):
    resp = requests.get(BITSTAMP_URL, params={'step': step, 'limit': limit})
    data = resp.json().get('data', {}).get('ohlc', [])
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)
    return df.set_index('timestamp')

# --- The corrected RSI trading job ---
def rsi_trading_job():
    try:
        # fetch & compute
        df = fetch_bitstamp_candles(1000,60)
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        df['RSI'] = compute_rsi(df['close'],14)
        last_rsi = df['RSI'].iloc[-1]

        # account & positions
        account   = trade_client.get_account()
        usd_avail = float(account.cash)
        positions = trade_client.get_all_positions()

        # normalize symbol
        clean_symbol = ALPACA_SYMBOL.replace("/","")
        btc_pos = next((p for p in positions if p.symbol==clean_symbol), None)
        current_qty = float(btc_pos.qty) if btc_pos else 0.0

        # === DUST FIX: round to 8 dec, treat tiny as zero ===
        current_qty = round(current_qty, 8)

        logging.info(f"RSI={last_rsi:.2f}, current_qty={current_qty:.8f}, cash=${usd_avail:.2f}")

        # BUY when oversold & flat
        if last_rsi <= 30 and current_qty == 0:
            logging.info("RSI ≤30 & flat → BUY")
            price      = df['close'].iloc[-1]
            target_usd = min(100, usd_avail)
            buy_qty    = round(target_usd/price - 1e-8, 8)
            if buy_qty > 0:
                mo = MarketOrderRequest(
                    symbol=ALPACA_SYMBOL,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    qty=buy_qty
                )
                resp = trade_client.submit_order(order_data=mo)
                logging.info(f"BUY executed: qty={buy_qty} @ {price:.2f}, order_id={resp.id}")
            else:
                logging.info("Not enough USD to buy any BTC")

        # SELL entire position when overbought
        elif last_rsi >= 70 and current_qty > 0:
            logging.info("RSI ≥70 & in position → SELL all")
            sell_qty = current_qty  # no dust leftover
            mo = MarketOrderRequest(
                symbol=ALPACA_SYMBOL,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                qty=sell_qty
            )
            resp = trade_client.submit_order(order_data=mo)
            logging.info(f"SELL executed: qty={sell_qty}, order_id={resp.id}")

        else:
            logging.info("No trade signal")

    except Exception as e:
        logging.error(f"RSI trading job error: {e}")

# schedule the job
scheduler = BackgroundScheduler(timezone='US/Eastern')
scheduler.add_job(rsi_trading_job, 'interval', minutes=1)
scheduler.start()

# --- Dash App (unchanged aside from imports) ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
brand_colors = {'background':'#2C3E50','text':'#ECF0F1'}

app.layout = dbc.Container([
    html.H2('Crypto Dashboard (BTC/USD)', style={'color':brand_colors['text']}),
    dbc.Row([
        dbc.Col([
            html.Label('BTC Qty',style={'color':brand_colors['text']}),
            dcc.Input(id='btc-qty',type='number',value=0.5,step=0.1),
            html.Br(),html.Br(),
            dbc.Button('Buy', id='buy-btc', color='success', className='me-2'),
            dbc.Button('Sell', id='sell-btc', color='danger'),
            html.Br(),html.Br(),
            html.Div(id='order-status', style={'color':brand_colors['text']})
        ], width=3),
        dbc.Col(dcc.Graph(id='price-chart'), width=9)
    ]),
    dbc.Row(dcc.Graph(id='rsi-chart'), className='mt-4'),
    dcc.Interval(id='interval', interval=30*1000, n_intervals=0),
    dbc.Row(dbc.Col(dash_table.DataTable(
        id='positions-table', page_size=10,
        style_header={'backgroundColor':brand_colors['background'],'color':brand_colors['text']},
        style_cell={'backgroundColor':brand_colors['background'],'color':brand_colors['text']}
    )), className='mt-4'),
    html.H4('Order Stream', style={'color':brand_colors['text'],'marginTop':'20px'}),
    dbc.Row(dbc.Col(dash_table.DataTable(
        id='orders-table', page_size=10,
        style_header={'backgroundColor':brand_colors['background'],'color':brand_colors['text']},
        style_cell={'backgroundColor':brand_colors['background'],'color':brand_colors['text']}
    )), className='mt-2')
], fluid=True, style={'backgroundColor':brand_colors['background'],'padding':'20px'})

# ... (your existing callbacks for charts, manual orders, tables) ...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
