import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

# Backtest logic only (for brevity)

def pattern_signal(df):
    if len(df) < 2:
        return None
    o, c = df.Open.iloc[-2], df.Close.iloc[-2]
    o1, c1 = df.Open.iloc[-1], df.Close.iloc[-1]
    if o1 > c1 and o < c and c1 < o and o1 >= c:
        return 'short'
    if o1 < c1 and o > c and c1 > o and o1 <= c:
        return 'long'
    return None


def sma_signal(df, fast, slow):
    if len(df) < slow:
        return None
    sma_f = df.Price.rolling(fast).mean()
    sma_s = df.Price.rolling(slow).mean()
    if sma_f.iloc[-2] < sma_s.iloc[-2] and sma_f.iloc[-1] > sma_s.iloc[-1]:
        return 'long'
    if sma_f.iloc[-2] > sma_s.iloc[-2] and sma_f.iloc[-1] < sma_s.iloc[-1]:
        return 'short'
    return None


def backtest(pair, tp, sl, strategy, fast, slow):
    # Fetch data
    ticker = {'EUR_USD':'EURUSD=X', 'AUD_USD':'AUDUSD=X'}[pair]
    hist = yf.download(ticker, period='6mo', interval='15m')
    hist = hist.reset_index()
    hist['Price'] = hist['Close']
    cash = 10000.0
    position = 0
    entry_price = 0.0
    trades = []
    eq = []

    for i in range(1, len(hist)):
        window = hist.iloc[:i+1]
        sig = pattern_signal(window) if strategy=='Pattern' else sma_signal(window, fast, slow)
        price = hist.Close.iloc[i]

        # Entry
        if sig == 'long' and position == 0:
            position = 1
            entry_price = price
            trades.append({'time': hist.time.iloc[i], 'side':'long', 'entry':price})
        elif sig == 'short' and position == 0:
            position = -1
            entry_price = price
            trades.append({'time': hist.time.iloc[i], 'side':'short', 'entry':price})

        # Exit
        if position != 0:
            ret = (price - entry_price)/entry_price * position
            if ret >= tp or ret <= -sl:
                cash *= (1 + ret)
                trades[-1].update({'exit_time': hist.time.iloc[i], 'exit': price, 'ret': ret, 'cash': cash})
                position = 0

        # Equity
        current_eq = cash * (1 + ((price - entry_price)/entry_price * position if position != 0 else 0))
        eq.append({'time': hist.time.iloc[i], 'equity': current_eq})

    trades_df = pd.DataFrame(trades)
    eq_df     = pd.DataFrame(eq)
    return hist, trades_df, eq_df

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    html.H2('Backtest Historical Strategy', style={'color':'#ECF0F1'}),
    dbc.Row([
        dbc.Col([
            html.Label('Pair'),
            dcc.Dropdown(['EUR_USD','AUD_USD'], 'EUR_USD', id='bt-pair'),
            html.Label('TP %'), dcc.Input(id='bt-tp', type='number', value=3),
            html.Label('SL %'), dcc.Input(id='bt-sl', type='number', value=5),
            html.Label('Strategy'),
            dcc.RadioItems(['Pattern','SMA'], 'Pattern', id='bt-strategy'),
            html.Label('SMA Fast'), dcc.Input(id='bt-sf', type='number', value=5),
            html.Label('SMA Slow'), dcc.Input(id='bt-ss', type='number', value=20),
            dbc.Button('Run Backtest', id='bt-run')
        ], width=4),
        dbc.Col(dcc.Graph(id='bt-price-chart'), width=8)
    ]),
    html.Div(id='bt-metrics', style={'color':'#ECF0F1'})
], fluid=True)

@app.callback(
    [Output('bt-price-chart','figure'), Output('bt-metrics','children')],
    [Input('bt-run','n_clicks')],
    [dash.dependencies.State('bt-pair','value'),
     dash.dependencies.State('bt-tp','value'),
     dash.dependencies.State('bt-sl','value'),
     dash.dependencies.State('bt-strategy','value'),
     dash.dependencies.State('bt-sf','value'),
     dash.dependencies.State('bt-ss','value')]
)
def run_backtest(n, pair, tp, sl, strategy, sf, ss):
    if not n:
        return go.Figure(), ''

    hist, trades, eq = backtest(pair, tp/100, sl/100, strategy, sf, ss)

    # Price chart with trades
    fig = go.Figure(data=[
        go.Candlestick(x=hist.time, open=hist.Open, high=hist.High, low=hist.Low, close=hist.Close)
    ])
    for _, t in trades.iterrows():
        fig.add_trace(go.Scatter(x=[t.time], y=[t.entry], mode='markers', marker=dict(color='#E74C3C', size=10)))

    # Metrics
    total_ret = eq.equity.iloc[-1]/10000 - 1 if not eq.empty else 0
    rrtn = html.Ul([
        html.Li(f"Trades: {len(trades)}"),
        html.Li(f"Return: {total_ret:.2%}")
    ])
    return fig, rrtn

if __name__=='__main__':
    app.run(debug=False)
