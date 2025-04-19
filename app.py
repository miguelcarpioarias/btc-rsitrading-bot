import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

# --- Signal Definitions ---
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

# --- Backtest Engine ---
def backtest(symbol, tp, sl, strategy, fast, slow):
    # For equities, use daily bars over 12 months to stay within Yahoo limits
    hist = yf.download(symbol, period='12mo', interval='1d')
    # Reset index to get Date column and rename to time
    hist = hist.reset_index().rename(columns={'Date':'time'})
    hist = hist.reset_index().rename(columns={'Datetime':'time'})
    hist['Price'] = hist['Close']
    cash = 10000.0
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    for i in range(1, len(hist)):
        window = hist.iloc[:i+1]
        sig = pattern_signal(window) if strategy == 'Pattern' else sma_signal(window, fast, slow)
        price = hist.Close.iloc[i]

        # Entry logic
        if sig == 'long' and position == 0:
            position = 1
            entry_price = price
            trades.append({'time': hist.time.iloc[i], 'side': 'long', 'entry': price})
        elif sig == 'short' and position == 0:
            position = -1
            entry_price = price
            trades.append({'time': hist.time.iloc[i], 'side': 'short', 'entry': price})

        # Exit logic
        if position != 0:
            ret = (price - entry_price) / entry_price * position
            if ret >= tp or ret <= -sl:
                cash *= (1 + ret)
                trades[-1].update({'exit_time': hist.time.iloc[i], 'exit': price, 'ret': ret, 'cash': cash})
                position = 0

        # Track equity
        current_eq = cash * (1 + ((price - entry_price)/entry_price * position if position != 0 else 0))
        equity.append({'time': hist.time.iloc[i], 'equity': current_eq})

    return hist, pd.DataFrame(trades), pd.DataFrame(equity)

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([
    html.H2('Equity Backtest Dashboard', style={'color': '#ECF0F1'}),
    dbc.Row([
        dbc.Col([
            html.Label('Ticker'),
            dcc.Dropdown(
                options=[
                    {'label': 'AMD', 'value': 'AMD'},
                    {'label': 'NVDA', 'value': 'NVDA'},
                    {'label': 'AAPL', 'value': 'AAPL'},
                    {'label': 'TSLA', 'value': 'TSLA'}
                ],
                value='AMD',
                id='bt-pair'
            ),
            html.Br(),
            html.Label('Take-Profit (%)'),
            dcc.Input(id='bt-tp', type='number', value=3),
            html.Br(),
            html.Label('Stop-Loss (%)'),
            dcc.Input(id='bt-sl', type='number', value=5),
            html.Br(),
            html.Label('Strategy'),
            dcc.RadioItems(
                options=[{'label': 'Pattern', 'value': 'Pattern'}, {'label': 'SMA Crossover', 'value': 'SMA'}],
                value='Pattern',
                id='bt-strategy'
            ),
            html.Br(),
            html.Label('SMA Fast Period'),
            dcc.Input(id='bt-sf', type='number', value=5),
            html.Br(),
            html.Label('SMA Slow Period'),
            dcc.Input(id='bt-ss', type='number', value=20),
            html.Br(),
            dbc.Button('Run Backtest', id='bt-run', color='secondary')
        ], width=4),
        dbc.Col(dcc.Graph(id='bt-price-chart'), width=8)
    ]),
    html.Div(id='bt-metrics', style={'color': '#ECF0F1', 'marginTop': '20px'})
], fluid=True)

@app.callback(
    Output('bt-price-chart', 'figure'),
    Output('bt-metrics', 'children'),
    Input('bt-run', 'n_clicks'),
    State('bt-pair', 'value'),
    State('bt-tp', 'value'),
    State('bt-sl', 'value'),
    State('bt-strategy', 'value'),
    State('bt-sf', 'value'),
    State('bt-ss', 'value')
)
def run_backtest(n_clicks, pair, tp, sl, strategy, sf, ss):
    if not n_clicks:
        return go.Figure(), ''

    hist, trades, eq = backtest(pair, tp/100, sl/100, strategy, sf, ss)

    # Price chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=hist.time, open=hist.Open, high=hist.High, low=hist.Low, close=hist.Close
        )
    ])
    # Add trade entries
    for _, t in trades.iterrows():
        fig.add_trace(go.Scatter(
            x=[t.time], y=[t.entry], mode='markers',
            marker=dict(color='#E74C3C', size=10, symbol='triangle-up' if t.side=='long' else 'triangle-down')
        ))
    fig.update_layout(
        title=f'{pair} Backtest', xaxis_rangeslider_visible=False,
        paper_bgcolor='#2C3E50', plot_bgcolor='#2C3E50', font_color='#ECF0F1'
    )

    # Metrics
    total_ret = eq.equity.iloc[-1] / 10000 - 1 if not eq.empty else 0
    returns = eq.equity.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if not returns.empty else 0
    max_dd = (eq.equity / eq.equity.cummax() - 1).min() if not eq.empty else 0
    metrics = html.Ul([
        html.Li(f"Trades: {len(trades)}"),
        html.Li(f"Total Return: {total_ret:.2%}"),
        html.Li(f"Sharpe Ratio: {sharpe:.2f}"),
        html.Li(f"Max Drawdown: {max_dd:.2%}")
    ])

    return fig, metrics

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(__import__('os').environ.get('PORT', 10000)), debug=False)
