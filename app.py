import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

# --- Signal Definitions ---
def pattern_signal(df):
    """Return 'long', 'short', or None based on two-bar reversal pattern."""
    if df.shape[0] < 2:
        return None
    # Extract previous and last bars as scalars
    prev = df.iloc[-2]
    last = df.iloc[-1]
    o_prev, c_prev = prev['Open'], prev['Close']
    o_last, c_last = last['Open'], last['Close']
    # Bearish reversal
    if (o_last > c_last) and (o_prev < c_prev) and (c_last < o_prev) and (o_last >= c_prev):
        return 'short'
    # Bullish reversal
    if (o_last < c_last) and (o_prev > c_prev) and (c_last > o_prev) and (o_last <= c_prev):
        return 'long'
    return None
    o_prev = df['Open'].iloc[-2]
    c_prev = df['Close'].iloc[-2]
    o_last = df['Open'].iloc[-1]
    c_last = df['Close'].iloc[-1]
    # Bearish reversal
    if o_last > c_last and o_prev < c_prev and c_last < o_prev and o_last >= c_prev:
        return 'short'
    # Bullish reversal
    if o_last < c_last and o_prev > c_prev and c_last > o_prev and o_last <= c_prev:
        return 'long'
    return None


def sma_signal(df, fast, slow):
    """Return 'long' or 'short' on SMA crossover, or None."""
    if df.shape[0] < slow:
        return None
    sma_fast = df['Price'].rolling(window=fast).mean()
    sma_slow = df['Price'].rolling(window=slow).mean()
    # Previous and last values
    prev_fast, curr_fast = sma_fast.iloc[-2], sma_fast.iloc[-1]
    prev_slow, curr_slow = sma_slow.iloc[-2], sma_slow.iloc[-1]
    if prev_fast < prev_slow and curr_fast > curr_slow:
        return 'long'
    if prev_fast > prev_slow and curr_fast < curr_slow:
        return 'short'
    return None

# --- Backtest Engine ---
def backtest(symbol, tp, sl, strategy, fast, slow):
    # Download daily bars over 6 months
    hist = yf.download(symbol, period='6mo', interval='1d')
    hist = hist.rename_axis('time').reset_index()
    hist['Price'] = hist['Close']

    cash = 10000.0
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    for i in range(1, len(hist)):
        window = hist.iloc[:i+1].copy()
        # Determine signal
        sig = pattern_signal(window) if strategy == 'Pattern' else sma_signal(window, fast, slow)
        price = hist['Close'].iloc[i]

        # Entry
        if sig == 'long' and position == 0:
            position = 1
            entry_price = price
            trades.append({'time': hist['time'].iloc[i], 'side': 'long', 'entry': price})
        elif sig == 'short' and position == 0:
            position = -1
            entry_price = price
            trades.append({'time': hist['time'].iloc[i], 'side': 'short', 'entry': price})

        # Exit
        if position != 0:
            ret = (price - entry_price) / entry_price * position
            if ret >= tp or ret <= -sl:
                cash *= (1 + ret)
                trades[-1].update({'exit_time': hist['time'].iloc[i], 'exit': price, 'ret': ret, 'cash': cash})
                position = 0

        # Equity
        current_equity = cash * (1 + ((price-entry_price)/entry_price * position if position != 0 else 0))
        equity.append({'time': hist['time'].iloc[i], 'equity': current_equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)
    return hist, trades_df, equity_df

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([
    html.H2('Equity Backtest Dashboard', style={'color': '#ECF0F1'}),
    dbc.Row([
        dbc.Col([
            html.Label('Ticker'),
            dcc.Dropdown(
                options=[{'label': t, 'value': t} for t in ['AMD','NVDA','AAPL','TSLA']],
                value='AMD', id='bt-pair'
            ),
            html.Br(),
            html.Label('Take-Profit (%)'), dcc.Input(id='bt-tp', type='number', value=3), html.Br(),
            html.Label('Stop-Loss (%)'),   dcc.Input(id='bt-sl', type='number', value=5), html.Br(),
            html.Label('Strategy'),
            dcc.RadioItems(
                options=[{'label':'Pattern','value':'Pattern'}, {'label':'SMA Crossover','value':'SMA'}],
                value='Pattern', id='bt-strategy'
            ), html.Br(),
            html.Label('SMA Fast Period'), dcc.Input(id='bt-sf', type='number', value=5), html.Br(),
            html.Label('SMA Slow Period'), dcc.Input(id='bt-ss', type='number', value=20), html.Br(),
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

    hist, trades, equity = backtest(pair, tp/100, sl/100, strategy, sf, ss)

    # Price chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=hist['time'], open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']
        )
    ])
    # Trade markers
    for _, t in trades.iterrows():
        fig.add_trace(go.Scatter(
            x=[t['time']], y=[t['entry']], mode='markers',
            marker=dict(color='#E74C3C', size=10,
                        symbol='triangle-up' if t['side']=='long' else 'triangle-down')
        ))
    fig.update_layout(
        title=f'{pair} Backtest', xaxis_rangeslider_visible=False,
        paper_bgcolor='#2C3E50', plot_bgcolor='#2C3E50', font_color='#ECF0F1'
    )

    # Metrics
    total_ret = equity['equity'].iloc[-1]/10000 - 1 if not equity.empty else 0
    rets = equity['equity'].pct_change().dropna()
    sharpe = (rets.mean()/rets.std()*np.sqrt(252)) if not rets.empty else 0
    max_dd = (equity['equity']/equity['equity'].cummax() - 1).min() if not equity.empty else 0

    metrics = html.Ul([
        html.Li(f"Trades: {len(trades)}"),
        html.Li(f"Total Return: {total_ret:.2%}"),
        html.Li(f"Sharpe Ratio: {sharpe:.2f}"),
        html.Li(f"Max Drawdown: {max_dd:.2%}")
    ])

    return fig, metrics

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
