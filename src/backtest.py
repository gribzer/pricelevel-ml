# src/backtest.py

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def run_backtest(df, levels=None, initial_capital=10000.0):
    df = df.copy()
    df.sort_index(inplace=True)
    df["position"] = 0
    df["trade_price"] = np.nan

    capital = initial_capital
    position = 0
    entry_price = 0.0
    equity_history = []
    trades = []
    current_trade = None

    if "signal" not in df.columns:
        df["signal"] = 0

    idx_list = df.index.to_list()
    n = len(idx_list)
    for i in range(n - 1):
        date_current = idx_list[i]
        sig = df["signal"].iloc[i]
        date_next = idx_list[i+1]
        open_next = df["open"].iloc[i+1]

        if position == 0:
            if sig == 1:
                entry_price = open_next
                position = 1
                df.loc[date_next, "trade_price"] = entry_price
                current_trade = {
                    "entry_time": date_next,
                    "entry_price": entry_price
                }
        elif position == 1:
            if sig == -1:
                sell_price = open_next
                profit = sell_price - entry_price
                capital += profit
                position = 0
                df.loc[date_next, "trade_price"] = sell_price

                if current_trade:
                    current_trade["exit_time"] = date_next
                    current_trade["exit_price"] = sell_price
                    current_trade["pnl"] = profit
                    trades.append(current_trade)
                current_trade = None

        equity_history.append((date_current, capital))

    if position == 1:
        last_date = idx_list[-1]
        last_close = df["close"].iloc[-1]
        profit = last_close - entry_price
        capital += profit
        if current_trade:
            current_trade["exit_time"] = last_date
            current_trade["exit_price"] = last_close
            current_trade["pnl"] = profit
            trades.append(current_trade)

    eq_curve = pd.DataFrame(equity_history, columns=["time", "Equity"])
    eq_curve.set_index("time", inplace=True)
    return df, trades, eq_curve

def plot_backtest_results(df, trades, equity_curve, levels=None, title="Backtest Results"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.02)

    x_data = df.index
    candles = go.Candlestick(
        x=x_data,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC"
    )
    fig.add_trace(candles, row=1, col=1)

    trade_df = df.dropna(subset=["trade_price"])
    buy_points = trade_df[trade_df["signal"] == 1]
    sell_points = trade_df[trade_df["signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_points.index, 
            y=buy_points["trade_price"],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points.index, 
            y=sell_points["trade_price"],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell'
        ),
        row=1, col=1
    )

    if levels:
        for lvl in levels:
            fig.add_hline(
                y=lvl,
                line=dict(color='blue', width=1, dash='dash'),
                annotation_text=f"{lvl:.2f}",
                annotation_position="top left",
                row=1, col=1
            )

    eq_line = go.Scatter(
        x=equity_curve.index, 
        y=equity_curve["Equity"],
        line=dict(color='purple', width=2),
        name="Equity"
    )
    fig.add_trace(eq_line, row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    fig.show()
