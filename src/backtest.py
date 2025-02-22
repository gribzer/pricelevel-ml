import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def run_backtest(df, levels=None, initial_capital=10000.0):
    """
    Пример бэктеста, ожидающего колонку 'Date'.
    Если нет 'Date', пытаемся найти дату/время в других колонках
    или в индексе, и создать 'Date'.
    """
    df = df.copy()

    # 1. Проверяем, есть ли 'Date'
    if "Date" not in df.columns:
        # Если есть наиболее вероятные названия, переименуем
        renamed_date_col = None
        for candidate in ["date", "datetime", "time", "timestamp"]:
            if candidate in df.columns:
                renamed_date_col = candidate
                break

        if renamed_date_col:
            df.rename(columns={renamed_date_col: "Date"}, inplace=True)
        else:
            # Если ни одной подходящей колонки нет, проверим индекс
            if isinstance(df.index, pd.DatetimeIndex):
                df["Date"] = df.index
            else:
                df.reset_index(inplace=True)
                if "index" in df.columns:
                    df.rename(columns={"index": "Date"}, inplace=True)

    # 2. Теперь сортируем по Date
    if "Date" not in df.columns:
        raise KeyError("Не удалось найти или создать колонку 'Date' в df!")

    df.sort_values("Date", inplace=True)
    df.set_index(pd.RangeIndex(len(df)), inplace=True)

    # 3. Простейшая логика бэктеста
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

    n = len(df)
    for i in range(n - 1):
        date_current = df["Date"].iloc[i]
        sig = df["signal"].iloc[i]
        open_next = df["Open"].iloc[i+1]

        if position == 0:
            if sig == 1:
                # Buy
                entry_price = open_next
                position = 1
                df.loc[i+1, "trade_price"] = entry_price
                current_trade = {
                    "entry_time": df["Date"].iloc[i+1],
                    "entry_price": entry_price
                }
        elif position == 1:
            if sig == -1:
                # Sell
                sell_price = open_next
                profit = sell_price - entry_price
                capital += profit
                position = 0
                df.loc[i+1, "trade_price"] = sell_price

                if current_trade:
                    current_trade["exit_time"] = df["Date"].iloc[i+1]
                    current_trade["exit_price"] = sell_price
                    current_trade["pnl"] = profit
                    trades.append(current_trade)
                current_trade = None

        equity_history.append((date_current, capital))

    # Если позиция осталась открытой
    if position == 1:
        last_close = df["Close"].iloc[-1]
        date_last = df["Date"].iloc[-1]
        profit = last_close - entry_price
        capital += profit
        position = 0
        if current_trade:
            current_trade["exit_time"] = date_last
            current_trade["exit_price"] = last_close
            current_trade["pnl"] = profit
            trades.append(current_trade)
        current_trade = None

    eq_curve = pd.DataFrame(equity_history, columns=["Date", "Equity"])
    eq_curve.set_index("Date", inplace=True)

    return df, trades, eq_curve


def plot_backtest_results(df, trades, equity_curve, levels=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.02)

    # Candlestick
    candles = go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC"
    )
    fig.add_trace(candles, row=1, col=1)

    # Показываем точки входа/выхода
    trade_df = df.dropna(subset=["trade_price"])
    buy_points = trade_df[trade_df["signal"] == 1]
    sell_points = trade_df[trade_df["signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_points["Date"], y=buy_points["trade_price"],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points["Date"], y=sell_points["trade_price"],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell'
        ),
        row=1, col=1
    )

    # Уровни (опционально)
    if levels is not None:
        for i, lvl in enumerate(levels):
            fig.add_hline(
                y=lvl,
                line=dict(color='blue', width=1, dash='dash'),
                annotation_text=f"Level {i+1}: {lvl:.2f}",
                annotation_position="top left",
                row=1, col=1
            )

    # Кривая капитала
    eq_line = go.Scatter(
        x=equity_curve.index, y=equity_curve["Equity"],
        line=dict(color='purple', width=2),
        name="Equity"
    )
    fig.add_trace(eq_line, row=2, col=1)

    fig.update_layout(
        title="Backtest Results",
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    fig.show()
