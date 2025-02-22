# src/incremental_fetcher.py

import os
import pandas as pd
from datetime import datetime
from .config import RAW_DATA_PATH, BYBIT_API_KEY, BYBIT_API_SECRET
from .data_fetcher import _fetch_bybit_kline

def load_kline_incremental(
    symbol,
    category,
    interval,
    start,
    end,
    csv_dir=RAW_DATA_PATH
):
    """
    Смотрит {symbol}_{category}_{interval}.csv.
    Если нет => качаем весь start..end.
    Если есть => догружаем недостающие слева/справа.
    Возвращаем подотрезок df_sub (start..end).
    """
    os.makedirs(csv_dir, exist_ok=True)
    csv_name = f"{symbol}_{category}_{interval}.csv"
    csv_path = os.path.join(csv_dir, csv_name)

    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)

    if os.path.exists(csv_path):
        df_full = pd.read_csv(csv_path, parse_dates=["open_time"], index_col="open_time")
        df_full.sort_index(inplace=True)
        if df_full.index.tz is not None:
            df_full.index = df_full.index.tz_convert(None)
        csv_start = df_full.index.min()
        csv_end   = df_full.index.max()
        print(f"[{symbol} {interval}] CSV exists. range=({csv_start}..{csv_end})")
    else:
        df_full = pd.DataFrame()
        csv_start, csv_end = None, None
        print(f"[{symbol} {interval}] CSV not found => create new.")

    # Если файл вообще пуст
    if df_full.empty:
        print(f"[{symbol} {interval}] CSV empty => fetch {start}..{end} in one shot.")
        df_new = _fetch_bybit_kline(symbol, category, interval, start, end,
                                    api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
        if not df_new.empty:
            df_full = df_new
            df_full.sort_index(inplace=True)
            df_full.to_csv(csv_path)
        else:
            print(f"[{symbol} {interval}] No data from Bybit for {start}..{end} => empty.")
        return df_full.loc[start_dt:end_dt].copy()

    # Догрузка слева
    if start_dt < csv_start:
        left_end_dt = csv_start - pd.Timedelta(days=1)
        if left_end_dt >= start_dt:
            fetch_start = start_dt.strftime("%Y-%m-%d")
            fetch_end   = left_end_dt.strftime("%Y-%m-%d")
            print(f"[{symbol} {interval}] Догружаем слева: {fetch_start}..{fetch_end}")
            if fetch_start<=fetch_end:
                df_left = _fetch_bybit_kline(symbol, category, interval,
                                             fetch_start, fetch_end,
                                             api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
                if not df_left.empty:
                    if df_left.index.tz is not None:
                        df_left.index = df_left.index.tz_convert(None)
                    df_full = pd.concat([df_left, df_full]).drop_duplicates()
                    df_full.sort_index(inplace=True)
                    df_full.to_csv(csv_path)
                else:
                    print(f"[{symbol} {interval}] no data from Bybit for left fetch => skip.")
        # обновим csv_start
        csv_start = df_full.index.min()
        csv_end   = df_full.index.max()

    # Догрузка справа
    if csv_end is not None and end_dt > csv_end:
        right_start_dt = csv_end + pd.Timedelta(days=1)
        if right_start_dt <= end_dt:
            fetch_start = right_start_dt.strftime("%Y-%m-%d")
            fetch_end   = end_dt.strftime("%Y-%m-%d")
            print(f"[{symbol} {interval}] Догружаем справа: {fetch_start}..{fetch_end}")
            if fetch_start<=fetch_end:
                df_right = _fetch_bybit_kline(symbol, category, interval,
                                              fetch_start, fetch_end,
                                              api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
                if not df_right.empty:
                    if df_right.index.tz is not None:
                        df_right.index = df_right.index.tz_convert(None)
                    df_full = pd.concat([df_full, df_right]).drop_duplicates()
                    df_full.sort_index(inplace=True)
                    df_full.to_csv(csv_path)
                else:
                    print(f"[{symbol} {interval}] no data from Bybit for right fetch => skip.")

    return df_full.loc[start_dt:end_dt].copy()
