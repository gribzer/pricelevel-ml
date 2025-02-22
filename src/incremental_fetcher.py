# src/incremental_fetcher.py

import os
import pandas as pd
from datetime import datetime, timedelta

# Предположим, что _fetch_bybit_kline - это низкоуровневая функция,
# уже у вас есть в data_fetcher.py
from .data_fetcher import _fetch_bybit_kline

def load_kline_incremental(
    symbol,
    category,
    interval,
    start,
    end,
    api_key="",
    api_secret="",
    csv_dir="data/raw"
):
    """
    Инкрементальная загрузка свечей (Bybit).
    Хранит общий CSV: {symbol}_{category}_{interval}.csv
    При запросе (start..end):
      - если CSV не существует, скачивает весь период
      - если CSV есть, догружает (слева/справа) недостающие даты
      - сохраняет результат
      - возвращает df_sub, т.е. нужный подотрезок

    Параметры:
      symbol   - e.g. "BTCUSDT"
      category - "spot", "linear", ...
      interval - "60", "D", etc
      start, end - строки "YYYY-MM-DD"
    """

    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{symbol}_{category}_{interval}.csv")

    # Прочитаем CSV, если есть
    if os.path.exists(csv_path):
        df_full = pd.read_csv(csv_path, parse_dates=["open_time"], index_col="open_time")
        df_full.sort_index(inplace=True)
        csv_start = df_full.index.min()
        csv_end   = df_full.index.max()
    else:
        df_full = pd.DataFrame()
        csv_start = None
        csv_end   = None

    # Преобразуем start,end в datetime
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)

    # Нужно ли догружать слева? (если просим start < csv_start)
    if (csv_start is None) or (start_dt < csv_start):
        fetch_start = start
        # берём на 1 день меньше csv_start, чтобы не пропустить граничную свечу
        if csv_start is not None:
            fetch_end_dt = csv_start - pd.Timedelta(days=1)
            fetch_end = fetch_end_dt.strftime("%Y-%m-%d")
        else:
            fetch_end = end  # если CSV пуст, качаем всё
        if pd.to_datetime(fetch_start) <= pd.to_datetime(fetch_end):
            print(f"Догружаем слева: {fetch_start}..{fetch_end}")
            df_new_left = _fetch_bybit_kline(
                symbol, category, interval,
                start=fetch_start, end=fetch_end,
                api_key=api_key, api_secret=api_secret
            )
            if not df_new_left.empty:
                df_full = pd.concat([df_new_left, df_full]).drop_duplicates()
                df_full.sort_index(inplace=True)

    # Нужно ли догружать справа? (если просим end > csv_end)
    if (csv_end is None) or (end_dt > csv_end):
        if csv_end is not None:
            fetch_start_dt = csv_end + pd.Timedelta(days=1)
            fetch_start = fetch_start_dt.strftime("%Y-%m-%d")
        else:
            fetch_start = start
        fetch_end = end
        if pd.to_datetime(fetch_start) <= pd.to_datetime(fetch_end):
            print(f"Догружаем справа: {fetch_start}..{fetch_end}")
            df_new_right = _fetch_bybit_kline(
                symbol, category, interval,
                start=fetch_start, end=fetch_end,
                api_key=api_key, api_secret=api_secret
            )
            if not df_new_right.empty:
                df_full = pd.concat([df_full, df_new_right]).drop_duplicates()
                df_full.sort_index(inplace=True)

    # Сохраняем обновлённый df_full, если не пуст
    if not df_full.empty:
        df_full.to_csv(csv_path)
    else:
        print("Похоже, что ничего не скачали, df_full.empty")

    # Возвращаем срез (start..end)
    df_sub = df_full.loc[start_dt:end_dt].copy()
    return df_sub
