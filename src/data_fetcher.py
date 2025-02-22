# src/data_fetcher.py

import os
import time
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP

from .config import (
    BYBIT_API_KEY, BYBIT_API_SECRET,
    BYBIT_CATEGORY,
    BYBIT_START_DATE, BYBIT_END_DATE,
    RAW_DATA_PATH,
    DAILY_INTERVAL, H4_INTERVAL, H1_INTERVAL
)

def interpret_interval_ms(interval_str: str) -> int:
    # ... (как прежде) ...
    interval_str = interval_str.upper()
    if interval_str.isdigit():
        return int(interval_str) * 60_000
    if interval_str == "D":
        return 24 * 60 * 60_000
    if interval_str == "W":
        return 7 * 24 * 60 * 60_000
    if interval_str == "M":
        return 30 * 24 * 60 * 60_000
    raise ValueError(f"Неизвестный формат интервала: {interval_str}")

def _fetch_bybit_kline(symbol, category, interval, start, end,
                       api_key, api_secret):
    # Функция низкого уровня: скачивает свечи
    # аналогична fetch_bybit_data, но можем назвать _fetch_bybit_kline
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)

    int_ms = interpret_interval_ms(interval)
    session = HTTP(api_key=api_key, api_secret=api_secret,
                   testnet=False, logging_level="WARNING")

    all_records = []
    limit = 200
    current_start = start_ms

    while True:
        try:
            resp = session.get_kline(
                category=category,
                symbol=symbol,
                interval=str(interval),
                start=current_start,
                end=end_ms,
                limit=limit
            )
        except Exception as e:
            print(f"Ошибка Bybit API kline для {symbol}, interval={interval}:", e)
            break

        if resp.get("retCode", -1) != 0:
            print("API вернул ошибку:", resp)
            break

        result = resp.get("result", {})
        items = result.get("list", [])
        if not items:
            break

        for record in items:
            if category in ["linear", "inverse"]:
                start_candle_ms = int(record[0])
                open_p   = record[2]
                high_p   = record[3]
                low_p    = record[4]
                close_p  = record[5]
                volume_p = record[6]
            elif category == "spot":
                start_candle_ms = int(record[0])
                open_p   = record[1]
                high_p   = record[2]
                low_p    = record[3]
                close_p  = record[4]
                volume_p = record[5]
            else:
                print(f"Неизвестная категория: {category}")
                return pd.DataFrame()

            all_records.append({
                "open_time": start_candle_ms,
                "open":      open_p,
                "high":      high_p,
                "low":       low_p,
                "close":     close_p,
                "volume":    volume_p
            })

        if category in ["linear", "inverse"]:
            last_candle_end_ms = int(items[-1][1])
            if last_candle_end_ms >= end_ms:
                break
            current_start = last_candle_end_ms + 1
        else:
            last_candle_start_ms = int(items[-1][0])
            next_start = last_candle_start_ms + int_ms
            if next_start >= end_ms:
                break
            current_start = next_start + 1

        time.sleep(0.25)

    if not all_records:
        print(f"Нет записей: {symbol} {interval}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df

def load_single_symbol_multi_timeframe(symbol, 
                                       category=BYBIT_CATEGORY,
                                       start=BYBIT_START_DATE,
                                       end=BYBIT_END_DATE,
                                       daily_interval=DAILY_INTERVAL,
                                       h4_interval=H4_INTERVAL,
                                       h1_interval=H1_INTERVAL):
    """
    Загружает 3 фрейма: (df_daily, df_4h, df_1h) для одного symbol'a.
    """
    # 1) Daily
    df_daily = _fetch_bybit_kline(symbol, category, daily_interval, start, end,
                                  BYBIT_API_KEY, BYBIT_API_SECRET)
    # 2) 4H
    df_4h = _fetch_bybit_kline(symbol, category, h4_interval, start, end,
                               BYBIT_API_KEY, BYBIT_API_SECRET)
    # 3) 1H
    df_1h = _fetch_bybit_kline(symbol, category, h1_interval, start, end,
                               BYBIT_API_KEY, BYBIT_API_SECRET)
    return df_daily, df_4h, df_1h
