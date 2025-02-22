# src/data_fetcher.py

import os
import time
import pandas as pd
from datetime import datetime
# Импорт для pybit >= 2.0 (Unified Trading API)
from pybit.unified_trading import HTTP

from .config import (
    BYBIT_API_KEY, BYBIT_API_SECRET,
    BYBIT_SYMBOL, BYBIT_CATEGORY,
    BYBIT_INTERVAL,
    BYBIT_START_DATE, BYBIT_END_DATE,
    RAW_DATA_PATH
)

def interpret_interval_ms(interval_str: str) -> int:
    """
    Преобразовать строку интервала ("60", "D", "W" и т.п.)
    в число миллисекунд. (Упрощённая версия)
    """
    interval_str = interval_str.upper()

    if interval_str.isdigit():
        minutes = int(interval_str)
        return minutes * 60_000
    if interval_str == "D":
        return 24 * 60 * 60_000
    if interval_str == "W":
        return 7 * 24 * 60 * 60_000
    if interval_str == "M":
        return 30 * 24 * 60 * 60_000

    raise ValueError(f"Неизвестный формат интервала: {interval_str}")


def fetch_bybit_data(
    symbol=BYBIT_SYMBOL,
    category=BYBIT_CATEGORY,
    interval=BYBIT_INTERVAL,
    start=BYBIT_START_DATE,
    end=BYBIT_END_DATE,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET
):
    """
    Загрузка исторических свечей (kline) через Bybit Unified v5 API.
    Возвращает DataFrame: [open, high, low, close, volume] c индексом open_time.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)

    interval_ms = interpret_interval_ms(str(interval))

    session = HTTP(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        logging_level="WARNING"
    )

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
            print("Ошибка при запросе Bybit API (get_kline):", e)
            break

        if resp.get("retCode", -1) != 0:
            print("API вернул ошибку:", resp)
            break

        result = resp.get("result", {})
        items = result.get("list", [])
        if not items:
            break

        for record in items:
            # linear/inverse => [start_ms, end_ms, open, high, low, close, volume, ...]
            # spot           => [openTime_ms, open, high, low, close, volume, ...]
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

        # Сдвиг current_start
        if category in ["linear", "inverse"]:
            last_candle_end_ms = int(items[-1][1])
            if last_candle_end_ms >= end_ms:
                break
            current_start = last_candle_end_ms + 1
        else:
            last_candle_start_ms = int(items[-1][0])
            next_start = last_candle_start_ms + interval_ms
            if next_start >= end_ms:
                break
            current_start = next_start + 1

        time.sleep(0.25)

    if not all_records:
        print("Нет записей.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


def load_bybit_data(
    symbol=BYBIT_SYMBOL,
    category=BYBIT_CATEGORY,
    interval=BYBIT_INTERVAL,
    start=BYBIT_START_DATE,
    end=BYBIT_END_DATE
):
    """
    Высокоуровневая функция для получения DataFrame со свечами Bybit (v5).
    Кэшируем результат в CSV (в папке RAW_DATA_PATH).
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    csv_fname = f"Bybit_{category}_{symbol}_{interval}_{start}_{end}.csv"
    csv_path = os.path.join(RAW_DATA_PATH, csv_fname)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["open_time"], index_col="open_time")
        return df

    df = fetch_bybit_data(symbol, category, interval, start, end)
    if not df.empty:
        df.to_csv(csv_path)
    return df


def load_multi_timeframe(
    symbol=BYBIT_SYMBOL,
    category=BYBIT_CATEGORY,
    daily_interval="D",
    h4_interval="240",
    h1_interval="60",
    daily_limit=None,
    h4_limit=None,
    h1_limit=None,
    start=BYBIT_START_DATE,
    end=BYBIT_END_DATE
):
    """
    Загрузка данных на нескольких таймфреймах (день, 4ч, 1ч).
    Если *limit передан, считаем start = end - limit (в днях).
    """
    def compute_start_date(base_end, days):
        end_dt = datetime.strptime(base_end, "%Y-%m-%d")
        start_dt = end_dt - pd.Timedelta(days=days)
        return start_dt.strftime("%Y-%m-%d")

    # Daily
    if daily_limit is not None:
        start_daily = compute_start_date(end, daily_limit)
    else:
        start_daily = start

    df_daily = load_bybit_data(
        symbol=symbol,
        category=category,
        interval=daily_interval,
        start=start_daily,
        end=end
    )

    # 4h
    if h4_limit is not None:
        start_4h = compute_start_date(end, h4_limit)
    else:
        start_4h = start

    df_4h = load_bybit_data(
        symbol=symbol,
        category=category,
        interval=h4_interval,
        start=start_4h,
        end=end
    )

    # 1h
    if h1_limit is not None:
        start_1h = compute_start_date(end, h1_limit)
    else:
        start_1h = start

    df_1h = load_bybit_data(
        symbol=symbol,
        category=category,
        interval=h1_interval,
        start=start_1h,
        end=end
    )

    return df_daily, df_4h, df_1h
