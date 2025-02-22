# src/data_fetcher.py

import os
import time
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP

from .config import BYBIT_API_KEY, BYBIT_API_SECRET

def _fetch_bybit_kline(
    symbol, category, interval,
    start, end,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET
):
    """
    Низкоуровневая функция (без кэширования).
    Скачивает свечи у Bybit (v5).
    Возвращает DataFrame (tz-naive index).
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)

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
            print(f"Ошибка Bybit API (get_kline) для {symbol} {interval}: {e}")
            break

        if resp.get("retCode", -1) != 0:
            print("API вернул ошибку:", resp)
            break

        result = resp.get("result", {})
        items = result.get("list", [])
        if not items:
            break

        for record in items:
            # linear/inverse или spot
            if category in ["linear","inverse"]:
                start_candle_ms = int(record[0])
                open_p   = record[2]
                high_p   = record[3]
                low_p    = record[4]
                close_p  = record[5]
                volume_p = record[6]
            elif category=="spot":
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
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume_p
            })

        if category in ["linear","inverse"]:
            last_candle_end_ms = int(items[-1][1])
            if last_candle_end_ms>=end_ms:
                break
            current_start = last_candle_end_ms+1
        else:
            last_candle_start_ms = int(items[-1][0])
            # interval_ms => ...
            # упрощённо:
            current_start = last_candle_start_ms+1

        time.sleep(0.25)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    # Снимаем таймзону
    df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)

    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    return df
