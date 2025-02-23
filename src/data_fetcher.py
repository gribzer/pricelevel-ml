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
    Скачивает свечи у Bybit (v5), избегая дубликатов. 
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
    max_loops = 1000
    loop_count = 0
    last_ts = -1

    print(f"[_fetch_bybit_kline] {symbol}, {interval}, {start}..{end}")
    while True:
        if loop_count >= max_loops:
            print(f"[{symbol} {interval}] Превышен лимит итераций={max_loops}, выходим.")
            break
        loop_count += 1

        resp = None
        try:
            print(f"  -> Request #{loop_count}, current_start={current_start}, end_ms={end_ms}")
            resp = session.get_kline(
                category=category,
                symbol=symbol,
                interval=str(interval),
                start=current_start,
                end=end_ms,
                limit=limit
            )
        except Exception as e:
            print(f"Bybit API error: {e}")
            break

        ret_code = resp.get("retCode", -1)
        if ret_code != 0:
            print(f"[{symbol} {interval}] retCode={ret_code}, resp={resp}")
            break

        items = resp.get("result", {}).get("list", [])
        print(f"     Получено {len(items)} записей.")
        if not items:
            break

        batch = []
        max_ts_in_batch = -1

        for record in items:
            if category in ["linear", "inverse"]:
                start_ms_  = int(record[0])
                end_ms_    = int(record[1])
                open_p     = record[2]
                high_p     = record[3]
                low_p      = record[4]
                close_p    = record[5]
                volume_p   = record[6]
                candle_ts  = end_ms_
            elif category == "spot":
                start_ms_  = int(record[0])
                open_p     = record[1]
                high_p     = record[2]
                low_p      = record[3]
                close_p    = record[4]
                volume_p   = record[5]
                candle_ts  = start_ms_
            else:
                print(f"[{symbol}] Неизвестная категория: {category}")
                return pd.DataFrame()

            if candle_ts > max_ts_in_batch:
                max_ts_in_batch = candle_ts
            if candle_ts <= last_ts:
                # уже видели
                continue

            batch.append({
                "open_time": start_ms_,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume_p
            })

        if not batch:
            print("     Все полученные записи повторные => прерываем.")
            break

        all_records.extend(batch)
        last_ts = max_ts_in_batch

        if max_ts_in_batch >= end_ms:
            print("     Достигнут end_ms => выходим.")
            break
        current_start = max_ts_in_batch + 1
        time.sleep(0.05)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    print(f"[{symbol} {interval}] Итог: {len(df)} строк.")
    return df
