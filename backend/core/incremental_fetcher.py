# core/incremental_fetcher.py

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP

from core.config import BYBIT_API_KEY, BYBIT_API_SECRET, RAW_DATA_PATH

def load_kline_incremental(symbol, category, interval,
                           start_str, end_str,
                           csv_dir=RAW_DATA_PATH):
    """
    Инкрементально грузит свечи в CSV (daily,4h,1h или любой другой interval).
    Если CSV уже есть, догружает недостающее справа.
    Возвращает df (start_str..end_str).
    """
    os.makedirs(csv_dir, exist_ok=True)
    csv_name = f"{symbol}_{category}_{interval}.csv"
    csv_path = os.path.join(csv_dir, csv_name)

    start_dt = pd.to_datetime(start_str)
    end_dt   = pd.to_datetime(end_str)

    # 1) Если CSV уже есть, читаем
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path, parse_dates=["open_time"], index_col="open_time")
        df_csv.sort_index(inplace=True)
        if not df_csv.empty:
            csv_start = df_csv.index[0]
            csv_end   = df_csv.index[-1]
            print(f"[{symbol} {interval}] CSV exists, range=({csv_start}..{csv_end}), rows={len(df_csv)}")
        else:
            print(f"[{symbol} {interval}] CSV is empty => init.")
            csv_start, csv_end = None, None
    else:
        df_csv = pd.DataFrame()
        print(f"[{symbol} {interval}] CSV not found => create new empty.")
        csv_start, csv_end = None, None

    # 2) Если пусто => качаем всё
    if df_csv.empty:
        print(f"[{symbol} {interval}] Fetch all: {start_dt}..{end_dt}")
        new_data = _fetch_bybit_kline(symbol, category, interval,
                                      start_dt, end_dt,
                                      api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET,
                                      limit=1000, sleep_sec=0.2, max_loops=1000)
        df_merged = new_data
    else:
        # Догрузка справа
        if csv_end < end_dt:
            fetch_start = csv_end + timedelta(seconds=1)
            if fetch_start < end_dt:
                print(f"[{symbol} {interval}] Need to fetch: {fetch_start}..{end_dt}")
                new_data = _fetch_bybit_kline(symbol, category, interval,
                                              fetch_start, end_dt,
                                              api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET,
                                              limit=1000, sleep_sec=0.2, max_loops=1000)
                if not new_data.empty:
                    df_merged = pd.concat([df_csv, new_data])
                    df_merged = df_merged[~df_merged.index.duplicated()]
                    df_merged.sort_index(inplace=True)
                else:
                    df_merged = df_csv
            else:
                df_merged = df_csv
        else:
            df_merged = df_csv

    if not df_merged.empty:
        df_merged.to_csv(csv_path)
    else:
        print(f"[{symbol} {interval}] No data for {start_str}..{end_str}")

    # 3) Возвращаем нужный подотрезок
    df_return = df_merged.loc[(df_merged.index >= start_dt) & (df_merged.index <= end_dt)].copy()
    return df_return


def _fetch_bybit_kline(symbol, category, interval,
                       start_dt, end_dt,
                       api_key=BYBIT_API_KEY,
                       api_secret=BYBIT_API_SECRET,
                       limit=1000,
                       sleep_sec=0.2,
                       max_loops=1000):
    """
    Универсальная функция для Bybit v5 (spot/linear/inverse),
    скачивает свечи за период start_dt..end_dt,
    избегая дубликатов (по таймстемпу).

    Аргументы:
      - symbol: e.g. "BTCUSDT"
      - category: "linear" / "inverse" / "spot"
      - interval: 60 / 240 / "D" и т.д.
      - start_dt, end_dt: datetime-объекты (или pd.Timestamp)
      - api_key, api_secret: ключи для доступа к Bybit API
      - limit: макс. кол-во свечей за 1 запрос (1..1000)
      - sleep_sec: пауза между запросами
      - max_loops: предельное кол-во итераций

    Возвращает pd.DataFrame c колонками [open, high, low, close, volume],
    индекс = DatetimeIndex (UTC, без tz)
    """

    # Преобразуем datetime -> ms
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)

    session = HTTP(api_key=api_key, api_secret=api_secret,
                   testnet=False)  # logging_level="WARNING" при желании

    all_records = []
    current_start = start_ms
    loop_count = 0
    last_ts = -1  # для проверки дубликатов

    print(f"[_fetch_bybit_kline] {symbol}, {interval}, {start_dt.date()}..{end_dt.date()}")
    while True:
        if loop_count >= max_loops:
            print(f"[{symbol} {interval}] Превышен лимит итераций={max_loops}, выходим.")
            break
        loop_count += 1

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
            # Пустой список => больше нечего загружать
            break

        batch = []
        max_ts_in_batch = -1

        for record in items:
            # У Bybit v5 формат немного разнится для spot vs linear
            if category in ["linear", "inverse"]:
                # record = [start_ms_, end_ms_, open, high, low, close, volume, turnover]
                start_ms_  = int(record[0])
                end_ms_    = int(record[1])
                open_p     = record[2]
                high_p     = record[3]
                low_p      = record[4]
                close_p    = record[5]
                volume_p   = record[6]
                candle_ts  = end_ms_
            elif category == "spot":
                # record = [openTime, open, high, low, close, volume, ...
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

            # Проверка дубликатов (last_ts)
            if candle_ts <= last_ts:
                # уже видели, пропускаем
                continue

            batch.append({
                "open_time": start_ms_,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume_p
            })
            if candle_ts > max_ts_in_batch:
                max_ts_in_batch = candle_ts

        if not batch:
            print("     Все полученные записи повторные => прерываем.")
            break

        all_records.extend(batch)
        last_ts = max_ts_in_batch

        # Если дошли до конца диапазона
        if max_ts_in_batch >= end_ms:
            print("     Достигнут end_ms => выходим.")
            break

        # Смещаем current_start
        current_start = max_ts_in_batch + 1
        time.sleep(sleep_sec)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)

    # Преобразуем колонки к float
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    print(f"[{symbol} {interval}] Итог: {len(df)} строк. (start={df.index[0] if not df.empty else None}, end={df.index[-1] if not df.empty else None})")
    return df
