# src/incremental_fetcher.py

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP

def load_kline_incremental(symbol, category, interval,
                           start_str, end_str,
                           csv_dir="data/raw"):
    """
    Инкрементально грузит свечи в CSV (daily,4h,1h). 
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
            csv_start, csv_end=None,None
    else:
        df_csv = pd.DataFrame()
        print(f"[{symbol} {interval}] CSV not found => create new empty.")
        csv_start, csv_end = None, None

    # 2) Если пусто => качаем всё
    if df_csv.empty:
        print(f"[{symbol} {interval}] Fetch all: {start_dt}..{end_dt}")
        new_data = _fetch_bybit_kline_full(symbol, category, interval,
                                           start_dt, end_dt)
        df_merged = new_data
    else:
        # Догрузка справа
        if csv_end < end_dt:
            fetch_start = csv_end + timedelta(seconds=1)
            if fetch_start < end_dt:
                print(f"[{symbol} {interval}] Need to fetch: {fetch_start}..{end_dt}")
                new_data = _fetch_bybit_kline_full(symbol, category, interval,
                                                   fetch_start, end_dt)
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
    df_return = df_merged.loc[(df_merged.index>=start_dt)&(df_merged.index<=end_dt)].copy()
    return df_return


def _fetch_bybit_kline_full(symbol, category, interval,
                            start_dt, end_dt,
                            limit=1000):
    """
    Универсальная функция для Bybit v5 (spot/linear).
    Формат: 
       list -> [[openTime, open, high, low, close, volume], [...], ...]
    Возвращаем DataFrame c колонками [open,high,low,close,volume], 
    индекс = DatetimeIndex (из openTime).
    """

    from pybit.unified_trading import HTTP
    import pandas as pd
    import time

    session = HTTP()
    start_ms = int(start_dt.timestamp()*1000)
    end_ms   = int(end_dt.timestamp()*1000)

    all_records = []
    current_start= start_ms
    request_count=0
    prev_last_ts=None

    while True:
        request_count += 1
        print(f"  -> Request #{request_count}, current_start={current_start}, end_ms={end_ms}...")

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
            print(f"[{symbol} {interval}] Error while get_kline: {e}")
            break

        if resp.get("retCode", -1) != 0:
            print("API retCode!=0 => break:", resp)
            break

        result = resp.get("result", {})
        items  = result.get("list", [])
        if not items:
            print("  -> No items => break.")
            break

        print(f"     Получено {len(items)} записей.")

        # По ответам Bybit v5, при category="spot" ИЛИ "linear" 
        # формат kline = [ openTime, open, high, low, close, volume ]
        for rec in items:
            # rec[0] => openTime (ms)
            # rec[1] => open (str float)
            # rec[2] => high
            # rec[3] => low
            # rec[4] => close
            # rec[5] => volume
            ot = int(rec[0])           # время (ms)
            o  = float(rec[1])         # open
            h  = float(rec[2])         # high
            l  = float(rec[3])         # low
            c  = float(rec[4])         # close
            vol= float(rec[5])         # volume

            all_records.append({
                "open_time": ot,
                "open":  o,
                "high":  h,
                "low":   l,
                "close": c,
                "volume": vol
            })

        # last_ts = openTime последней записи
        last_ts= int(items[-1][0])
        if last_ts>= end_ms:
            print("  -> Достигнут или превышен end_ms => выходим.")
            break

        # защита от зацикливания
        if prev_last_ts is not None and last_ts<=prev_last_ts:
            print(f"  -> last_ts не растёт (={last_ts}), prev={prev_last_ts} => break.")
            break
        prev_last_ts= last_ts

        current_start= last_ts + 1
        time.sleep(0.2)

        if request_count>=200:
            print(f"  -> Request #{request_count}, слишком много => break.")
            break

    if not all_records:
        return pd.DataFrame()

    df= pd.DataFrame(all_records)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)

    # убираем дубли
    df= df[~df.index.duplicated()]

    return df
