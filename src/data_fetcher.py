# src/data_fetcher.py

import os
import time
import pandas as pd
from datetime import datetime
from pybit import HTTP

from .config import (
    BYBIT_API_KEY, BYBIT_API_SECRET,
    BYBIT_SYMBOL, BYBIT_INTERVAL,
    BYBIT_START_DATE, BYBIT_END_DATE,
    RAW_DATA_PATH
)

def fetch_bybit_data(
    symbol=BYBIT_SYMBOL,
    interval=BYBIT_INTERVAL,
    start=BYBIT_START_DATE,
    end=BYBIT_END_DATE
):
    """
    Загружает исторические свечи (OHLCV) с Bybit для указанного symbol, 
    таймфрейма (interval, в минутах) и периода [start, end].
    Возвращает DataFrame с индексом по времени (UTC).
    
    ПРИМЕЧАНИЕ: Bybit отдает максимум 200 свечей за один запрос, поэтому
    придётся повторно вызывать API, пока не соберём весь нужный диапазон.
    """
    # Создаём HTTP-сессию pybit (для публичных данных ключи не требуются, 
    # но при желании можно использовать BYBIT_API_KEY/SECRET)
    session = HTTP("https://api.bybit.com", request_timeout=10)

    # Конвертируем start/end в timestamp (Bybit использует секунды)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    # За один запрос Bybit возвращает до 200 свечей => необходимо делать цикл
    all_records = []
    limit = 200
    current_from = start_ts

    while True:
        try:
            # Вызываем публичный метод /v2/public/kline/list 
            # (старая версия API Bybit; для v5 может потребоваться другая функция)
            resp = session.query_kline(
                symbol=symbol,
                interval=str(interval),
                limit=limit,
                _from=current_from
            )
        except Exception as e:
            print("Ошибка при запросе Bybit API:", e)
            break
        
        if resp.get("ret_code", 0) != 0:
            # API вернул ошибку (например, неверный символ), выходим
            print("API вернул ошибку:", resp)
            break

        result = resp["result"]
        if not result:
            # Пустой результат => данные закончились
            break

        # Сохраняем записи
        for record in result:
            all_records.append(record)

        # Получаем время последней свечи (в UNIX секундах)
        last_candle_time = result[-1]["open_time"]
        # Смещаем current_from для следующего запроса
        current_from = last_candle_time + interval * 60

        if current_from >= end_ts:
            # Достигли или превысили конечную дату
            break

        time.sleep(0.25)  # чтобы не превышать rate limit

    if not all_records:
        print("Не удалось получить данные с Bybit.")
        return pd.DataFrame()

    # Преобразуем список словарей в DataFrame
    df = pd.DataFrame(all_records)

    # Переименуем колонки, если они отличаются
    rename_map = {
        "open_time": "open_time",
        "open":      "open",
        "high":      "high",
        "low":       "low",
        "close":     "close",
        "volume":    "volume"
    }
    df.rename(columns=rename_map, inplace=True, errors="ignore")

    # Преобразуем open_time в datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    # Приведём ключевые поля к float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    return df


def load_bybit_data(
    symbol=BYBIT_SYMBOL,
    interval=BYBIT_INTERVAL,
    start=BYBIT_START_DATE,
    end=BYBIT_END_DATE
):
    """
    Высокоуровневая функция, возвращающая DataFrame с историческими свечами Bybit.
    Кэшируем в CSV, чтобы не скачивать один и тот же диапазон повторно.
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    csv_fname = f"Bybit_{symbol}_{interval}_{start}_{end}.csv"
    csv_path = os.path.join(RAW_DATA_PATH, csv_fname)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["open_time"], index_col="open_time")
        return df

    # Иначе качаем заново
    df = fetch_bybit_data(symbol, interval, start, end)
    if not df.empty:
        df.to_csv(csv_path)
    return df
