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
    Преобразовать строку интервала (\"60\", \"D\", \"W\" и т.п.)
    в число миллисекунд. Учитываем только самые популярные варианты.
    
    Если хотите поддерживать все форматы (\"5\", \"15\", \"30\", \"120\" и т.д.),
    добавляйте нужные ветки. Ниже — базовые примеры.
    """
    interval_str = interval_str.upper()
    
    # Если это чисто цифровое, считаем что это минуты.
    if interval_str.isdigit():
        # конвертируем в количество минут, умножаем на 60_000
        minutes = int(interval_str)
        return minutes * 60_000
    
    # Возможные форматы: D (день), W (неделя), M (месяц?)
    if interval_str == "D":
        return 24 * 60 * 60_000  # сутки
    if interval_str == "W":
        return 7 * 24 * 60 * 60_000  # неделя
    if interval_str == "M":
        # По документации Bybit \"M\" = месяц, 
        # но точное кол-во дней в месяце может различаться.
        # Для упрощения берём 30 дней:
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
    Загрузка исторических свечей (kline) через Bybit Unified v5 API (pybit >= 2.0).
    Возвращает DataFrame со столбцами:
      [open_time, open, high, low, close, volume]  (индекс = open_time, UTC).
    
    Аргументы:
      symbol    : тикер (BTCUSDT, ETHUSDT и т.д.)
      category  : 'linear', 'inverse' или 'spot'
      interval  : '1', '3', '5', '15', '30', '60', '120', '240', '720', 'D', 'W', 'M'
      start     : начальная дата (\"YYYY-MM-DD\")
      end       : конечная дата
      api_key   : ключ Bybit (если требуется для приватных запросов; market data публична)
      api_secret: секрет Bybit
    """

    # Переводим start/end в миллисекунды
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)

    # Сколько миллисекунд занимает 1 свеча? Нужно для спота, чтобы определить следующий шаг.
    interval_ms = interpret_interval_ms(str(interval))

    # Создаём HTTP-сессию
    session = HTTP(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,     # True, если нужно testnet
        logging_level="WARNING"
    )

    all_records = []
    limit = 200
    current_start = start_ms

    while True:
        try:
            # GET /v5/market/kline
            resp = session.get_kline(
                category=category,       # 'linear', 'inverse', 'spot'
                symbol=symbol,
                interval=str(interval),
                start=current_start,
                end=end_ms,
                limit=limit
            )
        except Exception as e:
            print("Ошибка при запросе Bybit API (get_kline):", e)
            break

        # Проверяем retCode
        if resp.get("retCode", -1) != 0:
            print("API вернул ошибку:", resp)
            break

        # Извлекаем список свечей
        result = resp.get("result", {})
        items = result.get("list", [])
        if not items:
            # Пусто => достигли конца
            break

        # Парсим каждую свечу в общий формат
        for record in items:
            # Для фьючерсов (linear/inverse): record = [start_ms, end_ms, open, high, low, close, volume, turnover,...]
            # Для spot: record = [openTime_ms, open, high, low, close, volume, turnover,...]
            if category in ["linear", "inverse"]:
                # пример: [1681370460000, 1681370519999, \"28285.00\", \"28288.00\", \"28284.00\", \"28284.50\", \"7.23\", \"204627.73\", ...]
                start_candle_ms = int(record[0])
                # end_candle_ms = int(record[1])  # не обязательно сохранять
                open_p   = record[2]
                high_p   = record[3]
                low_p    = record[4]
                close_p  = record[5]
                volume_p = record[6]
            elif category == "spot":
                # пример: [1681370460000, \"28285.00\", \"28288.00\", \"28284.00\", \"28284.50\", \"7.23\", \"204627.73\"]
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

        # Определяем, куда сдвигать current_start:
        #   - для фьючерсов (linear/inverse) используем end_ms последней свечи.
        #   - для спота берем openTime последней свечи + interval_ms (поскольку нет endTime).
        if category in ["linear", "inverse"]:
            last_candle_end_ms = int(items[-1][1])  # end_ms
            if last_candle_end_ms >= end_ms:
                break
            current_start = last_candle_end_ms + 1
        else:
            # spot
            last_candle_start_ms = int(items[-1][0])  # openTime_ms
            next_start = last_candle_start_ms + interval_ms
            if next_start >= end_ms:
                break
            current_start = next_start + 1

        # Пауза, чтобы не превышать rate-limit
        time.sleep(0.25)

    if not all_records:
        print("Не удалось получить данные или нет записей.")
        return pd.DataFrame()

    # Превращаем в DataFrame
    df = pd.DataFrame(all_records)
    
    # open_time => datetime (UTC)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    # Приводим к float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
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
    Кэшируем результат в CSV, чтобы при повторном вызове не перекачивать то же самое.
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    csv_fname = f"Bybit_{category}_{symbol}_{interval}_{start}_{end}.csv"
    csv_path = os.path.join(RAW_DATA_PATH, csv_fname)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["open_time"], index_col="open_time")
        return df

    # Иначе запрашиваем у Bybit
    df = fetch_bybit_data(symbol, category, interval, start, end)
    if not df.empty:
        df.to_csv(csv_path)
    return df
