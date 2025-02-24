# core/data_fetcher/bybit.py

import time
from pybit import HTTP

def get_http_session(api_key: str, api_secret: str, use_testnet: bool = False):
    """
    Создаёт HTTP-сессию для Bybit (v5).
    В pybit 5.x нет публичных параметров base_url/test в конструкторе,
    поэтому, если нужно, переключаемся на testnet через session._set_api_url(...).
    """
    session = HTTP(
        api_key=api_key,
        api_secret=api_secret
        # НЕ передаём test=... и base_url=...
    )

    # Меняем базовый URL при необходимости
    if use_testnet:
        session._set_api_url("https://api-testnet.bybit.com")
    else:
        session._set_api_url("https://api.bybit.com")

    return session


def fetch_historical_klines(session_http, symbol: str, interval: str, limit: int = 200):
    """
    Запрашивает исторические свечи через v5 GET /market/kline.
    Возвращает список словарей для lightweight-charts:
      [{time, open, high, low, close}, ...]
    """
    end_time = int(time.time() * 1000)
    # Пример: берем 6 месяцев истории (~ 15552000 секунд)
    start_time = end_time - (15552000 * 1000)

    resp = session_http.get_kline(
        category="linear",  # USDT-пары
        symbol=symbol,
        interval=interval,  # "1","15","60","240","D"
        limit=limit,
        startTime=start_time,
        endTime=end_time
    )
    # Проверяем retCode
    if resp.get("retCode") != 0:
        print("Bybit Kline Error:", resp)
        return []

    kline_list = resp["result"]["list"]  # список списков: [startTime, open, high, low, close, volume, ...]
    kline_list.sort(key=lambda x: x[0])  # сортируем по startTime

    results = []
    for item in kline_list:
        ts_ms = item[0]
        o = float(item[1])
        h = float(item[2])
        l = float(item[3])
        c = float(item[4])
        results.append({
            "time": ts_ms // 1000,  # UNIX timestamp (sec)
            "open": o,
            "high": h,
            "low": l,
            "close": c
        })

    return results
