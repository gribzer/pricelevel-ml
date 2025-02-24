# server.py
import os
import time
import json
import datetime
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# К примеру, Bybit mainnet
BYBIT_BASE_URL = "https://api.bybit.com"

# Настройки глубины для разных таймфреймов:
#  - D  => 6 months
#  - 4H => 3 months
#  - 1H => 1 month
#  - 15M => 1 week
TIMEFRAME_DEPTH = {
    "D": 6 * 30 * 24 * 3600,   # ~6 months (180 days)
    "4H": 3 * 30 * 24 * 3600,  # ~3 months
    "1H": 30 * 24 * 3600,      # ~1 month
    "15M": 7 * 24 * 3600,      # ~1 week
}

# Маппинг для Bybit
# Bybit intervals: "D" or "240" or "60" or "15"
TIMEFRAME_MAP = {
    "D": "D",     # Daily
    "4H": "240",  # 4 hours
    "1H": "60",   # 1 hour
    "15M": "15",  # 15 minutes
}


@app.route("/")
def index():
    return "Server is running. Use /api/history?symbol=BTCUSDT&timeframe=4H"


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Пример: /api/history?symbol=BTCUSDT&timeframe=4H
    Доступные таймфреймы: D, 4H, 1H, 15M
    """
    symbol = request.args.get("symbol", "BTCUSDT")
    timeframe = request.args.get("timeframe", "1H")
    # Вычислим глубину (сек) на основе таблицы
    depth_sec = TIMEFRAME_DEPTH.get(timeframe, 7 * 24 * 3600)  # если не найдёт, 1 week
    # Преобразуем во внутренний формат Bybit
    bybit_interval = TIMEFRAME_MAP.get(timeframe, "60")  # default 1H

    # Текущее время (ms)
    end_time = int(time.time() * 1000)
    start_time = end_time - (depth_sec * 1000)

    # Запрос v5/market/kline
    # Док: https://bybit-exchange.github.io/docs/v5/market/kline
    endpoint = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval,
        "start": start_time,
        "end": end_time,
        "limit": 1000  # макс 1000 за запрос
    }
    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0:
            print("Bybit Kline Error:", data)
            return jsonify({"candles": [], "error": data}), 400

        kline_list = data["result"]["list"] or []
        # пример структуры item: [startTime, open, high, low, close, volume, turnover]
        # Сортируем по startTime
        kline_list.sort(key=lambda x: x[0])
        results = []
        for item in kline_list:
            st_ms = int(item[0])
            o = float(item[1])
            h = float(item[2])
            l = float(item[3])
            c = float(item[4])
            results.append({
                "time": st_ms // 1000,  # UNIX timestamp (sec)
                "open": o,
                "high": h,
                "low": l,
                "close": c
            })
        return jsonify({"candles": results})
    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"candles": [], "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
