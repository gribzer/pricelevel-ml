# server.py
import os
import time
import json
import threading
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from websocket import WebSocketApp, WebSocketConnectionClosedException

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "ANY_SECRET_HERE"
socketio = SocketIO(app, cors_allowed_origins="*")

##########################################################
# Настройка смещения в часах (пример: 3 часа)
##########################################################
UTC_OFFSET_HOURS = 3
LOCAL_OFFSET_SECONDS = UTC_OFFSET_HOURS * 3600

##########################################################
# WebSocket к Bybit
##########################################################
WS_URL_MAINNET = "wss://stream.bybit.com/v5/public"
WS_URL_TESTNET = "wss://stream-testnet.bybit.com/v5/public"

ws_app = None
ws_thread = None

def on_message(ws, message_str):
    # Обработка real-time
    pass

def on_open(ws):
    print("[Bybit WS] Opened connection")

def on_close(ws, code, msg):
    print("[Bybit WS] Closed", code, msg)

def on_error(ws, err):
    print("[Bybit WS] Error:", err)

def run_ws_forever():
    if ws_app:
        ws_app.run_forever()

def init_ws(url):
    global ws_app, ws_thread
    ws_app = WebSocketApp(
        url=url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws_thread = threading.Thread(target=run_ws_forever, daemon=True)
    ws_thread.start()
    time.sleep(1)

def subscribe_kline(symbol, interval):
    global ws_app
    if ws_app is None:
        print("[INFO] Trying mainnet:", WS_URL_MAINNET)
        init_ws(WS_URL_MAINNET)

    sub_msg = {
        "op": "subscribe",
        "args": [f"kline.{interval}.{symbol}"]
    }
    try:
        ws_app.send(json.dumps(sub_msg))
    except Exception as e:
        print("[ERROR] Subscribing on mainnet, fallback testnet...", e)
        init_ws(WS_URL_TESTNET)
        time.sleep(0.5)
        ws_app.send(json.dumps(sub_msg))

##########################################################
# Получаем исторические свечи (REST)
##########################################################
def fetch_historical_klines(symbol: str, interval: str, limit: int = 10):
    base_url = "https://api.bybit.com"
    endpoint = f"{base_url}/v5/market/kline"

    # Вычитаем 3 часа
    current_time = int((time.time() - LOCAL_OFFSET_SECONDS) * 1000)
    # 6 месяцев назад
    six_months_ms = 15552000 * 1000
    end_time = current_time
    start_time = end_time - six_months_ms

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start": start_time,
        "end": end_time
    }
    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0:
            print("[ERROR] retCode:", data)
            return []
        kline_list = data["result"].get("list", [])
        kline_list.sort(key=lambda x: int(x[0]))
        results = []
        for item in kline_list:
            start_ms = int(item[0])
            o = float(item[1])
            h = float(item[2])
            l = float(item[3])
            c = float(item[4])
            results.append({
                "time": start_ms // 1000,
                "open": o,
                "high": h,
                "low": l,
                "close": c
            })
        return results
    except Exception as e:
        print("[ERROR] fetch_historical_klines exception:", e)
        return []

##########################################################
# Flask endpoints
##########################################################
@app.route("/")
def index():
    return "Server is running. Use /api/history?symbol=BTCUSDT&interval=1&limit=10"

@app.route("/api/history", methods=["GET"])
def api_history():
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "1")
    limit = min(int(request.args.get("limit", 10)), 1000)

    candles = fetch_historical_klines(symbol, interval, limit)
    subscribe_kline(symbol, interval)
    return jsonify({"candles": candles})

if __name__ == "__main__":
    socketio.run(app, port=5000, debug=True)
