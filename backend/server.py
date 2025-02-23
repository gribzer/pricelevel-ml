# backend/server.py
import os
import time
import json
import threading

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

from core.ws_clients.aggregator import CandleAggregator
from core.ws_clients.bybit_ws import BybitWSClient
from core.ws_clients.bingx_ws import BingXWSClient
from core.ws_clients.htx_ws   import HTXWSClient
from core.config import TIMEFRAME_SECONDS

app = Flask(__name__, static_folder='../webapp/tradingview')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Глобальные
global_aggregators = {}

@app.route('/')
def index():
    # отдаём tradingview/index.html
    return send_from_directory('../webapp/tradingview', 'index.html')

@socketio.on('connect', namespace='/live')
def on_connect():
    print("[server] A client connected on /live")

def broadcast_candle(source, candle):
    data = {"source": source, "candle": candle}
    socketio.emit('new_candle', json.dumps(data), namespace='/live')

# Пример: при закрытии свечи aggregator 
#   => broadcast_candle(...)
# Можно сделать это внутри самих ws-классов
#   или подписаться via callback.

def start_bybit():
    sym = "BTCUSDT"
    agg = CandleAggregator(TIMEFRAME_SECONDS)
    global_aggregators[sym] = agg

    def aggregator_hook(trade):
        closed = agg.add_trade(trade)
        if closed:
            broadcast_candle("BYBIT_"+sym, closed)

    client = BybitWSClient(symbol=sym, aggregator=None)
    # Здесь два варианта:
    # - Либо в BybitWSClient внутри on_message уже agg.add_trade,
    # - Либо создаём aggregator_hook
    # Для примера используем aggregator внутри BybitWSClient => simpler
    client.aggregator = agg
    client.connect()

def start_bingx():
    sym = "BTC-USDT"
    agg = CandleAggregator(TIMEFRAME_SECONDS)
    global_aggregators[sym] = agg

    client = BingXWSClient(symbol=sym, aggregator=agg)
    client.connect()

def start_htx():
    sym = "btcusdt"
    agg = CandleAggregator(TIMEFRAME_SECONDS)
    global_aggregators[sym] = agg

    client = HTXWSClient(symbol=sym, aggregator=agg)
    client.connect()

if __name__=="__main__":
    # Запуск WS клиентов
    threading.Thread(target=start_bybit, daemon=True).start()
    threading.Thread(target=start_bingx, daemon=True).start()
    threading.Thread(target=start_htx,   daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=8000, debug=True)
