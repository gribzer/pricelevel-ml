# run.py
import os
import threading
from dotenv import load_dotenv

load_dotenv()  # Подгружает .env (BYBIT_API_KEY, ...)

from realtime_data.aggregator import CandleAggregator
from realtime_data.ws_client import BybitWSListener
from core.pipeline.realtime_model import RealtimeModelTrainer
from webapp.app import app

from core.config import (
    BYBIT_API_KEY, BYBIT_API_SECRET,
    TOP_SYMBOLS, CHANNEL_TYPE
)

global_aggregators = {}
global_realtime_model = None

def start_ws(symbol="BTCUSDT"):
    agg = CandleAggregator(timeframe_s=3600)
    global_aggregators[symbol] = agg

    api_key = BYBIT_API_KEY or os.getenv("BYBIT_API_KEY", "")
    api_secret = BYBIT_API_SECRET or os.getenv("BYBIT_API_SECRET", "")
    is_testnet = (os.getenv("BYBIT_TESTNET", "false").lower() == "true")

    # channel_type="linear" => слушаем publicTrade для USDT-контрактов
    listener = BybitWSListener(
        symbol=symbol,
        channel_type=CHANNEL_TYPE,
        use_testnet=is_testnet,
        api_key=api_key,
        api_secret=api_secret
    )

    def on_msg(msg):
        # msg => {"topic":"publicTrade","data":[ {price, time, qty}, ... ]}
        data_list = msg.get("data", [])
        for trade in data_list:
            aggregator_input = {
                "price": trade["price"],
                "timestamp": trade["time"],  # ms
                "qty": trade["qty"]
            }
            closed_candle = agg.add_trade(aggregator_input)
            if closed_candle and global_realtime_model:
                global_realtime_model.train_on_new_candle(symbol, closed_candle)

    listener.on_message = on_msg
    listener.run()

if __name__ == "__main__":
    global_realtime_model = RealtimeModelTrainer()
    global_realtime_model.load_or_init_model()

    # Поднимаем потоки WS для каждого символа
    for sym in TOP_SYMBOLS:
        t = threading.Thread(target=start_ws, args=(sym,), daemon=True)
        t.start()

    # Запуск Dash
    app.run_server(debug=True)
