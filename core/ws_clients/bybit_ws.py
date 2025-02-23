# core/ws_clients/bybit_ws.py
import time
import threading
from pybit.unified_trading import WebSocket
from core.config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_CHANNEL_TYPE, IS_BYBIT_TESTNET

class BybitWSClient:
    """
    Подключение к Bybit v5 (publicTrade).
    Приходит "data":[ {price, time, qty}, ... ]
    При закрытии свечи => print(...) или callback.
    """

    def __init__(self, symbol="BTCUSDT", aggregator=None):
        self.symbol = symbol
        self.aggregator = aggregator
        self.api_key = BYBIT_API_KEY
        self.api_secret = BYBIT_API_SECRET
        self.channel_type = BYBIT_CHANNEL_TYPE  # "linear" / "inverse"
        self.use_testnet = IS_BYBIT_TESTNET
        self.ws = None

    def connect(self):
        self.ws = WebSocket(
            self.channel_type,
            testnet=self.use_testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
            ping_interval=30,
            ping_timeout=10
        )

        def internal_cb(msg):
            data_arr = msg.get("data", [])
            for trade in data_arr:
                if not self.aggregator:
                    continue
                closed = self.aggregator.add_trade({
                    "price": trade["price"],
                    "timestamp": trade["time"],
                    "qty": trade["qty"]
                })
                if closed:
                    print(f"[BybitWSClient {self.symbol}] Closed candle => {closed}")

        topic_str = "publicTrade.{symbol}"
        self.ws.subscribe(topic=topic_str, symbol=self.symbol, callback=internal_cb)
        print(f"[BybitWSClient] Subscribed => {self.symbol}")

        # Удерживаем поток
        thr = threading.Thread(target=self._hold, daemon=True)
        thr.start()

    def _hold(self):
        while True:
            time.sleep(1)

    def stop(self):
        if self.ws:
            self.ws.close()
