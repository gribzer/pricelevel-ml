# core/ws_clients/bingx_ws.py

import json
import zlib
import time
import threading
import websocket
from core.config import BINGX_WS_ENDPOINT

class BingXWSClient:
    """
    Подключение к BingX Swap:
    wss://open-api-swap.bingx.com/swap-market
    channel => trade#BTC-USDT
    Gzip binary => zlib.decompress
    """

    def __init__(self, symbol="BTC-USDT", aggregator=None):
        self.symbol = symbol
        self.aggregator = aggregator
        self.ws = None
        self._stop = False

    def connect(self):
        self._stop = False
        self.ws = websocket.WebSocketApp(
            BINGX_WS_ENDPOINT,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        t = threading.Thread(target=self.ws.run_forever, daemon=True)
        t.start()

    def _on_open(self, ws):
        print(f"[BingXWSClient] Opened => {self.symbol}")
        sub_req = {
            "action": "sub",
            "params": {
                "channel": f"trade#{self.symbol}"
            }
        }
        ws.send(json.dumps(sub_req))
        print(f"[BingXWSClient] Sent sub => trade#{self.symbol}")

    def _on_message(self, ws, raw_msg):
        try:
            if isinstance(raw_msg, bytes):
                msg = zlib.decompress(raw_msg, 16+zlib.MAX_WBITS)
                data = json.loads(msg.decode("utf-8"))
            else:
                data = json.loads(raw_msg)
        except Exception as e:
            print(f"[BingXWSClient {self.symbol}] parse error: {e}")
            return

        ch = data.get("ch")
        if ch and ch.startswith("trade#"):
            tick = data.get("tick", {})
            arr  = tick.get("data", [])
            for d in arr:
                if self.aggregator:
                    closed = self.aggregator.add_trade({
                        "price": d["price"],
                        "timestamp": d["ts"],
                        "qty": d["qty"]
                    })
                    if closed:
                        print(f"[BingXWSClient {self.symbol}] Closed => {closed}")

    def _on_error(self, ws, error):
        print(f"[BingXWSClient {self.symbol}] Error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[BingXWSClient {self.symbol}] Closed => code={code}, msg={msg}")

    def stop(self):
        self._stop = True
        if self.ws:
            self.ws.close()
