# core/ws_clients/htx_ws.py
import json
import zlib
import time
import threading
import websocket
from core.config import HTX_WS_ENDPOINT

class HTXWSClient:
    """
    Подключение к Huobi(HTX) Spot:
    wss://api.huobi.pro/ws
    sub => "market.btcusdt.trade.detail"
    Gzip, ping/pong
    """

    def __init__(self, symbol="btcusdt", aggregator=None):
        self.symbol = symbol
        self.aggregator = aggregator
        self.ws = None
        self._stop = False

    def connect(self):
        self._stop = False
        self.ws = websocket.WebSocketApp(
            HTX_WS_ENDPOINT,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        thr = threading.Thread(target=self.ws.run_forever, daemon=True)
        thr.start()

    def _on_open(self, ws):
        print(f"[HTXWSClient] Opened => {self.symbol}")
        sub_req = {
            "sub": f"market.{self.symbol}.trade.detail",
            "id": f"sub_{self.symbol}"
        }
        ws.send(json.dumps(sub_req))
        print(f"[HTXWSClient] Sent sub => {sub_req}")

    def _on_message(self, ws, raw_msg):
        try:
            if isinstance(raw_msg, bytes):
                msg = zlib.decompress(raw_msg, 16+zlib.MAX_WBITS)
                data = json.loads(msg.decode("utf-8"))
            else:
                data = json.loads(raw_msg)
        except Exception as e:
            print(f"[HTXWSClient {self.symbol}] parse error: {e}")
            return

        if "ping" in data:
            pong = {"pong": data["ping"]}
            ws.send(json.dumps(pong))
            return

        if "subbed" in data:
            print(f"[HTXWSClient {self.symbol}] Subscribed => {data}")
            return

        ch = data.get("ch","")
        if ".trade.detail" in ch:
            tick = data.get("tick", {})
            arr  = tick.get("data", [])
            for d in arr:
                if self.aggregator:
                    closed = self.aggregator.add_trade({
                        "price": d["price"],
                        "timestamp": d["ts"],
                        "qty": d["amount"]
                    })
                    if closed:
                        print(f"[HTXWSClient {self.symbol}] Candle => {closed}")

    def _on_error(self, ws, error):
        print(f"[HTXWSClient {self.symbol}] Error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[HTXWSClient {self.symbol}] Closed => code={code}, msg={msg}")

    def stop(self):
        self._stop = True
        if self.ws:
            self.ws.close()
