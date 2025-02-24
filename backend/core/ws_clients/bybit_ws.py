# core/ws_clients/bybit_ws.py

from pybit.unified_trading import WebSocket

class BybitWSManager:
    """
    Управляет подключением к WebSocket Bybit (v5 public).
    При создании можно указать testnet=True/False,
    и автоматически выберется нужный endpoint.
    """

    def __init__(self, api_key, api_secret, use_testnet=False, callback=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.callback = callback

        if use_testnet:
            self.endpoint = "wss://stream-testnet.bybit.com/v5/public"
        else:
            self.endpoint = "wss://stream.bybit.com/v5/public"

        # channel_type="linear" => USDT-пары
        self.ws = WebSocket(
            api_key=self.api_key,
            api_secret=self.api_secret,
            endpoint=self.endpoint,
            channel_type="linear",
            callback=self._on_message
        )
        # Подписки
        self.subscriptions = set()

    def _on_message(self, msg: dict):
        """
        Колбэк, вызываемый при каждом сообщении WS.
        """
        if not msg:
            return
        data_obj = msg.get("data", {})
        topic = data_obj.get("topic", "")

        if "kline" in topic:
            # data_obj["data"] = массив свечей
            kline_data = data_obj.get("data", [])
            for candle in kline_data:
                # Вызываем внешний callback
                if self.callback:
                    self.callback(topic, candle)

    def subscribe_kline(self, symbol: str, interval: str):
        """
        Подписка на kline.<interval>.<symbol>.
        pybit 5.x предоставляет ws.kline_stream(...).
        """
        sub_id = self.ws.kline_stream(symbol=symbol, interval=interval)
        self.subscriptions.add((symbol, interval))
        return sub_id
