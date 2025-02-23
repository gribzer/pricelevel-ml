# realtime_data/ws_client.py
import time
from pybit.unified_trading import WebSocket

class BybitWSListener:
    """
    Подключение к Bybit v5 WebSocket (публичные сделки) на "linear" канале.
    Топик: "publicTrade".
    """

    def __init__(
        self,
        symbol="BTCUSDT",
        channel_type="linear",  # "inverse" / "spot" и т.д.
        use_testnet=False,
        api_key=None,
        api_secret=None
    ):
        self.symbol = symbol
        self.channel_type = channel_type
        self.use_testnet = use_testnet
        self.api_key = api_key
        self.api_secret = api_secret
        self.on_message = None  # задаётся извне

    def run(self):
        """
        Создаём WebSocket: первый арг. => "linear".
        Затем ws.subscribe([...]) списком.
        Циклом удерживаем поток.
        """
        ws = WebSocket(
            self.channel_type,
            testnet=self.use_testnet,
            api_key=self.api_key,
            api_secret=self.api_secret
        )

        def internal_callback(msg):
            if self.on_message:
                self.on_message(msg)

        # ВНИМАНИЕ: передаём список [{"topic":"publicTrade", "symbol":"BTCUSDT"}]
        ws.subscribe(
            [
                {
                    "topic": "publicTrade",
                    "symbol": self.symbol
                }
            ],
            callback=internal_callback
        )

        # Запускаем вечный цикл, чтобы поток не завершался
        while True:
            time.sleep(1)
