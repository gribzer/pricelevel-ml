# core/ws_clients/aggregator.py

class CandleAggregator:
    """
    Собирает приходящие сделки (trade) в 1m/1h/... свечу.
    Когда свеча закрывается => возвращает closed_candle (dict).
    """
    def __init__(self, timeframe_s=60):
        self.timeframe_s = timeframe_s
        self.current_candle = None
        self.candles = []

    def add_trade(self, trade: dict):
        """
        trade = { 'price':..., 'timestamp':..., 'qty':... }
        timestamp (ms), price(float), qty(float)
        Возвращает closed_candle или None
        """
        ts_sec = float(trade["timestamp"]) / 1000.0
        price  = float(trade["price"])
        qty    = float(trade.get("qty", 0))

        if self.current_candle is None:
            self._start_new_candle(ts_sec, price, qty)
            return None

        start_ts = self.current_candle["start"]
        if ts_sec >= start_ts + self.timeframe_s:
            # ЗАКРЫТЬ
            closed = self.current_candle
            closed["close"] = price
            self.candles.append(closed)
            # Новая
            self._start_new_candle(ts_sec, price, qty)
            return closed
        else:
            c = self.current_candle
            c["high"]   = max(c["high"], price)
            c["low"]    = min(c["low"],  price)
            c["close"]  = price
            c["volume"] += qty
            return None

    def _start_new_candle(self, ts_sec, price, qty):
        self.current_candle = {
            "start":  ts_sec,
            "open":   price,
            "high":   price,
            "low":    price,
            "close":  price,
            "volume": qty
        }

