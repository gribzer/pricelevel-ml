# realtime_data/aggregator.py

class CandleAggregator:
    """
    Собирает сделки (trade) в свечи заданного таймфрейма (timeframe_s).
    Если свеча "закрылась" (ts >= start + timeframe_s),
    возвращает closed_candle (dict).
    """

    def __init__(self, timeframe_s=3600):
        self.timeframe_s = timeframe_s
        self.current_candle = None
        self.candles = []

    def add_trade(self, trade: dict):
        """
        trade = {
          "price": <str or float>,
          "timestamp": <int ms>,
          "qty": <str or float>
        }
        Возвращает закрытую свечу (dict) или None, 
        если текущая свеча ещё не закрылась.
        """
        ts_sec = trade["timestamp"] / 1000.0
        price  = float(trade["price"])
        qty    = float(trade.get("qty", 0))

        if self.current_candle is None:
            self._start_new_candle(ts_sec, price, qty)
            return None

        candle_start = self.current_candle["start"]
        if ts_sec >= candle_start + self.timeframe_s:
            # Закрываем предыдущую свечу
            closed = self.current_candle
            closed["close"] = price
            self.candles.append(closed)
            # Начинаем новую свечу
            self._start_new_candle(ts_sec, price, qty)
            return closed
        else:
            # Обновляем текущую свечу
            self.current_candle["high"]   = max(self.current_candle["high"], price)
            self.current_candle["low"]    = min(self.current_candle["low"], price)
            self.current_candle["close"]  = price
            self.current_candle["volume"] += qty
            return None

    def _start_new_candle(self, ts_sec, price, qty):
        self.current_candle = {
            "start": ts_sec,
            "open":  price,
            "high":  price,
            "low":   price,
            "close": price,
            "volume": qty
        }
