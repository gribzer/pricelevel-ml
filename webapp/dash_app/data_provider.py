# webapp/data_provider.py
from typing import List, Tuple, Dict

# Предположим, глобальные объекты
from realtime_data.aggregator import CandleAggregator
# from core.models.levels_model import SomeLevelsModel  # пример

global_aggregators = {
    "BTCUSDT": CandleAggregator(timeframe_s=60),
    "ETHUSDT": CandleAggregator(timeframe_s=60),
    # ...
}

global_levels = {
    "BTCUSDT": [],
    "ETHUSDT": [],
}

def get_candles_and_levels(symbol: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Возвращает tuple (candles, levels).
    candles -> [{open, high, low, close, start, volume}, ...]
    levels -> [{price, label}, ...]
    """
    aggregator = global_aggregators[symbol]
    candles = aggregator.candles[-200:]  # последние 200 свечей для графика

    return candles, global_levels[symbol]
