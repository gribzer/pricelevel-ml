# src/multi_tf_analysis.py

import pandas as pd
import numpy as np
from .config import (
    MIN_TOUCHES_FILTER, MAX_AGE_DAYS, ATR_BUFFER, VOLUME_FACTOR
)

def compute_atr(df_daily: pd.DataFrame, period=7) -> float:
    # ... как раньше ...
    highs = df_daily["high"].values
    lows = df_daily["low"].values
    closes = df_daily["close"].values

    trs = []
    for i in range(1, len(df_daily)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    atr_series = pd.Series(trs).rolling(period).mean()
    current_atr = atr_series.iloc[-1] if len(atr_series) >= period else np.nan
    return current_atr

def filter_levels(df_daily, df_4h, df_1h, raw_levels,
                  min_touches=MIN_TOUCHES_FILTER,
                  max_age_days=MAX_AGE_DAYS,
                  atr_buffer=ATR_BUFFER,
                  volume_factor=VOLUME_FACTOR):
    """
    Мультитаймфреймовая фильтрация.
    - df_daily: дневной фрейм
    - df_4h, df_1h: для проверки объёмов / скорости подхода
    """
    filtered = []
    now_ts = df_daily.index[-1]
    current_atr = compute_atr(df_daily, period=7)
    if pd.isna(current_atr):
        return []

    for lvl in raw_levels:
        touches = 0
        last_touch_ts = None
        for i in range(len(df_daily)):
            if abs(df_daily["close"].iloc[i] - lvl) < (atr_buffer * current_atr):
                touches += 1
                last_touch_ts = df_daily.index[i]
        if touches < min_touches:
            continue

        if last_touch_ts is None:
            continue
        diff_days = (now_ts - last_touch_ts).days
        if diff_days > max_age_days:
            continue

        # MTF-проверка. Пример: берём последние 3 дня до last_touch_ts в 4h
        date_threshold = last_touch_ts - pd.Timedelta(days=3)
        df_4h_slice = df_4h[(df_4h.index >= date_threshold) & (df_4h.index <= last_touch_ts)]
        if len(df_4h_slice) > 2:
            avg_4h_vol = df_4h_slice["volume"].mean()
            global_4h_vol = df_4h["volume"].mean()
            if avg_4h_vol < volume_factor * global_4h_vol:
                # объёмы не сильно повышены
                pass
        
        # Аналогично 1H (проверяем скорость подхода)
        df_1h_slice = df_1h[(df_1h.index >= date_threshold) & (df_1h.index <= last_touch_ts)]
        if len(df_1h_slice) > 2:
            price_start = df_1h_slice["close"].iloc[0]
            price_end   = df_1h_slice["close"].iloc[-1]
            # Пример: если price_end - price_start < 0.2 * current_atr => поджатие
            # ...

        filtered.append(lvl)
    return sorted(filtered)
