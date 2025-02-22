# src/multi_tf_analysis.py

import pandas as pd
import numpy as np

# Ужесточённые настройки
DEFAULT_MIN_TOUCHES = 5
DEFAULT_ATR_BUFFER = 0.25
DEFAULT_VOLUME_FACTOR = 1.5
DEFAULT_MAX_AGE_DAYS = 40

def compute_atr(df_daily: pd.DataFrame, period=7) -> float:
    """
    Простой расчёт ATR по rolling mean TR (за period=7 дней).
    """
    if df_daily.empty:
        return np.nan
    highs = df_daily["high"].values
    lows = df_daily["low"].values
    closes = df_daily["close"].values

    trs = []
    for i in range(1, len(df_daily)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)
    atr_series = pd.Series(trs).rolling(period).mean()
    current_atr = atr_series.iloc[-1] if len(atr_series) >= period else np.nan
    return current_atr

def filter_levels(df_daily, df_4h, df_1h,
                  raw_levels,
                  min_touches=DEFAULT_MIN_TOUCHES,
                  atr_buffer=DEFAULT_ATR_BUFFER,
                  volume_factor=DEFAULT_VOLUME_FACTOR,
                  max_age_days=DEFAULT_MAX_AGE_DAYS):
    """
    Ужесточённая фильтрация:
      - Нужно >=min_touches касаний
      - Уровень не старше max_age_days
      - Проверяем ATR-buffer
      - (Опционально) проверяем объёмы на 4h
    """
    filtered = []
    if df_daily.empty or not raw_levels:
        return []

    now_ts = df_daily.index[-1]
    current_atr = compute_atr(df_daily, period=7)
    if pd.isna(current_atr):
        return []

    for lvl in raw_levels:
        # Считаем, сколько касаний (|close - lvl| < atr_buffer*ATR)
        touches = 0
        last_touch_ts = None
        for i in range(len(df_daily)):
            if abs(df_daily["close"].iloc[i] - lvl) < (atr_buffer * current_atr):
                touches += 1
                last_touch_ts = df_daily.index[i]
        if touches < min_touches:
            continue

        # возраст уровня
        if last_touch_ts is None:
            continue
        diff_days = (now_ts - last_touch_ts).days
        if diff_days > max_age_days:
            continue

        # Проверка объёмов на 4H
        # (Смотрим 3 дня до last_touch_ts)
        if df_4h is not None and not df_4h.empty:
            date_threshold = last_touch_ts - pd.Timedelta(days=3)
            df_4h_slice = df_4h[(df_4h.index >= date_threshold) & (df_4h.index <= last_touch_ts)]
            if len(df_4h_slice) > 2:
                avg_4h_vol = df_4h_slice["volume"].mean()
                global_4h_vol = df_4h["volume"].mean()
                # Ужесточить?
                # if avg_4h_vol < volume_factor * global_4h_vol:
                #     continue  # отсеять
                pass

        # (Опционально) можно ещё что-то проверять

        filtered.append(lvl)

    return sorted(filtered)
