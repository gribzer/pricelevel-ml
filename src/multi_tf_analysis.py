# src/multi_tf_analysis.py

import pandas as pd
import numpy as np

def compute_atr(df_daily: pd.DataFrame, period=7) -> float:
    """
    Рассчитываем простой ATR за нужный период (дней).
    """
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

def find_primary_levels(df_daily: pd.DataFrame, window=5, touches_required=2):
    """
    Простейший поиск локальных экстремумов на df_daily. 
    Возвращает список уровней (float).
    """
    closes = df_daily["close"].values
    levels = []

    for i in range(window, len(closes) - window):
        local_window = closes[i-window : i+window+1]
        center = closes[i]
        # Проверим локальный максимум (пример).
        if center == max(local_window):
            levels.append(center)

        # Можно аналогично искать минимум, если надо.

    # Условная "кластеризация" близких уровней (очень упрощённо).
    levels = sorted(levels)
    cluster_distance = 0.001
    clustered = []
    current_cluster = [levels[0]] if levels else []

    for lvl in levels[1:]:
        if abs(lvl - current_cluster[-1]) / lvl < cluster_distance:
            current_cluster.append(lvl)
        else:
            mean_lvl = np.mean(current_cluster)
            clustered.append(mean_lvl)
            current_cluster = [lvl]
    if current_cluster:
        clustered.append(np.mean(current_cluster))

    return clustered

def filter_levels(
    df_daily, df_4h, df_1h,
    raw_levels,
    min_touches=2,
    max_age_days=60,
    atr_buffer=0.1,
    volume_factor=1.2
):
    """
    Фильтруем уровни, чтобы оставить только "актуальные" по множеству критериев:
      - Кол-во касаний (>= min_touches).
      - Возраст уровня (не старше max_age_days).
      - Наличие "свободной зоны".
      - Проверяем 4h и 1h на объёмы/поджатие (упрощённо).
    """
    filtered = []

    # Для дат используем .index[-1]
    now_ts = df_daily.index[-1]  # последний бар
    # Если нужно число дней, сравниваем (now_ts - last_touch_ts).days

    # Пример: посчитаем ATR дневной
    current_atr = compute_atr(df_daily, period=7)
    if pd.isna(current_atr):
        print("Не удалось вычислить ATR, мало данных?")
        return []

    for lvl in raw_levels:
        # 1) Считаем, сколько касаний
        touches = 0
        last_touch_ts = None
        for i in range(len(df_daily)):
            close_i = df_daily["close"].iloc[i]
            # если close в пределах atr_buffer * ATR => касание
            if abs(close_i - lvl) < (atr_buffer * current_atr):
                touches += 1
                last_touch_ts = df_daily.index[i]

        if touches < min_touches:
            continue

        # 2) Проверка на возраст
        if last_touch_ts is None:
            continue
        diff_days = (now_ts - last_touch_ts).days
        if diff_days > max_age_days:
            continue

        # 3) Ищем хотя бы одно дневное наблюдение с "близким закрытием" (минимум фитилей)
        # Условная логика
        close_enough = False
        for i in range(len(df_daily)):
            row = df_daily.iloc[i]
            if abs(row["close"] - lvl) < 0.5 * atr_buffer * current_atr:
                # допустим, проверяем фитиль
                wick_up = row["high"] - max(row["close"], row["open"])
                wick_down = min(row["close"], row["open"]) - row["low"]
                if wick_up < 0.5 * current_atr and wick_down < 0.5 * current_atr:
                    close_enough = True
                    break
        if not close_enough:
            continue

        # 4) Простая "свободная зона" — посмотрим, что было после последнего касания
        slice_df = df_daily[df_daily.index >= last_touch_ts]
        if slice_df["high"].max() - lvl < 1.0 * current_atr:
            # допустим, мало места вверх
            pass  # Можно не обрубать, зависит от вашей логики

        # 5) Мультитаймфрейм-проверка: на 4h пусть будут повышенные объёмы и "поджатие"
        date_threshold = last_touch_ts - pd.Timedelta(days=3)  # 3 дня до последнего касания
        df_4h_slice = df_4h[(df_4h.index >= date_threshold) & (df_4h.index <= last_touch_ts)]
        if len(df_4h_slice) > 1:
            avg_4h_vol = df_4h_slice["volume"].mean()
            mean_4h_vol = df_4h["volume"].mean()  # для всего 4h
            if (avg_4h_vol > volume_factor * mean_4h_vol):
                # Значит, объёмы действительно повыше
                pass
            # Проверка "поджатия" (скорая логика)
            price_start = df_4h_slice["close"].iloc[0]
            price_end   = df_4h_slice["close"].iloc[-1]
            # ...

        # Если дошли сюда без continue, принимаем уровень
        filtered.append(lvl)

    return filtered
