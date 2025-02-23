import pandas as pd
import numpy as np

def compute_atr(df_daily, period=5):
    if df_daily.empty:
        return np.nan
    highs = df_daily["high"].values
    lows  = df_daily["low"].values
    closes= df_daily["close"].values
    trs = []
    for i in range(1, len(df_daily)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    ser = pd.Series(trs).rolling(period).mean()
    return ser.iloc[-1] if len(ser) >= period else np.nan

def filter_levels(df_daily, raw_levels, 
                  min_touches=2,
                  atr_buffer=0.15,
                  max_age_days=999):
    if df_daily.empty or not raw_levels:
        return []
    now_ts = df_daily.index[-1]
    cur_atr = compute_atr(df_daily, period=5)
    if pd.isna(cur_atr):
        return raw_levels  # если ATR не посчитался, вернём исходные

    filtered = []
    for lvl in raw_levels:
        touches = 0
        last_touch_ts = None
        for i in range(len(df_daily)):
            c = df_daily["close"].iloc[i]
            if abs(c - lvl) <= atr_buffer * cur_atr:
                touches += 1
                last_touch_ts = df_daily.index[i]
        if touches < min_touches:
            continue
        if last_touch_ts is None:
            continue
        age_days = (now_ts - last_touch_ts).days
        if age_days > max_age_days:
            continue
        filtered.append(lvl)
    return sorted(filtered)

def select_best_level(df_daily, filtered_levels):
    """
    Супер-простой способ: выбираем тот уровень, который чаще "касался" цены 
    (в пределах 1% от среднего Close).
    """
    if not filtered_levels:
        return None

    mean_close = df_daily["close"].mean()
    best_level = None
    best_score = -999
    for lvl in filtered_levels:
        score = 0
        for c in df_daily["close"]:
            if abs(c - lvl) <= 0.01 * mean_close:
                score += 1
        if score > best_score:
            best_score = score
            best_level = lvl
    return best_level
