# core/liquidity_levels.py

import pandas as pd
import numpy as np
from datetime import date, timedelta

# Импортим из локальных модулей
from .incremental_fetcher import load_kline_incremental
from .config import BYBIT_CATEGORY, DAYS_D, DAYS_4H, DAYS_1H

def find_all_levels_for_symbol(symbol):
    """
    1) Загружаем данные:
       - дневные (180д) => df_d
       - 4h   ( 90д) => df_4h
       - 1h   ( 30д) => df_1h
    2) На daily ищем два ключевых уровня (d_sup, d_res).
    3) На 4h ищем уровни, сравниваем с дневными:
       - Если 4h-уровень близок к дневному (с учётом ATR/процент), то подтверждаем (при желании можно усреднить).
    4) На 1h ищем 2 уровня (h1_sup, h1_res), если «близко» к дневным - возвращаем, иначе None.
    5) Итог: (d_sup, d_res, h1_sup, h1_res, df_d, df_4h, df_1h).
    """

    today = date.today()
    end_dt_str = today.strftime("%Y-%m-%d")

    # 1) Дневные
    start_d = (today - timedelta(days=DAYS_D)).strftime("%Y-%m-%d")
    df_d = load_kline_incremental(symbol, BYBIT_CATEGORY, "D", start_d, end_dt_str)

    # 4h
    start_4h = (today - timedelta(days=DAYS_4H)).strftime("%Y-%m-%d")
    df_4h = load_kline_incremental(symbol, BYBIT_CATEGORY, "240", start_4h, end_dt_str)

    # 1h
    start_1h = (today - timedelta(days=DAYS_1H)).strftime("%Y-%m-%d")
    df_1h = load_kline_incremental(symbol, BYBIT_CATEGORY, "60", start_1h, end_dt_str)

    # Если нет дневных данных - смысла дальше нет
    if df_d.empty:
        return None, None, None, None, df_d, df_4h, df_1h

    # 2) На daily
    d_sup, d_res = find_liquid_levels(df_d)
    if (d_sup is None) or (d_res is None):
        # Если не смогли найти дневные уровни - всё, выходим
        return d_sup, d_res, None, None, df_d, df_4h, df_1h

    # 3) 4h => "подтверждаем" (если данные есть)
    if (df_4h is not None) and not df_4h.empty:
        h4_sup, h4_res = find_liquid_levels(df_4h)
        # Если 4h-суппорт «рядом» с d_sup (±0.5% или с учётом ATR) - считаем подтверждённым
        # Можно усреднять, чтобы скорректировать дневной уровень
        # Ниже упрощённая проверка (±0.5%)
        if h4_sup and abs(h4_sup - d_sup)/d_sup <= 0.005:
            # Пример: усредним
            d_sup = (d_sup + h4_sup)/2

        if h4_res and abs(h4_res - d_res)/d_res <= 0.005:
            d_res = (d_res + h4_res)/2

    # 4) 1h => если «близко» к d_sup/d_res, возвращаем (h1_sup,h1_res)
    h1_sup, h1_res = None, None
    if (df_1h is not None) and not df_1h.empty:
        tmp_sup, tmp_res = find_liquid_levels(df_1h)
        # Если 1h sup достаточно близок к d_sup, считаем его уточнённым
        if tmp_sup and abs(tmp_sup - d_sup) / d_sup <= 0.002:
            h1_sup = tmp_sup
        # Аналогично для res
        if tmp_res and abs(tmp_res - d_res) / d_res <= 0.002:
            h1_res = tmp_res

    return d_sup, d_res, h1_sup, h1_res, df_d, df_4h, df_1h


def find_liquid_levels(df, round_number=True):
    """
    Ищем (support, resistance), используя кастомную логику:
    - Считаем ATR,
    - Находим "сырые" потенциальные уровни (локальные экстремумы),
    - Ставим им "оценку" score,
    - Выбираем 2 лучших (ниже/выше текущей цены).
    """
    if len(df) < 10:
        return None, None

    daily_atr = compute_daily_atr(df, 6)
    current_price = df["close"].iloc[-1]

    raw_levels = find_potential_levels(df, daily_atr, round_number=round_number)
    if not raw_levels:
        return fallback_levels(df, current_price, daily_atr)

    scored = score_levels(df, raw_levels, current_price, daily_atr)
    if not scored:
        return fallback_levels(df, current_price, daily_atr)

    sup, res = pick_best_two_levels(scored, current_price)
    if sup is None or res is None:
        fsup, fres = fallback_levels(df, current_price, daily_atr)
        if sup is None:
            sup = fsup
        if res is None:
            res = fres
    return sup, res

def compute_daily_atr(df, period=6):
    if len(df) < period:
        return 0
    highs = df["high"].values
    lows  = df["low"].values
    closes= df["close"].values
    trs = []
    for i in range(1, len(df)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    ser = pd.Series(trs).rolling(period).mean()
    return ser.iloc[-1] if len(ser) >= period else 0

def find_potential_levels(df, daily_atr, round_number=True):
    """
    Находим локальные экстремумы (окно=5),
    отбрасываем слишком «старые» (age>90д),
    и при желании «округляем» уровень.
    """
    window = 5
    closes = df["close"].values
    idxs = df.index
    local_levels = set()

    for i in range(window, len(df) - window):
        seg = closes[i - window : i + window + 1]
        center = closes[i]
        # Если это локальный минимум или максимум
        if center == seg.max() or center == seg.min():
            # Пропускаем уровни старше 90д
            max_age_days = 90
            age = (idxs[-1] - idxs[i]).days
            if age > max_age_days:
                continue
            lvl = center
            if round_number:
                lvl = snap_to_round_price(lvl, daily_atr)
            local_levels.add(lvl)

    return sorted(local_levels)

def snap_to_round_price(price, daily_atr):
    """
    "Округляем" уровень (особенно для больших чисел).
    """
    if price < 1:
        return round(price, 3)
    elif price < 1000:
        step = 0.5
        return round(price / step) * step
    else:
        step = 50
        return round(price / step) * step

def score_levels(df, levels, current_price, daily_atr):
    """
    Вычисляем "очки" для каждого уровня:
      - кол-во касаний,
      - объёмы,
      - свободное пространство,
      - скорость подхода,
      - ложные пробои
    Фильтруем: score>=0.5
    """
    out = []
    for lvl in levels:
        sc = 0
        sc += 0.5 * count_touches(df, lvl, daily_atr)
        sc += check_volumes_on_touches(df, lvl, daily_atr)
        sc += measure_free_space(df, lvl, daily_atr)
        sc += check_approach_speed(df, lvl)
        sc += check_false_breaks(df, lvl, daily_atr)
        out.append({"level": lvl, "score": sc})

    threshold = 0.5
    filtered = [x for x in out if x["score"] >= threshold]
    return sorted(filtered, key=lambda x: x["level"])

def count_touches(df, lvl, daily_atr):
    tol = 0.2 * daily_atr
    c = df["close"].values
    cnt = np.sum(np.abs(c - lvl) <= tol)
    return min(cnt, 8) / 8

def check_volumes_on_touches(df, lvl, daily_atr):
    tol = 0.2 * daily_atr
    mask = np.abs(df["close"] - lvl) <= tol
    if not np.any(mask):
        return 0
    sel = df[mask]
    avgv = df["volume"].mean()
    ratio = (sel["volume"] > avgv * 1.2).mean()
    return ratio

def measure_free_space(df, lvl, daily_atr):
    """
    Проверяем, есть ли "запас хода" от уровня до экстремума (>=1.5*ATR).
    """
    cprice = df["close"].iloc[-1]
    if lvl > cprice:
        top = df["high"].max()
        return 0.5 if (top - lvl) >= 1.5 * daily_atr else 0
    else:
        bot = df["low"].min()
        return 0.5 if (lvl - bot) >= 1.5 * daily_atr else 0

def check_approach_speed(df, lvl):
    """
    Смотрим волатильность последних 5 свечей vs средней по df.
    Если вдруг сужение => +0.5
    """
    if len(df) < 5:
        return 0
    rng_all = (df["high"] - df["low"]).mean()
    last5 = df.iloc[-5:]
    rng5 = (last5["high"] - last5["low"]).mean()
    if rng5 < rng_all * 0.8:
        return 0.5
    return 0

def check_false_breaks(df, lvl, daily_atr):
    """
    Ложные пробои вокруг lvl (±0.2*ATR).
    """
    tol = 0.2 * daily_atr
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    fb = 0
    for i in range(1, len(df)):
        prev_close = c[i-1]
        # Если уровень выше предыдущей цены, смотрим, не было ли заброса и возврата
        if lvl > prev_close:
            if (h[i] >= lvl + tol) and (c[i] < lvl):
                fb += 1
        else:
            # Если уровень ниже, смотрим, не было ли прокола снизу
            if (l[i] <= lvl - tol) and (c[i] > lvl):
                fb += 1

    return min(1.0, fb * 0.5)

def pick_best_two_levels(scored, current_price):
    """
    Разделяем уровни на те, что ниже current_price (supp)
    и выше (res). Берём лучший из каждой категории.
    """
    supp = [x for x in scored if x["level"] <= current_price]
    ress = [x for x in scored if x["level"] >= current_price]

    if not supp and not ress:
        return None, None

    sup = None
    if supp:
        # Сортируем: чем выше score, тем лучше;
        # при равном score – чем ближе к price, тем лучше
        s = sorted(supp, key=lambda x: (x["score"], x["level"]), reverse=True)
        sup = s[0]["level"]

    res = None
    if ress:
        # Аналогично, но для уровней выше цены
        r = sorted(ress, key=lambda x: (x["score"], -x["level"]), reverse=True)
        res = r[0]["level"]

    return sup, res

def fallback_levels(df, current_price, daily_atr):
    """
    Если мы так и не нашли адекватных уровней,
    берём "запас" от текущей цены ±1.5*ATR.
    """
    sup = current_price - 1.5 * daily_atr
    res = current_price + 1.5 * daily_atr
    return sup, res
