import pandas as pd
import numpy as np
from datetime import date, timedelta
from .incremental_fetcher import load_kline_incremental
from .config import BYBIT_CATEGORY, DAYS_D, DAYS_4H, DAYS_1H

def find_all_levels_for_symbol(symbol):
    """
    1) Грузим df_d(180d), df_4h(90d), df_1h(30d)
    2) На daily => find_liquid_levels => (d_sup, d_res)
    3) 4h => find_liquid_levels => если близко, подтверждаем (не обрезаем)
    4) 1h => find_liquid_levels => если близко, возвращаем h1_sup/res, иначе None
    5) return (d_sup, d_res, h1_sup, h1_res, df_d, df_4h, df_1h)
    """
    today = date.today()
    end_dt_str = today.strftime("%Y-%m-%d")

    # daily
    start_d = (today - timedelta(days=DAYS_D)).strftime("%Y-%m-%d")
    df_d = load_kline_incremental(symbol, BYBIT_CATEGORY, "D", start_d, end_dt_str)

    # 4h
    start_4h = (today - timedelta(days=DAYS_4H)).strftime("%Y-%m-%d")
    df_4h = load_kline_incremental(symbol, BYBIT_CATEGORY, "240", start_4h, end_dt_str)

    # 1h
    start_1h = (today - timedelta(days=DAYS_1H)).strftime("%Y-%m-%d")
    df_1h = load_kline_incremental(symbol, BYBIT_CATEGORY, "60", start_1h, end_dt_str)

    if df_d.empty:
        return None, None, None, None, df_d, df_4h, df_1h

    # На daily
    d_sup, d_res = find_liquid_levels(df_d)
    if (d_sup is None) or (d_res is None):
        return d_sup, d_res, None, None, df_d, df_4h, df_1h

    # 4h => "подтверждаем"
    if (df_4h is not None) and not df_4h.empty:
        h4_sup, h4_res = find_liquid_levels(df_4h)
        # if h4_sup is close to d_sup => ok
        # ...
        # сейчас ничего не усредняем, только "проверяем"
        # (пропущено для краткости)
        pass

    # 1h => если close, return
    h1_sup, h1_res = None, None
    if (df_1h is not None) and not df_1h.empty:
        tmp_sup, tmp_res = find_liquid_levels(df_1h)
        if tmp_sup and abs(tmp_sup - d_sup) / d_sup <= 0.002:
            h1_sup = tmp_sup
        if tmp_res and abs(tmp_res - d_res) / d_res <= 0.002:
            h1_res = tmp_res

    return d_sup, d_res, h1_sup, h1_res, df_d, df_4h, df_1h


def find_liquid_levels(df, round_number=True):
    """
    Ищем (support, resistance).
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
    lows = df["low"].values
    closes = df["close"].values
    trs = []
    for i in range(1, len(df)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    ser = pd.Series(trs).rolling(period).mean()
    return ser.iloc[-1] if len(ser) >= period else 0

def find_potential_levels(df, daily_atr, round_number=True):
    window = 5
    closes = df["close"].values
    idxs = df.index
    local_levels = set()
    for i in range(window, len(df) - window):
        seg = closes[i - window:i + window + 1]
        center = closes[i]
        if center == seg.max() or center == seg.min():
            # skip if older than 90 days
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
    if price < 1:
        return round(price, 3)
    elif price < 1000:
        step = 0.5
        return round(price / step) * step
    else:
        step = 50
        return round(price / step) * step

def score_levels(df, levels, current_price, daily_atr):
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
    cprice = df["close"].iloc[-1]
    if lvl > cprice:
        top = df["high"].max()
        if (top - lvl) >= 1.5 * daily_atr:
            return 0.5
        else:
            return 0
    else:
        bot = df["low"].min()
        if (lvl - bot) >= 1.5 * daily_atr:
            return 0.5
        else:
            return 0

def check_approach_speed(df, lvl):
    if len(df) < 5:
        return 0
    rng_all = (df["high"] - df["low"]).mean()
    last5 = df.iloc[-5:]
    rng5 = (last5["high"] - last5["low"]).mean()
    if rng5 < rng_all * 0.8:
        return 0.5
    return 0

def check_false_breaks(df, lvl, daily_atr):
    tol = 0.2 * daily_atr
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    fb = 0
    for i in range(1, len(df)):
        if lvl > c[i - 1]:
            if (h[i] >= lvl + tol) and (c[i] < lvl):
                fb += 1
        else:
            if (l[i] <= lvl - tol) and (c[i] > lvl):
                fb += 1
    return min(1.0, fb * 0.5)

def pick_best_two_levels(scored, current_price):
    supp = [x for x in scored if x["level"] <= current_price]
    ress = [x for x in scored if x["level"] >= current_price]
    if not supp and not ress:
        return None, None
    sup = None
    if supp:
        s = sorted(supp, key=lambda x: (x["score"], x["level"]), reverse=True)
        sup = s[0]["level"]
    res = None
    if ress:
        r = sorted(ress, key=lambda x: (x["score"], -x["level"]), reverse=True)
        res = r[0]["level"]
    return sup, res

def fallback_levels(df, current_price, daily_atr):
    sup = current_price - 1.5 * daily_atr
    res = current_price + 1.5 * daily_atr
    return sup, res