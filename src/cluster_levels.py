# src/cluster_levels.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Параметры по умолчанию, можно брать из config или ставить жёстко
DEFAULT_WINDOW = 15
DEFAULT_EPS_FRAC = 0.01   # 1%
DEFAULT_MIN_SAMPLES = 5   # нужно минимум 5 экстремумов в кластере

def find_local_extrema(prices, window=DEFAULT_WINDOW):
    """
    Ищем локальные максимумы/минимумы по окну ±window.
    Возвращаем два списка: local_maxima, local_minima,
    в каждом [(time, price), ...].
    """
    local_maxima = []
    local_minima = []
    values = prices.values
    idxs = prices.index

    for i in range(window, len(values) - window):
        segment = values[i-window : i+window+1]
        center = values[i]
        if center == segment.max():
            local_maxima.append((idxs[i], center))
        if center == segment.min():
            local_minima.append((idxs[i], center))
    
    return local_maxima, local_minima

def cluster_extrema(maxima, minima,
                    eps_frac=DEFAULT_EPS_FRAC,
                    min_samples=DEFAULT_MIN_SAMPLES):
    """
    Объединяем maxima+minima, потом DBSCAN по цене.
    eps_frac => eps_val = mean_price * eps_frac
    min_samples => минимальное кол-во точек для кластера
    Возвращаем список уровней (float).
    """
    all_points = maxima + minima
    if not all_points:
        return []

    prices = np.array([p for (_, p) in all_points]).reshape(-1,1)
    mean_price = prices.mean()
    eps_val = mean_price * eps_frac

    db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(prices)
    labels = db.labels_
    unique_labels = set(labels)
    levels = []
    for lbl in unique_labels:
        if lbl == -1:
            # шум
            continue
        cluster_points = prices[labels == lbl]
        level_price = cluster_points.mean()
        levels.append(level_price)
    return sorted(levels)

def remove_too_close_levels(levels, min_dist_frac=0.01):
    """
    Удаляем уровни, которые ближе друг к другу, чем min_dist_frac*цена.
    Например, min_dist_frac=0.01 => 1%.
    """
    if not levels:
        return []
    levels = sorted(levels)
    filtered = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - filtered[-1]) / lvl >= min_dist_frac:
            filtered.append(lvl)
    return filtered

def make_binary_labels(df, levels, threshold_frac=0.001):
    """
    level_label=1, если close в пределах ± (threshold_frac * close) от уровня.
    """
    if not levels:
        return pd.Series(data=0, index=df.index, name='level_label')

    closes = df["close"].values
    labels = np.zeros_like(closes, dtype=int)
    for i in range(len(closes)):
        c = closes[i]
        for lvl in levels:
            if abs(c - lvl) <= c * threshold_frac:
                labels[i] = 1
                break
    return pd.Series(data=labels, index=df.index, name='level_label')
