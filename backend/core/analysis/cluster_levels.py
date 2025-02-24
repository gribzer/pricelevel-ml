# core/analysis/cluster_levels.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def find_local_extrema(prices, window=5):
    """
    Ищем локальные экстремумы в ряду 'prices' с помощью простого окна +/- window
    Возвращаем (список_локальных_max, список_локальных_min) = ([(dt,price),...],...)
    """
    local_maxima = []
    local_minima = []
    vals = prices.values
    idxs = prices.index
    for i in range(window, len(vals) - window):
        seg = vals[i - window : i + window + 1]
        center = vals[i]
        if center == seg.max():
            local_maxima.append((idxs[i], center))
        if center == seg.min():
            local_minima.append((idxs[i], center))
    return local_maxima, local_minima

def cluster_extrema(maxima, minima, eps_frac=0.01, min_samples=2):
    """
    Сливаем экстремумы в кластеры (DBSCAN).
    eps_frac=0.01 => радиус ~ 1% от средней цены
    """
    all_points = maxima + minima
    if not all_points:
        return []
    arr = np.array([p for (_, p) in all_points]).reshape(-1, 1)
    mean_val = arr.mean()
    eps_val = mean_val * eps_frac

    db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(arr)
    labels = db.labels_
    unq = set(labels)
    levels = []
    for l in unq:
        if l == -1:
            continue
        clust = arr[labels==l]
        lvl = clust.mean()
        levels.append(lvl)
    return sorted(levels)

def remove_too_close_levels(levels, min_dist_frac=0.005):
    """
    Убираем уровни, если они ближе чем 0.5% друг к другу (min_dist_frac=0.005).
    """
    if not levels:
        return []
    out = [levels[0]]
    for lvl in levels[1:]:
        prev = out[-1]
        rel_dist = abs(lvl - prev)/( (lvl+prev)/2 )
        if rel_dist > min_dist_frac:
            out.append(lvl)
    return out
