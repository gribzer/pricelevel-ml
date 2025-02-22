# src/cluster_levels.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from .config import EPS_PERCENT, MIN_SAMPLES, WINDOW_SIZE

def find_local_extrema(prices, window=WINDOW_SIZE):
    """
    Возвращает списки (индексы, цены) локальных максимумов и минимумов.
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

def cluster_extrema(maxima, minima, eps_frac=EPS_PERCENT, min_samples=MIN_SAMPLES):
    """
    DBSCAN-кластеризация экстремумов по оси цены.
    Возвращает список уровней (float).
    """
    all_points = maxima + minima
    if not all_points:
        return []

    prices = np.array([p for (_, p) in all_points]).reshape(-1, 1)
    mean_price = prices.mean()
    eps_val = mean_price * eps_frac  # радиус кластера

    db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(prices)
    labels = db.labels_
    unique_labels = set(labels)

    levels = []
    for lbl in unique_labels:
        if lbl == -1:
            continue
        cluster_points = prices[labels == lbl]
        level_price = cluster_points.mean()
        levels.append(level_price)
    levels = sorted(levels)
    return levels

def make_binary_labels(df, levels, threshold_frac=0.001):
    """
    Метка = 1, если close рядом с одним из levels (± threshold_frac).
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
