# src/cluster_levels.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from .config import EPS_PERCENT, MIN_SAMPLES

def find_local_extrema(prices, window=5):
    """
    Возвращает списки (индексы, цены) локальных максимумов и минимумов.
    prices - Series (индекс-дата, значение-цена).
    window - количество баров слева и справа для подтверждения экстремума.
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

def cluster_extrema(maxima, minima, eps_frac=0.005, min_samples=2):
    """
    Объединяем все экстремумы в один список и кластеризуем по цене.
    eps_frac - доля от средней цены, используемая как eps в DBSCAN.
    Возвращает список уровней (float), отсортированных по величине.
    """
    all_points = maxima + minima  # это список кортежей (time, price)
    if not all_points:
        return []

    # Массив только цен
    prices = np.array([p for (_, p) in all_points]).reshape(-1, 1)
    mean_price = prices.mean()
    eps_val = mean_price * eps_frac

    db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(prices)
    labels = db.labels_
    unique_labels = set(labels)
    levels = []
    for lbl in unique_labels:
        if lbl == -1:
            continue  # шум
        cluster_points = prices[labels == lbl]
        level_price = cluster_points.mean()
        levels.append(level_price)
    levels = sorted(levels)
    return levels

def make_binary_labels(df, levels, threshold_frac=0.001):
    """
    Создаем бинарную метку: 1, если close цены в df находится
    вблизи одного из уровней (± threshold_frac), иначе 0.
    
    df: DataFrame с колонкой 'close'.
    levels: список цен уровней.
    threshold_frac: допустимый люфт в долях от цены (или можно взять от уровня).
    Возвращаем Series (та же индексация, значения 0/1).
    """
    if not levels:
        return pd.Series(data=0, index=df.index)

    closes = df["close"].values
    labels = np.zeros_like(closes, dtype=int)
    for i in range(len(closes)):
        c = closes[i]
        # Проверяем близость цены c к какому-либо уровню
        # Здесь возьмем люфт как threshold_frac от c
        # Альтернативно можно взять threshold_frac от самого уровня
        for lvl in levels:
            if abs(c - lvl) <= c * threshold_frac:
                labels[i] = 1
                break
    return pd.Series(data=labels, index=df.index, name='level_label')
