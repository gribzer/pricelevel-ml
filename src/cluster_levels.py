# src/cluster_levels.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def find_local_extrema(prices, window=15):
    local_maxima=[]
    local_minima=[]
    vals=prices.values
    idxs=prices.index
    for i in range(window, len(vals)-window):
        seg=vals[i-window:i+window+1]
        center=vals[i]
        if center==seg.max():
            local_maxima.append((idxs[i], center))
        if center==seg.min():
            local_minima.append((idxs[i], center))
    return local_maxima, local_minima

def cluster_extrema(maxima, minima, eps_frac=0.01, min_samples=5):
    all_points=maxima+minima
    if not all_points:
        return []

    prices=np.array([p for (_,p) in all_points]).reshape(-1,1)
    mean_price=prices.mean()
    eps_val=mean_price*eps_frac

    db=DBSCAN(eps=eps_val, min_samples=min_samples).fit(prices)
    labels=db.labels_
    unq=set(labels)
    levels=[]
    for lbl in unq:
        if lbl==-1:
            continue
        cpoints=prices[labels==lbl]
        level=cpoints.mean()
        levels.append(level)
    return sorted(levels)

def remove_too_close_levels(levels, min_dist_frac=0.01):
    if not levels:
        return []
    levels=sorted(levels)
    filtered=[levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - filtered[-1])/lvl>=min_dist_frac:
            filtered.append(lvl)
    return filtered

def make_binary_labels(df, levels, threshold_frac=0.001):
    if not levels:
        return pd.Series(data=0, index=df.index, name="level_label")
    closes=df["close"].values
    arr=np.zeros_like(closes, dtype=int)
    for i in range(len(closes)):
        c=closes[i]
        for lvl in levels:
            if abs(c-lvl)<=c*threshold_frac:
                arr[i]=1
                break
    return pd.Series(arr, index=df.index, name="level_label")
