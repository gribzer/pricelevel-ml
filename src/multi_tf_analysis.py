# src/multi_tf_analysis.py

import pandas as pd
import numpy as np

def compute_atr(df_daily, period=7):
    if df_daily.empty:
        return np.nan
    highs=df_daily["high"].values
    lows =df_daily["low"].values
    closes=df_daily["close"].values
    trs=[]
    for i in range(1,len(df_daily)):
        tr=max(
            highs[i]-lows[i],
            abs(highs[i]-closes[i-1]),
            abs(lows[i]-closes[i-1])
        )
        trs.append(tr)
    ser=pd.Series(trs).rolling(period).mean()
    return ser.iloc[-1] if len(ser)>=period else np.nan

def filter_levels(df_daily, df_4h, df_1h,
                  raw_levels,
                  min_touches=5,
                  atr_buffer=0.25,
                  volume_factor=1.5,
                  max_age_days=40):
    filtered=[]
    if df_daily.empty or not raw_levels:
        return []
    now_ts=df_daily.index[-1]
    cur_atr=compute_atr(df_daily, period=7)
    if pd.isna(cur_atr):
        return []

    for lvl in raw_levels:
        touches=0
        last_touch_ts=None
        for i in range(len(df_daily)):
            c=df_daily["close"].iloc[i]
            if abs(c-lvl)<(atr_buffer*cur_atr):
                touches+=1
                last_touch_ts=df_daily.index[i]
        if touches<min_touches:
            continue
        if last_touch_ts is None:
            continue

        if (now_ts - last_touch_ts).days>max_age_days:
            continue

        # 4h
        if df_4h is not None and not df_4h.empty:
            dt_thresh = last_touch_ts - pd.Timedelta(days=3)
            slice_4h = df_4h[(df_4h.index>=dt_thresh)&(df_4h.index<=last_touch_ts)]
            if len(slice_4h)>2:
                avg_4h = slice_4h["volume"].mean()
                global_4h = df_4h["volume"].mean()
                # if avg_4h<(volume_factor*global_4h):
                #     continue
        filtered.append(lvl)

    return sorted(filtered)
