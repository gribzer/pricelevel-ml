# src/main.py

import os
import torch
import pandas as pd
import concurrent.futures

from .config import (
    TOP_SYMBOLS, BYBIT_CATEGORY,
    BYBIT_START_DATE, BYBIT_END_DATE,
    DAILY_INTERVAL, H4_INTERVAL, H1_INTERVAL,
    BATCH_SIZE
)
from .incremental_fetcher import load_kline_incremental
from .cluster_levels import (
    find_local_extrema,
    cluster_extrema,
    remove_too_close_levels,
    make_binary_labels
)
from .multi_tf_analysis import filter_levels
from .dataset import MultiSymbolDataset
from .train import train_model_multi, evaluate_model_multi
from .backtest import run_backtest, plot_backtest_results

def process_symbol(symbol_id, sym, category, start, end, d_intv, h4_intv, h1_intv):
    print(f"=== process_symbol({symbol_id}, {sym}) => загрузка {start}..{end}")
    # 1) Инкрементальная подгрузка
    df_d  = load_kline_incremental(sym, category, d_intv, start, end)
    df_4h = load_kline_incremental(sym, category, h4_intv, start, end)
    df_1h = load_kline_incremental(sym, category, h1_intv, start, end)
    if df_d.empty or df_1h.empty:
        return None

    # 2) Экстремумы (daily)
    local_maxima, local_minima = find_local_extrema(df_d["close"], window=15)
    raw_levels = cluster_extrema(local_maxima, local_minima, eps_frac=0.01, min_samples=5)
    raw_levels = remove_too_close_levels(raw_levels, min_dist_frac=0.01)

    final_levels = filter_levels(
        df_d, df_4h, df_1h, raw_levels,
        min_touches=5,
        atr_buffer=0.25,
        volume_factor=1.5,
        max_age_days=40
    )
    return (symbol_id, sym, df_d, df_4h, df_1h, final_levels)

def run_pipeline_multi(device="cpu"):
    print("=== [1] Параллельная загрузка (инкрементальная) для нескольких инструментов ===")
    symbol_list = TOP_SYMBOLS
    symbol_id_map = { s: i for i,s in enumerate(symbol_list)}

    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        fut_map={}
        for sym in symbol_list:
            sid=symbol_id_map[sym]
            fut=executor.submit(
                process_symbol,
                sid, sym,
                BYBIT_CATEGORY,
                BYBIT_START_DATE, BYBIT_END_DATE,
                DAILY_INTERVAL, H4_INTERVAL, H1_INTERVAL
            )
            fut_map[fut]=sym

        for fut in concurrent.futures.as_completed(fut_map):
            sym=fut_map[fut]
            try:
                res=fut.result()
                if res:
                    results.append(res)
                else:
                    print(f"Нет (daily/H1) данных для {sym}, пропускаем.")
            except Exception as e:
                print(f"Ошибка при загрузке {sym}: {e}")

    if not results:
        print("Нет валидных результатов.")
        return

    # Собираем daily => df_big
    df_list=[]
    symbol_levels_map={}
    for (sid, sym, df_d, df_4h, df_1h, final_levels) in results:
        symbol_levels_map[sid] = final_levels
        lab_daily = make_binary_labels(df_d, final_levels, threshold_frac=0.001)
        df_d["level_label"]=lab_daily
        df_d["symbol"]=sym
        df_d["symbol_id"]=sid
        df_list.append(df_d)

    df_big = pd.concat(df_list).sort_index()
    print(f"Сформирован df_big (daily), строчек: {len(df_big)}")

    # Dataset => обучаем
    dataset=MultiSymbolDataset(df_big, seq_len=50)
    from torch.utils.data import random_split,DataLoader
    n=len(dataset)
    tr_size=int(n*0.8)
    va_size=int(n*0.1)
    te_size=n-tr_size-va_size
    train_data,val_data,test_data = random_split(dataset,[tr_size,va_size,te_size])
    print(f"Dataset total={n}, train={tr_size}, val={va_size}, test={te_size}")

    num_symbols = df_big["symbol_id"].nunique()
    print(f"Обучение модели (num_symbols={num_symbols})...")
    model=train_model_multi(train_data,val_data,num_symbols=num_symbols,device=device,
                            learning_rate=0.0003,num_epochs=50,batch_size=BATCH_SIZE)

    # Тест
    test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)
    evaluate_model_multi(model,test_loader,device=device)

    # Бэктест (H1), последние 60 дней
    print("=== Бэктест + графики: часовой TF, последние 60 дней ===")
    cutoff_days=60
    for (sid, sym, df_d, df_4h, df_1h, final_levels) in results:
        if df_1h.empty:
            continue
        lab_h1 = make_binary_labels(df_1h, final_levels, threshold_frac=0.001)
        df_1h["level_label"]=lab_h1
        df_1h["signal"]=(df_1h["level_label"]==1).astype(int)

        last_date=df_1h.index.max()
        if pd.isna(last_date):
            continue
        cut_date=last_date - pd.Timedelta(days=cutoff_days)
        df_trim=df_1h[df_1h.index>=cut_date].copy()
        if df_trim.empty:
            continue

        df_bt,trades,eq_curve=run_backtest(df_trim,levels=final_levels,initial_capital=10000.0)
        title_str=f"{sym} (H1) - last {cutoff_days} days"
        plot_backtest_results(df_bt,trades,eq_curve,levels=final_levels,title=title_str)

    print("=== Готово! ===")

if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    run_pipeline_multi(device=device)
