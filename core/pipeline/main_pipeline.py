# core/pipeline/main_pipeline.py

import os
import torch
import concurrent.futures
import plotly.graph_objs as go
import pandas as pd

from ..config import (
    TOP_SYMBOLS, BYBIT_CATEGORY,
    DAYS_D, DAYS_4H, DAYS_1H,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
)
from ..incremental_fetcher import load_kline_incremental
from ..liquidity_levels import find_all_levels_for_symbol
from ..dataset import MultiSymbolDataset
from ..models.train import train_model_multi, evaluate_model_multi

def process_symbol(sym, sid):
    """
    find_all_levels_for_symbol => (d_sup, d_res, h1_sup,h1_res, df_d, df_4h, df_1h)
    Возвращаем (sym, sid, d_sup, d_res, h1_sup, h1_res, df_d, df_1h).
    """
    d_sup,d_res,h1_sup,h1_res, df_d, df_4h, df_1h= find_all_levels_for_symbol(sym)
    return (sym, sid, d_sup, d_res, h1_sup, h1_res, df_d, df_1h)

def run_pipeline_multi(device="cpu"):
    import datetime
    today= datetime.date.today()
    print(f"[CONFIG] Сегодня: {today},  диапазон: {today - datetime.timedelta(days=DAYS_D)}..{today}")

    print(f"Используем устройство: {device}")

    symbol_list= TOP_SYMBOLS
    symbol_id_map= {s:i for i,s in enumerate(symbol_list)}

    results=[]
    # Параллельная загрузка
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        fut_map={}
        for sym in symbol_list:
            sid= symbol_id_map[sym]
            fut= executor.submit(process_symbol, sym, sid)
            fut_map[fut]= sym
        for fut in concurrent.futures.as_completed(fut_map):
            sym= fut_map[fut]
            try:
                ret= fut.result()
                results.append(ret)
            except Exception as e:
                print(f"[{sym}] Error: {e}")

    # Склеиваем df_d => df_big
    df_list=[]
    symbol_data_map={}
    for (sym, sid, d_sup, d_res, h1_sup, h1_res, df_d, df_1h) in results:
        if df_d is None or df_d.empty:
            print(f"[{sym}] Нет дневных данных => skip.")
            continue
        df_d["symbol"]= sym
        df_d["symbol_id"]= sid

        # Сохраним h1, чтобы после обучения построить график
        symbol_data_map[sym]= (d_sup,d_res,h1_sup,h1_res,df_1h)

        df_list.append(df_d)

    if not df_list:
        print("Нет дневных данных => завершаем.")
        return

    df_big= pd.concat(df_list).sort_index()
    print("[INFO] df_big.shape=", df_big.shape)

    # Dataset => MultiSymbol
    from torch.utils.data import random_split, DataLoader
    dataset= MultiSymbolDataset(df_big, seq_len=50, feature_col="close", label_col="close")
    n= len(dataset)
    if n<200:
        print("Недостаточно данных для обучения => выходим.")
        return

    tr_size= int(n*0.7)
    va_size= int(n*0.15)
    te_size= n - tr_size - va_size
    train_ds, val_ds, test_ds= random_split(dataset, [tr_size, va_size, te_size])

    print(f"Dataset total={n}, train={tr_size}, val={va_size}, test={te_size}")
    num_symbols= df_big["symbol_id"].nunique()
    print(f"[Train model] num_symbols={num_symbols}")

    model= train_model_multi(train_ds, val_ds,
                             num_symbols=num_symbols,
                             device=device,
                             learning_rate=LEARNING_RATE,
                             num_epochs=NUM_EPOCHS,
                             batch_size=BATCH_SIZE)

    # Evaluate
    from torch.utils.data import DataLoader
    test_loader= DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_mse= evaluate_model_multi(model, test_loader, device=device)
    print(f"[Test] MSE={test_mse:.4f}")

    # Рисуем 1h график => 2 дневных + 2 часовых
    for sym in symbol_list:
        if sym not in symbol_data_map: 
            continue
        d_sup,d_res,h1_sup,h1_res, df_1h= symbol_data_map[sym]
        if df_1h is None or df_1h.empty:
            continue
        plot_1h_chart(sym, df_1h, d_sup, d_res, h1_sup, h1_res)


def plot_1h_chart(symbol, df_1h, d_sup, d_res, h1_sup, h1_res):
    df_1h= df_1h.copy()
    df_1h.sort_index(inplace=True)

    # Убедимся, что no duplicates
    df_1h= df_1h[~df_1h.index.duplicated()]
    print(f"[{symbol}] 1h rows={len(df_1h)}")

    fig= go.Figure()
    fig.add_trace(go.Candlestick(
        x= df_1h.index,                # Datetime
        open= df_1h["open"],
        high= df_1h["high"],
        low=  df_1h["low"],
        close=df_1h["close"],
        name=f"{symbol} 1H"
    ))
    # Дневные => синие
    if d_sup is not None:
        fig.add_hline(y=d_sup, line=dict(color='blue', dash='dash'),
                      annotation_text=f"D-Sup={d_sup:.2f}")
    if d_res is not None:
        fig.add_hline(y=d_res, line=dict(color='blue', dash='dash'),
                      annotation_text=f"D-Res={d_res:.2f}")

    # Часовые => зелёные
    if h1_sup is not None:
        fig.add_hline(y=h1_sup, line=dict(color='green', dash='dot'),
                      annotation_text=f"H1-Sup={h1_sup:.2f}")
    if h1_res is not None:
        fig.add_hline(y=h1_res, line=dict(color='green', dash='dot'),
                      annotation_text=f"H1-Res={h1_res:.2f}")

    fig.update_layout(
        title=f"{symbol} (1H) - daily + 1H levels",
        xaxis_rangeslider_visible=False
    )
    fig.show()


if __name__=="__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    run_pipeline_multi(device=device)
