# src/main.py

import os
import torch
import pandas as pd

from .config import (
    TOP_SYMBOLS, BYBIT_CATEGORY,
    BYBIT_START_DATE, BYBIT_END_DATE,
    DAILY_INTERVAL, H4_INTERVAL, H1_INTERVAL,
    BATCH_SIZE
)
from .data_fetcher import load_single_symbol_multi_timeframe
from .cluster_levels import find_local_extrema, cluster_extrema, make_binary_labels
from .multi_tf_analysis import filter_levels
from .dataset import MultiSymbolDataset
from .train import train_model_multi, evaluate_model_multi
from .backtest import run_backtest, plot_backtest_results

def run_pipeline_multi(device="cpu"):
    print("=== [1] Загружаем данные (D, 4h, 1h) для каждого инструмента ===")
    
    symbol_list = TOP_SYMBOLS  # 10+ символов
    df_list = []
    symbol_to_id = {}
    symbol_counter = 0

    for sym in symbol_list:
        print(f"-> Загрузка {sym} ...")
        df_d, df_4h, df_1h = load_single_symbol_multi_timeframe(
            symbol=sym,
            category=BYBIT_CATEGORY,
            start=BYBIT_START_DATE,
            end=BYBIT_END_DATE,
            daily_interval=DAILY_INTERVAL,
            h4_interval=H4_INTERVAL,
            h1_interval=H1_INTERVAL
        )
        if df_d.empty:
            print(f"Нет дневных данных для {sym}, пропускаем.")
            continue
        if df_4h.empty:
            print(f"Предупреждение: Нет 4H данных для {sym}, но всё равно продолжим.")
        if df_1h.empty:
            print(f"Предупреждение: Нет 1H данных для {sym}, но всё равно продолжим.")

        # Найдём экстремумы по df_d:
        from .cluster_levels import find_local_extrema, cluster_extrema
        local_maxima, local_minima = find_local_extrema(df_d["close"], window=10)
        raw_levels = cluster_extrema(local_maxima, local_minima)
        
        # Фильтруем
        final_levels = filter_levels(df_d, df_4h, df_1h, raw_levels)
        print(f"{sym}: исходных уровней={len(raw_levels)}, после фильтра={len(final_levels)}")

        # level_label
        label_s = make_binary_labels(df_d, final_levels, threshold_frac=0.001)
        df_d["level_label"] = label_s

        # добавим symbol, symbol_id
        df_d["symbol"] = sym
        df_d["symbol_id"] = symbol_counter
        symbol_to_id[sym] = symbol_counter
        symbol_counter += 1

        df_list.append(df_d)

    # Если df_list пуст, завершаем
    if not df_list:
        print("Нет данных ни по одному инструменту!")
        return

    # Объединяем дневные
    df_big = pd.concat(df_list).sort_index()

    # Убедимся, что есть нужные колонки
    for col in ["close","symbol","symbol_id","level_label"]:
        if col not in df_big.columns:
            raise ValueError(f"В df_big нет колонки {col}!")

    print(f"Итого сформирован df_big с {len(df_big)} строками, {symbol_counter} инструмент(ами).")

    # [2] Создаём MultiSymbolDataset
    from .dataset import MultiSymbolDataset
    dataset = MultiSymbolDataset(df_big)  # seq_len берется из config

    n = len(dataset)
    train_size = int(n*0.8)
    val_size = int(n*0.1)
    test_size = n - train_size - val_size

    from torch.utils.data import random_split, DataLoader
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset total={n}, train={train_size}, val={val_size}, test={test_size}")

    # [3] Обучаем MultiSymbolLSTM
    from .train import train_model_multi, evaluate_model_multi
    num_symbols = df_big["symbol_id"].nunique()
    print(f"Обучаем модель. num_symbols={num_symbols}")
    model = train_model_multi(train_data, val_data, num_symbols=num_symbols, device=device)

    # [4] Оцениваем на тесте
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    evaluate_model_multi(model, test_loader, device=device)

    # [5] Бэктест:
    #    Для демонстрации: строим ОТДЕЛЬНЫЙ график по каждому символу.
    print("=== Бэктест (упрощённо) по каждому символу ===")
    for sym_id in range(num_symbols):
        df_sym = df_big[df_big["symbol_id"]==sym_id].copy()
        if df_sym.empty:
            continue
        sym_name = df_sym["symbol"].iloc[0]

        # Простая логика: если level_label=1 => signal=1
        df_sym["signal"] = (df_sym["level_label"]==1).astype(int)

        from .backtest import run_backtest, plot_backtest_results
        df_bt, trades, eq_curve = run_backtest(df_sym, levels=None, initial_capital=10000.0)
        plot_backtest_results(df_bt, trades, eq_curve, levels=None, 
                              title=f"{sym_name} Backtest")

    print("=== Готово! ===")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    run_pipeline_multi(device=device)
