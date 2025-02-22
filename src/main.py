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
# Вместо старого "data_fetcher" импортируем новую функцию из "incremental_fetcher"
from .incremental_fetcher import load_kline_incremental

from .cluster_levels import (
    find_local_extrema, cluster_extrema, make_binary_labels
)
from .multi_tf_analysis import filter_levels
from .dataset import MultiSymbolDataset
from .train import train_model_multi, evaluate_model_multi
from .backtest import run_backtest, plot_backtest_results


def process_symbol(symbol_id, sym, category, start, end, d_intv, h4_intv, h1_intv):
    """
    Параллельная функция для одного символа:
      1) Использует load_kline_incremental(...) для инкрементальной подгрузки (D,4H,1H).
      2) Находит экстремумы по дневным df_d.
      3) Кластеризует (cluster_extrema), затем фильтрует (filter_levels).
    Возвращает (symbol_id, sym, df_d, df_4h, df_1h, final_levels).
    Если дневные или часовые данные пусты, возвращает None.
    """

    # Шаг 1: загружаем инкрементально три таймфрейма
    df_d  = load_kline_incremental(symbol=sym, category=category, interval=d_intv,
                                   start=start, end=end)
    df_4h = load_kline_incremental(symbol=sym, category=category, interval=h4_intv,
                                   start=start, end=end)
    df_1h = load_kline_incremental(symbol=sym, category=category, interval=h1_intv,
                                   start=start, end=end)
    # Проверяем, что daily и H1 не пусты
    if df_d.empty or df_1h.empty:
        return None

    # Экстремумы на daily
    local_maxima, local_minima = find_local_extrema(df_d["close"], window=10)
    raw_levels = cluster_extrema(local_maxima, local_minima)
    
    # Фильтрация (daily + 4H + 1H)
    final_levels = filter_levels(df_d, df_4h, df_1h, raw_levels)
    return (symbol_id, sym, df_d, df_4h, df_1h, final_levels)


def run_pipeline_multi(device="cpu"):
    print("=== [1] Параллельная загрузка (инкрементальная) для нескольких инструментов ===")

    symbol_list = TOP_SYMBOLS
    symbol_id_map = {sym: i for i, sym in enumerate(symbol_list)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {}
        for sym in symbol_list:
            sid = symbol_id_map[sym]
            fut = executor.submit(
                process_symbol,
                sid, sym,
                BYBIT_CATEGORY,
                BYBIT_START_DATE, BYBIT_END_DATE,
                DAILY_INTERVAL, H4_INTERVAL, H1_INTERVAL
            )
            future_map[fut] = sym

        for fut in concurrent.futures.as_completed(future_map):
            sym = future_map[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
                else:
                    print(f"Нет нужных данных для {sym}, пропускаем.")
            except Exception as e:
                print(f"Ошибка при загрузке {sym}: {e}")

    if not results:
        print("Нет данных ни по одному символу!")
        return

    # results => (sid, sym, df_d, df_4h, df_1h, final_levels)
    df_list = []
    symbol_levels_map = {}
    for (sid, sym, df_d, df_4h, df_1h, final_levels) in results:
        # Сохраняем уровни (для информации/бэктеста)
        symbol_levels_map[sid] = final_levels

        # Ставим level_label на дневной фрейм (для обучения)
        lab_daily = make_binary_labels(df_d, final_levels, threshold_frac=0.001)
        df_d["level_label"] = lab_daily
        df_d["symbol"] = sym
        df_d["symbol_id"] = sid
        df_list.append(df_d)

    df_big = pd.concat(df_list).sort_index()
    print(f"=== Сформирован df_big (daily) c {len(df_big)} строк, {len(results)} инструментов ===")

    from .dataset import MultiSymbolDataset
    dataset = MultiSymbolDataset(df_big)
    from torch.utils.data import random_split, DataLoader
    n = len(dataset)
    train_size = int(n*0.8)
    val_size   = int(n*0.1)
    test_size  = n - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset total(daily)={n}, train={train_size}, val={val_size}, test={test_size}")

    # Обучение
    num_symbols = df_big["symbol_id"].nunique()
    print(f"Обучаем модель (num_symbols={num_symbols})...")
    model = train_model_multi(train_data, val_data, num_symbols=num_symbols, device=device)

    # Тест
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    evaluate_model_multi(model, test_loader, device=device)

    # Бэктест и графики (часовые, последние 60 дней, по одному окну)
    print("=== Бэктест (H1) и графики (по одному окну на инструмент), последние ~60 дней ===")
    cutoff_days = 60
    for (sid, sym, df_d, df_4h, df_1h, final_levels) in results:
        if df_1h.empty:
            continue
        
        lab_1h = make_binary_labels(df_1h, final_levels, threshold_frac=0.001)
        df_1h["level_label"] = lab_1h
        df_1h["signal"] = (df_1h["level_label"] == 1).astype(int)

        last_date = df_1h.index.max()
        if pd.isna(last_date):
            continue
        cutoff_date = last_date - pd.Timedelta(days=cutoff_days)
        df_trim = df_1h[df_1h.index>=cutoff_date].copy()
        if df_trim.empty:
            continue

        from .backtest import run_backtest, plot_backtest_results
        df_bt, trades, eq_curve = run_backtest(
            df_trim, levels=final_levels, initial_capital=10000.0
        )
        title_str = f"{sym} (H1) - Last {cutoff_days} days"
        plot_backtest_results(df_bt, trades, eq_curve, levels=final_levels, title=title_str)

    print("=== Готово! ===")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    run_pipeline_multi(device=device)
