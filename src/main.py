# src/main.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split

from .data_fetcher import load_multi_timeframe
from .multi_tf_analysis import (
    find_primary_levels,
    filter_levels
)
from .backtest import run_backtest, plot_backtest_results

def run_pipeline(symbol="BTCUSDT", device='cpu'):
    print("=== [1] Загрузка мультитаймфреймовых данных... ===")
    df_daily, df_4h, df_1h = load_multi_timeframe(
        symbol=symbol,
        daily_limit=365,  # последний год дневных
        h4_limit=90,      # последний ~3 месяца 4H
        h1_limit=30       # последний ~1 месяц 1H
    )

    # Проверяем нужные колонки
    for df in [df_daily, df_4h, df_1h]:
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"В DataFrame нет колонки '{col}'! Проверьте source данных.")

    print("=== [2] Поиск первичных уровней на дневном ТФ... ===")
    raw_levels = find_primary_levels(df_daily, window=5, touches_required=2)
    print(f"Найдено сырых уровней (до фильтрации): {len(raw_levels)}")

    print("=== [3] Фильтрация уровней по мультитаймфреймовым критериям... ===")
    final_levels = filter_levels(
        df_daily=df_daily,
        df_4h=df_4h,
        df_1h=df_1h,
        raw_levels=raw_levels,
        min_touches=2,
        max_age_days=60,
        atr_buffer=0.15,     # 15% от ATR
        volume_factor=1.2    # объёмы должны быть на 20% выше среднего
    )
    print(f"Уровней после фильтрации: {len(final_levels)} => {final_levels}")

    # === [4] Подготовим DataFrame для бэктеста (берём df_daily)
    df_test = df_daily.copy()
    df_test["signal"] = 0

    # Пример: если закрытие предыдущего бара <= уровень, а текущего > уровень => сигнал Buy
    idx_list = df_test.index.to_list()
    for i in range(1, len(idx_list)):
        close_prev = df_test["close"].iloc[i-1]
        close_now  = df_test["close"].iloc[i]
        for lvl in final_levels:
            if close_prev <= lvl < close_now:
                df_test.loc[idx_list[i], "signal"] = 1

    # === [5] Запускаем бэктест
    df_bt, trades, eq_curve = run_backtest(df_test, levels=final_levels, initial_capital=10000.0)

    # === [6] Рисуем результат
    plot_backtest_results(df_bt, trades, eq_curve, levels=final_levels)

    print("=== Готово! ===")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    run_pipeline(symbol="BTCUSDT", device=device)
