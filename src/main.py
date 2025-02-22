# src/main.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split

from .config import (
    PROCESSED_DATA_PATH,
    SEQ_LEN,
    BATCH_SIZE,
    NUM_EPOCHS
)
from .data_fetcher import load_bybit_data
from .cluster_levels import (
    find_local_extrema,
    cluster_extrema,
    make_binary_labels
)
from .dataset import CryptoLevelsDataset
from .train import train_model, evaluate_model

# Импортируем наши функции бэктеста (добавьте этот импорт)
from .backtest import run_backtest, plot_backtest_results

def run_pipeline(device='cpu'):
    print("=== [1] Загрузка / чтение исторических данных Bybit... ===")
    df = load_bybit_data()  # Считываем параметры из config.py (symbol, interval, etc.)

    # Убедимся, что нужные колонки есть: [close] и т.п.
    # (примечание: в файлах может быть upper case. Убедитесь, что код полностью согласован)
    if 'close' not in df.columns:
        raise ValueError("DataFrame не содержит колонку 'close'.")

    print("=== [2] Поиск локальных экстремумов и кластеризация уровней... ===")
    local_maxima, local_minima = find_local_extrema(df['close'], window=5)
    levels = cluster_extrema(local_maxima, local_minima)
    print(f"Найдено кластеров уровней: {len(levels)}")
    print("Уровни:", levels)

    print("=== [3] Создание бинарных меток (0/1) по близости к уровню... ===")
    df['level_label'] = make_binary_labels(df, levels, threshold_frac=0.001)

    print("Пример данных с метками:")
    print(df[['close', 'level_label']].head(10))

    # Сохраним обработанные данные (опционально, в CSV или Parquet)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    processed_file = os.path.join(PROCESSED_DATA_PATH, "bybit_processed.csv")
    df.to_csv(processed_file)
    print(f"Обработанные данные сохранены в {processed_file}")

    print("=== [4] Подготовка Dataset и DataLoader для обучения... ===")
    dataset = CryptoLevelsDataset(df, label_col='level_label')

    # Разделим датасет: 80% train, 10% val, 10% test
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset size = {dataset_size}, train={train_size}, val={val_size}, test={test_size}")

    print("=== [5] Обучение модели... ===")
    model = train_model(train_data, val_data, device=device)

    print("=== [6] Тестирование модели... ===")
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    evaluate_model(model, test_loader, device=device, prefix="Test")

    print("=== [7] Бэктест и визуализация... ===")
    # Предположим, вы хотите использовать колонку 'level_label' или какую-то 'signal' 
    # для входа/выхода. Для примера: если level_label == 1 => Buy, иначе => 0
    # (Это очень упрощённая логика, подставьте реальную стратегию.)

    # Превратим df (после обучения) в формат c OHLC: ['Open','High','Low','Close'] 
    # Для наглядности — нужно в верхней части pipeline conv lower-case. 
    # Здесь же, предполагаем, что есть такие колонки:
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    }, inplace=True)

    # Создадим столбец 'signal' (Buy=+1, Sell=-1, 0=hold)
    df["signal"] = 0
    df.loc[df["level_label"] == 1, "signal"] = +1
    # (Это, конечно, очень наивная логика. Лучше применить реальную 
    # систему сигналов, завязанную на модели. Здесь для примера.)

    # Запускаем бэктест (см. backtest.py)
    from .backtest import run_backtest, plot_backtest_results
    df_bt, trades, equity_curve = run_backtest(df, levels=levels, initial_capital=10000.0)

    # Визуализируем
    plot_backtest_results(df_bt, trades, equity_curve, levels=levels)

    print("=== Процесс завершён. Модель обучена, протестирована и выполнен бэктест. ===")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    run_pipeline(device=device)
