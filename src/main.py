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

# Импортируем наши функции бэктеста
from .backtest import run_backtest, plot_backtest_results


def run_pipeline(device='cpu', threshold=0.5):
    """
    Основной pipeline:
      1) Загрузка и подготовка данных
      2) Поиск/кластеризация уровней
      3) Формирование целевых меток
      4) Создание PyTorch Dataset/DataLoader
      5) Обучение модели
      6) Оценка (evaluate_model)
      7) Инференс на новой модели + бэктест
    """

    print("=== [1] Загрузка / чтение исторических данных Bybit... ===")
    df = load_bybit_data()  # Считываем параметры из config.py (symbol, interval, etc.)

    # Убедимся, что нужные колонки есть (нижний регистр / upper-case — как в вашем коде)
    # Для примера считаем, что пришли колонки 'open','high','low','close'
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame не содержит колонку '{col}'. Колонки: {df.columns}")

    print("=== [2] Поиск локальных экстремумов и кластеризация уровней... ===")
    local_maxima, local_minima = find_local_extrema(df['close'], window=5)
    levels = cluster_extrema(local_maxima, local_minima)
    print(f"Найдено кластеров уровней: {len(levels)}")
    print("Уровни:", levels)

    print("=== [3] Создание бинарных меток (0/1) по близости к уровню... ===")
    df['level_label'] = make_binary_labels(df, levels, threshold_frac=0.001)
    print("Пример данных с метками:")
    print(df[['close', 'level_label']].head(10))

    # (Опционально) Сохраним обработанные данные
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    processed_file = os.path.join(PROCESSED_DATA_PATH, "bybit_processed.csv")
    df.to_csv(processed_file, index=False)
    print(f"Обработанные данные сохранены в {processed_file}")

    print("=== [4] Подготовка Dataset и DataLoader для обучения... ===")
    dataset = CryptoLevelsDataset(df, label_col='level_label')  # ваш класс датасета

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

    print("=== [7] Инференс на новой модели и бэктест... ===")

    ################################################################
    # Шаг 7A: Подготовим данные для инференса и получим предсказания
    ################################################################

    # Для инференса (упрощённо) предположим, что наша модель принимает
    # "окно цен" или набор фичей по df, как в train.py.
    # В реальном проекте лучше создать отдельный класс Dataset,
    # но в примере сделаем минимально.

    # Сначала переименуем open/high/low/close в заглавную форму,
    # чтобы бэктест потом корректно находил ['Open','High','Low','Close'].
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low':  'Low',
        'close': 'Close'
    }, inplace=True)

    # Допустим, модель предсказывает вероятность (0..1).
    # Ниже - простой цикл. Можно также сделать DataLoader, как в evaluate_model.
    # Считаем, что у нас есть метод model_inference(model, df, device).
    # Мы покажем inline-пример, как можно получить сигналы.

    model.eval()
    signals = [0]*len(df)  # Массив под сигналы
    with torch.no_grad():
        # Пример: идём по индексам df
        # Условно, берём "окно" SEQ_LEN=50 (как пример), передаём в модель.
        for i in range(50, len(df)):
            # Пример фичей: возьмём окно close-цен
            window = df["Close"].iloc[i-50:i].values
            # Превратим в тензор
            x_seq = torch.tensor(window, dtype=torch.float32).view(-1, 1).to(device)
            # Прогоняем через модель
            out = model(x_seq.unsqueeze(0))  # [1, seq_len, 1] => модель => [1, 1]
            p = out.item()  # Считаем, что out = вероятность

            # Если p >= threshold => сигнал на покупку
            if p >= threshold:
                signals[i] = 1  # buy

    # Запишем это в DataFrame
    df["signal"] = signals

    print(f"Число баров с signal=1 при threshold={threshold}: {sum(signals)}")

    ###################################################
    # Шаг 7B: Запуск бэктеста с нашим новым сигналом
    ###################################################
    df_bt, trades, eq_curve = run_backtest(df, levels=levels, initial_capital=10000.0)

    # Визуализируем результат
    plot_backtest_results(df_bt, trades, eq_curve, levels=levels)

    print("=== Процесс завершён: модель обучена, протестирована и прогноза на df выполнен. ===")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    run_pipeline(device=device)
