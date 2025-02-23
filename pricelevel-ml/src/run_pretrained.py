# src/run_pretrained.py

import sys
import os

import torch
import mlflow.pytorch
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from .data_fetcher import load_bybit_data
from .backtest import run_backtest, plot_backtest_results

def run_pretrained_pipeline(model_uri=None, device="cpu", threshold=0.5):
    """
    Запускает инференс и бэктест уже обученной модели, без повторного обучения.

    model_uri: 
      - Может быть локальным путём к .pth-файлу (e.g. 'model.pth')
      - Или URI MLflow (e.g. 'runs:/<RUN_ID>/models').

    threshold:
      - Если выход модели — вероятность (0..1), 
        то при p > threshold считаем signal=+1, иначе 0 (упрощённо).
    """

    # 1. Проверка аргумента
    if not model_uri:
        raise ValueError("Не задан model_uri! Пример вызова:\n"
                         "  python -m src.run_pretrained runs:/<RUN_ID>/models [threshold]\n"
                         "или\n"
                         "  python -m src.run_pretrained model.pth [threshold]")

    # 2. Загрузка модели
    print(f"=== Загрузка модели: {model_uri} ===")
    # Определяем, MLflow ли это, или локальный путь
    if model_uri.startswith("runs:/") or model_uri.startswith("mlflow:/"):
        print("MLflow URI обнаружен => mlflow.pytorch.load_model(...)")
        model = mlflow.pytorch.load_model(model_uri)
    else:
        print("Предполагаем локальный файл => torch.load(...)")
        # Вариант: модель может быть просто state_dict (обычно .pth),
        # или целая сохранённая модель. Нужно знать, как вы сохраняли.
        # Если это state_dict, придётся сначала создать архитектуру, а потом load_state_dict.
        # Если это целая модель (torch.save(model)), можно torch.load(...).
        # Допустим, вы сохраняли через torch.save(model).
        model = torch.load(model_uri, map_location=device)

    model.to(device)
    model.eval()
    print("Модель загружена и переведена в eval-режим.")

    # 3. Загрузка исторических данных
    print("=== Загрузка исторических данных (Bybit) ===")
    df = load_bybit_data()  # Предполагается, что в df колонки: open, high, low, close

    # Приведём названия к TitleCase
    if "open" in df.columns:
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        }, inplace=True)

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' не найдена в df! Есть колонки: {df.columns.tolist()}")

    # 4. Генерация сигналов путём инференса
    #    Допустим, модель принимает массив цен [seq_len, 1] => выдаёт p
    #    Сформируем DataLoader
    class SimplePriceDataset(Dataset):
        def __init__(self, df, seq_len=50):
            self.seq_len = seq_len
            self.prices = df["Close"].values
            self.n = len(self.prices)
        def __len__(self):
            return self.n - self.seq_len
        def __getitem__(self, idx):
            x_seq = self.prices[idx : idx+self.seq_len]
            x_seq = torch.tensor(x_seq, dtype=torch.float32).view(-1,1)
            return x_seq

    seq_len = 50
    dataset = SimplePriceDataset(df, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)  # [batch, 1], e.g. sigmoid
            p = out.squeeze().cpu().numpy()
            if p.ndim == 0:
                # Если batch=1, out.squeeze() будет скаляром
                p = [p]
            preds.extend(p.tolist())

    # Первые seq_len баров не имеют прогноза
    df["pred_proba"] = [None]*seq_len + preds
    # Простая логика: signal=+1, если p>threshold, иначе 0
    df["signal"] = 0
    df.loc[df["pred_proba"]>threshold, "signal"] = 1

    print(f"=== Генерация сигналов: p>={threshold} => Buy ===")
    print(f"Итоговое кол-во сигналов Buy: {(df['signal']==1).sum()}")

    # 5. Запускаем бэктест (run_backtest) и рисуем результат
    df_bt, trades, eq_curve = run_backtest(df, initial_capital=10000.0)
    plot_backtest_results(df_bt, trades, eq_curve)

    print("=== Бэктест завершён. График отобразится в Plotly-окне ===")


def main_pretrained():
    # Пример запуска:
    #   python -m src.run_pretrained runs:/f0a04f94b9ba4f9b8917a4cfaacdef0b/models 0.5
    # или
    #   python -m src.run_pretrained model.pth
    if len(sys.argv) < 2:
        print("Usage: python -m src.run_pretrained <model_uri> [threshold]")
        sys.exit(1)

    model_uri = sys.argv[1]
    threshold = 0.5
    if len(sys.argv) >= 3:
        threshold = float(sys.argv[2])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_pretrained_pipeline(model_uri=model_uri, device=device, threshold=threshold)


if __name__ == "__main__":
    main_pretrained()
