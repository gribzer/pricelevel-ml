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
    if not model_uri:
        raise ValueError(
            "Не задан model_uri! Пример:\n"
            "  python -m src.run_pretrained runs:/<RUN_ID>/models [threshold]\n"
            "или\n"
            "  python -m src.run_pretrained model.pth [threshold]"
        )

    print(f"=== Загрузка модели: {model_uri} ===")
    if model_uri.startswith("runs:/") or model_uri.startswith("mlflow:/"):
        print("MLflow URI => mlflow.pytorch.load_model")
        model = mlflow.pytorch.load_model(model_uri)
    else:
        print("Локальный файл => torch.load(...)")
        model = torch.load(model_uri, map_location=device)

    model.to(device)
    model.eval()

    # Загрузка данных
    df = load_bybit_data()
    # Убедимся, что есть нужные колонки
    for col in ["open","high","low","close"]:
        if col not in df.columns:
            raise ValueError(f"Нет колонки {col}")

    # Инференс
    class SimplePriceDataset(Dataset):
        def __init__(self, df, seq_len=50):
            self.seq_len = seq_len
            self.prices = df["close"].values
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
            out = model(batch_x)
            p = out.squeeze().cpu().numpy()
            if p.ndim == 0:
                p = [p]
            preds.extend(p.tolist())

    df["pred_proba"] = [None]*seq_len + preds
    df["signal"] = 0
    df.loc[df["pred_proba"] >= threshold, "signal"] = 1

    df_bt, trades, eq_curve = run_backtest(df, initial_capital=10000.0)
    plot_backtest_results(df_bt, trades, eq_curve)

def main_pretrained():
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
