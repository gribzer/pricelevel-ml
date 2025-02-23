# core/pipeline/run_pretrained.py

import sys
import os
import torch
import mlflow.pytorch
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from ..config import (
    BYBIT_CATEGORY,
    BYBIT_START_DATE, BYBIT_END_DATE,
    DAILY_INTERVAL, H4_INTERVAL, H1_INTERVAL,
    TOP_SYMBOLS
)
from ...data_fetcher.data_fetcher import load_single_symbol_multi_timeframe
from ..analysis.cluster_levels import find_local_extrema, cluster_extrema, make_binary_labels
from ..analysis.multi_tf_analysis import filter_levels
from ..backtest import run_backtest, plot_backtest_results

class InferenceDataset(Dataset):
    """
    Простой датасет для инференса на одном символе, используя мультисет-модель:
    - seq_close: (seq_len, 1)
    - symbol_id: single int (один инструмент)
    """
    def __init__(self, df, seq_len=50, symbol_id=0):
        self.seq_len = seq_len
        self.symbol_id = symbol_id  # один для всего датасета
        self.prices = df["close"].values
        self.n = len(self.prices)

    def __len__(self):
        return self.n - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.prices[idx : idx+self.seq_len]
        x_seq = torch.tensor(x_seq, dtype=torch.float32).view(-1, 1)
        # symbol_id = константа
        return x_seq, torch.tensor(self.symbol_id, dtype=torch.long)

def run_pretrained_pipeline(model_uri=None, symbol="BTCUSDT", device="cpu", threshold=0.5):
    """
    Выполняет инференс для одного инструмента (symbol) на часовых данных.
    Модель - мультиинструментная (MultiSymbolLSTM), но здесь symbol_id=0 (или другое).
    """
    if not model_uri:
        raise ValueError(
            "Не задан model_uri! Пример:\n"
            "  python -m src.run_pretrained runs:/<RUN_ID>/models <symbol> [threshold]\n"
            "или\n"
            "  python -m src.run_pretrained model.pth <symbol> [threshold]"
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

    print(f"=== Загрузка данных для символа: {symbol} ===")
    # Грузим дневные и часовые данные (df_d, df_h1). При желании можно грузить df_4h для фильтра, 
    # но здесь главное - часовой для инференса.
    df_d, df_4h, df_h1 = load_single_symbol_multi_timeframe(
        symbol=symbol,
        category=BYBIT_CATEGORY,
        start=BYBIT_START_DATE,
        end=BYBIT_END_DATE,
        daily_interval=DAILY_INTERVAL,
        h4_interval=H4_INTERVAL,
        h1_interval=H1_INTERVAL
    )
    if df_d.empty or df_h1.empty:
        print(f"Нет нужных данных (daily или h1) для {symbol}.")
        return

    # Можно (необязательно) искать уровни на daily и показать их на графике
    # (как пример новой логики).
    local_maxima, local_minima = find_local_extrema(df_d["close"], window=10)
    raw_levels = cluster_extrema(local_maxima, local_minima)
    final_levels = filter_levels(df_d, df_4h, df_h1, raw_levels)

    print(f"Найдено сырых уровней: {len(raw_levels)}, после фильтра: {len(final_levels)}")

    # Сконструируем Dataset (часовик)
    # Допустим, символ_id=0 (если модель обучалась на нескольких символах, 
    # нужно подставить реальный ID, соответствующий symbol.
    # Если модель ждёт, к примеру, "BTCUSDT => id=2", то здесь нужно использовать 2.
    # Для простоты пусть будет 0, 
    # если в вашей модели order= "BTCUSDT -> 0, ETHUSDT ->1, etc." - нужно знать mapping.
    symbol_id_for_inference = 0
    seq_len = 50
    infer_dataset = InferenceDataset(df_h1, seq_len=seq_len, symbol_id=symbol_id_for_inference)
    loader = DataLoader(infer_dataset, batch_size=64, shuffle=False)

    preds = []
    # Инференс
    model.eval()
    with torch.no_grad():
        for batch_x, batch_sym_id in loader:
            batch_x      = batch_x.to(device)
            batch_sym_id = batch_sym_id.to(device)
            out = model(batch_x, batch_sym_id)  # (batch, 1)
            p = out.squeeze().cpu().numpy()
            if p.ndim == 0:
                p = [p]
            preds.extend(p.tolist())

    # Запишем preds в df_h1
    # Первые seq_len баров не имеют прогноза
    df_h1["pred_proba"] = [None]*seq_len + preds

    # Простая логика: signal=1, если p>=threshold
    df_h1["signal"] = 0
    df_h1.loc[df_h1["pred_proba"] >= threshold, "signal"] = 1

    # Обрежем последние 60 дней для наглядного графика
    cutoff_days = 60
    last_date = df_h1.index.max()
    if pd.isna(last_date):
        print("df_h1 index empty? Нечего рисовать.")
        return
    cutoff_date = last_date - pd.Timedelta(days=cutoff_days)
    df_h1_trim = df_h1[df_h1.index >= cutoff_date].copy()
    if df_h1_trim.empty:
        print("За последние 60 дней нет данных?")
        return

    # Запускаем бэктест
    df_bt, trades, eq_curve = run_backtest(
        df_h1_trim,
        levels=final_levels,
        initial_capital=10000.0
    )

    # Рисуем график
    title_str = f"{symbol} (H1, Inference) - last {cutoff_days} days, threshold={threshold}"
    plot_backtest_results(
        df_bt, trades, eq_curve,
        levels=final_levels,
        title=title_str
    )

    print("=== Готово! ===")


def main_pretrained():
    """
    Пример использования:
      python -m src.run_pretrained <model_uri> <symbol> [threshold]
    Если <symbol> не указан, берём "BTCUSDT".
    Если threshold не указан, берём 0.5.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.run_pretrained <model_uri> [symbol] [threshold]")
        sys.exit(1)

    model_uri = sys.argv[1]

    # Пытаемся считать symbol
    if len(sys.argv) >= 3:
        # Если второй аргумент содержит '.', '/', 'runs:/', 'mlflow:/' - возможно это threshold
        # но логичнее считать, что второй арг - symbol
        symbol_candidate = sys.argv[2]
        # Проверим, не float ли это
        try:
            _ = float(symbol_candidate)
            # Значит это threshold, символ не задан
            symbol = "BTCUSDT"
            threshold = float(symbol_candidate)
            # Если есть третий арг? - check
            if len(sys.argv)>=4:
                # => ... 
                threshold = float(sys.argv[3])
        except ValueError:
            # Значит это символ
            symbol = symbol_candidate
            # threshold
            if len(sys.argv)>=4:
                threshold = float(sys.argv[3])
            else:
                threshold = 0.5
    else:
        # Нет второго аргумента
        symbol = "BTCUSDT"
        threshold = 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_pretrained_pipeline(model_uri=model_uri,
                            symbol=symbol,
                            device=device,
                            threshold=threshold)


if __name__ == "__main__":
    main_pretrained()
