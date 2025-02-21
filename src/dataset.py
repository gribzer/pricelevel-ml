# src/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from .config import SEQ_LEN, INPUT_SIZE

class CryptoLevelsDataset(Dataset):
    """
    Формируем выборку (X, y):
    X - последовательность длиной SEQ_LEN,
    y - метка 0/1, характеризующая последнюю точку последовательности.
    """
    def __init__(self, df, label_col='level_label', seq_len=SEQ_LEN, input_size=INPUT_SIZE):
        # Предполагаем, что df содержит столбец 'close' (и др.) + столбец label_col (метки)
        self.seq_len = seq_len

        # Можно взять несколько колонок. Предположим, что используем только 'close'.
        # Но можно расширять: ['close', 'volume', ...]
        self.feature_cols = ['close']
        data_np = df[self.feature_cols].values
        # Нормализация (упрощённая) - можно заменить на более сложный scaler
        mean = data_np.mean(axis=0)
        std = data_np.std(axis=0)
        self.data = (data_np - mean) / (std + 1e-8)

        # Метки
        self.labels = df[label_col].values.astype(int)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.data[idx : idx+self.seq_len]
        y_label = self.labels[idx+self.seq_len-1]  # метка на последнем баре
        # Превращаем в тензоры
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        # Вход нужно иметь форму (seq_len, input_size)
        return x_seq, y_label
