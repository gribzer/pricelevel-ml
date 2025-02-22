# src/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from .config import SEQ_LEN

class MultiSymbolDataset(Dataset):
    """
    Допустим, у нас большой df со всеми инструментами,
    где есть колонки:
      - close
      - symbol_id
      - level_label
    seq_len: длина последовательности.
    """
    def __init__(self, df, seq_len=SEQ_LEN):
        self.seq_len = seq_len

        # Предположим, что df уже более-менее отнормирован или 
        # хотя бы 'close' в одном масштабе. Если нет, нужно либо
        # делать groupby(symbol_id) и нормировать отдельно, либо 
        # global minmax/meanstd...
        
        self.close_arr = df["close"].values
        self.symbol_arr = df["symbol_id"].values  # int
        self.label_arr = df["level_label"].values.astype(float)
        
        self.n = len(df)

    def __len__(self):
        return self.n - self.seq_len

    def __getitem__(self, idx):
        # seq_len от idx до idx+seq_len
        seq_close = self.close_arr[idx : idx+self.seq_len]
        y_label = self.label_arr[idx + self.seq_len - 1]

        # берем symbol_id первого бара (или последнего — на ваше усмотрение)
        symbol_id = self.symbol_arr[idx]

        # Тензоры
        seq_close_t = torch.tensor(seq_close, dtype=torch.float32).view(-1, 1)
        symbol_id_t = torch.tensor(symbol_id, dtype=torch.long)
        y_label_t = torch.tensor(y_label, dtype=torch.float32)

        return seq_close_t, symbol_id_t, y_label_t
