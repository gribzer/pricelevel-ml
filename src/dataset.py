# src/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from .config import SEQ_LEN

class MultiSymbolDataset(Dataset):
    """
    Храним все инструменты в одном DF, 
    берем close, symbol_id, level_label.
    """
    def __init__(self, df, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.close_arr  = df["close"].values
        self.symbol_arr = df["symbol_id"].values  # int
        self.label_arr  = df["level_label"].values.astype(float)
        self.n = len(df)

    def __len__(self):
        return self.n - self.seq_len

    def __getitem__(self, idx):
        seq_close = self.close_arr[idx : idx+self.seq_len]
        symbol_id = self.symbol_arr[idx]
        y_label   = self.label_arr[idx + self.seq_len - 1]

        import torch
        seq_close_t = torch.tensor(seq_close, dtype=torch.float32).view(-1,1)
        symbol_id_t = torch.tensor(symbol_id, dtype=torch.long)
        y_label_t   = torch.tensor(y_label, dtype=torch.float32)
        return seq_close_t, symbol_id_t, y_label_t
