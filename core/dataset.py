# core/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset

class MultiSymbolDataset(Dataset):
    def __init__(self, df,
                 seq_len=50,
                 feature_col="close",
                 label_col="close"):
        self.seq_len = seq_len
        self.feature_col = feature_col
        self.label_col   = label_col

        if "symbol_id" not in df.columns:
            raise ValueError("Отсутствует symbol_id в DataFrame")

        self.symbol_arr = df["symbol_id"].values

        arr = df[feature_col].values.astype(np.float32).reshape(-1,1)
        self.min_ = arr.min()
        self.max_ = arr.max()
        self.feature_arr = (arr - self.min_)/(self.max_-self.min_+1e-9)

        lb = df[label_col].values.astype(np.float32).reshape(-1,1)
        self.lb_min = lb.min()
        self.lb_max = lb.max()
        self.label_arr= (lb - self.lb_min)/(self.lb_max-self.lb_min+1e-9)

        self.n = len(df)

    def __len__(self):
        return self.n - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.feature_arr[idx : idx+self.seq_len]   # (seq_len,1)
        sym_id= self.symbol_arr[idx]
        y_val = self.label_arr[idx + self.seq_len -1]      # (1,)

        import torch
        x_seq_t= torch.tensor(x_seq, dtype=torch.float32)  # (seq_len,1)
        sym_id_t= torch.tensor(sym_id, dtype=torch.long)
        y_t= torch.tensor(y_val, dtype=torch.float32)       # shape(1,)

        return x_seq_t, sym_id_t, y_t

    def inverse_transform_label(self, y_value):
        return y_value*(self.lb_max - self.lb_min)+self.lb_min
