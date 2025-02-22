# src/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np

class MultiSymbolDataset(Dataset):
    def __init__(self, df, seq_len=50, feature_col="close", label_col="level_label"):
        self.seq_len=seq_len
        self.feature_col=feature_col
        self.label_col=label_col

        self.close_arr = df[feature_col].values
        self.symbol_arr= df["symbol_id"].values
        self.label_arr = df[label_col].values.astype(float)
        self.n=len(df)

    def __len__(self):
        return self.n-self.seq_len

    def __getitem__(self, idx):
        x_seq=self.close_arr[idx:idx+self.seq_len]
        sym_id=self.symbol_arr[idx]
        y_label=self.label_arr[idx+self.seq_len-1]

        import torch
        x_seq_t=torch.tensor(x_seq,dtype=torch.float32).view(-1,1)
        sym_id_t=torch.tensor(sym_id,dtype=torch.long)
        y_t=torch.tensor(y_label,dtype=torch.float32)
        return x_seq_t,sym_id_t,y_t
