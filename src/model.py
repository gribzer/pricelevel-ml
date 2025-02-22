# src/model.py

import torch
import torch.nn as nn
from .config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE,
    EMB_DIM
)

class MultiSymbolLSTM(nn.Module):
    """
    Model: (close_seq, symbol_id) => [probability].
    symbol_id -> embedding -> concat with close -> LSTM -> FC -> sigmoid
    """
    def __init__(self, num_symbols, 
                 input_size=INPUT_SIZE,
                 hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS,
                 output_size=OUTPUT_SIZE,
                 emb_dim=EMB_DIM):
        super(MultiSymbolLSTM, self).__init__()
        self.symbol_emb = nn.Embedding(num_symbols, emb_dim)
        self.lstm = nn.LSTM(input_size+emb_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_close, symbol_id):
        """
        seq_close: (batch, seq_len, 1)
        symbol_id: (batch,)
        """
        batch_size, seq_len, _ = seq_close.shape
        emb = self.symbol_emb(symbol_id)  # (batch, emb_dim)
        emb_expanded = emb.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, emb_dim)

        combined = torch.cat((seq_close, emb_expanded), dim=2)
        out, _ = self.lstm(combined)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        out = self.sigmoid(out)
        return out
