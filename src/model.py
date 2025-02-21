# src/model.py

import torch
import torch.nn as nn
from .config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE

class LevelDetectionModel(nn.Module):
    def __init__(self, 
                 input_size=INPUT_SIZE, 
                 hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, 
                 output_size=OUTPUT_SIZE):
        super(LevelDetectionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # LSTM выход: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)
        # Берём последнюю временную позицию
        last_out = out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_out)   # (batch, output_size)
        out = self.sigmoid(out)   # вероятность
        return out
