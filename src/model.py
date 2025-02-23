"""
model.py

Модель MultiSymbolLSTM для регрессии: 
- Принимает последовательность цен (seq_close) + ID инструмента (sym_id),
- Возвращает float-значение (без ограничений [0..1]).

Если нужно использовать классификацию, можно вернуть Sigmoid, но тогда 
и лосс-функция должна быть BCELoss, а target иметь [0..1].
"""

import torch
import torch.nn as nn

class MultiSymbolLSTM(nn.Module):
    def __init__(self,
                 num_symbols,
                 input_size=1,     # количество входных фич (обычно 1, если только close)
                 hidden_size=64,
                 num_layers=2,
                 output_size=1,   # выходим одним числом (регрессия)
                 emb_dim=8):
        super(MultiSymbolLSTM, self).__init__()
        # Встраивание ID инструмента:
        self.symbol_emb = nn.Embedding(num_symbols, emb_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size + emb_dim,
            hidden_size,
            num_layers,
            batch_first=True
        )
        # Финальный слой => одно число
        self.fc = nn.Linear(hidden_size, output_size)

        # Для регрессии нам не нужна сигмоида:
        self.activation = nn.Identity()

    def forward(self, seq_close, sym_id):
        """
        seq_close: тензор (batch_size, seq_len, 1)
        sym_id:    тензор (batch_size,) с ID инструмента
        Возвращает shape (batch_size, 1) — предсказанное float-значение
        """
        batch_size, seq_len, _ = seq_close.shape

        # Получаем embedding инструмента: (batch, emb_dim)
        emb = self.symbol_emb(sym_id)  
        # Расширяем до (batch, seq_len, emb_dim)
        emb_expanded = emb.unsqueeze(1).repeat(1, seq_len, 1)

        # Склеиваем (seq_close, embedding)
        combined = torch.cat((seq_close, emb_expanded), dim=2)

        # Прогоняем через LSTM
        out, _ = self.lstm(combined)
        # Берём выход последнего шага
        last_out = out[:, -1, :]

        # Линейный слой
        out = self.fc(last_out)

        # activation = Identity, чтобы не сжимать [0..1]
        out = self.activation(out)
        return out  # shape (batch, 1)
