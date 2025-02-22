# src/model.py

import torch
import torch.nn as nn

class MultiSymbolLSTM(nn.Module):
    def __init__(self, num_symbols,
                 input_size=1, hidden_size=64, num_layers=2, output_size=1, emb_dim=8):
        super(MultiSymbolLSTM,self).__init__()
        self.symbol_emb = nn.Embedding(num_symbols, emb_dim)
        self.lstm = nn.LSTM(input_size+emb_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, seq_close, sym_id):
        batch_size, seq_len, _=seq_close.shape
        emb=self.symbol_emb(sym_id)  # (batch, emb_dim)
        emb_expanded=emb.unsqueeze(1).repeat(1, seq_len,1)
        combined=torch.cat((seq_close,emb_expanded),dim=2)
        out,_=self.lstm(combined)
        last_out=out[:,-1,:]
        out=self.fc(last_out)
        out=self.sigmoid(out)
        return out
