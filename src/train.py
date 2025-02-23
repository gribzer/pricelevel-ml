# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MultiSymbolLSTM(nn.Module):
    def __init__(self, num_symbols,
                 input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, emb_dim=8, dropout=0.2):
        super().__init__()
        self.symbol_emb= nn.Embedding(num_symbols, emb_dim)
        self.lstm= nn.LSTM(input_size+emb_dim, hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout)
        self.fc= nn.Linear(hidden_size, output_size)

    def forward(self, x_seq, sym_id):
        """
        x_seq: (batch, seq_len, 1)
        sym_id: (batch,)
        out => (batch,1)
        """
        b, s, _= x_seq.shape
        emb= self.symbol_emb(sym_id)   # (b, emb_dim)
        emb_exp= emb.unsqueeze(1).repeat(1, s, 1) # => (b, s, emb_dim)
        combined= torch.cat((x_seq, emb_exp), dim=2) # (b, s, input_size+emb_dim)
        out, _= self.lstm(combined)
        last_out= out[:,-1,:]   # (b, hidden_size)
        y= self.fc(last_out)    # (b,1)
        return y


def train_model_multi(train_ds, val_ds,
                      num_symbols,
                      device='cpu',
                      learning_rate=0.0003,
                      num_epochs=50,
                      batch_size=64):
    model= MultiSymbolLSTM(num_symbols,
                           input_size=1, hidden_size=128, num_layers=2,
                           output_size=1, emb_dim=8, dropout=0.2).to(device)
    criterion= nn.MSELoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader= DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_loss=0; count=0
        for x_seq, sym_id, y in train_loader:
            x_seq= x_seq.to(device)
            sym_id= sym_id.to(device)
            y= y.to(device)               # shape (b,1)
            optimizer.zero_grad()
            out= model(x_seq, sym_id)     # (b,1)
            loss= criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()* len(y)
            count += len(y)

        train_mse= total_loss/count if count>0 else 0
        val_mse= evaluate_model_multi(model,val_loader,device)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train MSE: {train_mse:.4f}, Val MSE={val_mse:.4f}")

    return model

def evaluate_model_multi(model, data_loader, device='cpu'):
    model.eval()
    criterion= nn.MSELoss(reduction='sum')
    total=0; n=0
    with torch.no_grad():
        for x_seq, sym_id, y in data_loader:
            x_seq= x_seq.to(device)
            sym_id= sym_id.to(device)
            y= y.to(device)
            out= model(x_seq, sym_id)
            loss= criterion(out, y)
            total+= loss.item()
            n+= len(y)
    if n==0: return 0
    mse= total/n
    return mse
