# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from .model import MultiSymbolLSTM

def train_model_multi(
    train_dataset,
    val_dataset,
    num_symbols,
    device='cpu',
    learning_rate=0.0003,
    num_epochs=50,
    batch_size=64
):
    model=MultiSymbolLSTM(num_symbols=num_symbols).to(device)
    criterion=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    with mlflow.start_run():
        mlflow.log_param("learning_rate",learning_rate)
        mlflow.log_param("batch_size",batch_size)
        mlflow.log_param("epochs",num_epochs)
        mlflow.log_param("num_symbols",num_symbols)

        for epoch in range(num_epochs):
            model.train()
            total_loss=0.0
            for x_seq, sym_id, y in train_loader:
                x_seq=x_seq.to(device)
                sym_id=sym_id.to(device)
                y=y.to(device)

                optimizer.zero_grad()
                outputs=model(x_seq, sym_id)
                loss=criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()

                total_loss+=loss.item()

            avg_loss=total_loss/len(train_loader)
            print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss",avg_loss,step=epoch)

            if val_loader and (epoch+1)%5==0:
                acc, prec, rec, f1=evaluate_model_multi(model,val_loader,device)
                mlflow.log_metric("val_accuracy",acc,step=epoch)
                mlflow.log_metric("val_precision",prec,step=epoch)
                mlflow.log_metric("val_recall",rec,step=epoch)
                mlflow.log_metric("val_f1",f1,step=epoch)

        mlflow.pytorch.log_model(model,artifact_path="models")
        print("Модель сохранена в MLflow.")
    return model

def evaluate_model_multi(model, data_loader, device='cpu'):
    model.eval()
    all_preds=[]
    all_labels=[]
    with torch.no_grad():
        for x_seq, sym_id, y in data_loader:
            x_seq=x_seq.to(device)
            sym_id=sym_id.to(device)
            y=y.to(device)
            outputs=model(x_seq, sym_id)
            preds=(outputs.squeeze()>=0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds=np.array(all_preds)
    all_labels=np.array(all_labels)
    accuracy=(all_preds==all_labels).mean()
    precision=precision_score(all_labels,all_preds,zero_division=0)
    recall=recall_score(all_labels,all_preds,zero_division=0)
    f1=f1_score(all_labels,all_preds,zero_division=0)
    print(f"Val Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return accuracy, precision, recall, f1
