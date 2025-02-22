# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import mlflow
import mlflow.pytorch

from .config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
)
from .model import MultiSymbolLSTM

def train_model_multi(train_dataset, val_dataset, num_symbols, device='cpu'):
    model = MultiSymbolLSTM(num_symbols=num_symbols).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with mlflow.start_run():
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0

            for seq_close, symbol_id, label in train_loader:
                seq_close = seq_close.to(device)   # (batch, seq_len, 1)
                symbol_id = symbol_id.to(device)   # (batch,)
                label = label.to(device)           # (batch,)

                optimizer.zero_grad()
                outputs = model(seq_close, symbol_id)
                loss = criterion(outputs.squeeze(), label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if val_loader and (epoch+1) % 5 == 0:
                accuracy, precision, recall, f1 = evaluate_model_multi(model, val_loader, device=device)
                mlflow.log_metric("val_accuracy", accuracy, step=epoch)
                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_recall", recall, step=epoch)
                mlflow.log_metric("val_f1", f1, step=epoch)

        mlflow.pytorch.log_model(model, artifact_path="models")
        print("Модель сохранена в MLflow.")

    return model


def evaluate_model_multi(model, data_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seq_close, symbol_id, label in data_loader:
            seq_close = seq_close.to(device)
            symbol_id = symbol_id.to(device)
            label = label.to(device)

            outputs = model(seq_close, symbol_id)
            preds = (outputs.squeeze() >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"Val Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1
