# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import mlflow  # <-- добавляем mlflow
import mlflow.pytorch  # для сохранения PyTorch-моделей

from .config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
)
from .model import LevelDetectionModel

def train_model(train_dataset, val_dataset=None, device='cpu'):
    """
    Запускает обучение модели, логирует результаты в MLflow.
    """
    model = LevelDetectionModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Инициируем run в MLflow
    with mlflow.start_run():
        # Логируем гиперпараметры
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_loss:.4f}")
            # Логируем метрику loss в MLflow
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Валидация (если есть)
            if val_loader and (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
                accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device=device, prefix="Val")
                # Логируем метрики в MLflow
                mlflow.log_metric("val_accuracy",  accuracy,  step=epoch)
                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_recall",    recall,    step=epoch)
                mlflow.log_metric("val_f1",        f1,        step=epoch)

        # По завершении обучения сохраним модель как артефакт
        mlflow.pytorch.log_model(model, artifact_path="models")
        print("Модель сохранена в MLflow.")

    return model

def evaluate_model(model, data_loader, device='cpu', prefix="Test"):
    """
    Вычисление метрик на заданном data_loader (accuracy, precision, recall, f1).
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            preds = (outputs.squeeze() >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"{prefix} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1
