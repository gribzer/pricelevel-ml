# core/pipeline/realtime_model.py
import torch
import pandas as pd
from core.models.train import build_new_model

class RealtimeModelTrainer:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.new_data_buffer = []

    def load_or_init_model(self):
        try:
            self.model = torch.load("my_model.pth", map_location=self.device)
            print("[RealtimeModelTrainer] Модель загружена из my_model.pth")
        except:
            print("[RealtimeModelTrainer] Файл не найден => создаём новую модель")
            self.model = build_new_model(num_symbols=1, device=self.device)

    def train_on_new_candle(self, symbol, candle: dict):
        """
        candle = { start, open, high, low, close, volume }
        """
        df = pd.DataFrame([{
            "timestamp": candle["start"],
            "open":  candle["open"],
            "high":  candle["high"],
            "low":   candle["low"],
            "close": candle["close"],
            "volume":candle["volume"],
            "symbol": symbol
        }])
        self.new_data_buffer.append(df)

        # Пример: дообучаемся каждые 24 свечи
        if len(self.new_data_buffer) >= 24:
            big_df = pd.concat(self.new_data_buffer)
            self.new_data_buffer = []

            print(f"[RealtimeModelTrainer] Дообучаемся на {len(big_df)} записях ...")
            # Логика mini-batch
            torch.save(self.model, "my_model.pth")
