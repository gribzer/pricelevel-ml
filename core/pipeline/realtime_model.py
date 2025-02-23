# core/pipeline/realtime_model.py
import torch
import pandas as pd

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
            # создайте/инициализируйте self.model
            self.model = None

    def train_on_new_candle(self, symbol, candle: dict):
        # candle = {"start":..., "open":..., "high":..., "low":..., "close":..., "volume":...}
        df = pd.DataFrame([{
            "timestamp": candle["start"],
            "open": candle["open"],
            "high": candle["high"],
            "low":  candle["low"],
            "close":candle["close"],
            "volume":candle["volume"],
            "symbol":symbol
        }])
        self.new_data_buffer.append(df)

        if len(self.new_data_buffer) >= 24:
            big_df = pd.concat(self.new_data_buffer)
            self.new_data_buffer = []
            print(f"[RealtimeModelTrainer] Дообучение на {len(big_df)} свечах ...")
            # ... mini-batch train logic
            if self.model:
                torch.save(self.model, "my_model.pth")
