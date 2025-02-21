# src/config.py

import os

# ---------------------------
# Параметры для Bybit
# ---------------------------
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Пример тикера и настроек для Bybit
BYBIT_SYMBOL = "BTCUSDT"
BYBIT_INTERVAL = 60  # в минутах (доступные варианты: 1, 3, 5, 15, 30, 60, 240, 720, 1440 и т.д.)
BYBIT_START_DATE = "2022-01-01"
BYBIT_END_DATE   = "2022-12-31"

# ---------------------------
# Пути к данным
# ---------------------------
RAW_DATA_PATH = "data/raw"             # где сохранять сырые выгрузки
PROCESSED_DATA_PATH = "data/processed" # куда класть Parquet/HDF5

# ---------------------------
# Параметры для кластеризации
# ---------------------------
EPS_PERCENT = 0.005   # 0.5% от средней цены (для DBSCAN)
MIN_SAMPLES = 2

# ---------------------------
# Параметры модели
# ---------------------------
INPUT_SIZE = 1    # кол-во признаков (например, close price)
HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 64
SEQ_LEN = 50      # длина окна входных данных (количество баров в последовательности)
