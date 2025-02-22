# src/config.py

import os

# -------------------------------------------------
# Параметры для Bybit (Unified v5)
# -------------------------------------------------
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Топ-10 крипто (пример)
TOP_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "AVAXUSDT",
    "SUIUSDT",
    "TONUSDT",
]

BYBIT_CATEGORY = "spot"
# Укажем разные интервалы: D, 4h (240), 1h (60)
DAILY_INTERVAL = "D"
H4_INTERVAL    = "240"
H1_INTERVAL    = "60"

BYBIT_START_DATE = "2023-01-01"
BYBIT_END_DATE   = "2025-01-01"

# -------------------------------------------------
# Пути к данным
# -------------------------------------------------
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# -------------------------------------------------
# Параметры для кластеризации/поиска уровней
# -------------------------------------------------
EPS_PERCENT = 0.005    
MIN_SAMPLES = 4       
WINDOW_SIZE = 12     
MIN_TOUCHES_FILTER = 4
MAX_AGE_DAYS = 90     
ATR_BUFFER = 0.20     
VOLUME_FACTOR = 1.3   

# -------------------------------------------------
# Параметры модели
# -------------------------------------------------
INPUT_SIZE = 1
HIDDEN_SIZE = 128
NUM_LAYERS = 3
OUTPUT_SIZE = 1
LEARNING_RATE = 0.0003
NUM_EPOCHS = 80
BATCH_SIZE = 64
SEQ_LEN = 80
EMB_DIM = 8  # размер embedding для symbol_id (см. модель)
