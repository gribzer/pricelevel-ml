# src/config.py

import os

# -------------------------------------------------
# Параметры для Bybit (Unified v5)
# -------------------------------------------------
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Пример тикера и настроек (USDT-перпетуал)
BYBIT_SYMBOL = "BTCUSDT"

# Bybit v5 требует указать тип контракта / рынка:
# - "linear"   => USDT-перпетуалы
# - "inverse"  => inverse-перпетуалы (BTCUSD, ETHUSD)
# - "spot"     => спот-рынок
# Ниже для BTCUSDT USDT-PERP выберем "linear"
BYBIT_CATEGORY = "spot"

# Интервал свечей (строка в минутах):
# Bybit v5 поддерживает: "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M" и т.д.
BYBIT_INTERVAL = "60"  # 60 мин = 1 час

BYBIT_START_DATE = "2025-01-01"
BYBIT_END_DATE   = "2025-02-20"

# -------------------------------------------------
# Пути к данным
# -------------------------------------------------
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# -------------------------------------------------
# Параметры для кластеризации (пример)
# -------------------------------------------------
EPS_PERCENT = 0.005   # 0.5% от средней цены (для DBSCAN)
MIN_SAMPLES = 2

# -------------------------------------------------
# Параметры модели
# -------------------------------------------------
INPUT_SIZE = 1    # количество признаков (close price)
HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 64
SEQ_LEN = 50
