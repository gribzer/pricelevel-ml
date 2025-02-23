# config.py

import os
import datetime

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# ================================
# Последние 6 месяцев
# ================================

TOP_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LTCUSDT",
]

BYBIT_CATEGORY = "linear"  # или "linear" для фьючерсов

# Периоды загрузки:
DAYS_D  = 180  # дневной  ~6 мес
DAYS_4H = 90   # 4ч      ~3 мес
DAYS_1H = 30   # часовой ~1 мес

# == Автоматически берём последние 6 месяцев относительно "сегодня"
today = datetime.date.today()
delta_6m = datetime.timedelta(days=180)
start_dt = today - delta_6m
end_dt   = today

BYBIT_START_DATE = start_dt.strftime("%Y-%m-%d")
BYBIT_END_DATE   = end_dt.strftime("%Y-%m-%d")

print(f"[CONFIG] Сегодня: {today}, диапазон: {BYBIT_START_DATE}..{BYBIT_END_DATE}")

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# Периоды в днях:
DAYS_D  = 180  # дневной
DAYS_4H = 90   # 4h
DAYS_1H = 30   # 1h

# Параметры обучения:
NUM_EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.0003