# core/config.py
import os
import datetime

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Список инструментов (линейные USDT) — можно менять
TOP_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT"
]

# ВНИМАНИЕ: для publicTrade линейного / инверсного WS в pybit v5 используем "linear"
CHANNEL_TYPE = "linear"

# Параметры для history fetch (дневные, 4ч, 1ч) — если используете incremental_fetcher
DAYS_D  = 180  
DAYS_4H = 90   
DAYS_1H = 30   

today = datetime.date.today()
delta_6m = datetime.timedelta(days=180)
start_dt = today - delta_6m
end_dt   = today

BYBIT_START_DATE = start_dt.strftime("%Y-%m-%d")
BYBIT_END_DATE   = end_dt.strftime("%Y-%m-%d")

print(f"[CONFIG] Сегодня: {today}, диапазон: {BYBIT_START_DATE}..{BYBIT_END_DATE}")

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

NUM_EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
