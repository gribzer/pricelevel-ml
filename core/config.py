# core/config.py
import os

# === API keys ===
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

BINGX_API_KEY = os.getenv("BINGX_API_KEY", "")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET", "")

HTX_API_KEY = os.getenv("HTX_API_KEY", "")
HTX_API_SECRET = os.getenv("HTX_API_SECRET", "")

# === Endpoints (WS) ===
BYBIT_CHANNEL_TYPE = "linear"  # или "inverse"
IS_BYBIT_TESTNET = False
BINGX_WS_ENDPOINT = "wss://open-api-swap.bingx.com/swap-market"
HTX_WS_ENDPOINT   = "wss://api.huobi.pro/ws"

# === Other settings ===
TIMEFRAME_SECONDS = 60  # пусть по умолчанию 1m
