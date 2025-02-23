# Проект `pricelevel-ml`

**pricelevel-ml** — это Python-проект для:

1. **Получения рыночных данных** (исторические и реальные потоки) по нескольким биржам:
        - Bybit (v5 Unified)
        - BingX (Swap V2)
        - Huobi/HTX (Spot)
2. **Сбора** этих данных в удобном виде (свечи OHLCV) и **определения** ключевых ценовых уровней (поддержка/сопротивление) на разных таймфреймах.
3. **Обучения** (пример) LSTM-модели на исторических данных (PyTorch), чтобы предсказывать закрытие свечи/разные сигналы.
4. **Визуализации** в **реальном времени**:
        - Через **TradingView Lightweight Charts** (или Chart.js / Highcharts) — чистый HTML+JS фронтенд, подтягивающий данные из Python-бэкенда (Flask+SocketIO/WebSocket).
        - Опционально через **Dash** — для более глубоких аналитических панелей.

## 1. Структура проекта

Пример дерева каталогов:

```bash
pricelevel-ml/
├── run.py                   # (опционально) общий entry point
├── backend/
│   ├── server.py            # Flask + SocketIO WebSocket server
│   ├── __init__.py
│   └── ...
├── core/
│   ├── __init__.py
│   ├── config.py            # Все ключи, настройки
│   ├── data_fetcher/        # Модули для историч. скачивания (REST)
│   │   ├── __init__.py
│   │   ├── bybit.py
│   │   ├── bingx.py
│   │   ├── htx.py
│   │   └── ...
│   ├── ws_clients/          # Модули для реального потока (WS)
│   │   ├── __init__.py
│   │   ├── aggregator.py    # CandleAggregator
│   │   ├── bybit_ws.py
│   │   ├── bingx_ws.py
│   │   ├── htx_ws.py
│   │   └── ...
│   ├── models/              # Логика обучения, модели
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── ...
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── main_pipeline.py
│   │   └── realtime_model.py
│   └── ...
├── webapp/
│   ├── dash_app/
│   │   ├── app.py           # Dash-приложение (аналитика)
│   ├── tradingview/         # Фронт c lightweight-charts
│   │   ├── index.html
│   │   └── app.js           # JS-код (подключ. к SocketIO)
│   ├── chartjs/             # (опцион.) Пример с chart.js
│   └── ...
├── data/
│   ├── raw/                 # csv-файлы с сырьём
│   └── processed/
├── requirements.txt
├── README.md                # Настоящий файл документации
└── ...
```

**Ключевые** элементы:

### 1.1. `core/config.py`

- Хранит API-ключи, эндпоинты:

        ```python
        BYBIT_API_KEY = ...
        BINGX_WS_ENDPOINT = "wss://open-api-swap.bingx.com/swap-market"
        HTX_WS_ENDPOINT = "wss://api.huobi.pro/ws"
        TIMEFRAME_SECONDS = 60  # или 3600
        ...
        ```

### 1.2. `core/data_fetcher/` (REST-fetch)

- `bybit.py`, `bingx.py`, `htx.py` — для **исторической** выгрузки свечей (через HTTP/REST).

### 1.3. `core/ws_clients/` (WebSocket потоки)

- `aggregator.py`: класс `CandleAggregator`, собирающий сделки → свечи.
- `bybit_ws.py`, `bingx_ws.py`, `htx_ws.py`: классы, которые подключаются к соответствующим WS-эндпоинтам, распаковывают gzip (при необходимости), парсят сделки, вызывают `aggregator.add_trade(...)`.

### 1.4. `core/models/`

- `train.py`: LSTM-модель, `train_model`, `evaluate_model`.
- При желании `inference.py`.

### 1.5. `core/pipeline/`

- `main_pipeline.py`: скрипты, где вы делаете:
        - Скачивание исторических данных,
        - Формирование датасета,
        - Обучение (train.py) + оценка,
        - Поиск уровней.
- `realtime_model.py`: пример дообучения «на лету», если хотим.

### 1.6. `webapp/tradingview/`

- `index.html`: HTML + подключение `lightweight-charts` + `socket.io`.
- `app.js`: JS, где создаётся chart, подключается к backend (`socket.io`), обрабатывает `new_candle` и обновляет график.

### 1.7. `backend/server.py`

- Flask + SocketIO, отдаёт статику (`index.html`), запускает потоки WS-клиентов (Bybit, BingX, HTX), рассылает новые закрытые свечи на фронтенд.

---

## 2. Как работает поток данных (WS)

Для **каждой** биржи (Bybit/BingX/HTX) есть отдельный класс, например `BybitWSClient`, который:

1. Подключается к нужному **WebSocket** URL.
2. Подписывается на **public trade** (либо candles, если нужно).
3. Приходит массив сделок → вызывает `aggregator.add_trade(...)`.
4. Когда aggregator закрывает свечу (timeframe истёк), формирует `{open,high,low,close,volume,...}`, optionally печатает/сохраняет.

На **бэкенде** (Flask/SocketIO) при закрытии свечи делаем `socketio.emit('new_candle', ...)`. Фронтенд ловит это событие → обновляет график TradingView/Chart.js.

---

## 3. Как получать исторические данные

Скрипты в `core/data_fetcher/bybit.py` (пример) позволяют скачивать **исторические** свечи (REST). При запуске, например:

```bash
cd pricelevel-ml
python -m core.data_fetcher.bybit
```

(или вызов из `main_pipeline.py`). В итоге сохраняете CSV в `data/raw/<symbol>_bybit_1h.csv`.

---

## 4. Поиск уровней (Liquidity Levels)

- `liquidity_levels.py` (или `cluster_levels.py`) + `incremental_fetcher.py` (для Bybit, если остался старый код).
- Алгоритм:
        1. Загружаем `df_d` (дневной),
        2. Находим 2 уровня (support, resist).
        3. Возможно уточняем 4h, 1h.
        4. Возвращаем `(d_sup, d_res, h1_sup, h1_res)`.

---

## 5. Обучение модели

В `core/models/train.py`:

- `MultiSymbolLSTM` (пример),
- `train_model_multi(...)`,
- `evaluate_model_multi(...)`.

В `core/pipeline/main_pipeline.py` вы делаете:

- Сбор исторических дневных df,
- `MultiSymbolDataset(...)`,
- `train_model_multi(...)`,
- MSE → вывод.

---

## 6. Запуск “всего” проекта

### 6.1. Шаги

1. **Установить** зависимости:

        ```bash
        pip install -r requirements.txt
        ```

2. **Запустить** бэкенд (WS + SocketIO) + фоновый сбор Bybit/BingX/HTX: В консоли увидите логи:

        ```bash
        cd pricelevel-ml
        python -m backend.server
        ```

        ```csharp
        [BybitWSClient] Subscribed => BTCUSDT
        [BingXWSClient] Opened => ...
        [HTXWSClient] ...
        * Serving Flask app 'server'
        ...
        ```

3. **Открыть** браузер `http://127.0.0.1:8000/` (или другой порт), который отдаёт `webapp/tradingview/index.html`.
4. Если всё корректно, по мере прихода сделок будут формироваться свечи, и `new_candle` будет отправляться фронту. TradingView Lightweight Charts обновится.

### 6.2. Отдельное обучение на истории

```bash
cd pricelevel-ml
python -m core.pipeline.main_pipeline
```

(где `main_pipeline.py` качает исторические данные, формирует DataFrame, вызывает `train.py`, и т.д.)

---

## 7. TradingView / Chart.js / Highcharts

В папке `webapp/tradingview/`:

- **`index.html`**: Подключает `lightweight-charts`, `socket.io`, создаёт candlestickSeries.
- При получении события `'new_candle'` (через SocketIO) → парсит JSON → `bars.push(bar)`, `series.setData(bars)`.

Если вместо TradingView вы используете Chart.js или Highcharts, логика та же: JS + WebSocket → обновляете график.

---

## 8. Частые проблемы

1. **ModuleNotFoundError**: Нужно запускать `python -m backend.server` из корневой папки (`pricelevel-ml`) + `__init__.py` файлы в папках.
2. **Connection is already closed**: Bybit/BingX иногда рвут WS, нужно автопереподключение.
3. **График пуст**: Возможно нет сделок (testnet?), timeframe слишком большой, или не рассылаете `new_candle`.
4. **MSE слишком маленькое**: Нормально для нормированных данных. Смотрите реальную (денежную) ошибку.
5. **“handler not found”** (Bybit) / “invalid start byte”** (gzip)**: Нужно правильно подписываться, распаковывать gzip.

---

## 9. Итого

Проект **pricelevel-ml** теперь:

1. Имеет **многобиржевой** сбор данных: Bybit, BingX, HTX.
2. **В реальном времени** формирует свечи, может обучать (дообучать) модель LSTM.
3. Выводит графики на **TradingView Lightweight** (HTML+JS), либо Chart.js / Highcharts, а также может предоставлять **Dash** панели.

Это даёт гибкую основу для экспериментов с детектированием ценовых уровней, обучением ML-моделей и реальным потоком рыночных данных.
