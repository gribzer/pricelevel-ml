# Проект pricelevel-ml

pricelevel-ml — это Python-проект для:

- Загрузки исторических данных (свечей OHLCV) по ряду инструментов (крипто) с помощью Bybit API (Unified v5).
- Определения ключевых ценовых уровней (поддержка/сопротивление) на разных таймфреймах (D, 4h, 1h).
- Обучения простой LSTM-модели (пример) на данных дневного таймфрейма (или другом), чтобы предсказывать (в демо-версии) close.
- Отрисовки интерактивных графиков (Plotly) с отмеченными уровнями.

Проект рассчитан на демонстрацию подхода:

- инкрементальная подгрузка данных
- нахождение «двух» ближайших к цене уровней (поддержка, сопротивление)
- простая модель (LSTM) на PyTorch

## 1. Структура проекта

Дерево:

```plaintext
pricelevel-ml/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── incremental_fetcher.py
│   ├── liquidity_levels.py
│   ├── cluster_levels.py     (при необходимости)
│   ├── dataset.py
│   ├── train.py
│   ├── main.py               (главный entry point)
│   ├── ...
├── data/
│   ├── raw/                  (csv-файлы с сырыми свечами)
│   └── processed/
├── requirements.txt
├── README.md (или docs)
└── ...
```

### Основные модули

#### config.py

Содержит глобальные параметры:

- BYBIT_CATEGORY ("spot" или "linear")
- TOP_SYMBOLS (список: "BTCUSDT", "ETHUSDT" ...)
- Периоды: DAYS_D (сколько дней назад для дневок), DAYS_4H (для 4h), DAYS_1H (для 1h)
- Параметры обучения: NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE и др.

#### incremental_fetcher.py

Модуль, в котором реализована логика загрузки свечей в CSV-файл. Главная функция — `load_kline_incremental(...)`, которая:

- Проверяет, есть ли уже CSV
- Если нет, скачивает всё (`_fetch_bybit_kline_full`)
- Если есть, догружает недостающее справа,
- Возвращает DataFrame за запрошенный период.

Здесь же находится `_fetch_bybit_kline_full` — функция «глубокой» выгрузки с помощью Bybit v5. Важно:

- Обрабатывает category=="spot" и category=="linear"
- Парсит ответ массива [openTime, open, high, low, close, volume]
- Защита от зацикливания, если last_ts не растёт
- Складывает всё в DataFrame (open_time → DatetimeIndex).

#### liquidity_levels.py

Логика, объединяющая:

- вызов `load_kline_incremental(symbol, category, interval, ...)` для дневного / 4h / 1h,
- функции `find_liquid_levels` (которую вы можете взять из `find_liquid_core` или `cluster_levels`) для определения двух уровней (support/resistance).

#### dataset.py

Определяет класс `MultiSymbolDataset`, который:

- Формирует выборку (seq_len исторических цен → предсказываем target),
- Нормирует признаки (min/max) (необязательно),
- Возвращает (x_seq, symbol_id, y).

#### train.py

- Модель `MultiSymbolLSTM` (LSTM + Embedding символа),
- Функции `train_model_multi`, `evaluate_model_multi`.

#### main.py

Главный entry point. Логика:

- Для каждого символа из TOP_SYMBOLS:
        - Запрашивает дневные, 4h, 1h данные (последние DAYS_D, DAYS_4H, DAYS_1H)
        - Ищет уровни (два на дневке, при желании ещё два на 1h)
        - Сохраняет всё в общий список
- Склеивает все дневные df_d → df_big, создаёт `MultiSymbolDataset`,
- Обучает LSTM,
- Отображает графики 1h (plotly) c 2 дневными уровнями + 2 (например) h1-уровнями.

## 2. Установка и запуск

Склонировать репозиторий:

```bash
git clone https://github.com/gribzer/pricelevel-ml.git
cd pricelevel-ml
```

Установить зависимости (в venv):

```bash
pip install -r requirements.txt
```

Настроить переменные (при необходимости) в `src/config.py`, например:

```python
BYBIT_CATEGORY = "linear"
TOP_SYMBOLS = ["BTCUSDT","ETHUSDT",...]
```

Запустить:

```bash
python -m src.main
```

В консоли будут логи:

- Загрузка данных (если csv нет),
- Поиск уровней,
- Обучение модели (эпохи),
- Вывод финального MSE,
- Появятся интерактивные окошки Plotly c графиками.

## 3. Пояснения к загрузке данных

### _fetch_bybit_kline_full

- Делает `session.get_kline(category=..., symbol=..., interval=..., start=..., end=..., limit=...)`.
- Парсит ответ, где list содержит массив шести полей: [openTime, open, high, low, close, volume], актуально для spot и linear.
- openTime (в мс) идёт в индекс DataFrame.
- Если Bybit возвращает 181 бар (примерно 6 месяцев для дневок) и last_ts не растёт — значит более старых данных нет, или API вернул ограниченный набор.
- Защита от бесконечного цикла: если last_ts не увеличивается, break.

Почему last_ts может не расти?

- Bybit может «обрезать» исторические данные. Например, для дневных USDT-перп может возвращать только полгода.

## 4. Поиск уровней

Метод `find_liquid_levels(df)` (в `liquidity_levels.py` или `cluster_levels.py`):

- Находит локальные экстремумы,
- Считает «очки» уровня (кол-во касаний, ложные пробои, объём и т.д.),
- Отбирает 2 ближайших (support, resistance).

Пример кода (упрощённо):

```python
def find_liquid_levels(df):
                # 1) daily_atr= ...
                # 2) собираем raw_levels (экстремумы)
                # 3) score_levels -> оставляем top
                # 4) pick_best_two_levels -> (support, resist)
                return support_level, resistance_level
```

## 5. Обучение модели

- Формируется df_big (из дневных df по разным символам), добавляются колонки symbol_id.
- `MultiSymbolDataset`:
        - Разбивает временной ряд на (seq_len,1) вход, + symbol_id, → предсказываем close.
        - train/test split (или random_split).
- `MultiSymbolLSTM` (LSTM вход = concat(close, embedding(symbol))) → FC.
- `train_model_multi`:
        - Optim=Adam, Loss=MSE, num_epochs=150 (к примеру).

Лог показывает уменьшение MSE:

```plaintext
[Epoch 1/150] Train MSE=0.07, Val=0.08
...
[Epoch 150/150] Train MSE=0.0003, Val=0.0001
[Test] MSE=0.0001
```

Значит, модель фактически «очень хорошо» предсказала close на историческом отрезке (с учётом нормировки, etc.).

## 6. Вывод графиков

В конце `main.py` для каждого символа:

- Берётся 1h df,
- Plotly Candlestick:

```python
go.Candlestick(
         x=df_1h.index,
         open=df_1h["open"],
         high=df_1h["high"],
         low= df_1h["low"],
         close=df_1h["close"],
)
```

- Добавляются `fig.add_hline(y=d_sup,...)` и `fig.add_hline(y=d_res,...)`.
- Если df_1h имеет уникальный datetime (и не пустой), появится нормальный свечной график.

## 7. Типовые проблемы

- invalid literal for int(): '...some float...'

- last_ts не растёт => break

Bybit API ограничивает глубину или возвращает однообразные timestamps.

- Model MSE очень маленькое (0.0001)

Это нормально на нормированных данных. Проверять реальную ошибку в долларах, возможно:

- Plotly рисует не свечи, а «столбики»

Убедитесь, что `df.index` — DatetimeIndex без дублей, `go.Candlestick` + `x= df.index`.

## 8. Резюме

Таким образом, pricelevel-ml позволяет:

- Инкрементально скачать OHLCV с Bybit v5, (поддержка spot/linear).
- Найти 2 ключевых уровня (поддержка, сопротивление) на дневном TF, опционально уточнить 1h.
- Обучить простую LSTM-модель на собранных данных (дневных).
- Построить интерактивные графики (4h или 1h), отметив уровни.

При запуске `python -m src.main` в консоли вы увидите логи загрузки, обучения и появятся интерактивные окна Plotly
