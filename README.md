# Техническая документация

Ниже представлена техническая документация по проекту, который обучает одну модель (LSTM) сразу на нескольких криптовалютных инструментах с учётом мультитаймфрейма, а также выполняет бэктест и визуализацию уровней поддержки/сопротивления. Документация раскрывает логику модулей, структуру данных, процесс обучения, а также даёт пояснения по запуску и расширению проекта.

## Общий обзор

Проект решает задачу автоматизированного определения ключевых уровней (поддержки/сопротивления) на рынке криптовалют, используя:

- Мультитаймфреймовый анализ: дневные (D), 4-часовые (H4) и 1-часовые (H1) свечи.
- Глубокое обучение (LSTM-модель), которая учится на нескольких инструментах сразу (например, топ-10 криптовалют) и позволяет совместно анализировать общие паттерны.
- Кластеризацию локальных экстремумов для вычленения потенциальных уровней.
- Бэктест (упрощённый), чтобы визуально оценить, как формирующиеся уровни коррелируют с сигналами входа/выхода.

Проект состоит из нескольких модулей в папке `src/`, которые совместно образуют конвейер (pipeline):

- `config.py`: Хранит гиперпараметры, списки инструментов, даты, настройки обучения и т.д.
- `data_fetcher.py`: Логика загрузки исторических данных с Bybit (Unified v5 API) для отдельных таймфреймов.
- `cluster_levels.py`: Методы нахождения локальных экстремумов и кластеризации уровней (например, DBSCAN).
- `multi_tf_analysis.py`: Дополнительные функции анализа (например, filter_levels), учитывающие разные таймфреймы.
- `dataset.py`: Описание PyTorch Dataset, который собирает признаки (close, symbol_id) и метки (level_label).
- `model.py`: Модель глубинного обучения (MultiSymbolLSTM), которая принимает на вход и цены, и идентификатор инструмента (embedding).
- `train.py`: Функции обучения модели (train_model_multi) и валидации (evaluate_model_multi).
- `backtest.py`: Код для упрощённого бэктеста сигналов (run_backtest) и их отрисовки (plot_backtest_results).
- `main.py`: Основной скрипт (pipeline), который:
  - Загружает данные для нескольких символов;
    - Находит уровни, ставит метки;
    - Создаёт общий Dataset;
    - Обучает модель на GPU;
    - Выполняет бэктест и рисует результаты.

## Структура проекта

Предположим, что каталог выглядит так:

```plaintext
pricelevel-ml/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_fetcher.py
│   ├── cluster_levels.py
│   ├── multi_tf_analysis.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── backtest.py
│   ├── main.py
│   └── run_pretrained.py   (необязательно, если используется)
├── data/
│   ├── raw/
│   └── processed/
├── venv/ (или другая виртуализация)
├── requirements.txt
└── ...
```

- `data/raw/` используется для кэширования CSV с историческими свечами.
- `data/processed/` может хранить уже агрегированные/предобработанные данные.
- `venv/` – виртуальное окружение Python (необязательно, но рекомендуется).

## Зависимости (requirements)

Пример списка пакетов, необходимых для работы:

- python >= 3.8
- torch >= 1.13 (с поддержкой CUDA, если нужно обучение на GPU)
- pandas >= 1.3
- numpy >= 1.21
- mlflow >= 2.2
- scikit-learn >= 1.0
- plotly >= 5.6
- pybit >= 2.0 (Unified Trading API)
- requests, tqdm (при необходимости)

В файле `requirements.txt` можно задать точные версии. Пример:

```ini
torch==1.13.1
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.1
plotly==5.11.0
pybit==2.2.1
mlflow==2.3.2
```

## Файл config.py

Этот модуль содержит глобальные параметры, которые контролируют проект. Например:

```python
BYBIT_API_KEY = ...
BYBIT_API_SECRET = ...

TOP_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", ...
]

BYBIT_CATEGORY = "spot"      # либо "linear"
DAILY_INTERVAL = "D"         # день
H4_INTERVAL = "240"          # 4 часа
H1_INTERVAL = "60"           # 1 час
BYBIT_START_DATE = "2023-01-01"
BYBIT_END_DATE   = "2025-01-01"

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

EPS_PERCENT = 0.005
MIN_SAMPLES = 4
WINDOW_SIZE = 12
MIN_TOUCHES_FILTER = 4
MAX_AGE_DAYS = 90
ATR_BUFFER = 0.20
VOLUME_FACTOR = 1.3

INPUT_SIZE = 1
HIDDEN_SIZE = 128
NUM_LAYERS = 3
OUTPUT_SIZE = 1
LEARNING_RATE = 0.0003
NUM_EPOCHS = 80
BATCH_SIZE = 64
SEQ_LEN = 80
EMB_DIM = 8
```

- `TOP_SYMBOLS` — список криптовалют, для которых будет загружаться история.
- `BYBIT_CATEGORY` — "spot" или "linear" (USDT-перп).
- `(DAILY_INTERVAL, H4_INTERVAL, H1_INTERVAL)` — интервалы свечей.
- Параметры кластеризации (`EPS_PERCENT, WINDOW_SIZE ...`) — влияют на алгоритм поиска уровней.
- Параметры LSTM-модели (`HIDDEN_SIZE, NUM_LAYERS и т.д.`) — влияют на архитектуру сети.

## Файл data_fetcher.py

Здесь описано, как качаются данные с Bybit через pybit (Unified v5). Вариант реализации:

- Функция `_fetch_bybit_kline(...)`: скачивает котировки для заданного символа, интервала и дат (start, end).
- Функция `load_single_symbol_multi_timeframe(symbol, ...)`: трижды вызывает `_fetch_bybit_kline`, получая `df_daily`, `df_4h`, `df_1h`.

Например:

```python
def load_single_symbol_multi_timeframe(symbol, 
                                                                             category=BYBIT_CATEGORY,
                                                                             start=BYBIT_START_DATE,
                                                                             end=BYBIT_END_DATE,
                                                                             daily_interval=DAILY_INTERVAL,
                                                                             h4_interval=H4_INTERVAL,
                                                                             h1_interval=H1_INTERVAL):
        """
        Возвращает (df_daily, df_4h, df_1h) для одного инструмента.
        """
        df_d  = _fetch_bybit_kline(...)
        df_4h = _fetch_bybit_kline(...)
        df_1h = _fetch_bybit_kline(...)
        return df_d, df_4h, df_1h
```

## Файл cluster_levels.py

Реализует поиск локальных экстремумов и их кластеризацию.

- `find_local_extrema(prices, window)`: находит индексы и цены локальных максимумов/минимумов в окне ±window.
- `cluster_extrema(maxima, minima, eps_frac, min_samples)`: объединяет точки (цены экстремумов) в кластеры (например, DBSCAN), возвращая список средних цен кластеров.
- `make_binary_labels(df, levels, threshold_frac=0.001)`: формирует бинарную метку (1, если close в пределах threshold_frac * close от одного из уровней).

## Файл multi_tf_analysis.py

Описывает дополнительные функции анализа, учитывающие разные ТФ:

- `compute_atr(df_daily, period=7)`: простой ATR по дневному фрейму.
- `filter_levels(df_daily, df_4h, df_1h, raw_levels, ...)`:
  - Базовая логика: проверить касания на дневном ТФ, возраст уровня, объёмы на 4h, скорость подхода на 1h и т.д.
    - Возвращает список «отфильтрованных» уровней.

## Файл dataset.py

Здесь задаётся PyTorch Dataset, который собирает последовательности (для LSTM) и метки. Для мультиинструментного обучения используют класс `MultiSymbolDataset`:

```python
class MultiSymbolDataset(Dataset):
        def __init__(self, df, seq_len=SEQ_LEN):
                # Сохраняем df["close"], df["symbol_id"], df["level_label"]
                # ...
        def __getitem__(self, idx):
                # Возвращаем (seq_close, symbol_id, label)
```

## Файл model.py

Хранит LSTM-модель с учётом `symbol_id`:

```python
class MultiSymbolLSTM(nn.Module):
        def __init__(self, num_symbols, ...):
                # self.symbol_emb = nn.Embedding(num_symbols, emb_dim)
                # self.lstm = nn.LSTM(input_size+emb_dim, hidden_size, ...)
                # self.fc = nn.Linear(...)
                # ...
        def forward(self, seq_close, symbol_id):
                # embedding(symbol_id) => concat => lstm => fc => sigmoid
```

## Файл train.py

Описывает процесс обучения и валидации:

```python
def train_model_multi(train_dataset, val_dataset, num_symbols, device='cpu'):
        model = MultiSymbolLSTM(num_symbols).to(device)
        # ...
        with mlflow.start_run():
                # for epoch in range(NUM_EPOCHS):
                #   ...
                #   loss.backward()
                #   ...
                # mlflow.pytorch.log_model(model, "models")
        return model
```

MLflow используется для логирования параметров (learning_rate, batch_size, epochs) и метрик (train_loss, val_accuracy и т.д.), а также для сохранения модели.

```python
def evaluate_model_multi(model, data_loader, device='cpu'):
        # Проходит по батчам, вычисляет accuracy, precision, recall, f1.
        # ...
```

## Файл backtest.py

Содержит функции для упрощённого бэктеста сигналов:

- `run_backtest(df, levels=None, initial_capital=10000.0)`:
  - Предполагается, что в `df` есть колонка `signal` (1 => Buy, -1 => Sell).
    - Итог: возвращает DataFrame с `trade_price`, список сделок `trades`, и `eq_curve` (equity во времени).

- `plot_backtest_results(df, trades, eq_curve, levels=None, title="")`:
  - Использует Plotly.
    - Рисует свечи OHLC, точки покупок/продаж, опционально горизонтальные линии `levels`, и снизу кривую капитала.

## Файл main.py

Главный скрипт, в котором:

- Загружаются несколько инструментов (из `config.TOP_SYMBOLS`).
- Для каждого:
  - Вызываются `load_single_symbol_multi_timeframe(...)` → `(df_d, df_4h, df_1h)`.
    - На `df_d` ищутся экстремумы → получаем `raw_levels` → фильтруем через `filter_levels(df_d, df_4h, df_1h, raw_levels)`.
    - Формируем `level_label`.
    - Помечаем `df_d["symbol_id"]`, складываем в общий `df_big`.
- Создаём `MultiSymbolDataset` из `df_big`.
- Разделяем на train/val/test, вызываем `train_model_multi(...)`.
- Проводим валидацию (`evaluate_model_multi`).
- Для каждого символа отдельно создаём бэктест: упрощённая логика сигналов (`level_label` => `signal`), и рисуем график.
- В конце выводится серия графиков, по одному на символ.

## Запуск и использование

Чтобы запустить проект:

1. Установите зависимости (см. `requirements.txt`).
2. Создайте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate   # или .\venv\Scripts\activate в Windows
pip install -r requirements.txt
```

1. Настройте (при необходимости) переменные среды `BYBIT_API_KEY`, `BYBIT_API_SECRET` (или оставьте пустыми, если нужны только публичные данные).
2. Запустите:

```bash
python -m src.main
```

Если CUDA доступна, код отобразит `Используем устройство: cuda`.

Произойдёт скачивание данных, поиск уровней, обучение, бэктест.

## Результаты

В MLflow (если запущен через `mlflow ui`) вы увидите новую запись (Run), где есть:

- Параметры обучения (learning_rate, batch_size, epochs).
- Метрики (train_loss, val_accuracy и т.д.).
- Сохранённая модель (artifact_path="models").

В консоли и в окне Plotly будут показаны графики бэктеста (candlestick + уровни + сигналы) для каждого инструмента.

## Расширение и настройка

- Добавить больше инструментов: нужно лишь расширить `TOP_SYMBOLS` в `config.py`.
- Ускорить скачивание: при большом числе символов, API Bybit можно «задушить». Возможно, стоит добавить параллелизм или кэширование CSV для 4h/1h.
- Сложная логика сигналов: вместо `df["level_label"]` => `signal=1`, можно использовать обученную модель (LSTM) для предсказания вероятности пробоя/отбоя уровня и генерировать `signal` на основе `p >= threshold`.
- Улучшить фильтрацию: в `filter_levels` добавить более точный учёт объёмов/ATR, ретестов, фитилей и т.д.
- Регуляризация: если модель переобучается, можно добавить dropout, early stopping, уменьшить `HIDDEN_SIZE` и т.д.
- Docker: при необходимости запаковать всё в Docker-образ, автоматизировать запуск.

## Вывод

Данный проект даёт сквозной пример:

- Сбор данных (мультисет, несколько таймфреймов).
- Поиск уровней (экстремумы + кластеризация + фильтры).
- Обучение единой RNN-модели сразу на нескольких криптоинструментах.
- Логирование (MLflow).
- Бэктест (Candlestick + сигналы) для визуального контроля.

Благодаря гибкой структуре, можно быстро менять логику (параметры уровней, архитектуру модели, стратегию сигналов) и тестировать гипотезы о трейдинге и анализе уровней на разных рынках.
