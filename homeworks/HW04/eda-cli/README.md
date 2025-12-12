# S04 – eda_cli: HTTP-сервис качества датасетов (FastAPI)

Расширенная версия проекта `eda-cli` из Семинара 03.

К существующему CLI-приложению для EDA добавлен **HTTP-сервис на FastAPI** с эндпоинтами `/health`, `/quality`  `/quality-from-csv`.  
Используется в рамках Семинара 04 курса «Инженерия ИИ».

---

## Связь с S03

Проект в S04 основан на том же пакете `eda_cli`, что и в S03:

- сохраняется структура `src/eda_cli/` и CLI-команда `eda-cli`;
- добавлен модуль `api.py` с FastAPI-приложением;
- в зависимости добавлены `fastapi` и `uvicorn[standard]`.

Цель S04 – показать, как поверх уже написанного EDA-ядра поднять простой HTTP-сервис.


## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему
- Браузер (для Swagger UI `/docs`) или любой HTTP-клиент:
  - `curl` / HTTP-клиент в IDE / Postman / Hoppscotch и т.п.

---

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.


## Просмотр первых n строк в наборе данных

```bash
uv run eda-cli head data/example.csv --n 4
```

Параметры:
    -path(required) - путь до csv файла
    -n(optional) - количество строк для вывода


## Тесты

```bash
uv run pytest -q
```
## Список тестов

1. test_summarize_dataset_basic - тестирует базовые анализ всего датасета
2. test_missing_table_and_quality_flags - тестирует функцию формирования словаря флагов качества
3. test_correlation_and_top_categories - тестирует корректность нахождения коэффициенто корреляции между столбцами
4. test_future_dates_detection - тестирует работоспособность функции нахождения некорректных дат
5. test_future_date_quality_flags - проверяет принадлежность флага к множеству [True, False]

## Запуск HTTP-сервиса

HTTP-сервис реализован в модуле `eda_cli.api` на FastAPI.

### Запуск Uvicorn

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Пояснения:

- `eda_cli.api:app` - путь до объекта FastAPI `app` в модуле `eda_cli.api`;
- `--reload` - автоматический перезапуск сервера при изменении кода (удобно для разработки);
- `--port 8000` - порт сервиса (можно поменять при необходимости).

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```

---
## Эндпоинты сервиса

### 1. `GET /health`

Простейший health-check.

**Запрос:**

```http
GET /health
```

**Ожидаемый ответ `200 OK` (JSON):**

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

Пример проверки через `curl`:

```bash
curl http://127.0.0.1:8000/health
```

---

### 2. Swagger UI: `GET /docs`

Интерфейс документации и тестирования API:

```text
http://127.0.0.1:8000/docs
```

Через `/docs` можно:

- вызывать `GET /health`;
- вызывать `POST /quality` (форма для JSON);
- вызывать `POST /quality-from-csv` (форма для загрузки файла).

---

### 3. `POST /quality` – заглушка по агрегированным признакам

Эндпоинт принимает **агрегированные признаки датасета** (размеры, доля пропусков и т.п.) и возвращает эвристическую оценку качества.

**Пример запроса:**

```http
POST /quality
Content-Type: application/json
```

Тело:

```json
{
  "n_rows": 0,
  "n_cols": 0,
  "max_missing_share": 1,
  "numeric_cols": 0,
  "categorical_cols": 0
}
```

**Пример ответа `200 OK`:**

```json
{
  
  "ok_for_model": false,
  "quality_score": 0,
  "message": "Качество данных недостаточно, требуется доработка (по текущим эвристикам).",
  "latency_ms": 0.02269999822601676,
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": true,
    "no_numeric_columns": true,
    "no_categorical_columns": true
  },
  "dataset_shape": {
    "n_rows": 0,
    "n_cols": 0
  }
}
```

**Пример вызова через `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 10000, "n_cols": 12, "max_missing_share": 0.15, "numeric_cols": 8, "categorical_cols": 4}'
```

---

### 4. `POST /quality-from-csv` – оценка качества по CSV-файлу

Эндпоинт принимает CSV-файл, внутри:

- читает его в `pandas.DataFrame`;
- вызывает функции из `eda_cli.core`:

  - `summarize_dataset`,
  - `missing_table`,
  - `compute_quality_flags`;
- возвращает оценку качества датасета в том же формате, что `/quality`.

**Запрос:**

```http
POST /quality-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

Через Swagger:

- в `/docs` открыть `POST /quality-from-csv`,
- нажать `Try it out`,
- выбрать файл (например, `data/example.csv`),
- нажать `Execute`.

**Пример вызова через `curl` (Linux/macOS/WSL):**

```bash
curl -X POST "http://127.0.0.1:8000/quality-from-csv" \
  -F "file=@data/example.csv"
```

**Пример ответа `200 OK`:**

```json
{
  "ok_for_model": false,
  "quality_score": 0.6944444444444444,
  "message": "CSV требует доработки перед обучением модели (по текущим эвристикам).",
  "latency_ms": 37.02519999933429,
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "excessive_zeros": false,
    "has_constant_columns": false,
    "has_future_dates": false,
    "has_suspicious_id_duplicates": true
  },
  "dataset_shape": {
    "n_rows": 36,
    "n_cols": 14
  }
}
```


4. `POST /metrics` - вывод метрик по работе сервиса

Эндпоинт не имеет входных параметров

**Пример ответа `200 OK`:**

```json
{
  "total_requests": 12,
  "avg_latency_ms": 24.12,
  "last_ok_for_model": false,
  "ok_ratio": 0,
  "latencies_ms": [
    26.249300019117072,
    20.591000007698312,
    16.686999995727092,
    19.41790001001209,
    54.06240001320839,
    14.52559998142533,
    16.669999982696027,
    28.35069998400286,
    31.647399999201298,
    19.652299990411848,
    14.635700004873797,
    26.892099995166063
  ],
  "csv_read": null,
  "wrong_files_count": 7,
  "csv_read_errors": 0
}
```
