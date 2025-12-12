# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

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