
from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)


METRICS = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "last_ok_for_model": None,  # True/False/None
    "ok_count": 0,             # Сколько раз модель ответила "ОК"
    "latencies_ms": [],        # Список всех задержек для усреднения
    "csv_read": 0,            # Количество прочитанных csv файлов
    "wrong_files_count": 0,   # Количество файлов неправильного типа
    "csv_read_errors": 0,    # Количество ошибок чтения csv файлов
}

def _update_metrics(latency: float, is_ok: bool):
    """Вспомогательная функция для обновления счетчиков."""
    METRICS["total_requests"] += 1
    METRICS["total_latency_ms"] += latency
    METRICS["last_ok_for_model"] = is_ok
    if is_ok:
        METRICS["ok_count"] += 1
    METRICS["latencies_ms"].append(latency)

# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


class MetricsResponse(BaseModel):
    """Модель ответа для статистики сервиса."""
    total_requests: int
    avg_latency_ms: float
    last_ok_for_model: bool | None = Field(
        description="Результат последнего вызова оценки качества (True/False). None, если вызовов не было."
    )
    ok_ratio: float | None = Field(
        default=None,
        description="Доля успешных (ok_for_model=True) датасетов среди всех запросов."
    )
    latencies_ms: list[float] | None = Field(
        default=None,
        description="Список всех задержек запросов в миллисекундах"
    )
    csv_read: int | None = Field(
        default=None, 
        description="Количество успешно прочитанных CSV-файлов."
    )
    wrong_files_count: int | None = Field(
        default=None,
        description="Количество запросов с неправильным типом файла."
    )
    csv_read_errors: int | None = Field(
        default=None,
        description="Количество ошибок при чтении CSV-файлов."
    )
# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0
    _update_metrics(latency_ms, ok_for_model)
    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Примитивный лог — на семинаре можно обсудить, как это превратить в нормальный logger
    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.

    Именно это по сути связывает S03 (CLI EDA) и S04 (HTTP-сервис).
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        METRICS["wrong_files_count"] += 1
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
        METRICS["csv_read"] += 1
    except Exception as exc:  # noqa: BLE001
        METRICS["csv_read_errors"] += 1
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        METRICS["csv_read_errors"] += 1
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0
    _update_metrics(latency_ms, ok_for_model)
    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


@app.post(
    "/head",
    tags=["exploration"],
    summary="Показать первые N строк CSV-файла",
    description="Принимает CSV файл и число N. Возвращает первые N строк в формате JSON."
)
async def head(
    n: int = 5,                
    file: UploadFile = File(...)
) -> list[dict]:
    """
    HTTP-аналог команды `eda-cli head`.
    """
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        return []
    
    result_df = df.head(n)
    return result_df.to_dict(orient="records")


@app.get("/metrics", response_model=MetricsResponse, tags=["system"])
def get_metrics() -> MetricsResponse:
    """
    Возвращает статистику использования сервиса (накопленную в памяти).
    """
    count = METRICS["total_requests"]
    total_lat = METRICS["total_latency_ms"]
    
    # Считаем среднюю задержку (защита от деления на 0)
    avg_lat = (total_lat / count) if count > 0 else 0.0
    
    # Считаем долю "хороших" датасетов
    ok_ratio = (METRICS["ok_count"] / count) if count > 0 else 0.0

    return MetricsResponse(
        total_requests=count,
        avg_latency_ms=round(avg_lat, 2),
        last_ok_for_model=METRICS["last_ok_for_model"],
        ok_ratio=round(ok_ratio, 2),
        latencies_ms=METRICS["latencies_ms"] if count > 0 else None,
        wrong_files_count=METRICS["wrong_files_count"],
        csv_read_errors=METRICS["csv_read_errors"],
    )
