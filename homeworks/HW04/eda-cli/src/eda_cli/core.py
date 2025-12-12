from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    
    # Типы данных
    is_numeric: bool
    is_date: bool
    
    # Статистики числовые
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    
    # Статистики нулей
    zero_count: int = 0
    zero_share: float = 0.0
    
    # Специфичные проверки
    is_constant: bool = False       # True, если 1 уникальное значение
    future_date_count: int = 0      # Количество дат из будущего

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]
    
    # Глобальные метрики
    total_zeros: int
    global_zero_share: float
    excessive_zeros: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
            "total_zeros": self.total_zeros,
            "global_zero_share": self.global_zero_share,
            "excessive_zeros": self.excessive_zeros,
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета.
    Включает проверки на типы, нули, константные колонки и даты из будущего.
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []
    
    total_zeros_in_dataset = 0
    now = pd.Timestamp.now()  # Текущее время для проверки дат

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))
        
        # 1. Проверка на константность (есть значение, и оно всего одно)
        is_constant = (unique == 1) and (non_null > 0)

        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        is_date = bool(ptypes.is_datetime64_any_dtype(s))
        
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None
        
        col_zero_count = 0
        col_zero_share = 0.0
        future_date_count = 0

        if is_numeric:
            if non_null > 0:
                min_val = float(s.min())
                max_val = float(s.max())
                mean_val = float(s.mean())
                std_val = float(s.std())
            
            col_zero_count = (s == 0).sum()
            col_zero_share = float(col_zero_count / n_rows) if n_rows > 0 else 0.0

        if is_date and non_null > 0:
            future_date_count = (s > now).sum()

        total_zeros_in_dataset += col_zero_count

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                is_date=is_date,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
                zero_count=col_zero_count,
                zero_share=col_zero_share,
                is_constant=is_constant,
                future_date_count=future_date_count
            )
        )

    # Глобальные показатели
    total_cells = n_rows * n_cols
    global_zero_share = 0.0
    if total_cells > 0:
        global_zero_share = float(total_zeros_in_dataset / total_cells)

    excessive_zeros = global_zero_share > 0.20

    return DatasetSummary(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        total_zeros=total_zeros_in_dataset,
        global_zero_share=global_zero_share,
        excessive_zeros=excessive_zeros
    )


def compute_quality_flags(summary: DatasetSummary, missing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Расширенные эвристики качества данных.
    """
    flags: Dict[str, Any] = {}
    
    #Базовые проверки размера
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    #Пропуски
    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5
    
    #Нули
    flags["excessive_zeros"] = summary.excessive_zeros

    # --- НОВЫЕ ЭВРИСТИКИ ---

    #Константные колонки (не несут информации)
    constant_cols = [c.name for c in summary.columns if c.is_constant]
    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns_list"] = constant_cols

    #Даты из будущего
    future_date_cols = [c.name for c in summary.columns if c.future_date_count > 0]
    flags["has_future_dates"] = len(future_date_cols) > 0
    flags["future_dates_columns"] = future_date_cols

    #Подозрительные дубликаты ID
    # Логика: если колонка называется как ID, но unique < non_null, значит есть дубликаты
    suspicious_ids = []
    for c in summary.columns:
        name_lower = c.name.lower()
        # Эвристика по имени: содержит id, code, guid, pk
        if any(x in name_lower for x in ['id', 'code', 'guid', 'pk']):
            # Если не все значения уникальны (среди заполненных)
            if c.non_null > 0 and c.unique < c.non_null:
                suspicious_ids.append(c.name)
    
    flags["has_suspicious_id_duplicates"] = len(suspicious_ids) > 0
    flags["suspicious_id_columns"] = suspicious_ids

    score = 1.0
    score -= max_missing_share
    
    if summary.n_rows < 100:
        score -= 0.1
        
    if summary.excessive_zeros:
        score -= 0.1
        
    if flags["has_constant_columns"]:
        score -= 0.1  # Штраф за наличие "мусорных" колонок
        
    if flags["has_future_dates"]:
        score -= 0.2  # Серьезная ошибка логики данных
        
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.15 # Потенциальная проблема с ключами

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "is_date": col.is_date,          # Новое поле
                "min": col.min,
                "max": col.max,
                "zero_share": col.zero_share,
                "is_constant": col.is_constant,  # Новое поле
                "future_cnt": col.future_date_count # Новое поле
            }
        )
    return pd.DataFrame(rows)

def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])
    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame({"missing_count": total, "missing_share": share})
        .sort_values("missing_share", ascending=False)
    )
    return result

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)

def top_categories(df: pd.DataFrame, max_columns: int = 5, top_k: int = 5) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []
    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)
    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame({
            "value": vc.index.astype(str),
            "count": vc.values,
            "share": share.values,
        })
        result[name] = table
    return result
