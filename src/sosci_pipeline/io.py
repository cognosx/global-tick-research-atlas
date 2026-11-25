"""Data contract (core columns every stage can rely on existing) and
format-agnostic read/write helpers. Parquet is the primary format; xlsx
is optional and gated behind a row-count cap to avoid memory blowups."""
from __future__ import annotations
import os
import re
import pandas as pd

CORE_COLUMNS = [
    "record_id", "title", "abstract", "authors", "year", "doi",
    "journal", "keywords", "document_type", "source_database",
    "source_record_id", "citation_count", "url",
]

_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    for c in CORE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df


def read_table(path: str, **kwargs) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    if path.endswith(".csv.gz") or path.endswith(".csv"):
        return pd.read_csv(path, dtype=str, **kwargs).fillna("")
    if path.endswith(".xlsx") or path.endswith(".xls"):
        sheet = kwargs.pop("sheet_name", 0)
        return pd.read_excel(path, sheet_name=sheet, dtype=str, **kwargs).fillna("")
    raise ValueError(f"Unsupported format for read_table: {path}")


def write_table(df: pd.DataFrame, path_no_ext: str, fmt: str = "parquet",
                 also_xlsx: bool = False, xlsx_max_rows: int = 200000,
                 sheet_name: str = "data") -> str:
    out_path = path_no_ext
    if not out_path.endswith((".parquet", ".csv.gz", ".csv", ".xlsx")):
        out_path = path_no_ext + (".parquet" if fmt == "parquet" else f".{fmt}")

    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    elif out_path.endswith(".csv.gz"):
        df.to_csv(out_path, index=False, compression="gzip")
    elif out_path.endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    if also_xlsx and len(df) <= xlsx_max_rows:
        xlsx_path = re.sub(r"\.(parquet|csv\.gz|csv)$", "", out_path) + ".xlsx"
        safe = df.copy()
        for c in safe.columns:
            safe[c] = safe[c].map(lambda v: _ILLEGAL_XLSX.sub(" ", str(v))[:32000])
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            safe.to_excel(xw, sheet_name=sheet_name, index=False)
    return out_path


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: _ILLEGAL_XLSX.sub(" ", str(v)) if isinstance(v, str) else v)
    return out
