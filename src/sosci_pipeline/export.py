"""Export utilities for turning pipeline outputs into human-explorable files.

The per-record corpus files are wide (full abstracts, enrichment namespaces,
embeddings-adjacent columns) and large — dumping every column to xlsx makes an
unwieldy file. `export_corpus_xlsx` writes a curated, exploration-friendly
column subset, respects Excel's row limit (splitting across sheets if needed),
and sanitises illegal control characters. `export_analysis_workbook` bundles the
small aggregate result tables (topic summary, taxonomy, region crosstab, odds
ratios, climate trend, country table) into one multi-sheet workbook.
"""
from __future__ import annotations
import os
import re
import pandas as pd

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
XLSX_MAX_ROWS = 1_048_575  # Excel hard limit (minus header)

# columns worth keeping for exploration, in a sensible order (only those present are used)
EXPLORE_COLS = [
    "record_id", "study_id", "year", "title", "abstract",
    "journal", "doi", "document_type", "oa_language",
    "authors", "affiliations", "oa_countries", "oa_institutions",
    "funders_unified", "is_funded", "citation_count",
    "source_database", "found_in", "n_sources",
    "tick_relevant", "tick_match_type",
    "topic_id", "topic_label", "topic_clean_label", "theme_group",
    "screen_verdict", "screen_reason",
    "title_en", "abstract_en", "translated",
]


def _sanitize(df):
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: _ILLEGAL.sub(" ", str(v))[:32000] if isinstance(v, str) else v)
    return out


def export_corpus_xlsx(in_path, out_path, columns=None, max_abstract_chars=2000):
    """Export a per-record corpus file to xlsx with a curated column subset.
    Splits into data_1, data_2, ... sheets if the row count exceeds Excel's limit."""
    df = pd.read_parquet(in_path) if in_path.endswith(".parquet") else pd.read_csv(in_path)
    cols = columns or [c for c in EXPLORE_COLS if c in df.columns]
    df = df[cols].copy()
    if "abstract" in df.columns and max_abstract_chars:
        df["abstract"] = df["abstract"].astype(str).str.slice(0, max_abstract_chars)
    if "abstract_en" in df.columns and max_abstract_chars:
        df["abstract_en"] = df["abstract_en"].astype(str).str.slice(0, max_abstract_chars)
    df = _sanitize(df)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        if len(df) <= XLSX_MAX_ROWS:
            df.to_excel(xw, sheet_name="corpus", index=False)
        else:
            n_sheets = (len(df) + XLSX_MAX_ROWS - 1) // XLSX_MAX_ROWS
            for i in range(n_sheets):
                chunk = df.iloc[i * XLSX_MAX_ROWS:(i + 1) * XLSX_MAX_ROWS]
                chunk.to_excel(xw, sheet_name=f"corpus_{i+1}", index=False)
    print(f"exported {len(df)} rows x {len(cols)} cols -> {out_path}")
    return out_path


def export_analysis_workbook(work_dir, out_path):
    """Bundle the aggregate result CSVs in work_dir into one multi-sheet xlsx.
    Any missing file is skipped with a note (so partial runs still export)."""
    sheets = {
        "topic_taxonomy": "topic_taxonomy.csv",
        "topic_summary": "topic_summary.csv",
        "region_crosstab": "topic_region_crosstab.csv",
        "region_oddsratios": "topic_region_oddsratios.csv",
        "region_figure_data": "topic_region_figure_data.csv",
        "climate_by_year": "climate_fraction_by_year.csv",
        "country_output": "country_output_table.csv",
        "coverage": "canonical_coverage.csv",
    }
    written = 0
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for sheet, fname in sheets.items():
            path = os.path.join(work_dir, fname)
            # country_output may be parquet
            if not os.path.exists(path) and fname.endswith(".csv"):
                alt = path.replace(".csv", ".parquet")
                if os.path.exists(alt):
                    df = pd.read_parquet(alt)
                    _sanitize(df).to_excel(xw, sheet_name=sheet[:31], index=False)
                    written += 1
                    continue
            if os.path.exists(path):
                df = pd.read_csv(path)
                _sanitize(df).to_excel(xw, sheet_name=sheet[:31], index=False)
                written += 1
            else:
                print(f"  (skip {sheet}: {fname} not found)")
    print(f"analysis workbook: {written} sheets -> {out_path}")
    return out_path


def run_export(ctx, which="both"):
    """Stage entry: export corpus and/or the analysis workbook for the project.
    which = 'corpus' | 'workbook' | 'both'."""
    from .runner import F
    outputs = []
    if which in ("corpus", "both"):
        src = ctx.find(F["analysis_ready"], F["topic_modeled"], required=False)
        if src:
            out = ctx.out("corpus_explore.xlsx")
            export_corpus_xlsx(src, out)
            outputs.append(out)
        else:
            print("no analysis_ready/topic_modeled file found to export as corpus.")
    if which in ("workbook", "both"):
        out = ctx.out("analysis_results.xlsx")
        export_analysis_workbook(ctx.work, out)
        outputs.append(out)
    return outputs
