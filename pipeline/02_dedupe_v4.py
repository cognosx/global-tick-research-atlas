# Auto-generated mirror of notebooks/01_corpus/02_dedupe_v4.ipynb

# # 02 · Deduplication v4 — fast, OpenAlex-only (GTRA v1.0)
# 
# Input: `works.parquet` from notebook 01 (one row per OpenAlex work).
# Code: `src/gtra/dedupe_fast.py` (tested on 77k synthetic records: ~35 s for matching, ~12 s for saving).
# 
# **Changes vs v3.1**
# - Full outputs are **parquet only**. The slow full-size `.xlsx` files are gone. One small QC workbook (≤500 sampled rows per sheet) is written with xlsxwriter in a few seconds.
# - No global `fillna("")`, so number columns keep their types.
# - Fuzzy title matching is vectorised (`rapidfuzz.process.cdist`, all CPU cores) and compares only years y and y+1 inside each block. It runs on the first **and** last content word of the title.
# - Every merge is saved as an edge (pair, stage, score) in `merge_edges.parquet`, so you can audit or undo any merge.
# - **Preprint ↔ published versions are linked, not merged** (`version_group`, `is_primary_version`).
# - Correction / erratum / retraction notices are never merged on title.
# - Fuzzy merges are blocked when titles differ by a number, a part numeral (Part I / Part II) or a swapped tick species.
# - Preprint-server and repository DOIs (bioRxiv, Research Square, Zenodo, figshare…) are "soft": they do not block a merge with the journal DOI.
# 
# **Merge ladder:** 0 same OpenAlex ID → 1 same DOI → 1b same PMID → 2a title+authors+year → 2b title+journal(ISSN-L)+year → 3 title+year with a shared author → 4 fuzzy title (≥95, year ±2, shared author; or same journal at ≥95).

# 1 · Setup
from google.colab import drive
drive.mount('/content/drive')

pass  # shell: pip -q install rapidfuzz xlsxwriter pyarrow
import os, sys, json, time
if not os.path.exists('/content/gtra'):
    pass  # shell: git clone -q https://github.com/cognosx/global-tick-research-atlas /content/gtra
else:
    pass  # shell: git -C /content/gtra pull -q
sys.path.insert(0, '/content/gtra/src')

import pandas as pd
from gtra.dedupe_fast import FastDeduper, DedupeConfig, save_outputs
pd.set_option('display.max_columns', None)

# 2 · Configuration
ROOT    = "/content/drive/MyDrive/COGNOSX/TICKS/ticks_research"     # <-- your Drive folder
RUN     = "openalex_2026-11-26"                                     # harvest folder from notebook 01
IN_PATH = f"{ROOT}/data_raw/{RUN}/A_ticks/works.parquet"
OUT_DIR = f"{ROOT}/preprocessing_data/{RUN}/dedupe"

cfg = DedupeConfig(
    fuzzy_threshold=95,          # title similarity (token_sort_ratio) with a shared author
    fuzzy_journal_threshold=95,  # stricter when only the journal confirms
    fuzzy_year_tol=2,            # year gap allowed for a true duplicate
    min_title_tokens=5,          # shorter titles skip title-based matching
    version_year_tol=3,          # year gap allowed for preprint -> published
    version_threshold=92,        # title similarity to link versions
)

# 3 · Load (keep native types: no fillna(""))
t0 = time.perf_counter()
df = pd.read_parquet(IN_PATH)
print(f"{len(df):,} rows × {df.shape[1]} cols in {time.perf_counter()-t0:.1f}s")
print(df["oa_type"].value_counts().to_string())
print("repeated oa_id:", df["oa_id"].duplicated().sum())

# 4 · Run
dd  = FastDeduper(cfg)
out = dd.run(df)
rep = dd.report(out)
print(json.dumps(rep, indent=1))

# 5 · Look at the fuzzy merges before trusting them (sorted from weakest)
edges = dd.edges_frame(out)
print(edges.groupby(["kind", "stage"]).size().to_string())
(edges[edges["stage"].str.startswith("4_") & edges["kind"].eq("merge")]
   .sort_values("score")[["score", "oa_title_a", "oa_title_b", "oa_year_a", "oa_year_b"]].head(25))

# 6 · Preprint ↔ published links (kept as two records, flagged)
vl = edges[edges["kind"].eq("version_link")]
vl[["score", "oa_type_a", "oa_year_a", "oa_title_a", "oa_type_b", "oa_year_b", "oa_title_b"]].head(15)

# 7 · Save (parquet + one small QC workbook)
paths = save_outputs(out, dd, OUT_DIR, qc_rows=500)
json.dump({"input": IN_PATH, "config": vars(cfg), "report": rep, "outputs": paths},
          open(f"{OUT_DIR}/dedupe_manifest.json", "w"), indent=1, default=str)

# **Outputs** (in `preprocessing_data/<run>/dedupe/`)
# 
# | File | What it is |
# |---|---|
# | `works_canonical.parquet` | one row per unique study, all `oa_*` columns + `study_id`, `n_records`, `version_group`, `is_primary_version` → **input for screening** |
# | `dedupe_map.parquet` | every input record → `study_id` (light: no abstracts) |
# | `merge_edges.parquet` | every merge and version link with stage and score |
# | `qc_dedupe_review.xlsx` | sampled sheets for a quick human check |
# | `dedupe_manifest.json` | input, settings, counts, timings |
# 
# For analyses that count papers, filter `is_primary_version == True` so a preprint and its published version count once. For screening, keep all canonical records.
