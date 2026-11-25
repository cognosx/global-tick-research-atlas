# ============================================================
#  YEAR CUTOFF FILTER (>= 2000)  (Colab)
#  Applies the publication-year inclusion limit on the ENRICHED master,
#  BEFORE dedup, so the cutoff propagates cleanly through the rest of
#  the pipeline (dedup -> coverage -> screening readiness -> screening).
#
#  Rules (per decision):
#    * KEEP year >= 2000
#    * KEEP records with missing/unparseable year (do not drop on absence)
#    * DROP year <= 1999
#  Uses best available year: `year` then OpenAlex `oa_year` fallback.
#
#  Input : master_enriched.xlsx
#  Output: master_enriched_2000.xlsx  -> feed THIS into dedup (08_dedupe_oa_aware)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pandas openpyxl

import os, re
import pandas as pd, numpy as np

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
IN_PATH = os.path.join(OUT_DIR, "master_enriched.xlsx")
OUT_PATH= os.path.join(OUT_DIR, "master_enriched_2000.xlsx")
CUTOFF  = 2000   # inclusive: keep >= 2000

df = pd.read_excel(IN_PATH, sheet_name="enriched_master", dtype=str).fillna("")
print("Loaded enriched master:", len(df))

def parse_year(s):
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(s or ""))   # 1800-2099 plausible range
    return int(m.group(0)) if m else None

# best year: prefer `year`, fall back to oa_year
yr = df["year"].map(parse_year)
if "oa_year" in df.columns:
    yr = yr.where(yr.notna(), df["oa_year"].map(parse_year))
df["_year_best"] = yr

n_total   = len(df)
n_known    = df["_year_best"].notna().sum()
n_missing  = df["_year_best"].isna().sum()
n_pre      = (df["_year_best"] < CUTOFF).sum()
n_post     = (df["_year_best"] >= CUTOFF).sum()

# KEEP: year >= cutoff OR year missing
keep_mask = (df["_year_best"] >= CUTOFF) | (df["_year_best"].isna())
kept = df[keep_mask].copy()
dropped = df[~keep_mask].copy()

print("\n=== YEAR FILTER (>= %d) ===" % CUTOFF)
print(f"  total records:        {n_total}")
print(f"  year known:           {n_known}  (missing/unparseable: {n_missing} -> KEPT)")
print(f"  pre-{CUTOFF} (dropped): {n_pre}")
print(f"  >= {CUTOFF} (kept):     {n_post}")
print(f"  KEPT (incl. missing): {len(kept)}")
print(f"  DROPPED:              {len(dropped)}")

# year distribution of what we keep (sanity)
print("\nKept records by year (every 5y, sampled):")
ky = kept["_year_best"].dropna().astype(int)
if len(ky):
    print(f"  min {ky.min()}  max {ky.max()}  median {int(ky.median())}")
    bins = ky.value_counts().sort_index()
    for y in range(CUTOFF, ky.max()+1, 5):
        c = bins.loc[y:y+4].sum() if len(bins.loc[y:y+4]) else 0
        print(f"   {y}-{y+4}: {c}")

# pre-2000 dropped, by year (so you can confirm the cutoff worked)
print("\nDropped pre-%d by year (top 10):" % CUTOFF)
dy = dropped["_year_best"].dropna().astype(int).value_counts().sort_index()
print(dy.tail(10).to_string() if len(dy) else "  (none)")

def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.drop(columns=[c for c in ["_year_best"] if c in frame.columns]).copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)
safe_xlsx(kept, OUT_PATH, "enriched_master")
# audit of dropped
safe_xlsx(dropped[[c for c in ["record_id","source_database","title","year","oa_year","doi"] if c in dropped.columns]],
          os.path.join(OUT_DIR,"qc_dropped_pre2000.xlsx"),"dropped")
print(f"\nSaved {OUT_PATH} ({len(kept)} records) -> feed into dedup (08_dedupe_oa_aware.py)")
print(f"Saved qc_dropped_pre2000.xlsx ({len(dropped)} records) for your PRISMA pre-screening exclusion count")
print("\nNEXT: in 08_dedupe_oa_aware.py set IN_PATH to master_enriched_2000.xlsx, then re-run")
print("dedup -> 09_canonical_coverage -> (re-enrich aff) -> 12_screening_readiness -> 13 backfill.")
