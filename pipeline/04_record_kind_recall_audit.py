# Auto-generated mirror of notebooks/01_corpus/04_record_kind_recall_audit.ipynb

# # 04 · Record kinds + recall audit (GTRA v1.0)
# 
# **A. Record kinds (free, seconds).** Flags records that are not studies: Zenodo taxonomic treatments ("Amblyomma ovale Koch 1844", about 1,200, mostly 2025), figures/supplements, notices, other repository items. Adds `record_kind` and `count_as_study` to `works_screened.parquet`. Without it, 2025 shows a false jump (3,096 included → 1,936 real studies).
# 
# **B. Recall audit (~30 min API + screening).** 11,197 records of the old human-checked set are not in the new harvest. For each one, OpenAlex itself says why (`not_found`, `year_out`, `type_out`, `query_match_now`, `fulltext_only`), then the same screening prompt decides whether it is tick-focused. Result: **harvest recall** for tick-focused papers, overall and within scope.
# 
# Code: `src/gtra/record_kind.py`, `src/gtra/recall_audit.py`, `src/gtra/screen_llm.py`.

# 1 · Setup
from google.colab import drive, userdata
drive.mount('/content/drive')
pass  # shell: pip -q install openai pandas pyarrow openpyxl xlsxwriter requests
import os, sys, json, importlib
if not os.path.exists('/content/gtra'):
    pass  # shell: git clone -q https://github.com/cognosx/global-tick-research-atlas /content/gtra
else:
    pass  # shell: git -C /content/gtra pull -q
sys.path.insert(0, '/content/gtra/src')
for k in ["OPENALEX_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]:
    try: os.environ[k] = userdata.get(k); print(k, "set")
    except Exception: print(k, "MISSING")

import numpy as np, pandas as pd
from pathlib import Path
import gtra.record_kind as RK, gtra.recall_audit as RA, gtra.screen_llm as G
from gtra.openalex_harvester import OpenAlexHarvester
for m in (RK, RA, G): importlib.reload(m)
pd.set_option('display.max_columns', None); pd.set_option('display.max_colwidth', 110)

# 2 · Configuration (same paths as notebooks 01-03)
ROOT   = "/content/drive/Othercomputers/My laptop/D:/COGNOS X bon/TICKS/ticks_research"   # <-- yours
RUN    = "openalex_2026-11-26"
PRE    = Path(f"{ROOT}/preprocessing_data/{RUN}")
SCREEN = PRE / "screening"
AUDIT  = PRE / "recall_audit"; AUDIT.mkdir(parents=True, exist_ok=True)
MANIFEST = json.load(open(f"{ROOT}/data_raw/{RUN}/harvest_manifest.json"))

# screening settings = notebook 03 (same prompt version, same cache -> nothing is paid twice)
G.PROVIDERS["gpt_6_luna"]["model"] = "gpt-6-luna"
G.PROVIDERS["deepseek"]["model"]   = "deepseek-v4-flash"
G.PRIMARY, G.TIEBREAK = ["gpt_6_luna", "deepseek"], "gpt_6_luna"
G.DRIVE_CACHE = str(SCREEN / "gtra_screen_cache.jsonl"); G.LOCAL_CACHE = "/content/gtra_screen_cache.jsonl"
G.BACKUP_SECONDS = 30
print("harvest:", MANIFEST["search_field"], MANIFEST["filters"], "| prompt", G.PROMPT_VERSION)

# ## A · Record kinds

scr = pd.read_parquet(SCREEN / "works_screened.parquet")
scr = RK.add_record_kind(scr)
scr.to_parquet(SCREEN / "works_screened.parquet", index=False)        # adds record_kind + count_as_study
print(pd.crosstab(scr["record_kind"], scr["gtra_layer"], margins=True).to_string())
yr = pd.to_numeric(scr["oa_year"], errors="coerce")
inc = scr["gtra_include"].fillna(False).astype(bool)
t = pd.DataFrame({"included (gtra_include)": inc.groupby(yr).sum(),
                  "studies (count_as_study)": scr.groupby(yr)["count_as_study"].sum()}).astype(int)
t.tail(8)

# look before you trust: 20 treatments and 20 repository_other records the models included
show = ["oa_year", "oa_doi", "oa_title", "gtra_class"]
print(scr[scr["record_kind"] == "taxonomic_treatment"][show].sample(20, random_state=1))
print(scr[(scr["record_kind"] == "repository_other") & scr["gtra_include"]][show].head(20))

# ## B · Recall audit against the old human-checked set

gold = pd.read_parquet("/content/gtra/data/gold/gold_included_ids.parquet")["oa_id"].map(G.short_id)
harvested = set(pd.read_parquet(PRE / "dedupe/dedupe_map.parquet", columns=["oa_id"])["oa_id"].map(G.short_id))
missing_ids = sorted(set(gold) - harvested)
print(f"old gold: {gold.nunique():,} | in new harvest: {len(set(gold) & harvested):,} | missing: {len(missing_ids):,}")

# OpenAlex explains each miss (results saved; delete the file to redo)
raw_path = AUDIT / "recall_audit_raw.parquet"
if raw_path.exists():
    aud = pd.read_parquet(raw_path)
else:
    h = OpenAlexHarvester(api_key=os.environ.get("OPENALEX_API_KEY"))
    y0, y1 = map(int, MANIFEST["filters"]["publication_year"].split("-"))
    aud = RA.audit(h, missing_ids, MANIFEST["block_A"], (y0, y1), MANIFEST["filters"]["type"],
                   MANIFEST["search_field"])
    aud.astype({"audit_reason": str}).to_parquet(raw_path, index=False)
aud["audit_reason"].value_counts()

# Screen every miss that OpenAlex still returns, with the SAME prompt (cached, resumable)
SAMPLE = None          # e.g. 2000 for a quicker estimate; None = all
todo = aud[aud["audit_reason"] != "not_found"].copy()
if SAMPLE and len(todo) > SAMPLE:
    todo = todo.sample(SAMPLE, random_state=11)
prep = G.prepare(todo)
res = G.screen(prep)
aud2 = aud.merge(res, on="oa_id", how="left")
aud2 = RK.add_record_kind(aud2)
aud2.to_parquet(AUDIT / "recall_audit_screened.parquet", index=False)
pd.crosstab(aud2["audit_reason"], aud2["gtra_layer"].fillna("(not screened)"), margins=True)

# Harvest recall for tick-focused papers (core + review), using real studies only
present = scr[scr["oa_id"].map(G.short_id).isin(set(gold)) & scr["record_kind"].isin(["research", "repository_other"])]
miss = aud2[aud2["record_kind"].isin(["research", "repository_other"]) | aud2["gtra_layer"].isna()]
rec = RA.harvest_recall(present["gtra_layer"], miss)
rec.to_csv(AUDIT / "harvest_recall.csv", index=False)
rec

# The misses that matter: tick-focused AND inside the harvest scope
gap = aud2[aud2["gtra_layer"].isin(["core", "review"]) & aud2["audit_reason"].isin(["query_match_now", "fulltext_only"])]
print(f"{len(gap):,} tick-focused papers missed inside scope")
print(gap["audit_reason"].value_counts().to_string())
gap[["audit_reason", "oa_year", "oa_type", "oa_title", "gtra_class", "gtra_evidence"]].sample(min(30, len(gap)), random_state=3)

# **How to read it**
# - `query_match_now` + core/review: the harvest **should** have found these. If there are many, re-run the harvest (index changed) or add them as a supplement.
# - `fulltext_only` + core/review: tick-focused by their abstract, but the tick term was only in the full text according to OpenAlex (often non-English abstracts or missing abstracts). Decide: accept as a known limit, or add as a supplement flagged `source = recall_audit`.
# - `year_out` / `type_out`: out of scope by design (pre-2000, book chapters…). Report, don't add.

# OPTIONAL: save the in-scope tick-focused misses as a supplement for v1.0 (flagged, not merged yet)
supp = gap.assign(source="recall_audit")
supp.to_parquet(AUDIT / "works_supplement_recall_audit.parquet", index=False)
print(f"{len(supp):,} records -> {AUDIT / 'works_supplement_recall_audit.parquet'}")
