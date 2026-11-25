# ============================================================
#  SCREENING-READINESS CHECK  (Colab)
#  Measures whether title / abstract / keywords are USABLE for
#  LLM title-abstract screening (not just non-empty). Flags
#  screening-weak records and quantifies what a backfill would fix.
#  Run on master_canonical_enriched.xlsx (or master_canonical.xlsx).
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pandas openpyxl

import os, re
import pandas as pd, numpy as np

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
for cand in ["master_canonical_enriched.xlsx","master_canonical_clean.xlsx","master_canonical.xlsx"]:
    p=os.path.join(OUT_DIR,cand)
    if os.path.exists(p): IN_PATH=p; break
print("Using:", os.path.basename(IN_PATH))
df = pd.read_excel(IN_PATH, sheet_name="canonical", dtype=str).fillna("")
print("Canonical studies:", len(df))

# ---- define USABLE (not just present) ----
JUNK_ABS = re.compile(r'^(no abstract|abstract not available|not available|n/?a|none|\[.*\])\.?$', re.I)
def usable_abstract(s):
    s=str(s or "").strip()
    if len(s) < 100: return False           # screening needs real text; <100 chars is a stub
    if JUNK_ABS.match(s): return False
    # mostly-non-alpha (e.g. leftover markup/refs) -> not usable
    alpha=sum(c.isalpha() for c in s)
    return alpha/max(len(s),1) > 0.5
def usable_title(s):
    s=str(s or "").strip()
    return len(s) >= 12 and s.lower() not in {"untitled","no title","title"}
def has_keywords(s):
    return len(str(s or "").strip()) > 0

df["ok_title"]    = df["title"].map(usable_title)
df["ok_abstract"] = df["abstract"].map(usable_abstract)
df["ok_keywords"] = df["keywords"].map(has_keywords) if "keywords" in df.columns else False

# language flag (screening is usually English; non-English abstracts need handling)
lang_col = "oa_language" if "oa_language" in df.columns else None
df["non_english"] = (df[lang_col].ne("") & df[lang_col].ne("en")) if lang_col else False

# ---- screening capability per record ----
def screen_mode(r):
    if r["ok_title"] and r["ok_abstract"]: return "title+abstract (strong)"
    if r["ok_title"] and not r["ok_abstract"]: return "title-only (weak)"
    if not r["ok_title"]: return "no usable title (very weak)"
    return "other"
df["screen_mode"] = df.apply(screen_mode, axis=1)

n=len(df)
print("\n=== USABLE field coverage (screening-grade, not just non-empty) ===")
print(f"  usable title:    {df['ok_title'].mean()*100:5.1f}%")
print(f"  usable abstract: {df['ok_abstract'].mean()*100:5.1f}%   (vs non-empty {(df['abstract'].str.len()>0).mean()*100:.1f}%)")
print(f"  any keywords:    {df['ok_keywords'].mean()*100:5.1f}%")
if lang_col:
    print(f"  non-English:     {df['non_english'].mean()*100:5.1f}%  (need translation or multilingual LLM)")

print("\n=== Screening capability of the corpus ===")
print(df["screen_mode"].value_counts().to_string())
weak = df[df["screen_mode"].str.contains("weak")]
print(f"\nScreening-WEAK studies (title-only or worse): {len(weak)} ({len(weak)/n*100:.1f}%)")

# which weak records COULD be fixed (have a DOI/PMID to backfill from)
fixable = weak[(weak.get("doi","").str.len()>0) | (weak.get("oa_pmid","").str.len()>0) if "oa_pmid" in weak else weak.get("doi","").str.len()>0]
print(f"  ...of which have a DOI/PMID to attempt abstract backfill: {len(fixable)}")
print(f"  ...truly unfixable (no identifier, no abstract): {len(weak)-len(fixable)}")

# ---- export screening-weak list for targeted backfill / manual handling ----
def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)
cols=[c for c in ["study_id","record_id","screen_mode","title","abstract","keywords","doi","oa_pmid","oa_language","non_english"] if c in df.columns]
safe_xlsx(weak[cols], os.path.join(OUT_DIR,"screening_weak_records.xlsx"),"weak")
safe_xlsx(df[cols+["ok_title","ok_abstract","ok_keywords"]], os.path.join(OUT_DIR,"screening_readiness.xlsx"),"readiness")
print("\nSaved screening_readiness.xlsx (full per-record flags)")
print("      screening_weak_records.xlsx (the records to backfill or screen on title)")
print("\nDECISION: if 'fixable' weak records are many, run a targeted abstract backfill")
print("(Europe PMC + Semantic Scholar by DOI/PMID) on JUST that list before screening.")
