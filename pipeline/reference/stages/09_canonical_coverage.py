# ============================================================
#  CANONICAL-SET COVERAGE + PRISMA  (Colab)
#  Run on master_dedupe_audit.xlsx. Measures the TRUE analytical
#  dataset (canonical rows only) and produces PRISMA identification
#  numbers. Decides whether a targeted re-enrichment is needed.
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pandas openpyxl

import os, re
import pandas as pd, numpy as np

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
df = pd.read_excel(os.path.join(OUT_DIR,"master_dedupe_audit.xlsx"), sheet_name="dedupe_audit", dtype=str).fillna("")

df["is_canonical"] = df["is_canonical"].astype(str).str.lower().isin(["true","1","yes"])
canon = df[df["is_canonical"]].copy()
print(f"All records: {len(df)} | Canonical studies: {len(canon)}")

def cov(frame, col): 
    return round((frame[col].astype(str).str.len()>0).mean()*100,1) if col in frame.columns else None

# fields powering your ranked bibliometric priorities
FIELDS = {
 # topic evolution (priority 1)
 "oa_topic":"topic","oa_subfield":"subfield","oa_field":"field","oa_domain":"domain",
 # geography/institutions (priority 2)
 "oa_countries":"countries","oa_institutions":"institutions","affiliations":"affiliations",
 # funding (priority 3)
 "funders_unified":"funders","is_funded":"funded_flag",
 # co-authorship (priority 4)
 "authors":"authors","oa_author_orcids":"author_orcids",
 # citations (priority 5)
 "citation_count":"citation_count",
 # core
 "title":"title","abstract":"abstract","year":"year","doi":"doi","oaid_norm":"openalex_id",
 "journal":"journal","document_type":"document_type","oa_language":"language",
}
rows=[]
for col,name in FIELDS.items():
    rows.append((name, col, cov(df,col), cov(canon,col)))
covtbl = pd.DataFrame(rows, columns=["field","column","all_records_pct","canonical_pct"])
print("\n=== COVERAGE: all records vs CANONICAL (your analytical set) ===")
print(covtbl.to_string(index=False))

# flag fields that are weak on the canonical set, ranked by your priorities
print("\n=== Re-enrichment guidance (for your top bibliometric fields) ===")
priority_fields = ["oa_topic","oa_field","oa_countries","oa_institutions","funders_unified","is_funded"]
for col in priority_fields:
    c = cov(canon, col)
    if c is None: continue
    verdict = "STRONG — analyze as-is" if c>=85 else ("OK — usable, minor gaps" if c>=75 else "WEAK — consider targeted re-enrichment")
    print(f"  {col:18s} {c:5.1f}%  -> {verdict}")

# PRISMA identification numbers
nr_all = len(df); ns = len(canon)
print("\n=== PRISMA identification ===")
if "source_database" in df.columns:
    print("Records identified per database:")
    print(df["source_database"].value_counts().to_string())
print(f"\nRecords after de-duplication (= records to screen): {ns}")
if "n_sources" in canon.columns:
    print("\nStudies by # databases:")
    print(canon["n_sources"].astype(str).value_counts().sort_index().to_string())

# save
def safe_xlsx(frame,path,sheet):
    with pd.ExcelWriter(path,engine="openpyxl") as xw: frame.to_excel(xw,sheet_name=sheet,index=False)
safe_xlsx(covtbl, os.path.join(OUT_DIR,"canonical_coverage.xlsx"), "coverage")
print(f"\nSaved canonical_coverage.xlsx")
print("\nDECISION: any field flagged WEAK above is a candidate for targeted re-enrichment")
print("BEFORE the bibliometric analysis. Fields >=85% are ready to analyze now.")
