# ============================================================
#  FUNDER FIX + TRUE COVERAGE  (Colab) — run on master_enriched.xlsx
#  No re-enrichment needed. Fixes two issues:
#    1. oa_funders ~0%  -> OpenAlex grants[] is sparse; Crossref is the real
#       funder source. Build a unified `funders` column (OpenAlex | Crossref).
#    2. has_funding 82.7% was MISLEADING (it counted "determination made" =
#       match rate, not "% funded"). Replace with an honest funded flag and
#       report the TRUE positive rate.
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pandas openpyxl

import os, re
import pandas as pd, numpy as np

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
IN_PATH = os.path.join(OUT_DIR, "master_enriched.xlsx")

df = pd.read_excel(IN_PATH, sheet_name="enriched_master", dtype=str).fillna("")
print("Loaded:", len(df), "records")

def nz(col): return df[col].astype(str).str.len() > 0 if col in df.columns else pd.Series(False, index=df.index)

# ---- 1. TRUE funder picture (what actually populated) ----
print("\n=== RAW funder/affiliation coverage (the columns the report hid) ===")
for c in ["oa_funders","cr_funders","oa_has_funding","cr_has_funding",
          "oa_affiliations_raw","cr_affiliations","epmc_affiliation"]:
    if c in df.columns:
        print(f"  {c:22s} non-empty: {nz(c).mean()*100:5.1f}%  "
              f"| =='yes': {(df[c].eq('yes')).mean()*100:5.1f}%" if 'has_funding' in c
              else f"  {c:22s} non-empty: {nz(c).mean()*100:5.1f}%")

# ---- 2. Unified funder column: OpenAlex grants -> Crossref funder registry ----
df["funders_unified"] = df["oa_funders"] if "oa_funders" in df.columns else ""
if "cr_funders" in df.columns:
    m = df["funders_unified"].eq("") & df["cr_funders"].ne("")
    df.loc[m, "funders_unified"] = df.loc[m, "cr_funders"]

# ---- 3. HONEST funded flag: yes if any funder/grant evidence, no if matched-but-none, "" if unmatched ----
matched = nz("oa_match_method") if "oa_match_method" in df.columns else nz("oa_id")
funded_evidence = (
    nz("funders_unified")
    | (df.get("oa_has_funding","").eq("yes") if "oa_has_funding" in df.columns else False)
    | (df.get("cr_has_funding","").eq("yes") if "cr_has_funding" in df.columns else False)
)
df["is_funded"] = np.where(funded_evidence, "yes", np.where(matched, "no", "unknown"))

print("\n=== HONEST funding numbers ===")
print(f"  funders_unified non-empty: {nz('funders_unified').mean()*100:.1f}%")
vc = df["is_funded"].value_counts()
for k in ["yes","no","unknown"]:
    n = int(vc.get(k,0)); print(f"  is_funded == {k:8s}: {n:6d} ({n/len(df)*100:.1f}%)")
print("  (Among matched records, 'yes' is your real funded rate; 'unknown' = couldn't determine)")

# ---- 4. save back (new columns only; everything else preserved) ----
def safe_xlsx(frame, path, sheet):
    ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out = frame.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: ILLEGAL.sub(" ", str(v))[:32000])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name=sheet, index=False)

safe_xlsx(df, IN_PATH, "enriched_master")   # overwrite in place with the two fixed columns added
print(f"\nSaved fixed file (added: funders_unified, is_funded) -> {IN_PATH}")
print("Use 'funders_unified' and 'is_funded' downstream instead of oa_funders/has_funding.")
