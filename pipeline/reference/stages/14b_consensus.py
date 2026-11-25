# ============================================================
#  CONSENSUS COMBINER  (Colab) — run after 1, 2, or 3 model passes
#  Reads whatever {model}_verdict columns exist in screening_results.xlsx
#  and computes the final verdict. Rule (your decision):
#    INCLUDE if ANY model includes; any UNCERTAIN or include/exclude
#    DISAGREEMENT -> human_review. Works with 1, 2, or 3 models present.
#  Re-runnable: re-run after each new model pass to refresh consensus.
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pandas openpyxl

import os, re
import pandas as pd, numpy as np

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
RESULTS = os.path.join(OUT_DIR, "screening_results.xlsx")
HUMANQ  = os.path.join(OUT_DIR, "screening_for_human_review.xlsx")

df = pd.read_excel(RESULTS, sheet_name="screening", dtype=str).fillna("")
MODELS = [m for m in ["claude","gpt","gemini"] if f"{m}_verdict" in df.columns and df[f"{m}_verdict"].ne("").any()]
print("Models present:", MODELS, f"({len(df)} records)")
if not MODELS: raise SystemExit("No model verdict columns found — run at least one 14a pass first.")

def consensus(row):
    vs=[row[f"{m}_verdict"].lower() for m in MODELS if row[f"{m}_verdict"] in ("include","exclude","uncertain")]
    n_done=len(vs)
    if n_done==0: return pd.Series(["pending","no_model_verdict_yet","no"])
    inc=vs.count("include"); exc=vs.count("exclude"); unc=vs.count("uncertain")
    if unc>0:                 return pd.Series(["human_review","uncertain_present","yes"])
    if inc>0 and exc>0:       return pd.Series(["human_review","model_disagreement","yes"])
    if inc>0:                 return pd.Series(["include","any_model_include","no"])
    return pd.Series(["exclude","all_exclude","no"])

df[["screen_verdict","screen_detail","screen_disagreement"]]=df.apply(consensus,axis=1)
# majority reason code across present models (for PRISMA "excluded with reasons")
def maj_reason(row):
    rs=[row[f"{m}_reason"] for m in MODELS if row.get(f"{m}_reason","")]
    return max(set(rs),key=rs.count) if rs else ""
df["screen_reason"]=df.apply(maj_reason,axis=1)
# title-only includes still need eligibility review (earlier decision)
if "eligibility_review_required" in df.columns:
    m=df["eligibility_review_required"].eq("yes") & df["screen_verdict"].eq("include")
    df.loc[m,"screen_detail"]=df.loc[m,"screen_detail"]+"+title_only_eligibility_review"

print(f"\n=== CONSENSUS on {len(MODELS)} model(s) ===")
print(df["screen_verdict"].value_counts().to_string())
print("\nExclusion reasons:")
print(df[df["screen_verdict"].eq("exclude")]["screen_reason"].value_counts().to_string())
n_pending=(df["screen_verdict"].eq("pending")).sum()
if n_pending: print(f"\n{n_pending} records still PENDING (no model has screened them yet)")

def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)
safe_xlsx(df,RESULTS,"screening")
human=df[df["screen_verdict"].isin(["human_review"]) | df["screen_detail"].str.contains("eligibility_review")]
cols=[c for c in ["study_id","record_id","title","abstract","screen_verdict","screen_detail","screen_reason"]+
      [f"{m}_verdict" for m in MODELS]+[f"{m}_reason" for m in MODELS] if c in df.columns]
safe_xlsx(human[cols],HUMANQ,"human_review")
print(f"\nUpdated {RESULTS}\nWrote {HUMANQ} ({len(human)} for human review)")
print(f"\nNOTE: with {len(MODELS)} model(s), this consensus is preliminary unless all 3 are done.")
print("Re-run this after each additional model pass to refresh.")
