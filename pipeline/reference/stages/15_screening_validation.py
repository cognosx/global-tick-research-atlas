# ============================================================
#  SCREENING VALIDATION  (Colab)
#  Measures LLM-consensus accuracy against a HUMAN-labelled random
#  sample. Produces sensitivity/specificity/PPV/NPV and Cohen's kappa
#  (LLM vs human, and per-model vs human). This is what makes the
#  LLM-assisted screening publishable.
#
#  WORKFLOW:
#   1. Run STEP A to draw a stratified random sample -> validation_sample.xlsx
#   2. YOU hand-screen the 'human_label' column (include/exclude) in that file
#   3. Run STEP B to compute agreement metrics
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pandas openpyxl scikit-learn

import os, re
import pandas as pd, numpy as np

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
RESULTS = os.path.join(OUT_DIR,"screening_results.xlsx")
SAMPLE  = os.path.join(OUT_DIR,"validation_sample.xlsx")

STEP = "A"          # "A" = draw sample;  "B" = compute metrics after you label it
SAMPLE_SIZE = 250   # 200-300 typical; stratified by verdict for balance

def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)

if STEP=="A":
    df = pd.read_excel(RESULTS, sheet_name="screening", dtype=str).fillna("")
    # stratified sample across verdicts so we validate include AND exclude AND human-review
    parts=[]
    rng=np.random.RandomState(2024)
    for v,g in df.groupby("screen_verdict"):
        k=min(len(g), max(20, int(SAMPLE_SIZE*len(g)/len(df))))
        parts.append(g.sample(k, random_state=rng))
    samp=pd.concat(parts).sample(frac=1, random_state=rng)   # shuffle
    samp["human_label"]=""        # <-- YOU fill: include / exclude
    samp["human_notes"]=""
    cols=[c for c in ["study_id","record_id","title","abstract","keywords",
          "screen_verdict","screen_reason","human_label","human_notes"] if c in samp.columns]
    safe_xlsx(samp[cols], SAMPLE, "validate")
    print(f"Drew {len(samp)} records -> {SAMPLE}")
    print("NOW: open it, fill 'human_label' with include/exclude for each, save, set STEP='B', re-run.")
    print("Screen BLIND if you can: hide screen_verdict while labelling to avoid bias.")

elif STEP=="B":
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    s = pd.read_excel(SAMPLE, sheet_name="validate", dtype=str).fillna("")
    s["human_label"]=s["human_label"].str.strip().str.lower()
    s=s[s["human_label"].isin(["include","exclude"])]
    print(f"Labelled records: {len(s)}")
    # collapse LLM verdict to include vs not-include (human_review counts as include for recall safety)
    s["llm_inc"]=s["screen_verdict"].isin(["include","human_review"]).astype(int)
    s["hum_inc"]=s["human_label"].eq("include").astype(int)
    tp=int(((s.llm_inc==1)&(s.hum_inc==1)).sum()); fp=int(((s.llm_inc==1)&(s.hum_inc==0)).sum())
    fn=int(((s.llm_inc==0)&(s.hum_inc==1)).sum()); tn=int(((s.llm_inc==0)&(s.hum_inc==0)).sum())
    sens=tp/(tp+fn) if tp+fn else float('nan')    # recall on includes — MOST important for screening
    spec=tn/(tn+fp) if tn+fp else float('nan')
    ppv =tp/(tp+fp) if tp+fp else float('nan')
    npv =tn/(tn+fn) if tn+fn else float('nan')
    kappa=cohen_kappa_score(s["hum_inc"],s["llm_inc"])
    print("\n=== LLM CONSENSUS vs HUMAN ===")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  Sensitivity (recall, catch true includes): {sens:.3f}   <-- want HIGH (>0.95)")
    print(f"  Specificity:                                {spec:.3f}")
    print(f"  PPV (precision):                            {ppv:.3f}")
    print(f"  NPV:                                        {npv:.3f}")
    print(f"  Cohen's kappa:                              {kappa:.3f}")
    if fn>0:
        print(f"\n  WARNING: {fn} studies the humans included were EXCLUDED by the LLM.")
        print("  These are the dangerous errors in a systematic review. Inspect them:")
        miss=s[(s.llm_inc==0)&(s.hum_inc==1)]
        for _,r in miss.head(10).iterrows(): print("   -", str(r.get("title",""))[:70])
    # per-model kappa if columns present
    for m in ["claude","gpt","gemini"]:
        col=f"{m}_verdict"
        if col in s.columns and s[col].ne("").any():
            mi=s[col].isin(["include"]).astype(int)
            try: print(f"  kappa {m} vs human: {cohen_kappa_score(s['hum_inc'],mi):.3f}")
            except Exception: pass
    print("\nReport sensitivity + kappa in your methods. Sensitivity>0.95 supports LLM-assisted screening;")
    print("lower -> widen human review (e.g. send all 'exclude' with low model confidence to humans too).")
