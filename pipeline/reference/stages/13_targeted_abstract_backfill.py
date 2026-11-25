# ============================================================
#  TARGETED ABSTRACT BACKFILL + SCREENING FLAGS  (Colab)
#  Runs on the ~846 fixable screening-weak records (have DOI/PMID,
#  missing usable abstract). Sources: Europe PMC -> Semantic Scholar.
#  Then assigns final screening flags for the whole canonical set:
#    - screen_input : "title+abstract" | "title_only"
#    - screen_confidence : "normal" | "low"
#    - eligibility_review_required : "yes" for title_only (your decision)
#  Output: master_canonical_screen_ready.xlsx  -> input to the LLM harness
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install requests pandas openpyxl

import os, re, time
import pandas as pd, numpy as np, requests

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
# full canonical set (enriched)
for cand in ["master_canonical_enriched.xlsx","master_canonical_clean.xlsx","master_canonical.xlsx"]:
    p=os.path.join(OUT_DIR,cand)
    if os.path.exists(p): IN_PATH=p; break
print("Canonical file:", os.path.basename(IN_PATH))
df = pd.read_excel(IN_PATH, sheet_name="canonical", dtype=str).fillna("")
print("Studies:", len(df))

POLITE_EMAIL="cognosxx@gmail.com"
S2_API_KEY=""   # optional
CKPT=os.path.join(OUT_DIR,"backfill_checkpoint.parquet")

# ---- usable-abstract test (same as readiness check) ----
JUNK_ABS=re.compile(r'^(no abstract|abstract not available|not available|n/?a|none|\[.*\])\.?$',re.I)
def usable_abstract(s):
    s=str(s or "").strip()
    if len(s)<100: return False
    if JUNK_ABS.match(s): return False
    a=sum(c.isalpha() for c in s); return a/max(len(s),1)>0.5
def usable_title(s):
    s=str(s or "").strip()
    return len(s)>=12 and s.lower() not in {"untitled","no title","title"}
def norm_doi(d):
    d=str(d or "").strip().lower()
    for p in ("https://doi.org/","doi:"): d=d.replace(p,"")
    return d if d.startswith("10.") else ""
def get_pmid(row):
    for c in ["oa_pmid","pmid"]:
        if c in row and re.fullmatch(r'\d{5,9}', str(row.get(c,"")).strip()): return str(row[c]).strip()
    return ""

for c in ["abstract","doi","oa_pmid","oa_language","abstract_source"]:
    if c not in df.columns: df[c]=""

# ---- identify fixable: weak (no usable abstract) but usable title + has DOI/PMID ----
df["_ok_abs"]=df["abstract"].map(usable_abstract)
df["_pmid"]=df.apply(get_pmid,axis=1)
fixable = df[(~df["_ok_abs"]) & ((df["doi"].map(norm_doi)!="") | (df["_pmid"]!=""))].copy()
print(f"Fixable weak records to attempt: {len(fixable)}")

# resume
done=set()
if os.path.exists(CKPT):
    prev=pd.read_parquet(CKPT).set_index("record_id")
    df=df.set_index("record_id"); df.update(prev); df=df.reset_index()
    done=set(prev.index.tolist()); print("Resumed:",len(done))
    df["_ok_abs"]=df["abstract"].map(usable_abstract)
    fixable=fixable[~fixable["record_id"].isin(done)]

sess=requests.Session(); sess.headers.update({"User-Agent":f"backfill/1.0 (mailto:{POLITE_EMAIL})"})
def get(url,**kw):
    try:
        r=sess.get(url,timeout=30,**kw)
        if r.status_code==200: return r.json()
    except Exception: return None
    return None

rec_epmc=rec_s2=0
# ---- Europe PMC pass ----
for n,(idx,row) in enumerate(fixable.iterrows()):
    doi=norm_doi(row["doi"]); pmid=row["_pmid"]
    q = f'DOI:"{doi}"' if doi else f'EXT_ID:{pmid} AND SRC:MED'
    j=get("https://www.ebi.ac.uk/europepmc/webservices/rest/search"
          f"?query={requests.utils.quote(q)}&resultType=core&format=json&pageSize=1")
    time.sleep(0.1)
    try:
        res=(j or {}).get("resultList",{}).get("result",[])
        if res:
            ab=res[0].get("abstractText","") or ""
            if usable_abstract(ab):
                df.loc[df["record_id"].eq(row["record_id"]),"abstract"]=ab
                df.loc[df["record_id"].eq(row["record_id"]),"abstract_source"]="europepmc_backfill"
                rec_epmc+=1
    except Exception: pass
    if n and n%200==0:
        df.to_parquet(CKPT,index=False); print(f"  EPMC ...{n}/{len(fixable)} (recovered {rec_epmc})")

# ---- Semantic Scholar batch for still-missing ----
df["_ok_abs"]=df["abstract"].map(usable_abstract)
still=df[(~df["_ok_abs"]) & ((df["doi"].map(norm_doi)!="")|(df["_pmid"]!=""))]
ids=[]; idx_by={}
for idx,row in still.iterrows():
    k=f'DOI:{norm_doi(row["doi"])}' if norm_doi(row["doi"]) else f'PMID:{row["_pmid"]}'
    ids.append(k); idx_by.setdefault(k,[]).append(row["record_id"])
print(f"\nSemantic Scholar fallback for {len(ids)} still-missing...")
hdr={"x-api-key":S2_API_KEY} if S2_API_KEY else {}
for i in range(0,len(ids),500):
    batch=ids[i:i+500]
    try:
        r=sess.post("https://api.semanticscholar.org/graph/v1/paper/batch?fields=abstract,tldr",
                    json={"ids":batch},headers=hdr,timeout=60)
        time.sleep(1.1 if not S2_API_KEY else 0.2)
        if r.status_code==200:
            for k,obj in zip(batch,r.json()):
                if not obj: continue
                ab=obj.get("abstract") or ((obj.get("tldr") or {}).get("text","") if obj.get("tldr") else "")
                if usable_abstract(ab):
                    for rid in idx_by.get(k,[]):
                        df.loc[df["record_id"].eq(rid),"abstract"]=ab
                        df.loc[df["record_id"].eq(rid),"abstract_source"]="s2_backfill"
                        rec_s2+=1
    except Exception as e: print("  S2 error",e)
    df.to_parquet(CKPT,index=False)
print(f"Recovered: EuropePMC +{rec_epmc} | Semantic Scholar +{rec_s2} | total +{rec_epmc+rec_s2}")

# ---- FINAL screening flags for the WHOLE canonical set ----
df["_ok_abs"]=df["abstract"].map(usable_abstract)
df["_ok_title"]=df["title"].map(usable_title)
df["screen_input"]=np.where(df["_ok_abs"],"title+abstract","title_only")
df["screen_confidence"]=np.where(df["_ok_abs"],"normal","low")
# your decision: title-only -> mandatory human review at eligibility
df["eligibility_review_required"]=np.where(df["_ok_abs"],"no","yes")
# records with no usable title at all -> always human review
df.loc[~df["_ok_title"],"eligibility_review_required"]="yes"
df.loc[~df["_ok_title"],"screen_confidence"]="very_low"

print("\n=== FINAL screening readiness (after backfill) ===")
print(f"  usable abstract now: {df['_ok_abs'].mean()*100:.1f}%  (was 88.7%)")
print(df["screen_input"].value_counts().to_string())
print(f"  flagged for mandatory eligibility review (title-only): {(df['eligibility_review_required'].eq('yes')).sum()}")

def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.drop(columns=[c for c in ["_ok_abs","_ok_title","_pmid"] if c in frame.columns]).copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)
OUTP=os.path.join(OUT_DIR,"master_canonical_screen_ready.xlsx")
safe_xlsx(df,OUTP,"canonical")
print(f"\nSaved {OUTP} -> input to the multi-LLM screening harness")
print("Columns added: screen_input, screen_confidence, eligibility_review_required")
