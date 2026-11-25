# ============================================================
#  CROSS-DATABASE DEDUPLICATION v4 — OpenAlex-ID aware  (Colab)
#  Tick-Borne Diseases / Climate / Sub-Saharan Africa review
#
#  RUN ON: master_enriched.xlsx (after enrichment + fix_funders)
#
#  Upgrades over v3, enabled by enrichment:
#   * Stage 0: exact OpenAlex work id (oa_id) — authoritative primary key.
#              Merges preprint+published (shared oa_id) per review decision.
#   * Smarter guard: SAME oa_id always merges; DIFFERENT oa_ids NEVER merge
#     (provably distinct works); DOI guard still applies when oa_ids absent.
#   * Matching ladder: oa_id -> DOI -> title+authors_fp+year ->
#                      title+journal+year -> title+year(author-gated) -> fuzzy
#   * Uses OpenAlex-NORMALIZED fields where present (oa_title, oa_authors,
#     oa_issn) so fingerprints are consistent, not 8 different formats.
#   * Canonical row = richest science-of-science metadata, oa_id-matched preferred.
#
#  Output: master_dedupe_audit.xlsx, master_canonical.xlsx, qc/*.xlsx
# ============================================================

## ---- 1. setup ----
from google.colab import drive
drive.mount('/content/drive')
!pip -q install rapidfuzz openpyxl pandas

import os, re, time, unicodedata
import pandas as pd, numpy as np
from rapidfuzz import fuzz
pd.set_option('display.max_columns', None)

## ---- 2. config ----
FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
IN_PATH = os.path.join(OUT_DIR, "master_enriched.xlsx")
QC_DIR  = os.path.join(OUT_DIR, "qc"); os.makedirs(QC_DIR, exist_ok=True)
AUDIT_PATH = os.path.join(OUT_DIR, "master_dedupe_audit.xlsx")
CANON_PATH = os.path.join(OUT_DIR, "master_canonical.xlsx")

FUZZY_THRESHOLD      = 90
FUZZY_YEAR_TOLERANCE = 2
MIN_TITLE_TOKENS     = 3
MIN_TITLE_CHARS      = 15
AUTHOR_FP_MODE       = "last_initial"

## ---- 3. load ----
df = pd.read_excel(IN_PATH, sheet_name="enriched_master", dtype=str).fillna("").reset_index(drop=True)
print("Loaded:", len(df), "records,", len(df.columns), "columns")
if "record_id" not in df.columns:
    df["record_id"] = [f"REC-{i+1:06d}" for i in range(len(df))]

## ---- 4. choose best available field (OpenAlex-normalized preferred) ----
def best(col_oa, col_orig):
    a = df[col_oa] if col_oa in df.columns else pd.Series("", index=df.index)
    b = df[col_orig] if col_orig in df.columns else pd.Series("", index=df.index)
    return a.where(a.astype(str).str.len() > 0, b)
df["m_title"]   = best("oa_title", "title")
df["m_authors"] = best("oa_authors", "authors")
df["m_journal"] = best("oa_journal", "journal")
df["m_year"]    = best("oa_year", "year")

## ---- 5. normalizers ----
_WS=re.compile(r"\s+"); _PUNCT=re.compile(r"[^\w\s]",re.UNICODE)
NA_LIKE={"","none","null","nan","n/a","[]","{}","na"}
def strip_acc(x): return "".join(c for c in unicodedata.normalize("NFKD",x) if not unicodedata.combining(c))
def clean_title(s):
    s=strip_acc(str(s or "").lower().strip())
    if s in NA_LIKE: return ""
    s=_PUNCT.sub(" ",s); return _WS.sub(" ",s).strip()
def norm_doi(d):
    d=str(d or "").strip().lower()
    for p in ("https://doi.org/","http://dx.doi.org/","https://dx.doi.org/","doi:"): d=d.replace(p,"")
    d=d.strip(); return d if d.startswith("10.") else ""
def norm_oaid(x):
    x=str(x or "").strip().lower()
    m=re.search(r'(w\d{5,})', x)            # extract the W-number from any oa id URL/string
    return m.group(1) if m else ""
def norm_year(y):
    m=re.search(r"(19|20)\d{2}",str(y or "")); return m.group(0) if m else ""

GENERIC_TITLES=frozenset({"front matter","back matter","summary","summaries","reviews in brief",
 "publications received","editorial","introduction","preface","contents","table of contents",
 "index","book review","letter","news","announcement","abstract","poster","supplement","no title",
 "retracted","bibliographie","corrigendum","erratum","obituary","in this issue","highlights",
 "acknowledgements","author index","subject index","calendar","meeting","conference","abstracts"})
def is_risky(t):
    if not t: return True
    if t in GENERIC_TITLES: return True
    if len(t.split())<MIN_TITLE_TOKENS: return True
    if len(t)<MIN_TITLE_CHARS: return True
    return False

JOURNAL_STOP={"the","of","and","for","in","on","a","journal","review","reviews","annual",
 "international","official","series","section","part","online","print","electronic","edition",
 "proceedings","bulletin","archives","annals","acta","revista","rev","jnl","j"}
def clean_journal(s):
    s=strip_acc(str(s or "").lower().strip())
    if s in {"","none","null","nan","n/a","na","med"}: return ""
    s=re.sub(r"\(.*?\)"," ",s); s=_PUNCT.sub(" ",s); s=_WS.sub(" ",s).strip()
    return " ".join(t for t in s.split() if t and t not in JOURNAL_STOP)
def journal_agree(a,b):
    if not a or not b: return False
    if a==b: return True
    return fuzz.token_set_ratio(a,b)>=85

def author_fp(authors_flat, mode=AUTHOR_FP_MODE):
    a=str(authors_flat or "").strip()
    if a.lower() in NA_LIKE: return ""
    out=[]
    for au in re.split(r"[;]",a):
        au=strip_acc(au.lower().strip())
        if not au or au in NA_LIKE: continue
        if "," in au:
            left,right=au.split(",",1)
            left=_WS.sub(" ",_PUNCT.sub(" ",left)).strip(); right=_WS.sub(" ",_PUNCT.sub(" ",right)).strip()
            lt=[t for t in left.split() if t]; rt=[t for t in right.split() if t]
            if not lt: continue
            if mode=="last": out.append(lt[0])
            else:
                if not rt: continue
                out.append("_".join(sorted(lt+rt)) if mode=="tokenbag" else f"{lt[0]}|{rt[0][:1]}")
        else:
            au=_WS.sub(" ",_PUNCT.sub(" ",au)).strip(); toks=[t for t in au.split() if t]
            if len(toks)<2:
                if mode=="last" and toks: out.append(toks[-1])
                continue
            if mode=="last": out.append(toks[-1])
            elif mode=="tokenbag": out.append("_".join(sorted(toks)))
            else: out.append(f"{toks[-1]}|{toks[0][:1]}")
    return ";".join(sorted(set(out)))
def fp_overlap(a,b):
    if not a or not b: return 0
    return len(set(a.split(";"))&set(b.split(";")))

t0=time.perf_counter()
df["title_clean"]=df["m_title"].map(clean_title)
df.loc[df["title_clean"].map(is_risky),"title_clean"]=""
df["oaid_norm"]=df["oa_id"].map(norm_oaid) if "oa_id" in df.columns else ""
df["doi_norm"]=df["doi"].map(norm_doi)
df["year_norm"]=df["m_year"].map(norm_year)
df["authors_fp"]=df["m_authors"].map(author_fp)
df["journal_clean"]=df["m_journal"].map(clean_journal)
df["title_first_token"]=df["title_clean"].map(lambda t:t.split(" ")[0] if t else "")
print(f"Normalization {time.perf_counter()-t0:.1f}s | risky blanked {(df['title_clean']=='').sum()} "
      f"| oa_id present {(df['oaid_norm']!='').sum()}")

## ---- 6. union-find with BOTH oa_id and DOI cluster tracking ----
parent={}; cl_oa={}; cl_doi={}
def find(x):
    parent.setdefault(x,x); r=x
    while parent[r]!=r: r=parent[r]
    while parent[x]!=r: parent[x],x=r,parent[x]
    return r
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb:
        parent[ra]=rb
        cl_oa.setdefault(rb,set()).update(cl_oa.get(ra,set()))
        cl_doi.setdefault(rb,set()).update(cl_doi.get(ra,set()))
def oa_conflict(a,b):
    oa_a=cl_oa.get(find(a),set()); oa_b=cl_oa.get(find(b),set())
    return bool(oa_a and oa_b and not (oa_a & oa_b))     # different oa_ids => distinct works
def doi_conflict(a,b):
    da=cl_doi.get(find(a),set()); db=cl_doi.get(find(b),set())
    return bool(da and db and da!=db and not (da & db))
def blocked(a,b):                                         # combined guard
    return oa_conflict(a,b) or doi_conflict(a,b)

for i in df.index:
    find(i)
    o=df.at[i,"oaid_norm"]; d=df.at[i,"doi_norm"]
    if o: cl_oa[i]={o}
    if d: cl_doi[i]={d}

M={}
## Stage 0: exact OpenAlex work id (authoritative; merges preprint+published)
n=0
for oaid,g in df[df["oaid_norm"].ne("")].groupby("oaid_norm"):
    ix=g.index.tolist()
    for j in ix[1:]:
        if find(ix[0])!=find(j): union(ix[0],j); n+=1
M["0_openalex_id"]=n

## Stage 1: exact DOI (guarded so it never crosses an oa_id boundary)
n=0
for doi,g in df[df["doi_norm"].ne("")].groupby("doi_norm"):
    ix=g.index.tolist()
    for j in ix[1:]:
        if find(ix[0])!=find(j) and not oa_conflict(ix[0],j): union(ix[0],j); n+=1
M["1_doi_exact"]=n

## Stage 2a: title + authors_fp + year
n=0
el=df["title_clean"].ne("")&df["authors_fp"].ne("")&df["year_norm"].ne("")
key=df["title_clean"]+"||"+df["authors_fp"]+"||"+df["year_norm"]
for k,g in df[el].groupby(key[el]):
    ix=g.index.tolist()
    for j in ix[1:]:
        if find(ix[0])!=find(j) and not blocked(ix[0],j): union(ix[0],j); n+=1
M["2a_title_fp_year"]=n

## Stage 2b: title + journal + year
n=0
el=df["title_clean"].ne("")&df["journal_clean"].ne("")&df["year_norm"].ne("")
key=df["title_clean"]+"||"+df["journal_clean"]+"||"+df["year_norm"]
for k,g in df[el].groupby(key[el]):
    ix=g.index.tolist()
    for j in ix[1:]:
        if find(ix[0])!=find(j) and not blocked(ix[0],j): union(ix[0],j); n+=1
M["2b_title_journal_year"]=n

## Stage 3: title + year, gated by >=1 shared author
n=0
el=df["title_clean"].ne("")&df["year_norm"].ne("")
key=df["title_clean"]+"||"+df["year_norm"]
for k,g in df[el].groupby(key[el]):
    ix=g.index.tolist()
    if len(ix)<2: continue
    for a in range(len(ix)):
        for b in range(a+1,len(ix)):
            ia,ib=ix[a],ix[b]
            if find(ia)==find(ib) or blocked(ia,ib): continue
            fa,fb=df.at[ia,"authors_fp"],df.at[ib,"authors_fp"]
            if fa and fb and fp_overlap(fa,fb)<1: continue
            union(ia,ib); n+=1
M["3_title_year_authorgate"]=n

## Stage 4: fuzzy title, year+/-tol, shared-author OR agreeing-journal
n=0
cand=df[df["title_clean"].ne("")]
for tok,block in cand.groupby("title_first_token"):
    if tok=="": continue
    ix=block.index.tolist()
    if len(ix)<2: continue
    info=[(i,df.at[i,"title_clean"],df.at[i,"year_norm"],df.at[i,"authors_fp"],df.at[i,"journal_clean"]) for i in ix]
    for a in range(len(info)):
        ia,ta,ya,fa,ja=info[a]
        for b in range(a+1,len(info)):
            ib,tb,yb,fb,jb=info[b]
            if find(ia)==find(ib) or blocked(ia,ib): continue
            if ya and yb and abs(int(ya)-int(yb))>FUZZY_YEAR_TOLERANCE: continue
            if not ((fa and fb and fp_overlap(fa,fb)>=1) or journal_agree(ja,jb)): continue
            if abs(len(ta)-len(tb))>25: continue
            if fuzz.token_sort_ratio(ta,tb)>=FUZZY_THRESHOLD: union(ia,ib); n+=1
M["4_fuzzy_gated"]=n

## ---- 7. assign study_id, canonical (richest + oa_id-matched preferred), provenance ----
df["cluster"]=[find(i) for i in df.index]
SCI_CORE=["title","abstract","keywords","authors","oa_author_orcids","affiliations",
 "oa_institutions","oa_countries","citation_count","journal","oa_issn","document_type",
 "oa_language","funders_unified","is_funded","year","doi","oaid_norm","oa_topic",
 "oa_subfield","oa_field","oa_domain","oa_concepts"]
SCI_CORE=[c for c in SCI_CORE if c in df.columns]
df["_sci"]=sum((df[c].astype(str).str.len()>0).astype(int) for c in SCI_CORE)
TYPE_PRI={"journal article":10,"research article":10,"article":10,"review":8,"review article":8,
 "systematic review":8,"book chapter":5,"book":5,"conference paper":4,"proceedings":4,
 "preprint":3,"posted content":3,"report":2,"thesis":2,"dataset":1}
def tpri(t):
    t=str(t or "").lower(); h=[v for k,v in TYPE_PRI.items() if k in t]; return max(h) if h else 0
df["_score"]=(df["oaid_norm"].ne("").astype(int)*30      # prefer oa_id-matched (full payload)
            + df["doi_norm"].ne("").astype(int)*100
            + df["document_type"].map(tpri)*3
            + df["_sci"]*5
            + (df["abstract"].astype(str).str.len()>0).astype(int)*6)
canon_idx=df.sort_values(["_score","record_id"],ascending=[False,True]).groupby("cluster").head(1).index
df["is_canonical"]=df.index.isin(canon_idx)

order={c:k for k,c in enumerate(df["cluster"].tolist())}
uniq=sorted(df["cluster"].unique(),key=lambda c:order[c])
cl2stu={c:f"STU-{i+1:06d}" for i,c in enumerate(uniq)}
df["study_id"]=df["cluster"].map(cl2stu)
if "source_database" in df.columns:
    found=df.groupby("study_id")["source_database"].apply(lambda s:"; ".join(sorted(set(s))))
    df["found_in"]=df["study_id"].map(found)
    df["n_sources"]=df.groupby("study_id")["source_database"].transform("nunique")
df["n_records"]=df.groupby("study_id")["record_id"].transform("count")

## ---- 8. report ----
nr,ns=len(df),df["study_id"].nunique()
print("\n=== STAGE MERGES ==="); 
for k,v in M.items(): print(f"  {k}: {v}")
print(f"\nRecords: {nr} | Unique studies: {ns} | Duplicates removed: {nr-ns} ({(nr-ns)/nr*100:.1f}%)")
print(f"Largest study: {df.groupby('study_id').size().max()} records")
# how many merges were ONLY possible thanks to oa_id (diff DOI but same work)
oa_only = df.groupby("study_id").filter(
    lambda g: g["oaid_norm"].nunique()==1 and g["doi_norm"].replace("",np.nan).nunique()>1)
print(f"Studies merged across DIFFERENT DOIs via shared oa_id (e.g. preprint+published): "
      f"{oa_only['study_id'].nunique()}")

## ---- 9. QC ----
study_has_doi=df.groupby("study_id")["doi_norm"].transform(lambda s:s.ne("").any())
qc_canon_no_doi=df[df["is_canonical"]&df["doi_norm"].eq("")&study_has_doi]
tt=df.groupby("study_id")["title_clean"].nunique()
qc_multi_title=df[df["study_id"].isin(tt[tt>1].index)]
qc_big=df[df["n_records"]>=8]
print(f"\nQC: canonical-missing-DOI {len(qc_canon_no_doi)} | multi-title-studies {len(qc_multi_title)} rows | large(>=8) {len(qc_big)} rows")

## ---- 10. save ----
def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)

front=["study_id","record_id","is_canonical","n_sources","n_records","found_in","oaid_norm",
 "source_database","source_record_id","title","authors","year","doi","journal","abstract",
 "citation_count","affiliations","oa_countries","funders_unified","is_funded","document_type",
 "oa_topic","oa_subfield","oa_field","oa_domain"]
front=[c for c in front if c in df.columns]
helper=["title_clean","doi_norm","oaid_norm","year_norm","authors_fp","title_first_token",
 "cluster","_sci","_score","m_title","m_authors","m_journal","m_year"]
rest=[c for c in df.columns if c not in front+helper]
audit=df[front+rest]
safe_xlsx(audit,AUDIT_PATH,"dedupe_audit")
safe_xlsx(audit[df["is_canonical"]],CANON_PATH,"canonical")
safe_xlsx(qc_canon_no_doi[front],os.path.join(QC_DIR,"qc_canonical_missing_doi.xlsx"),"qc")
safe_xlsx(qc_multi_title[front],os.path.join(QC_DIR,"qc_multi_title_studies.xlsx"),"qc")
safe_xlsx(qc_big[front],os.path.join(QC_DIR,"qc_large_studies.xlsx"),"qc")
print(f"\nSaved: {AUDIT_PATH}\n       {CANON_PATH} ({df['is_canonical'].sum()} studies)\n       {QC_DIR}")
