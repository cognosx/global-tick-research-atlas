# ============================================================
#  TYPE AUDIT — find ineligible types that slipped the filter
#  (e.g. the book review in the 29-record cluster). Run on
#  master_canonical.xlsx. Reports + optionally removes them.
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pandas openpyxl

import os, re
import pandas as pd

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
canon = pd.read_excel(os.path.join(OUT_DIR,"master_canonical.xlsx"), sheet_name="canonical", dtype=str).fillna("")
print("Canonical studies:", len(canon))

# same two-tier vocabulary as the type filter, applied to whatever type label the canonical row carries
OVERRIDE_DROP = {"retracted","retraction","erratum","published erratum","corrigendum","correction",
 "expression of concern","editorial","comment","news","news item","letter","book review",
 "biographical","bibliography","interview","discussion","patent","meeting abstract",
 "introductory journal article","study guide","seminar","congress"}
KEEP_HINT = {"article","journal article","research article","review","review article",
 "systematic review","book","book chapter","chapter","conference paper","proceedings",
 "preprint","case report","case reports","clinical trial"}

def norm(t):
    n=re.sub(r"[^a-z ]+"," ",str(t or "").lower()); return re.sub(r"\s+"," ",n).strip()
def has(text,vocab): return any(re.search(r"\b"+re.escape(w)+r"\b",text) for w in vocab)

# check BOTH the document_type and the oa_type if present, plus title (catches "Book review ..." titles)
def slipped(row):
    parts=[]
    for c in ["document_type","oa_type"]:
        if c in row and row[c]: parts += re.split(r"[;/|]", str(row[c]).lower())
    joined=norm(" ; ".join(parts))
    title=norm(row.get("title",""))
    # override-drop wins even if a keep word is present
    if has(joined, OVERRIDE_DROP): return "override_drop_type"
    # title-based catch for book reviews mislabeled as 'book'/'article'
    if title.startswith("book review") or " book review " in title: return "title_looks_book_review"
    return ""

canon["_slip"]=canon.apply(slipped, axis=1)
slipped_rows = canon[canon["_slip"].ne("")]
print(f"\nPotentially ineligible canonical studies found: {len(slipped_rows)}")
print("\nBy reason:")
print(slipped_rows["_slip"].value_counts().to_string())
print("\nBy type label:")
col = "document_type" if "document_type" in canon.columns else "oa_type"
print(slipped_rows[col].replace("","(empty)").value_counts().head(15).to_string())
print("\nSample (first 12):")
show=[c for c in ["study_id","document_type","oa_type","title"] if c in slipped_rows.columns]
for _,r in slipped_rows[show].head(12).iterrows():
    print("  ", " | ".join(str(r[c])[:40] for c in show))

# write the list for your review; also write a cleaned canonical with them removed
def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)

safe_xlsx(slipped_rows.drop(columns=["_slip"]), os.path.join(OUT_DIR,"qc_slipped_types.xlsx"),"slipped")
clean = canon[canon["_slip"].eq("")].drop(columns=["_slip"])
safe_xlsx(clean, os.path.join(OUT_DIR,"master_canonical_clean.xlsx"),"canonical")
print(f"\nSaved qc_slipped_types.xlsx ({len(slipped_rows)} to review)")
print(f"Saved master_canonical_clean.xlsx ({len(clean)} studies, ineligible removed)")
print("\nREVIEW qc_slipped_types.xlsx before using the clean file — confirm these are")
print("genuinely ineligible (some 'review' hits may be real review ARTICLES you want to keep).")
