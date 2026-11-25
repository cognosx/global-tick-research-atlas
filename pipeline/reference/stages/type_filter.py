# ============================================================
#  DOCUMENT-TYPE FILTER  (Google Colab) — run BEFORE dedup
#  Tick-Borne Diseases / Climate / Sub-Saharan Africa review
#
#  Keeps original-literature types; drops grants, datasets, patents,
#  theses, editorials, notes, meeting abstracts, etc.
#
#  Three layers:
#    1. Local type     - use each source's native type where present
#                        (WoS Document Type + WOS:<INDEX>: id-prefix rule,
#                         Lens Publication Type, Dimensions Publication_Type)
#    2. API enrichment - for records with NO type (OpenAlex/EuropePMC/GIM/
#                        type-less PubMed), look up type by DOI via
#                        OpenAlex API, fall back to Crossref
#    3. Uniform filter - one keep-list across the unified document_type
#
#  Run this on the master AFTER ids are added, BEFORE dedup.
#  Input : master_search_results_with_ids.xlsx
#  Output: master_filtered_by_type.xlsx  (+ type_filter_audit.xlsx)
# ============================================================

## ---- 1. setup ----
from google.colab import drive
drive.mount('/content/drive')
!pip -q install requests openpyxl pandas

import os, re, time, json
import pandas as pd, numpy as np
import requests

## ---- 2. config ----
FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
IN_PATH = os.path.join(OUT_DIR, "master_search_results_with_ids.xlsx")
OUT_PATH   = os.path.join(OUT_DIR, "master_filtered_by_type.xlsx")
AUDIT_PATH = os.path.join(OUT_DIR, "type_filter_audit.xlsx")

POLITE_EMAIL   = "cognosxx@gmail.com"   # OpenAlex "polite pool" — faster, kinder
ENRICH_VIA_API = True      # set False to skip API enrichment (local-type only)
KEEP_UNKNOWN   = True      # True = keep records whose type can't be determined (safer recall);
                           # False = drop unknowns (stricter precision). Reported either way.
MAX_ENRICH     = None      # cap API lookups for a test run (e.g. 500); None = all

# --- Two-tier decision vocabulary ---
# KEEP: substantive original literature (case reports KEPT per review scope).
KEEP_TYPES = {
    "article", "journal article", "research article", "research-article",
    "review", "review article", "review-article", "systematic review",
    "scoping review", "meta analysis", "book", "book chapter", "chapter",
    "monograph", "edited book", "conference paper", "proceedings", "proceeding",
    "proceedings paper", "conference proceedings", "conference proceedings article",
    "preprint", "posted content", "posted-content", "case reports", "case report",
    "case-report", "data paper", "clinical trial", "randomized controlled trial",
    "controlled clinical trial", "observational study", "comparative study",
    "evaluation study", "multicenter study", "validation study", "historical article",
    "practice guideline", "guideline", "consensus statement", "clinical trial protocol",
    "early access", "short survey",
}
# OVERRIDE_DROP: fires even when a keep-word is also present. This removes
# retractions/errata/corrections and editorial-adjacent content that MEDLINE
# tags alongside "Journal Article" (e.g. "Journal Article; Retracted Publication").
OVERRIDE_DROP = {
    "retracted", "retraction", "erratum", "published erratum", "corrigendum",
    "correction", "expression of concern", "editorial", "comment", "news",
    "news item", "letter", "book review", "biographical", "bibliography",
    "interview", "discussion", "patent", "meeting abstract",
    "introductory journal article", "study guide", "seminar", "congress",
}
# PLAIN_DROP: dropped only when NO keep-word is present.
DROP_TYPES = {
    "grant", "awarded grant", "dataset", "data set", "data study", "patent",
    "dissertation", "thesis", "reprint", "software", "report", "note", "paratext",
    "meeting", "journal issue", "non conventional", "non-conventional",
    "conference review",
}

## ---- 3. load ----
df = pd.read_excel(IN_PATH, sheet_name="master_data", dtype=str).fillna("").reset_index(drop=True)
print("Loaded:", len(df), "rows")
df["doc_type_raw"]    = ""    # the source-native type string we found
df["doc_type_source"] = ""    # where we got it: local | openalex_api | crossref_api | none

## ---- 4. normalizers ----
def norm_doi(d):
    d = str(d or "").strip().lower()
    for p in ("https://doi.org/","http://dx.doi.org/","https://dx.doi.org/","doi:"):
        d = d.replace(p, "")
    d = d.strip()
    return d if d.startswith("10.") else ""
df["doi_norm"] = df["doi"].map(norm_doi)

# ---- WoS ID-prefix rule: these indices are categorically non-article ----
WOS_DROP_PREFIX = {"GRANTS": "grant", "DIIDW": "patent", "DRCI": "dataset"}
WOS_THESIS_PREFIX = {"PQDT": "thesis"}   # dropped unless you add thesis to KEEP_TYPES
def wos_prefix(srid):
    m = re.match(r"WOS:([A-Z]+):", str(srid or ""))
    return m.group(1) if m else ""

## ---- 5. LAYER 1: local type per source ----
# Map: source_database -> the namespaced column holding its native type (if any)
LOCAL_TYPE_COL = {
    "Web of Science": "webofscience__Document_Type",   # adjust if your builder named it differently
    "Lens.org":       "lens__Publication_Type",
    "Dimensions":     "dimensions__Publication_Type",
}
# tolerant column finder (handles slight name variations from the builder)
def find_col(cands):
    low = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in low: return low[cand.lower()]
    return None

resolved = {}
for src, guess in LOCAL_TYPE_COL.items():
    col = find_col([guess,
                    guess.split("__")[0] + "__Document_Type",
                    guess.split("__")[0] + "__Publication_Type",
                    guess.split("__")[0] + "__type"])
    resolved[src] = col
print("Resolved local type columns:", resolved)

for src, col in resolved.items():
    mask = df["source_database"].eq(src)
    if col and col in df.columns:
        df.loc[mask, "doc_type_raw"]    = df.loc[mask, col]
        df.loc[mask, "doc_type_source"] = "local"

# also use the common document_type column if the builder already filled it
common_mask = df["document_type"].ne("") & df["doc_type_raw"].eq("")
df.loc[common_mask, "doc_type_raw"]    = df.loc[common_mask, "document_type"]
df.loc[common_mask, "doc_type_source"] = "local"

# WoS id-prefix override (authoritative for the junk indices)
wm = df["source_database"].eq("Web of Science")
df.loc[wm, "_wospfx"] = df.loc[wm, "source_record_id"].map(wos_prefix)
for pfx, label in {**WOS_DROP_PREFIX, **WOS_THESIS_PREFIX}.items():
    sel = wm & df["_wospfx"].eq(pfx)
    df.loc[sel, "doc_type_raw"]    = label
    df.loc[sel, "doc_type_source"] = "local"
print("WoS prefix counts:", df.loc[wm, "_wospfx"].value_counts().to_dict())

## ---- 6. LAYER 2: API enrichment for type-less records ----
need = df["doc_type_raw"].eq("") & df["doi_norm"].ne("")
if ENRICH_VIA_API and need.any():
    dois = df.loc[need, "doi_norm"].unique().tolist()
    if MAX_ENRICH: dois = dois[:MAX_ENRICH]
    print(f"\nEnriching {len(dois)} unique DOIs via OpenAlex (then Crossref fallback)...")
    type_by_doi = {}

    # OpenAlex batched (up to 50 DOIs per request via filter=doi:a|b|c)
    sess = requests.Session()
    B = 40
    for i in range(0, len(dois), B):
        batch = dois[i:i+B]
        url = ("https://api.openalex.org/works"
               f"?filter=doi:{'|'.join(batch)}&per-page={B}"
               f"&select=doi,type&mailto={POLITE_EMAIL}")
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200:
                for w in r.json().get("results", []):
                    d = norm_doi((w.get("doi") or ""))
                    if d and w.get("type"): type_by_doi[d] = ("openalex_api", w["type"])
        except Exception as e:
            print("  OpenAlex batch error:", e)
        time.sleep(0.12)   # ~8 req/s, within polite pool
        if (i//B) % 25 == 0: print(f"  ...{i+len(batch)}/{len(dois)}")

    # Crossref fallback for any DOI OpenAlex didn't resolve
    missing = [d for d in dois if d not in type_by_doi]
    print(f"OpenAlex resolved {len(type_by_doi)}; Crossref fallback for {len(missing)}...")
    for j, d in enumerate(missing):
        try:
            r = sess.get(f"https://api.crossref.org/works/{d}?mailto={POLITE_EMAIL}", timeout=30)
            if r.status_code == 200:
                t = r.json().get("message", {}).get("type", "")
                if t: type_by_doi[d] = ("crossref_api", t)
        except Exception:
            pass
        time.sleep(0.05)
        if j and j % 200 == 0: print(f"  crossref ...{j}/{len(missing)}")

    # write back
    for idx in df.index[need]:
        d = df.at[idx, "doi_norm"]
        if d in type_by_doi:
            srcflag, t = type_by_doi[d]
            df.at[idx, "doc_type_raw"]    = t
            df.at[idx, "doc_type_source"] = srcflag

df.loc[df["doc_type_source"].eq(""), "doc_type_source"] = "none"

## ---- 7. LAYER 3: uniform keep/drop decision ----
def norm_type(t):
    n = re.sub(r"[^a-z ]+", " ", str(t or "").lower())
    return re.sub(r"\s+", " ", n).strip()

def _has_word(text, vocab):
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in vocab)

def decide(t):
    raw = str(t or "").lower()
    parts = [norm_type(p) for p in re.split(r"[;/|]", raw) if norm_type(p)]
    if not parts:
        return "unknown"
    joined = " ; ".join(parts)
    if _has_word(joined, OVERRIDE_DROP):          # 1) override-drop wins over everything
        return "drop"
    if any(_has_word(p, KEEP_TYPES) for p in parts):  # 2) any keep component keeps it
        return "keep"
    if _has_word(joined, DROP_TYPES):             # 3) plain drop
        return "drop"
    return "drop"                                  # present but unrecognized -> drop

df["type_norm"]     = df["doc_type_raw"].map(norm_type)
df["keep_decision"] = df["doc_type_raw"].map(decide)
if KEEP_UNKNOWN:
    df.loc[df["keep_decision"].eq("unknown"), "keep_decision"] = "keep_unknown"

keep_mask = df["keep_decision"].isin(["keep", "keep_unknown"])
kept = df[keep_mask].copy()
dropped = df[~keep_mask].copy()

## ---- 8. report ----
print("\n=== TYPE-FILTER SUMMARY ===")
print(f"Input records:   {len(df)}")
print(f"KEPT:            {len(kept)}")
print(f"DROPPED:         {len(dropped)}")
print("\nType source breakdown:")
print(df["doc_type_source"].value_counts().to_string())
print("\nDecision breakdown:")
print(df["keep_decision"].value_counts().to_string())
print("\nDropped — by normalized type (top 15):")
print(dropped["type_norm"].replace("", "(empty)").value_counts().head(15).to_string())
print("\nKept — by source database:")
print(kept["source_database"].value_counts().to_string())
print(f"\nUnknown-type records: {(df['keep_decision'].eq('keep_unknown')).sum()} "
      f"({'kept' if KEEP_UNKNOWN else 'dropped'})")

## ---- 9. save ----
def safe_xlsx(frame, path, sheet):
    ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out = frame.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: ILLEGAL.sub(" ", str(v))[:32000])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name=sheet, index=False)

drop_helpers = ["_wospfx", "type_norm"]
kept_out = kept.drop(columns=[c for c in drop_helpers if c in kept.columns])
safe_xlsx(kept_out, OUT_PATH, "filtered_master")

# audit: every record + what we decided and why
audit_cols = ["record_id","source_database","source_record_id","doi","title",
              "doc_type_raw","doc_type_source","keep_decision"]
audit_cols = [c for c in audit_cols if c in df.columns]
safe_xlsx(df[audit_cols], AUDIT_PATH, "type_audit")

print("\nSaved:")
print(" ", OUT_PATH, f"({len(kept_out)} records -> feed THIS into dedup)")
print(" ", AUDIT_PATH, "(full keep/drop decisions for QC)")
print("\nNEXT: point 07_dedupe_synthesis at master_filtered_by_type.xlsx")
