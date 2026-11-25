# ============================================================
#  STAGE A — Tick-relevance gatekeeping (rule-based, fast, no API)
#  Confirms each record is genuinely about ticks/tick-borne research
#  before topic classification. Flags rather than drops, for audit.
#
#  Input : master_canonical_pass1.parquet  (or master_dedupe_pass1)
#  Output: adds column `tick_relevant` (bool) + `tick_match_term` (str)
#          writes master_tick_checked.parquet
# ============================================================
import os, re
import pandas as pd

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET  = os.path.join(WORK, "master_canonical_pass1.parquet")
OUT_PARQUET = os.path.join(WORK, "master_tick_checked.parquet")

# Tick genera (Ixodidae = hard ticks, Argasidae = soft ticks) + common families/synonyms.
# Kept broad on purpose: false negatives here are worse than false positives, since
# Stage B/C (topic modeling) will further refine relevance.
TICK_GENERA = [
    "ixodes","amblyomma","rhipicephalus","boophilus","dermacentor","haemaphysalis",
    "hyalomma","ornithodoros","argas","otobius","carios","nuttalliella",
    "rhipicentor","margaropus","anomalohimalaya","cosmiomma","nosomma",
]
TICK_FAMILY_TERMS = ["ixodidae","argasidae","ixodida","nuttalliellidae","acari"]
TICK_GENERIC = [r"\btick\b", r"\bticks\b", r"\btick-borne\b", r"\btick borne\b"]

# common tick-borne pathogens/diseases — a record can be tick-RELEVANT via the
# pathogen even if "tick" itself isn't in the title/abstract (e.g. a Borrelia
# genomics paper that assumes vector context). Used as a SECONDARY signal only.
TICK_BORNE_DISEASE_TERMS = [
    "lyme disease","borreliosis","anaplasmosis","babesiosis","ehrlichiosis",
    "rickettsiosis","tick-borne encephalitis","crimean-congo","theileriosis",
    "tularemia","relapsing fever","q fever","tick paralysis","heartland virus",
    "bourbon virus","powassan virus","alpha-gal syndrome",
]

def _compile(terms, word_boundary=True):
    if word_boundary:
        return re.compile("|".join(r"\b"+re.escape(t)+r"\b" for t in terms), re.I)
    return re.compile("|".join(re.escape(t) for t in terms), re.I)

_genera_re  = _compile(TICK_GENERA)
_family_re  = _compile(TICK_FAMILY_TERMS)
_generic_re = re.compile("|".join(TICK_GENERIC), re.I)
_disease_re = _compile(TICK_BORNE_DISEASE_TERMS, word_boundary=False)

def classify_relevance(title, abstract, keywords=""):
    text = f"{title or ''} {abstract or ''} {keywords or ''}"
    if _generic_re.search(text):
        m = _generic_re.search(text)
        return True, "tick_word", m.group(0)
    if _genera_re.search(text):
        m = _genera_re.search(text)
        return True, "genus", m.group(0)
    if _family_re.search(text):
        m = _family_re.search(text)
        return True, "family", m.group(0)
    if _disease_re.search(text):
        m = _disease_re.search(text)
        return True, "tick_borne_disease_only", m.group(0)
    return False, "none", ""

def run():
    df = pd.read_parquet(IN_PARQUET)
    print(f"Loaded {len(df)} rows")
    for c in ["title","abstract","keywords"]:
        if c not in df.columns: df[c] = ""
    results = df.apply(lambda r: classify_relevance(r["title"], r["abstract"], r.get("keywords","")), axis=1)
    df["tick_relevant"]   = [x[0] for x in results]
    df["tick_match_type"] = [x[1] for x in results]
    df["tick_match_term"] = [x[2] for x in results]

    n = len(df); n_rel = df["tick_relevant"].sum()
    print(f"\n=== TICK RELEVANCE CHECK ===")
    print(f"  relevant:     {n_rel} ({n_rel/n*100:.1f}%)")
    print(f"  NOT relevant: {n-n_rel} ({(n-n_rel)/n*100:.1f}%)  <- review these")
    print("\nMatch type breakdown (relevant only):")
    print(df[df.tick_relevant]["tick_match_type"].value_counts().to_string())

    # write the irrelevant subset for manual spot-check (small enough to skim)
    irrelevant = df[~df["tick_relevant"]]
    audit_cols = [c for c in ["record_id","study_id","title","abstract","source_database"] if c in df.columns]
    irrelevant[audit_cols].to_parquet(os.path.join(WORK,"qc_not_tick_relevant.parquet"), index=False)
    print(f"\nSaved qc_not_tick_relevant.parquet ({len(irrelevant)} rows) — spot-check a sample of these.")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved {OUT_PARQUET} ({len(df)} rows, +3 cols: tick_relevant, tick_match_type, tick_match_term)")
    return df

if __name__ == "__main__":
    run()
