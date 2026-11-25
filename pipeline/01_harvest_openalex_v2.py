# Mirror of notebooks/01_corpus/01_OpenAlexHarvester_Ticks_v2.ipynb (shell lines commented out)

# # 01 · OpenAlex harvest — tick-focused corpus (GTRA v1.0)
# 
# **Scope** (decided 2026-11-25, see `docs/decisions.md`): research in which ticks, tick parts or tick-derived material were studied, including pathogens detected in ticks, ticks on hosts, control tested on ticks and distribution or climate models built from tick records. Disease-only studies are excluded at screening and kept as a flagged layer. The harvest stays broad; screening applies the rules.
# 
# **What this notebook does**
# 1. Counts each query block before spending credits.
# 2. Harvests Block A (ticks) with a title + abstract + keywords search, not full text.
# 3. Pulls a 500-record random sample of pathogen/disease papers that Block A misses (recall probe).
# 4. Saves raw records (lossless), three tables (`works`, `authorships`, `awards`), a manifest and a QC summary.
# 
# **Schema notes (OpenAlex, 2026):** `grants` was removed, so funding comes from `funders` + `awards`. `is_paratext` is deprecated, so paratext is excluded through the `type` filter. New fields captured: study designs, full-text availability (PDF, GROBID XML), APC list/paid, corresponding institutions, citation percentiles, counts by year, related works.

# %% ------------------------------------------------------------------
# 1 · Setup: Drive, code, API key
from google.colab import drive, userdata
drive.mount('/content/drive')

pass  # shell: !pip -q install requests pandas pyarrow
import os, sys, json, datetime
if not os.path.exists('/content/gtra'):
    pass  # shell: !git clone -q https://github.com/cognosx/global-tick-research-atlas /content/gtra
else:
    pass  # shell: !git -C /content/gtra pull -q
sys.path.insert(0, '/content/gtra/src')

import pandas as pd
from gtra.openalex_harvester import OpenAlexHarvester, boolean_or
pd.set_option('display.max_columns', None)

h = OpenAlexHarvester(api_key=userdata.get('OPENALEX_API_KEY'),
                      include_xpac=False)   # True adds DataCite/repository works (theses, reports); test the count first

# %% ------------------------------------------------------------------
# 2 · Configuration
ROOT   = "/content/drive/MyDrive/COGNOSX/TICKS/ticks_research"   # <-- your Drive folder
RUN_ID = datetime.date.today().isoformat()
OUTDIR = f"{ROOT}/data_raw/openalex_{RUN_ID}"
os.makedirs(OUTDIR, exist_ok=True)

# Block A — ticks (harvested in full)
GENERA = ["ixodes", "amblyomma", "rhipicephalus", "boophilus", "dermacentor", "haemaphysalis",
          "hyalomma", "ornithodoros", "argas", "otobius", "carios", "nuttalliella", "rhipicentor",
          "margaropus", "anomalohimalaya", "cosmiomma", "nosomma",
          "bothriocroton", "antricola", "nothoaspis", "aponomma"]
FAMILIES = ["ixodidae", "argasidae", "ixodida", "nuttalliellidae", "ixodid", "argasid"]  # no "acari": mites
WORDS = ["tick", "ticks", "tick-borne", "tick borne", "tick bite", "tick-bite", "hard tick", "soft tick",
         "tique", "tiques", "carrapato", "carrapatos", "garrapata", "garrapatas",     # FR, PT, ES
         "zecca", "zecche", "zecke", "zecken"]                                        # IT, DE
CONTROL = ["acaricide", "acaricides", "acaricidal"]
BLOCK_A = boolean_or(GENERA + FAMILIES + WORDS + CONTROL)

# Block B — pathogens and diseases (recall probe only; mostly disease-only papers, out of scope)
BLOCK_B = boolean_or(["babesia", "babesiosis", "theileria", "theileriosis", "east coast fever",
    "anaplasma", "anaplasmosis", "ehrlichia", "ehrlichiosis", "heartwater", "cowdriosis",
    "lyme disease", "lyme borreliosis", "borrelia burgdorferi", "neuroborreliosis", "spotted fever",
    "rickettsia africae", "rickettsia conorii", "rickettsia rickettsii", "crimean-congo", "cchf", "cchfv",
    "tbev", "powassan", "kyasanur forest", "severe fever with thrombocytopenia", "sfts", "sftsv", "alpha-gal"])
PROBE_SIZE, SEED = 500, 2026

SEARCH_FIELD = "title_abstract_keywords"       # not full text; stricter: "title_and_abstract"
FILTERS = {
    "publication_year": "2000-2027",                        # harvest window
    "type": ["article", "review", "letter", "preprint"],    # excludes paratext (is_paratext is deprecated)
}
V1_YEARS = (2000, 2025)                                     # frozen v1.0 analysis window

INCLUSION = {
    "question": "Did the study use, examine or measure ticks or tick-derived material?",
    "include": ["pathogen_in_tick", "host_infestation", "tick_derived", "tick_control", "tick_organism",
                "tick_distribution_model"],
    "include_with_flag": ["review"],
    "exclude_keep_as_layer": ["disease_only"],
}
print("Block A:", len(BLOCK_A), "chars | Block B:", len(BLOCK_B), "chars")

# %% ------------------------------------------------------------------
# 3 · Size checks (one call each) — run before the harvest
counts = {
    "A_ticks": h.count(FILTERS, BLOCK_A, SEARCH_FIELD),
    "B_pathogens": h.count(FILTERS, BLOCK_B, SEARCH_FIELD),
    "B_not_A": h.count(FILTERS, f"{BLOCK_B} NOT {BLOCK_A}", SEARCH_FIELD),
    "A_old_fulltext_search": h.count(FILTERS, BLOCK_A, "all"),     # what full-text matching would add
    "A_title_and_abstract_only": h.count(FILTERS, BLOCK_A, "title_and_abstract"),
}
for k, v in counts.items():
    print(f"{k:28s} {v:>10,}")
print(f"\nBlock A ≈ {counts['A_ticks'] / 200:,.0f} list calls")

# %% ------------------------------------------------------------------
# 4 · Harvest Block A (resumable: re-run this cell after any interruption)
works = h.harvest(FILTERS, BLOCK_A, SEARCH_FIELD, outdir=f"{OUTDIR}/A_ticks")
works["in_v1_window"] = pd.to_numeric(works["oa_year"], errors="coerce").between(*V1_YEARS)
works["query_block"] = "A_ticks"
works.to_parquet(f"{OUTDIR}/A_ticks/works.parquet", index=False)
print(works.shape, "| in v1 window:", int(works["in_v1_window"].sum()))

# %% ------------------------------------------------------------------
# 5 · Recall probe: random sample of pathogen/disease papers that Block A does not find
raw = h.sample(PROBE_SIZE, FILTERS, f"{BLOCK_B} NOT {BLOCK_A}", SEARCH_FIELD, seed=SEED)
probe = pd.DataFrame([h.flatten_work(w) for w in raw])[
    ["oa_id", "oa_doi", "oa_title", "oa_abstract", "oa_year", "oa_journal", "oa_mesh", "oa_keywords"]]
probe["probe_label"] = ""     # fill by hand or LLM: tick_focused / disease_only / not_tick
probe.to_parquet(f"{OUTDIR}/B_probe_sample.parquet", index=False)
probe.to_excel(f"{OUTDIR}/B_probe_sample.xlsx", index=False)
print(len(probe), "probe records of", f"{counts['B_not_A']:,}")
# Missed tick-focused papers ≈ share labelled tick_focused × counts['B_not_A']

# %% ------------------------------------------------------------------
# 6 · QC summary and provenance manifest
w = works
qc = {
    "works": len(w), "in_v1_window": int(w["in_v1_window"].sum()),
    "by_type": w["oa_type"].value_counts().to_dict(),
    "share_with_abstract": round(w["oa_abstract"].notna().mean(), 3),
    "share_with_doi": round((w["oa_doi"] != "").mean(), 3),
    "share_with_pmid": round((w["oa_pmid"] != "").mean(), 3),
    "share_with_countries": round((w["oa_countries"] != "").mean(), 3),
    "share_with_funding": round(w["oa_has_funding"].mean(), 3),
    "share_with_mesh": round((w["oa_mesh"] != "").mean(), 3),
    "share_with_study_design": round((w["oa_study_designs"] != "").mean(), 3),
    "share_with_grobid_xml": round(w["oa_has_grobid_xml"].fillna(False).mean(), 3),
    "share_non_english": round((w["oa_language"].fillna("") != "en").mean(), 3),
    "share_retracted": round(w["oa_is_retracted"].fillna(False).mean(), 4),
}
manifest = {
    "run_id": RUN_ID,
    "harvested_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "endpoint": "https://api.openalex.org/works", "search_field": SEARCH_FIELD,
    "include_xpac": h.include_xpac, "filters": FILTERS, "v1_years": V1_YEARS,
    "block_A": BLOCK_A, "block_B_probe": BLOCK_B, "probe_size": PROBE_SIZE, "seed": SEED,
    "counts_before_harvest": counts, "qc": qc, "inclusion_rules": INCLUSION,
    "outputs": ["A_ticks/raw/*.jsonl.gz", "A_ticks/works.parquet", "A_ticks/authorships.parquet",
                "A_ticks/awards.parquet", "B_probe_sample.parquet"],
}
json.dump(manifest, open(f"{OUTDIR}/harvest_manifest.json", "w"), indent=2)
pd.Series({k: v for k, v in qc.items() if k != "by_type"})

# %% ------------------------------------------------------------------
# 7 · Look at the tables
authorships = pd.read_parquet(f"{OUTDIR}/A_ticks/authorships.parquet")
awards = pd.read_parquet(f"{OUTDIR}/A_ticks/awards.parquet")
print("authorships:", authorships.shape, "| awards:", awards.shape)
works.sample(3, random_state=1)
