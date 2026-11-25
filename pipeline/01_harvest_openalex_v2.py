# GTRA v1.0 — OpenAlex harvest (Colab cell). Replaces the search= cell of 01_OpenAlexHarvester_Ticks.
#
# What changed and why
#  1. Scoped search: title + abstract + keywords only (not full text). The old `search=` matched full
#     text too: 247,356 of 334,277 records (74%) had no tick term in their metadata.
#  2. Two blocks, harvested separately and tagged: A = tick taxa and words, B = tick-borne pathogens,
#     diseases and control. B finds papers that never say "tick" (e.g. Babesia in cattle, CCHF clinics).
#  3. "acari" removed (it pulls in mites: ~11,400 records); ixodid/argasid word forms and 4 genera added.
#  4. Non-English words added (French, Portuguese, Spanish, Italian, German) for African and Latin American
#     journals. Keywords also catch papers with no English abstract.
#  5. Fixed cut-off 2000–2025 for the v1.0 freeze (2026 is partial; update later with from_updated_date).
#  6. Provenance: query, filters, counts and date saved next to the data; dated folders, never overwritten.
#  7. Recall check: counts how many records only the old full-text search would add.

import json, datetime, os
import pandas as pd
from google.colab import userdata
# from openalex_harvester import OpenAlexHarvester   # the class defined earlier in the notebook

ROOT    = "/content/drive/MyDrive/COGNOSX/ticks"          # <-- your Drive folder
RUN_ID  = datetime.date.today().isoformat()               # e.g. 2026-11-25
OUTDIR  = f"{ROOT}/data_raw/openalex_{RUN_ID}"
os.makedirs(OUTDIR, exist_ok=True)

def q(term):  # quote multi-word or hyphenated terms so they stay phrases
    return f'"{term}"' if (" " in term or "-" in term) else term

def boolean_or(terms):
    return "(" + " OR ".join(q(t) for t in dict.fromkeys(terms)) + ")"

# --- Block A: ticks -----------------------------------------------------------------------------
GENERA = ["ixodes", "amblyomma", "rhipicephalus", "boophilus", "dermacentor", "haemaphysalis",
          "hyalomma", "ornithodoros", "argas", "otobius", "carios", "nuttalliella", "rhipicentor",
          "margaropus", "anomalohimalaya", "cosmiomma", "nosomma",
          "bothriocroton", "antricola", "nothoaspis", "aponomma"]           # 4 added
FAMILIES = ["ixodidae", "argasidae", "ixodida", "nuttalliellidae",
            "ixodid", "argasid"]                                           # "acari" removed
WORDS = ["tick", "ticks", "tick-borne", "tick borne", "tick bite", "tick-bite",
         "hard tick", "soft tick",
         "tique", "tiques",                   # French
         "carrapato", "carrapatos",           # Portuguese
         "garrapata", "garrapatas",           # Spanish
         "zecca", "zecche", "zecke", "zecken"]  # Italian, German
BLOCK_A = boolean_or(GENERA + FAMILIES + WORDS)

# --- Block B: tick-borne pathogens, diseases and control (papers that may never say "tick") -------
PATHOGENS = ["babesia", "babesiosis", "theileria", "theileriosis", "east coast fever",
             "anaplasma", "anaplasmosis", "ehrlichia", "ehrlichiosis", "heartwater", "cowdriosis",
             "lyme disease", "lyme borreliosis", "borrelia burgdorferi", "neuroborreliosis",
             "spotted fever", "rickettsia africae", "rickettsia conorii", "rickettsia rickettsii",
             "crimean-congo", "cchf", "cchfv", "tbev", "powassan", "kyasanur forest",
             "severe fever with thrombocytopenia", "sfts", "sftsv", "alpha-gal",
             "acaricide", "acaricides", "acaricidal"]
BLOCK_B = boolean_or(PATHOGENS)

FILTERS = {
    "publication_year": "2000-2025",                       # v1.0 freeze
    "type": ["article", "review", "letter", "preprint"],   # research, reviews, letters, preprints
    "is_paratext": "false",                                # drop front matter
}
SCOPE = "title_abstract_keywords.search"   # stricter alternative: "title_and_abstract.search"

for name, query in (("A_ticks", BLOCK_A), ("B_pathogens", BLOCK_B)):
    print(f"{name}: {len(query)} chars")    # keep each block well under ~4,000 (URL limit)

h = OpenAlexHarvester(api_key=userdata.get("OPENALEX_API_KEY"))

# 1) Size checks first (one call each)
counts = {}
for name, query in (("A_ticks", BLOCK_A), ("B_pathogens", BLOCK_B)):
    counts[name] = h.count(filters={**FILTERS, SCOPE: query})
counts["B_not_A"] = h.count(filters={**FILTERS, SCOPE: f"{BLOCK_B} NOT {BLOCK_A}"})
counts["old_fulltext_search_A"] = h.count(filters=FILTERS, search=BLOCK_A)   # recall check only
for k, v in counts.items():
    print(f"{k:24s} {v:>10,}")

# 2) Harvest each block into its own dated folder
frames = []
for name, query in (("A_ticks", BLOCK_A), ("B_pathogens", BLOCK_B)):
    df = h.harvest(filters={**FILTERS, SCOPE: query}, search=None,
                   outdir=f"{OUTDIR}/{name}", max_records=None)
    df["query_block"] = name
    frames.append(df)
    print(name, df.shape)

# 3) Combine: one row per work, with the blocks that found it
raw = pd.concat(frames, ignore_index=True)
blocks = raw.groupby("oa_id")["query_block"].agg(lambda s: "|".join(sorted(set(s))))
combined = raw.drop_duplicates("oa_id").drop(columns="query_block").merge(
    blocks.rename("query_blocks"), left_on="oa_id", right_index=True)
combined.to_parquet(f"{OUTDIR}/combined.parquet", index=False)
print("combined:", combined.shape, combined["query_blocks"].value_counts().to_dict())

# 4) Provenance manifest (goes in docs/ and the data paper)
manifest = {
    "run_id": RUN_ID, "harvested_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    "endpoint": "https://api.openalex.org/works", "scope": SCOPE, "filters": FILTERS,
    "block_A": BLOCK_A, "block_B": BLOCK_B, "counts_before_harvest": counts,
    "rows_per_block": {f["query_block"].iat[0]: len(f) for f in frames},
    "rows_combined": len(combined),
}
json.dump(manifest, open(f"{OUTDIR}/harvest_manifest.json", "w"), indent=2)
print(json.dumps(manifest, indent=2)[:1500])
