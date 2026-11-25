# GTRA v1.0 — OpenAlex harvest (Colab cell). Replaces the search= cell of 01_OpenAlexHarvester_Ticks.
#
# Scope (decided 2026-11-25, see docs/decisions.md and the "GTRA inclusion criteria" Sheet):
#   TICK-FOCUSED research only: studies that used ticks, tick body parts, tick-derived material
#   (proteins, genes, saliva, cell lines, anti-tick vaccines), pathogens detected IN ticks, ticks on hosts,
#   control or repellents tested ON ticks, and distribution/climate models built from tick records.
#   Disease-only studies (patients, serosurveys, pathogen vaccines in hosts) are excluded at screening,
#   but kept as a flagged layer. The harvest stays broad; screening applies these rules.
#
# What changed from the first cell and why
#  1. Scoped search: title + abstract + keywords only. The old `search=` also matched full text:
#     247,356 of 334,277 records (74%) had no tick term in their metadata.
#  2. Block A only is harvested in full. Pathogen/disease names (Block B) mostly bring disease-only papers,
#     which are out of scope, so B is used only as a RECALL PROBE: we count "B NOT A" and harvest a small
#     random sample of it to measure how many tick-focused papers Block A misses.
#  3. Acaricide terms moved into Block A (tick control is in scope). "acari" removed (it pulls in mites).
#  4. Word forms (ixodid, argasid, hard/soft tick, tick bite), 4 more genera, and the word "tick" in French,
#     Portuguese, Spanish, Italian and German, for African and Latin American journals.
#  5. Years 2000–2027 harvested so the newest online-first papers are kept; each record gets
#     `in_v1_window` (2000–2025) so the frozen v1.0 analysis set is one filter away.
#  6. Provenance: queries, filters, counts, scope rules and date saved in a manifest; dated folder,
#     never overwritten.

import json, datetime, os
import pandas as pd
from google.colab import userdata
# from openalex_harvester import OpenAlexHarvester   # the class defined earlier in the notebook

ROOT    = "/content/drive/MyDrive/COGNOSX/TICKS/ticks_research"   # <-- your Drive folder
RUN_ID  = datetime.date.today().isoformat()                        # e.g. 2026-11-25
OUTDIR  = f"{ROOT}/data_raw/openalex_{RUN_ID}"
os.makedirs(OUTDIR, exist_ok=True)

def q(term):  # quote multi-word or hyphenated terms so they stay phrases
    return f'"{term}"' if (" " in term or "-" in term) else term

def boolean_or(terms):
    return "(" + " OR ".join(q(t) for t in dict.fromkeys(terms)) + ")"

# --- Block A: ticks (harvested in full) ---------------------------------------------------------------
GENERA = ["ixodes", "amblyomma", "rhipicephalus", "boophilus", "dermacentor", "haemaphysalis",
          "hyalomma", "ornithodoros", "argas", "otobius", "carios", "nuttalliella", "rhipicentor",
          "margaropus", "anomalohimalaya", "cosmiomma", "nosomma",
          "bothriocroton", "antricola", "nothoaspis", "aponomma"]
FAMILIES = ["ixodidae", "argasidae", "ixodida", "nuttalliellidae", "ixodid", "argasid"]
WORDS = ["tick", "ticks", "tick-borne", "tick borne", "tick bite", "tick-bite",
         "hard tick", "soft tick",
         "tique", "tiques",                     # French
         "carrapato", "carrapatos",             # Portuguese
         "garrapata", "garrapatas",             # Spanish
         "zecca", "zecche", "zecke", "zecken"]  # Italian, German
CONTROL = ["acaricide", "acaricides", "acaricidal"]   # tick control is in scope
BLOCK_A = boolean_or(GENERA + FAMILIES + WORDS + CONTROL)

# --- Block B: recall probe only (pathogens and diseases; most results will be disease-only) -----------
PATHOGENS = ["babesia", "babesiosis", "theileria", "theileriosis", "east coast fever",
             "anaplasma", "anaplasmosis", "ehrlichia", "ehrlichiosis", "heartwater", "cowdriosis",
             "lyme disease", "lyme borreliosis", "borrelia burgdorferi", "neuroborreliosis",
             "spotted fever", "rickettsia africae", "rickettsia conorii", "rickettsia rickettsii",
             "crimean-congo", "cchf", "cchfv", "tbev", "powassan", "kyasanur forest",
             "severe fever with thrombocytopenia", "sfts", "sftsv", "alpha-gal"]
BLOCK_B = boolean_or(PATHOGENS)
PROBE_SIZE = 500          # random "B NOT A" records to screen by hand or LLM for missed tick-focused papers

FILTERS = {
    "publication_year": "2000-2027",                       # harvest window (v1.0 analysis: 2000–2025)
    "type": ["article", "review", "letter", "preprint"],   # research, reviews, letters, preprints
    "is_paratext": "false",                                # drop front matter
}
V1_YEARS = (2000, 2025)
SCOPE = "title_abstract_keywords.search"   # stricter alternative: "title_and_abstract.search"

# Inclusion rules applied at screening (kept with the data for provenance)
INCLUSION = {
    "question": "Did the study use, examine or measure ticks or tick-derived material?",
    "include": {
        "pathogen_in_tick": "Pathogen detected in field-collected ticks (the tick is the sample)",
        "host_infestation": "Ticks collected or counted on hosts",
        "tick_derived": "Anti-tick vaccines (Bm86), saliva, salivary proteins, genes, tick cell lines",
        "tick_control": "Acaricides, repellents, biocontrol tested on ticks",
        "tick_organism": "Tick biology, taxonomy, ecology; clinical tick-bite case with the tick identified",
        "tick_distribution_model": "Distribution or climate models built from tick records",
    },
    "include_with_flag": {"review": "Reviews of tick biology or control, analysed separately"},
    "exclude_keep_as_layer": {
        "disease_only": "Patients, serosurveys, pathogen vaccines in hosts, pathogen-only lab work, "
                        "alpha-gal clinical studies without tick material",
    },
}

for name, query in (("A_ticks", BLOCK_A), ("B_probe", BLOCK_B)):
    print(f"{name}: {len(query)} chars")    # keep each block well under ~4,000 (URL limit)

h = OpenAlexHarvester(api_key=userdata.get("OPENALEX_API_KEY"))

# 1) Size checks first (one call each)
counts = {
    "A_ticks": h.count(filters={**FILTERS, SCOPE: BLOCK_A}),
    "B_pathogens": h.count(filters={**FILTERS, SCOPE: BLOCK_B}),
    "B_not_A": h.count(filters={**FILTERS, SCOPE: f"{BLOCK_B} NOT {BLOCK_A}"}),
    "old_fulltext_search_A": h.count(filters=FILTERS, search=BLOCK_A),   # what full-text search would add
}
for k, v in counts.items():
    print(f"{k:24s} {v:>10,}")

# 2) Harvest Block A in full
dfA = h.harvest(filters={**FILTERS, SCOPE: BLOCK_A}, search=None,
                outdir=f"{OUTDIR}/A_ticks", max_records=None)
dfA = dfA.drop_duplicates("oa_id")
year = pd.to_numeric(dfA.get("oa_year", dfA.get("publication_year")), errors="coerce")
dfA["in_v1_window"] = year.between(*V1_YEARS)
dfA["query_block"] = "A_ticks"
dfA.to_parquet(f"{OUTDIR}/combined.parquet", index=False)
print("Block A:", dfA.shape, "| in v1 window:", int(dfA["in_v1_window"].sum()))

# 3) Recall probe: a random sample of "B NOT A" (OpenAlex `sample` + `seed`, so only 500 records are
#    pulled, not the whole disease literature). Screen it with the same inclusion rules.
import requests

def reconstruct_abstract(inv):
    if not inv:
        return ""
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))

flt = ",".join([f"publication_year:{FILTERS['publication_year']}",
                "type:" + "|".join(FILTERS["type"]),
                f"is_paratext:{FILTERS['is_paratext']}",
                f"{SCOPE}:{BLOCK_B} NOT {BLOCK_A}"])
rows, page = [], 1
while len(rows) < PROBE_SIZE and page <= 10:
    r = requests.get("https://api.openalex.org/works", timeout=60, params={
        "filter": flt, "sample": PROBE_SIZE, "seed": 2026, "per_page": 200, "page": page,
        "api_key": userdata.get("OPENALEX_API_KEY")})
    r.raise_for_status()
    res = r.json().get("results", [])
    if not res:
        break
    for w in res:
        rows.append({"oa_id": w["id"].rsplit("/", 1)[-1], "oa_title": w.get("display_name"),
                     "oa_year": w.get("publication_year"),
                     "oa_journal": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                     "oa_abstract": reconstruct_abstract(w.get("abstract_inverted_index"))})
    page += 1
probe = pd.DataFrame(rows).drop_duplicates("oa_id").head(PROBE_SIZE)
probe["probe_label"] = ""        # fill: tick_focused / disease_only / not_tick
probe.to_parquet(f"{OUTDIR}/B_probe_sample.parquet", index=False)
print("Recall probe sample:", len(probe), "of", counts["B_not_A"])
# Missed tick-focused papers ≈ (share labelled tick_focused in the probe) × counts["B_not_A"]

# 4) Provenance manifest (goes in docs/ and the data paper)
manifest = {
    "run_id": RUN_ID,
    "harvested_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "endpoint": "https://api.openalex.org/works", "scope": SCOPE, "filters": FILTERS,
    "v1_years": V1_YEARS, "block_A": BLOCK_A, "block_B_probe": BLOCK_B, "probe_size": PROBE_SIZE,
    "counts_before_harvest": counts, "rows_block_A": len(dfA),
    "rows_in_v1_window": int(dfA["in_v1_window"].sum()), "inclusion_rules": INCLUSION,
}
json.dump(manifest, open(f"{OUTDIR}/harvest_manifest.json", "w"), indent=2)
print(json.dumps({k: manifest[k] for k in ("run_id", "counts_before_harvest", "rows_block_A",
                                           "rows_in_v1_window")}, indent=2))
