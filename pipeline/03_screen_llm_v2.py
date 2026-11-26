# Auto-generated mirror of notebooks/01_corpus/03_screen_llm_v2.ipynb

# # 03 · LLM screening — tick-focused (GTRA v1.0, prompt `gtra-screen-v1`)
# 
# **Question:** what was studied in relation to ticks? The model names a class; inclusion is derived in code.
# 
# | class | GTRA layer |
# |---|---|
# | tick_organism · tick_derived · pathogen_in_tick · tick_control · host_infestation · tick_distribution_model | **core** (or **review** if it is a review) |
# | disease_only | **disease_only** (excluded, kept as a layer) |
# | not_tick | excluded |
# | uncertain | human queue |
# 
# **Design** (from the GenAI facet-1 engine, κ 0.92): two models, `gpt-6-luna` + `deepseek-v4-flash`, luna breaks ties; no adjudicator. Answers are cached per (record, model, prompt version), failed calls are never cached, every quote is checked against the text. Code: `src/gtra/screen_llm.py`.
# 
# **Order:** prepare → fill abstracts from PubMed → **pilot 300 (you code) → score → fix prompt → freeze** → dry run → full run → human queue → blind validation.

# 1 · Setup
from google.colab import drive, userdata
drive.mount('/content/drive')
pass  # shell: pip -q install openai pandas pyarrow openpyxl xlsxwriter requests
import os, sys, json, glob, time, hashlib, importlib
if not os.path.exists('/content/gtra'):
    pass  # shell: git clone -q https://github.com/cognosx/global-tick-research-atlas /content/gtra
else:
    pass  # shell: git -C /content/gtra pull -q
sys.path.insert(0, '/content/gtra/src')
for k in ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "NCBI_API_KEY"]:
    try: os.environ[k] = userdata.get(k); print(k, "set")
    except Exception: print(k, "missing" + (" (optional)" if k == "NCBI_API_KEY" else ""))

import numpy as np, pandas as pd
from pathlib import Path
import gtra.screen_llm as G
importlib.reload(G)
pd.set_option('display.max_columns', None)

# 2 · Configuration + provider check
ROOT  = "/content/drive/MyDrive/COGNOSX/TICKS/ticks_research"      # <-- your Drive folder
RUN   = "openalex_2026-11-26"
CANON = f"{ROOT}/preprocessing_data/{RUN}/dedupe/works_canonical.parquet"
OUT   = Path(f"{ROOT}/preprocessing_data/{RUN}/screening"); OUT.mkdir(parents=True, exist_ok=True)

G.PROVIDERS["gpt_6_luna"]["model"] = "gpt-6-luna"
G.PROVIDERS["deepseek"]["model"]   = "deepseek-v4-flash"
G.PRIMARY, G.TIEBREAK = ["gpt_6_luna", "deepseek"], "gpt_6_luna"
G.DRIVE_CACHE = str(OUT / "gtra_screen_cache.jsonl")
print("prompt", G.PROMPT_VERSION, "| primary", G.PRIMARY, "| tiebreak", G.TIEBREAK)
G.check_providers()

# 3 · Load canonical works (primary versions only) + rule flags
w = pd.read_parquet(CANON)
w = w[w["is_primary_version"]].copy() if "is_primary_version" in w.columns else w
print(f"{len(w):,} records to screen")

pm_path = OUT / "pubmed_abstracts.parquet"
pm = pd.read_parquet(pm_path) if pm_path.exists() else pd.DataFrame(columns=["oa_id", "abstract"])
df = G.prepare(w, abstract_extra=dict(zip(pm["oa_id"], pm["abstract"])))
print(df["screen_input"].value_counts().to_string())
print("\nwhere the tick word sits:\n" + df["rule_tick_where"].value_counts().to_string())
print("\nnoise flags:\n" + df["rule_noise"].replace("", "(none)").value_counts().head(10).to_string())

# ## 4 · Fill missing abstracts from PubMed (run once)
# Records with no OpenAlex abstract but a PMID. Add an `NCBI_API_KEY` secret for 10 requests/second (free at ncbi.nlm.nih.gov/account). Then **re-run cell 3**.

need = df[(df["screen_input"] == "title_only") & df["oa_pmid"].fillna("").astype(str).str.strip().ne("")]
print(f"{len(need):,} title-only records have a PMID")
if len(need) and not pm_path.exists():
    pmid = need["oa_pmid"].astype(str).str.extract(r"(\d{4,9})\s*$")[0]
    got = G.fetch_pubmed_abstracts(pmid.dropna(), api_key=os.environ.get("NCBI_API_KEY"))
    pm = pd.DataFrame({"oa_id": need["oa_id"], "pmid": pmid}).assign(abstract=lambda t: t["pmid"].map(got)).dropna()
    pm.to_parquet(pm_path, index=False)
    print(f"{len(pm):,} abstracts saved -> re-run cell 3")

# ## 5 · Old labels: free comparison, not final labels
# The 56,051 human-verified IDs answered a broader question (tick relevance, disease papers included). Use them to stratify the pilot and to check agreement at the end.

gold = set(pd.read_parquet("/content/gtra/data/gold/gold_included_ids.parquet")["oa_id"].map(G.short_id))
df["old_gold"] = df["oa_id"].isin(gold)
print(f"in old gold: {df['old_gold'].sum():,} of {len(df):,}")
pd.crosstab(df["rule_tick_where"], df["old_gold"], margins=True)

# ## 6 · Pilot: 300 records you code blind
# Stratified by where the tick word sits × noise flag × old gold × abstract present (square-root allocation, so hard strata get enough rows). The sheet is written once; code the `class` and `review` columns (dropdowns) without looking at the hidden key sheet.

SHEET = OUT / "gtra_pilot_coding.xlsx"
if not SHEET.exists():
    pilot = G.draw_pilot(df, n=300, gold_ids=gold)
    G.write_pilot_sheet(pilot, OUT)
    pilot[["record_no", "oa_id", "_stratum", "_weight"]].to_csv(OUT / "gtra_pilot_key.csv", index=False)
pilot_ids = set(pd.read_csv(OUT / "gtra_pilot_key.csv")["oa_id"])
print(len(pilot_ids), "pilot records ->", SHEET)

# 7 · Run both models on the pilot, then score against YOUR codes
pres = G.screen(df[df["oa_id"].isin(pilot_ids)])
pres.to_parquet(OUT / f"gtra_pilot_llm_{G.PROMPT_VERSION}.parquet", index=False)
tab = G.score(str(SHEET), pres)      # needs the coded sheet; re-run after coding

# **Gate before the full run** (write the numbers in `docs/decisions.md`):
# - consensus **sensitivity ≥ 0.95** for include/exclude (missing tick papers is the costly error)
# - consensus **class κ ≥ 0.80**, include κ ≥ 0.85
# - read every include/exclude disagreement. If a rule in the prompt caused it, edit `SYSTEM_PROMPT`, **bump `PROMPT_VERSION`**, re-run cell 7 (old answers stay cached under the old version).

# 8 · Freeze the run: the coding set + prompt define the folder, so a rerun resumes
FROZEN_COLS = ["oa_id", "oa_title", "coding_abstract", "coding_abstract_from", "screen_input", "oa_keywords",
               "oa_year", "oa_type", "oa_language", "rule_tick_where", "rule_taxon", "rule_noise", "old_gold"]
frozen = df[[c for c in FROZEN_COLS if c in df.columns]].drop_duplicates("oa_id").sort_values("oa_id").reset_index(drop=True)
sig = hashlib.sha1(("|".join(frozen["oa_id"]) + G.PROMPT_VERSION).encode()).hexdigest()[:10]
RUN_DIR = OUT / "full" / f"{G.PROMPT_VERSION}_{sig}"; (RUN_DIR / "parts").mkdir(parents=True, exist_ok=True)
if not (RUN_DIR / "input.parquet").exists():
    frozen.to_parquet(RUN_DIR / "input.parquet", index=False)
    (RUN_DIR / "manifest.json").write_text(json.dumps({
        "input": CANON, "n": len(frozen), "prompt_version": G.PROMPT_VERSION,
        "prompt_sha1": hashlib.sha1(G.SYSTEM_PROMPT.encode()).hexdigest(),
        "providers": {p: G.PROVIDERS[p]["model"] for p in G.PRIMARY}, "tiebreak": G.TIEBREAK,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
print(f"{len(frozen):,} records -> {RUN_DIR}")

# 9 · Dry run: 60 records, timed (answers are reused by the full run)
cache = G.restore_cache()
probe = frozen.sample(60, random_state=1)
todo = sum(G.cache_key(i, p) not in cache for i in probe["oa_id"] for p in G.PRIMARY)
t0 = time.time(); r = G.screen(probe, cache=cache); sec = time.time() - t0
left = sum(G.cache_key(i, p) not in cache for i in frozen["oa_id"] for p in G.PRIMARY)
per = sec / max(todo, 1)
print(f"{todo} calls in {sec:.0f}s -> {per:.2f}s each | remaining {left:,} -> ~{left*per/3600:.1f} h")
r["gtra_layer"].value_counts()

# ## 10 · Full run in resumable chunks
# Each chunk is saved to `parts/` as soon as it finishes. **If Colab disconnects: re-run cells 1, 2, 3, 8 and this one.** The run stops by itself if a model answers fewer than half of a chunk (wrong id, key or quota).

CHUNK = 3000
parts = RUN_DIR / "parts"
chunks = [frozen.iloc[i:i + CHUNK] for i in range(0, len(frozen), CHUNK)]
cache = G.restore_cache(); t_all = time.time()
for k, ch in enumerate(chunks):
    f = parts / f"part_{k:04d}.parquet"
    if f.exists(): continue
    t0 = time.time(); res = G.screen(ch, cache=cache); res.to_parquet(f, index=False); G.backup_cache()
    print(f"[{len(list(parts.glob('part_*')))}/{len(chunks)}] {len(ch):,} in {(time.time()-t0)/60:.1f} min | "
          f"agree {res['gtra_models_agree'].mean():.1%} | include {res['gtra_include'].mean():.1%} | "
          f"elapsed {(time.time()-t_all)/3600:.2f} h")
print("all done" if len(list(parts.glob("part_*"))) == len(chunks) else "NOT finished -- re-run this cell")

# 11 · Assemble: screened works + preprint partners inherit their published version's decision
codes = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part_*.parquet"))]).drop_duplicates("oa_id", keep="last")
allw = pd.read_parquet(CANON); allw["oa_id"] = allw["oa_id"].map(G.short_id)
scr = allw.merge(codes, on="oa_id", how="left")
if "version_group" in scr.columns:
    vg = scr[scr["gtra_class"].notna() & scr["version_group"].fillna("").ne("")].drop_duplicates("version_group").set_index("version_group")
    gcols = [c for c in codes.columns if c.startswith("gtra_")]
    miss = scr["gtra_class"].isna() & scr["version_group"].isin(vg.index)
    scr.loc[miss, gcols] = vg.loc[scr.loc[miss, "version_group"], gcols].values
    scr.loc[miss, "gtra_status"] = "inherited_version"
print(f"{len(scr):,} records | unscreened: {scr['gtra_class'].isna().sum():,}")
scr.to_parquet(OUT / "works_screened.parquet", index=False)
scr["gtra_layer"].value_counts(dropna=False)

# 12 · Quality checks
ok = scr["gtra_class"].notna()
print("status:\n" + scr.loc[ok, "gtra_status"].value_counts(normalize=True).round(3).to_string())
print(f"\nmodels agree (class)   {scr.loc[ok, 'gtra_models_agree'].mean():.1%}")
print(f"models agree (include) {scr.loc[ok, 'gtra_agree_include'].mean():.1%}")
print(f"evidence verbatim      {(scr.loc[ok, 'gtra_evidence_ok'] == True).mean():.1%}")
print(f"human queue            {scr.loc[ok, 'gtra_needs_human'].sum():,}")
x = scr.merge(df[["oa_id", "rule_tick_where", "rule_noise", "screen_input"]], on="oa_id", how="left")
print(pd.crosstab(x["rule_tick_where"], x["gtra_layer"], margins=True))
print(pd.crosstab(x["rule_noise"].replace("", "(none)"), x["gtra_layer"]))
print(pd.crosstab(x["screen_input"], x["gtra_agree_include"], normalize="index").round(3))
print(pd.crosstab(x["oa_id"].isin(gold).rename("old_gold"), x["gtra_layer"], margins=True))

# 13 · Human review queue (xlsxwriter, truncated text -> seconds)
q = scr[scr["gtra_needs_human"] == True].merge(frozen[["oa_id", "coding_abstract"]], on="oa_id", how="left")
cols = ["oa_id", "oa_year", "oa_title", "coding_abstract", "gpt_6_luna_class", "deepseek_class",
        "gtra_class", "gtra_status", "gtra_evidence"]
q = q[[c for c in cols if c in q.columns]].assign(your_class="", memo="")
q["coding_abstract"] = q["coding_abstract"].fillna("").str.slice(0, 1500)
q.to_excel(OUT / "gtra_human_queue.xlsx", index=False, engine="xlsxwriter")
print(f"{len(q):,} records -> {OUT / 'gtra_human_queue.xlsx'}")

# 14 · Blind validation sample (600): stratified by layer x model agreement, pilot records excluded
v = scr[ok & ~scr["oa_id"].isin(pilot_ids)].merge(frozen[["oa_id", "coding_abstract"]], on="oa_id")
v["_stratum"] = v["gtra_layer"].astype(str) + "|" + np.where(v["gtra_models_agree"], "agree", "split")
sizes = v["_stratum"].value_counts(); alloc = np.sqrt(sizes); alloc = (alloc / alloc.sum() * 600).round().astype(int).clip(5, None).clip(upper=sizes)
val = pd.concat([v[v["_stratum"] == s].sample(k, random_state=7) for s, k in alloc.items()]).sample(frac=1, random_state=7)
val["_weight"] = val["_stratum"].map(sizes / alloc); val["record_no"] = range(1, len(val) + 1)
G.write_pilot_sheet(val, OUT, name="gtra_validation_coding.xlsx")
val[["record_no", "oa_id", "_stratum", "_weight"]].to_csv(OUT / "gtra_validation_key.csv", index=False)
# after coding (ideally two reviewers): tab = G.score(str(OUT / "gtra_validation_coding.xlsx"), codes)
