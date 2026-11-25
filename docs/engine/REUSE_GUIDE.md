# Reusing sosci-pipeline for a new project

The engine (`sosci_pipeline/`) is generic. Everything project-specific lives in
three places, so starting a new study means editing those — never the engine.

## The three project-specific things
1. **A config** in `configs/` — copy `project.example.yaml`, set `paths`,
   `sources`, `year_filter`, `screening.question` + `criteria`, and (if you have
   a domain relevance filter) `topic_model.only_tick_relevant`.
2. **A project folder** in `projects/<name>/` — holds a `README.md` (findings,
   paper plan) and, once you've labelled topics, a `topic_taxonomy.csv`.
3. **A relevance filter** (optional) — only needed if your corpus, like the
   ticks one, was pulled with broad keywords that catch homonyms. For a new
   domain, either write a small rule-based classifier (see
   `sosci_pipeline/analysis_stages.py::_classify_relevance`) or rely on the
   topic model + a taxonomy to flag noise clusters after the fact.

## Pipeline order (what runs, in sequence)
Preprocessing (`OPTIMIZED_ORDER`):
```
type_filter → year_filter → dedupe_pass1 → enrich → fix_funders →
dedupe_pass2 → coverage → type_audit → reenrich → readiness → backfill
```
Then screening: `screen` (per model) → `consensus` → `validate` → `prisma`.

Analysis (`ANALYSIS_ORDER`, optional):
```
tick_relevance → topic_model → finalize_corpus → layer1_burden →
layer2_topic_region → layer3_climate_niche
```
Utility stages (run any time): `flows`, `translate`, `export`.

## Key design choices baked in
- **Memory-safe enrichment.** `enrich` reads only needed columns in
  CHUNK_SIZE-row slices via direct parquet **row-group access** (not a
  from-zero rescan), so each chunk costs the same regardless of position.
  Resumable at the chunk level; re-run the same cell after a disconnect.
- **Straggler re-enrichment.** `reenrich_stragglers` retries only records left
  unmatched by API rate limits, optionally under a different contact email —
  idempotent, backs up before overwriting.
- **Multilingual topic modelling by default.** `embed_model` defaults to
  `paraphrase-multilingual-mpnet-base-v2` so non-English papers cluster by
  scientific theme, not language. No translation/API cost. (Translation is a
  separate optional `translate` stage if you want permanent English text.)
- **Topic modelling as a QC filter.** Homonym contamination self-clusters;
  `finalize_corpus` drops the flagged topics (via the project taxonomy) and
  re-applies the year window on enrichment-corrected years.
- **Two flow diagrams.** `flows` emits a corpus-construction flowchart (for a
  bibliometric paper) and a PRISMA-2020 (for the SR arm), robust to re-run
  duplicate log entries.
- **Exploration exports.** `export` writes a curated per-record `corpus_explore.xlsx`
  and an `analysis_results.xlsx` bundling the aggregate result tables.

## Minimal new-project checklist
1. `cp configs/project.example.yaml configs/myproject.yaml` and edit it.
2. `mkdir projects/myproject` and add a `README.md`.
3. Put your ingested master in the WORK dir as
   `master_search_results_with_ids.parquet`.
4. `sosci all -c configs/myproject.yaml` (preprocessing), then screen/analysis
   stages as needed.
5. After `topic_model`, eyeball `topic_summary.csv`, build a
   `topic_taxonomy.csv`, drop it in the WORK dir, then run `finalize_corpus`.
