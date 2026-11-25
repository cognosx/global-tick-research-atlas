# Changelog

## v2.3 — reuse-ready reorganization + analysis completion
- **New stages**
  - `finalize_corpus` — drops NON-TICK homonym topics (via project taxonomy),
    re-applies the year window on enrichment-corrected years, attaches clean
    labels + theme groups → `master_analysis_ready`.
  - `translate` — optional non-English title/abstract translation (Claude API,
    resumable). Use only for permanent English text; multilingual embeddings
    handle topic clustering without it.
  - `export` / `export_corpus` / `export_workbook` — curated per-record
    `corpus_explore.xlsx` + aggregate `analysis_results.xlsx`.
  - `flows` — dual flow builder (corpus-construction + PRISMA-2020) in
    `provenance.build_flows`, robust to duplicate run-log entries.
- **Multilingual topic modelling is now the default** (`embed_model:
  paraphrase-multilingual-mpnet-base-v2`); embeddings are L2-normalised.
- **Layers 1–3 now read the cleaned `master_analysis_ready`** corpus first.
- **Repo reorganised for reuse across projects:**
  - `projects/<name>/` for per-project assets (taxonomy, findings README).
  - `notebooks/` for the Colab runner.
  - `docs/` with a REUSE_GUIDE.
  - `reference_scripts/{stages,analysis}/` (merged old `standalone_scripts/` +
    `analysis/`).
- New `export.py` module; `analysis.max_year`, `analysis.topic_taxonomy_csv`,
  `analysis.noise_topic_ids` config keys.

## v2.2 — straggler re-enrichment + model toggles
- `reenrich_stragglers` stage (retry rate-limited records, alt email, idempotent).
- `tick_relevance` prefers the richest `screen_ready` input.
- Runner notebook: per-model screening toggles.

## v2.1 — chunked enrichment fix + analysis layer
- Row-group-direct chunked enrichment (fixes O(n^2) growing per-chunk time).
- Tick-relevance, BERTopic, and Layers 1–3 as package stages.
