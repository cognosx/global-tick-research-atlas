# Notebooks

These are the Colab notebooks that built the GTRA data, copied from Google Drive on 2026-11-25. Before publishing:
- **outputs were cleared**
- **paths now point to** `/content/drive/MyDrive/COGNOSX/…` (change to your own)
- **personal emails were replaced** with `cognosxx@gmail.com`
- **keys:** none were found in the code; all keys are read from Colab secrets

Run them in the order of [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md), which gives each notebook's inputs, outputs, settings and recorded counts.

| Folder | Notebooks | Stage |
| --- | --- | --- |
| `01_corpus/` | `01_OpenAlexHarvester_Ticks`, `02_dedupe`, `02_dedupe_synthesis`, `03_tick_screen`, `03_screening_diagnostics`, `04_llm_screen`, `04_screen_validation`, `3a_llm_screen_single_model_gpt` (older), `apply_human_verified` | Harvest → dedup → rule + LLM screen → human labels → gold set |
| `02_enrichment/` | `05_harvest_authorships`, `05b_ror_enrichment`, `05c_orcid_enrichment`, `05_gender_enrichment`, `05_funding_enrichment`, `05_unpaywall_opencitations_enrichment`, `05_venue_enrichment`, `pmcid_backfill`, `ext_worldbank_capacity` | Authors, countries, careers, funders, OA, venues, IDs, capacity |
| `03_extraction_benchmarks/` | `06_extract_study_location`, `07_extract_methodology_and_triples`, `06_ext_gbd_burden`, `06_ext_b_gbif_occurrences`, `tick_classify5` (one shard; shards 1–4 are the same code), `tick_classify_all`, `tick_screen_prompt_v3`, `merge_human_and_class` | Study location, methods, entities, burden, tick occurrences, object of study |
| `04_structure_analysis/` | `citation_clustering`, `bertopic_multilingual_v2`, `paper3_equity_metrics`, `paper3_funder_network` | Clusters, topics, Paper 3 |
| `05_fulltext/` | `fulltext_preflight`, `fulltext_harvester_v5`, `reparse_offline_v2`, `diagnose_coverage`, `01c_fulltext_affiliations` | Full-text harvest and section parsing |
| `06_side_analyses/` | `biomining`, `biomining2_`, `en_ner_bc5cdr_md`, `02_pathogen_disease_NER_ticks`, `05_Methodology_Profile_Pipeline_v6b`, `crossfill_opportunity` | NER and relations, disease NER (phage-paper method), methodology profile, source cross-fill |
| `track_a_climate_runners/` | `sosci_pipeline_runner_v2.3`, `sosci_pipeline_runner_v2.2` | Climate review track (engine in `src/sosci_pipeline`) |
| (root) | `00_sosci_pipeline_runner.ipynb` | Runner shipped with the engine package |

Not included: older copies (`Copy of …`, `Copie de …`, `tick_classify_OLD`, fulltext harvester v1–v4, `03_tick_screen_old`) and notebooks from other projects (GenAI, preprints).
