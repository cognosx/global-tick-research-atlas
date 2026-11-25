# Data

Small, shareable tables are kept here. They let anyone check the corpus, the validation and the benchmarks without the large working files. Every table joins on the OpenAlex work ID (`oa_id`, e.g. `W2755950973`) or on country codes (`iso2` / `iso3`).

**Removed before publishing:** abstract columns (copyright varies by publisher) and BERTopic `Representative_Docs`. Titles and other OpenAlex metadata are CC0. Full texts are never shared.

## In this repository

| File | Rows | What it is | Made by |
| --- | ---: | --- | --- |
| `gold/gold_included_ids.parquet` | 56,051 | **The human-verified tick corpus**: the `oa_id` of every included paper | `apply_human_verified` |
| `validation/screener_validation_by_region.csv` | 20 | Precision, recall and F1 of each screener vs the human labels, by bloc (SSA, Maghreb, other) | `apply_human_verified` |
| `validation/llm_screen_validation_sample.csv` | 1,100 | Stratified sample with GPT, Gemini and DeepSeek labels, confidence, reasons and human labels | `04_screen_validation` |
| `validation/tickclass_validation.csv` (+ `_v1`) | 240 | Object-of-study classes vs gold class, by bloc | `tick_classify_all` |
| `validation/tickclass_review_disagreements_v2.csv` | 356 | Rule vs OpenAI vs DeepSeek disagreements with human class and notes | `merge_human_and_class` |
| `validation/methods_triples_validation.csv` | 1,121 | Method stage and tick × pathogen × host triples with gold columns to fill | `07_extract_methodology_and_triples` |
| `validation/study_location_validation.csv` | 720 | Extracted study countries with gold columns to fill | `06_extract_study_location` |
| `topics/topic_info_gold.csv` | 108 | BERTopic topics on the gold set: id, size, name, top words | `bertopic_multilingual_v2` |
| `topics/bertopic_topics_gold.parquet` | 42,555 | Topic of each gold paper (papers with abstracts) | `bertopic_multilingual_v2` |
| `topics/citation_clusters_gold.parquet` | 39,666 | Leiden bibliographic-coupling cluster of each gold paper | `citation_clustering` |
| `topics/cluster_top_refs_gold.parquet` | 180 | Most-cited references per cluster | `citation_clustering` |
| `lookups/topic_taxonomy.csv` | 46 | July (track A runner) topics: clean labels, theme groups, noise flags, SSA odds ratios | `sosci-pipeline` |
| `external/gbd_burden_country_year.parquet` | 4,080 | GBD DALY rates by country and year (encephalitis, NTDs), `tbd_daly_rate` | `06_ext_gbd_burden` |
| `external/wb_capacity_country_year.parquet` | 9,540 | World Bank R&D % GDP, researchers per million, GDP, population, education spend | `ext_worldbank_capacity` |
| `external/wb_country_meta.parquet` | 217 | Country names, ISO codes, income group, World Bank region | `ext_worldbank_capacity` |
| `external/gbif_species_country.parquet` | 93,271 | GBIF occurrence counts per tick species and country | `06_ext_b_gbif_occurrences` |
| `external/gbif_country_richness.parquet` | 252 | Tick species richness and total occurrences per country | `06_ext_b_gbif_occurrences` |
| `external/openalex_funders.parquet` | 45,639 | OpenAlex funders with country, Crossref and ROR IDs | `05_funding_enrichment` |
| `fulltext/preflight_audit.csv` | 35,107 | Full-text status per included study: source, sections, target sections | `fulltext_preflight` |
| `fulltext/fulltext_manifest.parquet` | 24,599 | Full-text harvest log: source, success, sections, references, attempts | `fulltext_harvester_v5` |
| `fulltext/reparse_report.csv` | 11,632 | GROBID / PyMuPDF re-parse results per PDF | `reparse_offline_v2` |
| `fulltext/missing_ids.csv` | 21,997 | Included studies still without full text | `fulltext_preflight` |
| `fulltext/pmcid_map.parquet` | 35,107 | DOI / PMID → PMCID matches and how they were found | `pmcid_backfill` |

## Kept in Google Drive only (too large or not shareable)

All in [`ticks/preprocessing_data`](https://drive.google.com/drive/folders/1Z8moDNA-ZKam-TiArOi1DQdu9m1XNYLi) unless noted. They will go to Zenodo as a versioned release once frozen (abstracts removed).

| File | Size | Why not here |
| --- | ---: | --- |
| `master_canonical.parquet` (334,277 deduplicated records) | 577 MB | Size |
| `canonical_screened.parquet` (rule-screen tiers) | 577 MB | Size |
| `screened_llm.parquet` (LLM screen, 103,737) | 585 MB | Size, abstracts |
| `llm_screen_human_validation.xlsx` (human review, `final_include`) | 171 MB | Size, abstracts |
| `screened_llm_gold.parquet` and the other `*_gold.parquet` enrichments | 9–93 MB each | Size, abstracts |
| `with_study_location`, `with_methods_triples`, `with_risk` (334,277 rows) | 585–589 MB | Size |
| `funding_enrich`, `with_upw_oc`, `with_venue`, `with_pmcid`, `with_burden` | 179–182 MB | Size |
| `authorships_reharvested.csv` (per-author countries) | 38 MB | Size |
| `with_orcid` / `_orcid_features` (researcher careers) | 16 MB | Personal data |
| `with_gender` / `gender_validation.csv` (inferred gender) | 24 MB | Sensitive personal data |
| `tickclass2/screened_llm_v3.*`, `merged_human_class2*` | 185–440 MB | Size, abstracts |
| `fulltext*/raw`, `sections/*.json` | — | Copyright |
| `doaj_journalcsv_*.csv`, `scimagojr_2025.csv` | 25 / 11 MB | Third-party files: download from [DOAJ](https://doaj.org/csv) and [SCImago](https://www.scimagojr.com/journalrank.php) |

**Not found anywhere** (see `docs/REPRODUCIBILITY.md`): raw harvest `data_raw/combined.parquet`, `with_livestock.parquet`, `with_doi.parquet`, the raw GBD export `gbd_daly_export.csv`.
