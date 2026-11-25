# Global Tick Research Atlas (GTRA)

An open, deduplicated and enriched map of the world's research on ticks and tick-borne diseases.

**Question behind the atlas:** does global tick research go where ticks and tick-borne diseases do harm?

> Status: under construction. Release `v1.0` (cut-off 31 Dec 2025) is in preparation.

## Key numbers (working version, to be frozen in v1.0)

| Item | Value |
| --- | --- |
| Raw records harvested (7 sources) | ~337,500 |
| Unique works after deduplication | ~334,300 |
| Confirmed tick papers (2000–2026) | ~82,000 |
| Topics (BERTopic) | 46, of which 8 noise topics removed |

## What is inside

- **Pipeline**: harvest → clean → deduplicate → tick-relevance screening → topics → enrichment → analysis.
- **Enrichment**: authorships and countries (ROR), funders, ORCID, venues (DOAJ, Scimago), open-access status (Unpaywall), citations (OpenCitations), study location, tick × pathogen × host entities.
- **Benchmarks** (planned): disease burden (GBD), outbreaks (WOAH-WAHIS), livestock (FAO), tick occurrence (GBIF).

## Repository layout

```
config/                gtra.yaml (broad corpus), legacy climate config, example
src/sosci_pipeline/    the engine: dedup, enrichment, LLM screening, topics, provenance
pipeline/reference/    original per-stage Colab scripts (stages/ and analysis/)
pipeline/fulltext/     tick_sections.py — multilingual section detection in full texts
notebooks/             Colab runner, exploration only
data/sample/           small sample to run everything end to end
data/lookups/          topic_taxonomy.csv (46 topics, theme groups, noise topics, SSA odds ratios)
validation/            blind validation sample, labels, agreement metrics
papers/                one folder per paper: analysis, figures, tables
docs/                  REPRODUCIBILITY.md (step-by-step build), data dictionary, decision log, engine docs
tests/                 engine tests (pytest)
explorer/              static web explorer
```

## Quick start

```bash
pip install -e ".[all]"      # or: pip install -e .  for the core only
pytest tests                 # engine tests
sosci --help                 # stage runner, reads config/gtra.yaml
```

In Colab, open `notebooks/00_sosci_pipeline_runner.ipynb` and set `base_dir` in `config/gtra.yaml`.
API keys are read from environment variables or Colab secrets, never from files in this repository.

## Data

The full dataset is too large for GitHub. It will be published on Zenodo with a DOI for each release.
Only identifiers and openly licensed metadata are shared. Raw exports from licensed databases (WoS, Scopus, Dimensions, Lens) and full texts are not redistributed.

## How to cite

See [`CITATION.cff`](CITATION.cff). A dataset DOI will be added at the v1.0 release.

## Team

Led by Maxime Descartes Mbogning Fonkou (CognosX), with domain expertise from tick researchers in Africa.

## Licence

- Code: MIT (see `LICENSE`)
- Derived data and documentation: CC BY 4.0 (see `DATA_LICENSE`)
