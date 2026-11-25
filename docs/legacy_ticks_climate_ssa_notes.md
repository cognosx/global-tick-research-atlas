# Project: Ticks / Climate / Sub-Saharan Africa

This folder holds the **project-specific** assets for the tick-borne-disease
study. The reusable engine lives in `../../sosci_pipeline/`; the config is
`../../configs/ticks_climate_ssa.yaml`. Keeping project assets here (separate
from the engine) is what makes the pipeline reusable for other projects.

## Corpus
All studies **related to ticks** (retrieved with tick-only keywords across 7
sources). Climate-tick relevance is identified by a downstream LLM screening
step; the broad corpus itself powers the bibliometric / science-of-science
analysis.

## Files here
- `topic_taxonomy.csv` — the 47 BERTopic topics relabelled into clean names +
  13 theme groups, each flagged keep/DROP, with the SSA odds ratio merged in.
  8 topics (~11,170 records, ~13.6%) are NON-TICK homonyms ("tick box", "tick
  size", "tick-tock", clock/quantum/archaeology) that BERTopic self-clustered —
  the `finalize_corpus` stage drops these to yield the clean ~70,944-paper
  corpus. Upload a copy of this file into the WORK dir so the stage can read it
  (path set in config: `analysis.topic_taxonomy_csv`).

## Headline findings (from the first full run)
- **Corpus:** 314,061 records → 114,013 unique studies → 108,899 canonical →
  82,114 tick-relevant → ~70,944 clean analysis-ready.
- **Layer 1 (output vs burden):** SSA = 6.22% of global tick output across 196
  countries; South Africa alone leads SSA (1,481 papers). BARI regression needs
  a burden denominator (livestock population recommended over the Sambado file).
- **Layer 2 (composition):** χ²≈3,592, p≈0, Cramér's V=0.265. SSA is
  OVER-represented in livestock/locally-endemic topics (heartwater OR 12.8,
  cattle OR 13.3, ASF OR 4.9, camels/Hyalomma OR 3.0, CCHF, acaricide
  resistance, babesiosis) and UNDER-represented in molecular/vaccine work
  (salivary proteins OR 0.61) and Northern-Hemisphere human diseases (Lyme, TBE,
  Powassan). **The sharpest story:** SSA tick science is applied / agricultural /
  surveillance-oriented, under-invested in mechanistic/vaccine capability.
- **Layer 3 (climate niche):** increasing trend τ=0.527, p=0.0001, but small —
  ~1% of tick papers in 2000 rising to ~2.9% in 2025. Motivates the SR.
- **Coverage caveat:** Layer 2 used 51,208/82,114 records (38% lacked
  country/affiliation). A targeted affiliation re-enrichment would strengthen it.

## Planned papers
1. **Flagship (Layers 1+2):** SSA tick science shaped by the livestock economy /
   burden → *PLOS NTD* / *BMJ Global Health* / *Lancet Planetary Health*.
2. **Atlas (topic map + homonym-filter method):** *Ticks and Tick-borne
   Diseases* / *Parasites & Vectors*.
3. **Climate niche + SR:** *Lancet Planetary Health* / *Environmental Research*.

## Dashboards
Corpus explorer (topic × country × year × pathogen), research-equity choropleth
(output vs livestock burden), topic-trends view. Built from the exported CSVs +
`corpus_explore.xlsx`.
