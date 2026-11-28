# Decision log

Every threshold and scope choice, with date and reason. Newest first.

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11-25 | **Scope: tick-focused research only.** Include studies that used ticks, tick parts or tick-derived material, pathogens detected in ticks, ticks on hosts, control or repellents tested on ticks, and distribution/climate models built from tick records (all 10 edge cases confirmed in the "GTRA inclusion criteria" Sheet). Reviews included with a flag. Disease-only studies excluded but kept as a flagged layer. | A clear construct (research on the vector), matching `tick_screen_prompt_v3`; the disease-only layer lets us show how much tick-borne disease research never touches a tick. |
| 2026-11-25 | Harvest: title + abstract + keywords search, Block A in full; pathogen/disease Block B used only as a 500-record recall probe. | Full-text search added 74% noise; Block B mostly brings out-of-scope disease-only papers. |
| 2026-11-25 | Work on the broad corpus (all tick research). Climate becomes a flag (`theme_climate`), not a separate corpus. | Broader questions, one master dataset, no duplicate pipelines. |
| 2026-11-25 | Freeze v1.0 at 31 Dec 2025; treat 2026 as partial. | Avoid an artificial dip in trends. |
| 2026-11-25 | Blind validation: n ≈ 600, two reviewers, Cohen's kappa, 95% CIs. | Earlier metrics came from reviewing LLM disagreements only, which may inflate them. |
| 2026-11-25 | Re-cluster BERTopic topic 0 (~24,300 papers). | One catch-all topic hides structure. |
| 2026-08 | Drop 8 noise topics ("tick box", "tick size", clock tick, quantum, archaeology, circadian, OCR noise, robots/games). | Non-tick meanings of "tick". |
| 2026-06 | Dedup ladder: OpenAlex ID → DOI → title+author+year → fuzzy title ≥ 90 (year ± 2). | Different OpenAlex IDs never merge; same ID always merges. |

## 2026-11-25 · Deduplication v4
- Full outputs are parquet only; Excel only for a sampled QC workbook (old xlsx export of 334k rows was the bottleneck).
- Preprint and published versions are linked (`version_group`), not merged. Counting analyses use `is_primary_version`.
- Correction, erratum and retraction notices are never merged on title.
- Every merge is logged in `merge_edges.parquet` with stage and score.
- Fuzzy merges blocked for series/numbered titles and swapped tick species; preprint-server and repository DOIs do not block merges.

## 2026-11-25 · Screening design (gtra-screen-v1)
- Engine reused from the GenAI facet-1 pipeline (κ 0.92 on 300 records): gpt-6-luna + deepseek-v4-flash, luna breaks ties, no adjudicator.
- The model names the object of study (9 classes, from the old tick_classify taxonomy); inclusion and layer (core / review / disease_only / excluded / uncertain) are derived in code.
- Rule flags (where the tick word sits, finance / plant / checkbox / mites / tic / archaeology noise) never exclude; they stratify the pilot and the audit.
- Only primary versions are screened; preprint partners inherit the decision.
- Gate before the full run: pilot of 300 hand-coded records, consensus sensitivity ≥ 0.95, class κ ≥ 0.80.
- Open decision for the pilot: acaricide residue / toxicity studies with ticks as the declared target are coded tick_control (tick_examined = false).

## 2026-11-27 · Record kinds and recall audit
- Full screening ran on 76,339 records (gtra-screen-v1) without the pilot gate; pilot and validation sheets are still uncoded, so all counts are provisional.
- `record_kind` flags non-studies: ~1,200 Zenodo taxonomic treatments (mostly 2025, one record per taxon name), figures/supplements, notices, other repository items. Counts use `count_as_study`; 2025 included drops from 3,096 to 1,936.
- Recall audit: 11,197 old-gold IDs are absent from the new harvest. Notebook 04 asks OpenAlex why (not_found / year_out / type_out / query_match_now / fulltext_only), screens them with the same prompt, and reports harvest recall for tick-focused papers.

## 2026-11-28 · Validation result (gtra-screen-v1)
- 598 blind, stratified records coded by coder 1; the 44 include/exclude disagreements adjudicated by coder 1 with both models' answers visible (to be confirmed by a blind second reviewer).
- Weighted to the corpus, adjudicated reference, "uncertain" counted as exclude: sensitivity 0.977 [0.961–0.989], specificity 0.993, precision 0.991. After human review of "uncertain": sensitivity 0.989. Class accuracy 0.815.
- Gate passed; prompt v1 stands. "Uncertain" records (968) go to human review; include/exclude splits keep luna's tie-break.
- Scope clarification from adjudication: knowledge/attitude/practice surveys about ticks and tick protection are tick research (include); mite-only studies and pathogen reviews without ticks are not.
- Labels: data/validation/gtra_v1_validation_labels.parquet (no abstracts).
