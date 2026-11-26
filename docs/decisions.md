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
