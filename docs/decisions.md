# Decision log

Every threshold and scope choice, with date and reason. Newest first.

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11-25 | Work on the broad corpus (all tick research). Climate becomes a flag (`theme_climate`), not a separate corpus. | Broader questions, one master dataset, no duplicate pipelines. |
| 2026-11-25 | Freeze v1.0 at 31 Dec 2025; treat 2026 as partial. | Avoid an artificial dip in trends. |
| 2026-11-25 | Blind validation: n ≈ 600, two reviewers, Cohen's kappa, 95% CIs. | Earlier metrics came from reviewing LLM disagreements only, which may inflate them. |
| 2026-11-25 | Re-cluster BERTopic topic 0 (~24,300 papers). | One catch-all topic hides structure. |
| 2026-08 | Drop 8 noise topics ("tick box", "tick size", clock tick, quantum, archaeology, circadian, OCR noise, robots/games). | Non-tick meanings of "tick". |
| 2026-06 | Dedup ladder: OpenAlex ID → DOI → title+author+year → fuzzy title ≥ 90 (year ± 2). | Different OpenAlex IDs never merge; same ID always merges. |
