# sosci-pipeline v2.3

Reusable preprocessing + bibliometric analysis for science-of-science /
meta-research projects: multi-source ingestion -> PRISMA-ready, screened
dataset -> topic classification -> equity/composition/trend analysis.
Driven by one YAML config.

## What's in v2.1
- **No monkey-patching.** Stages are config-aware (`context.py`): they read
  paths and format from the config, read/write Parquet natively, and chain
  by canonical filenames.
- **Optimized preprocessing order.** Year-filter runs *before* enrichment,
  and enrichment runs on the *deduplicated* canonical set — far fewer API
  calls. A second oa_id-aware dedup pass after enrichment recovers
  preprint<->published merges.
- **Memory-safe, crash-resilient enrichment** (`enrich_chunked.py`). Processes
  CHUNK_SIZE rows at a time; each chunk is a self-contained, resumable file.
  Uses **direct parquet row-group access** to read each chunk's rows — NOT
  a re-scan from the start of the file — so chunk N costs the same as chunk
  1 regardless of position (a from-zero rescan makes later chunks take
  several times longer than earlier ones; this avoids that entirely).
  Combine streams chunks into the final file without ever holding the whole
  dataset in RAM.
- **Bibliometric analysis layer** (`analysis_stages.py`): tick/domain-relevance
  gatekeeping, BERTopic classification, and three analysis layers —
  research output vs. burden-proxy equity (BARI-style), topical composition
  by region (chi-square + odds ratios), and climate-as-emerging-niche
  (Mann-Kendall trend test).

## Pipeline (optimized order)
```
ingest -> type_filter -> year_filter -> dedupe_pass1(raw) -> enrich(chunked, canonical)
-> fix_funders -> dedupe_pass2(oa_id-aware) -> coverage -> type_audit -> reenrich
-> readiness -> backfill -> screen xN -> consensus -> validate -> prisma
-> [optional] tick_relevance -> topic_model -> layer1_burden -> layer2_topic_region
-> layer3_climate_niche
```

## Install
```bash
pip install -e .                  # core engine
pip install -e ".[llm]"           # + screening (anthropic/openai/google-genai)
pip install -e ".[topics]"        # + BERTopic classification
pip install -e ".[analysis]"      # + the three bibliometric layers
pip install -e ".[all]"           # everything
```

## Run (CLI)
```bash
sosci all -c configs/ticks_climate_ssa.yaml --work-subdir preprocessing_data_v2
# or stage by stage:
sosci type_filter -c configs/ticks_climate_ssa.yaml --work-subdir preprocessing_data_v2
sosci enrich      -c configs/ticks_climate_ssa.yaml --work-subdir preprocessing_data_v2
sosci screen      -c configs/ticks_climate_ssa.yaml --model openai
sosci consensus   -c configs/ticks_climate_ssa.yaml
sosci prisma      -c configs/ticks_climate_ssa.yaml
# analysis layer (after screening, or directly on the enriched/canonical set):
sosci tick_relevance      -c configs/ticks_climate_ssa.yaml
sosci topic_model         -c configs/ticks_climate_ssa.yaml
sosci layer1_burden       -c configs/ticks_climate_ssa.yaml
sosci layer2_topic_region -c configs/ticks_climate_ssa.yaml
sosci layer3_climate_niche -c configs/ticks_climate_ssa.yaml
```

## On Colab
```python
import sys; sys.path.insert(0, '/content/sosci')   # after unzipping
from sosci_pipeline import runner
ctx = runner.build_ctx('/content/sosci/configs/ticks_climate_ssa.yaml',
                        work_subdir='preprocessing_data_v2')
runner.run_stage('enrich', ctx)   # re-run the SAME cell to resume after a disconnect
```

## The enrichment chunking fix, explained
Earlier versions read each chunk's wide-file rows by iterating
`pf.iter_batches()` from the start of the file and discarding everything
before the chunk's window. That means chunk 2 re-reads and discards chunk 1's
rows, chunk 3 re-reads and discards chunks 1 AND 2, etc. — an O(n^2) pattern
across the whole file where chunk N's wasted work grows with N (this is
exactly what causes "chunk 1 took 4 minutes, chunk 2 took 13 minutes, chunk 3
took even longer"). `enrich_chunked.py` fixes this by reading parquet
**row-groups directly** via `ParquetFile.read_row_groups()`, seeking straight
to the rows needed — so every chunk costs roughly the same regardless of
position in the file.

## Reuse on a new project
Copy `configs/project.example.yaml`, set paths/sources/year/criteria, run.
Engine unchanged.

## Architecture
- `config.py` — load/validate YAML -> Config
- `context.py` — runtime paths/format shared by stages (set once by runner)
- `io.py` — data contract (canonical columns) + parquet/csv/xlsx read+write
- `apiutil.py` — HTTP session + `parallel_map` bounded thread pool
- `dedupe.py` — OpenAlex-ID-aware union-find dedup engine
- `enrich_chunked.py` — memory-safe, resumable, row-group-direct enrichment
- `stages.py` — `run_<stage>(ctx)` for every preprocessing step
- `analysis_stages.py` — tick-relevance, BERTopic, and the three bibliometric
  analysis layers (equity / composition / climate-niche)
- `screen.py` — LLM screening + consensus + validation (config-driven)
- `provenance.py` — run log + automatic PRISMA numbers (hardened against
  non-JSON-serializable `extra` values)
- `runner.py` — orchestrates the optimized order
- `cli.py` — `sosci <stage>` entry point

## Bibliometric analysis layers (background)
Built for a study of how tick research is organized: who studies it, what
they study, and how that's changed over time — specifically asking whether
Sub-Saharan Africa's tick research output and composition match its
tick-borne disease burden, and how the climate-tick niche has grown within
the broader tick literature. See `analysis/` for the original standalone,
independently-tested versions of each layer's logic, and the project's
chat history for the literature-landscape analysis that scoped these layers
against existing work (notably Su et al. 2025 on global tick research trends,
and Sambado & Ryan 2026 on tick-borne virus reporting capacity) to avoid
duplicating published findings.
