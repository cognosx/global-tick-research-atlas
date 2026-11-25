"""Pipeline orchestrator. Runs stages in the OPTIMIZED order, sets the shared
context, logs provenance, and chains stages by canonical filenames (parquet).

Optimized order (year-filter before enrich; enrich the deduped canonical set):
  ingest -> type_filter -> year_filter -> dedupe(pass1, raw) -> enrich(canonical)
  -> fix_funders -> dedupe(pass2, oa_id-aware) -> coverage -> type_audit
  -> reenrich -> readiness -> backfill -> screen -> consensus -> validate -> prisma
"""
from __future__ import annotations
import os
from . import context as _context
from . import provenance as _prov
from .config import load_config

F = {
    "master": "master_search_results_with_ids",
    "typed": "master_filtered_by_type",
    "year": "master_year_filtered",
    "dedupe1": "master_dedupe_pass1",
    "canon1": "master_canonical_pass1",
    "tick_checked": "master_tick_checked",
    "topic_modeled": "master_topic_modeled",
    "analysis_ready": "master_analysis_ready",
    "translated": "master_analysis_ready_translated",
    "enriched": "master_enriched",
    "dedupe_audit": "master_dedupe_audit",
    "canonical": "master_canonical",
    "canon_clean": "master_canonical_clean",
    "canon_enriched": "master_canonical_enriched",
    "screen_ready": "master_canonical_screen_ready",
    "screen_results": "screening_results",
}

OPTIMIZED_ORDER = [
    "type_filter", "year_filter", "dedupe_pass1", "enrich", "fix_funders",
    "dedupe_pass2", "coverage", "type_audit", "reenrich", "readiness",
    "backfill",
]

ANALYSIS_ORDER = [
    "tick_relevance", "topic_model", "finalize_corpus", "layer1_burden",
    "layer2_topic_region", "layer3_climate_niche",
]


def build_ctx(config_path, work_subdir=None):
    cfg = load_config(config_path)
    work = cfg.work_dir
    if work_subdir:
        work = os.path.join(cfg.base_dir, work_subdir)
    ctx = _context.Ctx(base=cfg.base_dir, data=cfg.data_dir, work=work,
                        fmt=cfg.get("io", "format", default="parquet"),
                        email=cfg.email, cfg=cfg)
    _context.set_ctx(ctx)
    print(f"[runner] project={cfg.name}")
    print(f"[runner] work dir={work}")
    return ctx


def run_stage(name, ctx):
    from . import stages
    fn = getattr(stages, f"run_{name}", None)
    if fn is None:
        from . import analysis_stages
        fn = getattr(analysis_stages, f"run_{name}", None)
    if fn is None:
        from . import export
        fn = getattr(export, f"run_{name}", None)
    if fn is None:
        raise ValueError(f"Unknown stage: {name}")
    print(f"\n{'='*60}\n[stage] {name}\n{'='*60}")
    return fn(ctx)


def run_all(config_path, work_subdir=None, stop_after=None, include_analysis=False):
    ctx = build_ctx(config_path, work_subdir)
    order = OPTIMIZED_ORDER + (ANALYSIS_ORDER if include_analysis else [])
    for st in order:
        run_stage(st, ctx)
        if stop_after and st == stop_after:
            print(f"[runner] stopping after {st} as requested.")
            break
    print("\n[runner] generating PRISMA numbers...")
    P, path = _prov.write_prisma(ctx.work)
    print(f"[runner] PRISMA -> {path}")
    return ctx
