"""CLI: sosci <stage> -c config.yaml [--work-subdir preprocessing_data_v2]
Stages run in the optimized order. `all` runs the full preprocessing chain."""
from __future__ import annotations
import argparse
from . import runner as _runner

PRE_STAGES = _runner.OPTIMIZED_ORDER
ANALYSIS_STAGES = _runner.ANALYSIS_ORDER


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sosci",
                                  description="Science-of-science / meta-research preprocessing pipeline.")
    ap.add_argument("stage", help="stage name, 'all', 'screen', 'consensus', 'validate', or 'prisma'")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--work-subdir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--include-analysis", action="store_true")
    a = ap.parse_args(argv)

    if a.stage == "all":
        _runner.run_all(a.config, work_subdir=a.work_subdir, stop_after=a.stop_after,
                         include_analysis=a.include_analysis)
        return
    ctx = _runner.build_ctx(a.config, a.work_subdir)
    if a.stage in PRE_STAGES or a.stage in ANALYSIS_STAGES:
        _runner.run_stage(a.stage, ctx)
    elif a.stage == "screen":
        from . import screen
        screen.run_screen(ctx, model_label=a.model, max_records=a.max_records)
    elif a.stage == "consensus":
        from . import screen
        screen.run_consensus(ctx)
    elif a.stage == "validate_sample":
        from . import screen
        screen.draw_validation_sample(ctx)
    elif a.stage == "validate_compute":
        from . import screen
        screen.compute_validation(ctx)
    elif a.stage == "prisma":
        from . import provenance
        P, p = provenance.write_prisma(ctx.work)
        print("PRISMA ->", p)
    elif a.stage == "flows":
        from . import provenance
        provenance.build_flows(ctx.work)
        print("flows -> flow_corpus_construction.json/.mmd, flow_prisma_sr.json")
    elif a.stage == "translate":
        from . import analysis_stages
        analysis_stages.run_translate(ctx)
    elif a.stage in ("export", "export_corpus", "export_workbook"):
        from . import export
        which = {"export": "both", "export_corpus": "corpus", "export_workbook": "workbook"}[a.stage]
        export.run_export(ctx, which=which)
    else:
        raise SystemExit(f"unknown stage {a.stage}; valid: {PRE_STAGES + ANALYSIS_STAGES + ['all','screen','consensus','validate_sample','validate_compute','prisma','flows','translate','export']}")


if __name__ == "__main__":
    main()
