"""Run-log provenance + automatic PRISMA number generation. log_stage() is
hardened against non-JSON-serializable values (sets, tuples, Ellipsis, etc.)
so a stray value in `extra` can never crash the whole pipeline run."""
from __future__ import annotations
import os
import json
import datetime


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _safe(o):
    """Coerce values into JSON-serializable forms (sets/tuples -> lists, etc.)."""
    if isinstance(o, dict):
        return {str(k): _safe(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, list):
        return [_safe(v) for v in o]
    if o is ...:
        return None
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _log_path(work_dir):
    return os.path.join(work_dir, "run_log.json")


def _load_log(work_dir):
    p = _log_path(work_dir)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return []
    return []


def log_stage(work_dir, stage, inputs=None, output=None, rows_in=None, rows_out=None,
              params=None, notes="", extra=None):
    os.makedirs(work_dir, exist_ok=True)
    log = _load_log(work_dir)
    entry = _safe({
        "stage": stage, "timestamp": _now(),
        "inputs": inputs or [], "output": output,
        "rows_in": rows_in, "rows_out": rows_out,
        "params": params or {}, "notes": notes or "", "extra": extra or {},
    })
    log.append(entry)
    json.dump(log, open(_log_path(work_dir), "w"), indent=2, default=str)
    return entry


def build_prisma(work_dir):
    log = _load_log(work_dir)
    by_stage = {e["stage"]: e for e in log}

    def rows_out(stage):
        e = by_stage.get(stage)
        return e.get("rows_out") if e else None

    ingest = rows_out("ingest")
    type_filter = rows_out("type_filter")
    year_filter = rows_out("year_filter")
    dedupe1 = rows_out("dedupe_pass1")
    dedupe2 = rows_out("dedupe")
    screen = by_stage.get("screen", {})
    screen_extra = screen.get("extra", {}) if screen else {}

    P = {
        "records_identified": ingest,
        "records_after_type_filter": type_filter,
        "records_after_year_filter": year_filter,
        "records_after_dedup_pass1": dedupe1,
        "records_after_dedup_final": dedupe2,
        "records_screened": dedupe2 or dedupe1,
        "included": screen_extra.get("include"),
        "excluded": screen_extra.get("exclude"),
        "human_review": screen_extra.get("human_review"),
        "exclusion_reasons": screen_extra.get("reasons"),
    }
    return P


def write_prisma(work_dir):
    P = build_prisma(work_dir)
    path = os.path.join(work_dir, "prisma_numbers.json")
    json.dump(P, open(path, "w"), indent=2, default=str)
    return P, path


def _last_by_stage(log):
    out = {}
    for e in log:
        out[e["stage"]] = e
    return out


def build_flows(work_dir):
    """Emit BOTH flow artifacts a bibliometric+SR project needs:
      (A) corpus-construction flow  ingest -> ... -> tick_relevance ->
          topic_model -> finalize -> analysis-ready  (JSON + Mermaid)
      (B) PRISMA-2020 for the SR arm (populates once screening runs).
    Robust to the run_log holding repeated stage entries (re-runs):
    always uses the LAST occurrence of each stage."""
    log = _load_log(work_dir)
    S = _last_by_stage(log)

    def ro(stage):
        e = S.get(stage)
        return e.get("rows_out") if e else None

    ingest = ro("ingest"); typef = ro("type_filter"); yearf = ro("year_filter")
    dd1 = ro("dedupe_pass1"); dd2 = ro("dedupe")
    tr = S.get("tick_relevance", {}); tr_in = tr.get("rows_in"); tr_out = tr.get("rows_out")
    tm = ro("topic_model")
    fin = S.get("finalize_corpus", {})
    fin_out = fin.get("rows_out")
    fin_noise = (fin.get("extra", {}) or {}).get("dropped_noise")
    per_source = (S.get("ingest", {}).get("extra", {}) or {}).get("per_source", {})
    merges = (S.get("dedupe_pass1", {}).get("extra", {}) or {}).get("merges", {})

    flowA = {
        "records_identified_total": ingest, "records_by_source": per_source,
        "after_type_filter": typef, "after_year_filter": yearf,
        "unique_studies_after_dedup_pass1": dd1, "dedup_pass1_merges": merges,
        "canonical_after_enrich_and_dedup_pass2": dd2,
        "input_to_tick_relevance": tr_in, "tick_relevant": tr_out,
        "not_tick_relevant_removed": (tr_in - tr_out) if (tr_in and tr_out) else None,
        "topic_modeled": tm, "nontick_noise_topics_removed": fin_noise,
        "final_analysis_ready_corpus": fin_out,
    }
    json.dump(flowA, open(os.path.join(work_dir, "flow_corpus_construction.json"), "w"),
              indent=2, default=str)

    def fmt(n):
        return f"{n:,}" if isinstance(n, int) else "?"

    mm = ["flowchart TD",
          f'  A["Records identified (all sources)<br/>n = {fmt(ingest)}"]',
          f'  A --> B["Document-type filter<br/>n = {fmt(typef)}"]',
          f'  B --> C["Year filter<br/>n = {fmt(yearf)}"]',
          f'  C --> D["Cross-source de-duplication<br/>unique studies n = {fmt(dd1)}"]',
          f'  D --> E["Enrichment + oa_id-aware merge<br/>canonical n = {fmt(dd2)}"]',
          f'  E --> F["Tick-relevance filter (rule-based)<br/>tick-relevant n = {fmt(tr_out)}"]']
    if tr_in and tr_out:
        mm.append(f'  F -. "excluded not-tick n = {fmt(tr_in - tr_out)}" .-> X1[ ]')
    mm.append(f'  F --> G["BERTopic topic modelling<br/>n = {fmt(tm)}"]')
    if fin_noise:
        mm.append(f'  G -. "non-tick homonym topics n = {fmt(fin_noise)}" .-> X2[ ]')
    mm.append(f'  G --> H["Analysis-ready tick corpus<br/>n = {fmt(fin_out)}"]')
    open(os.path.join(work_dir, "flow_corpus_construction.mmd"), "w").write("\n".join(mm))

    screen = S.get("screen", {}); sx = screen.get("extra", {}) if screen else {}
    prisma = {
        "identification_records": ingest, "records_by_source": per_source,
        "after_type_filter": typef, "after_year_filter": yearf,
        "records_after_duplicates_removed": dd2 or dd1,
        "included": sx.get("include"), "excluded": sx.get("exclude"),
        "sent_to_human_review": sx.get("human_review"),
        "exclusion_reasons": sx.get("reasons"),
        "SCREENING_STATUS": "COMPLETE" if sx else "NOT RUN YET",
    }
    json.dump(prisma, open(os.path.join(work_dir, "flow_prisma_sr.json"), "w"),
              indent=2, default=str)
    return flowA, prisma
