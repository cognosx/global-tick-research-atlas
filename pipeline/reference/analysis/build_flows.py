# ============================================================
#  Build BOTH flow diagrams the project needs:
#   (A) corpus_construction_flow  — ingest -> ... -> tick_relevance
#       -> topic_model -> cleanup -> analysis-ready  (bibliometric atlas)
#   (B) prisma_sr                 — the PRISMA-2020 for the systematic
#       review arm (needs screening to have run; gracefully notes if not)
#  Emits JSON + Mermaid (paste-ready for papers/dashboards).
#  Robust to the run_log having repeated stage entries (multiple runs):
#  it always takes the LAST occurrence of each stage.
# ============================================================
import os, json

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

RUN_LOG   = os.path.join(WORK, "run_log.json")
CORPUS_FLOW = os.path.join(WORK, "corpus_construction_flow.json")  # from finalize_corpus.py

def _last_by_stage(log):
    """run_log may contain several entries per stage (re-runs). Keep the LAST."""
    out = {}
    for e in log:
        out[e["stage"]] = e   # later entries overwrite earlier
    return out

def run():
    log = json.load(open(RUN_LOG))
    S = _last_by_stage(log)
    def ro(stage):
        e = S.get(stage); return e.get("rows_out") if e else None

    ingest   = ro("ingest")
    typef    = ro("type_filter")
    yearf    = ro("year_filter")
    dd1      = ro("dedupe_pass1")
    dd2      = ro("dedupe")            # pass2 logs as "dedupe"
    # tick_relevance logs rows_out = n tick-relevant; rows_in = n screened
    tr = S.get("tick_relevance", {})
    tr_in  = tr.get("rows_in")
    tr_out = tr.get("rows_out")
    tm     = ro("topic_model")
    per_source = (S.get("ingest", {}).get("extra", {}) or {}).get("per_source", {})
    merges = (S.get("dedupe_pass1", {}).get("extra", {}) or {}).get("merges", {})

    corpus_extra = json.load(open(CORPUS_FLOW)) if os.path.exists(CORPUS_FLOW) else {}
    final_corpus = corpus_extra.get("final_analysis_corpus")
    noise_removed = corpus_extra.get("removed_nontick_noise")

    # ---------- (A) corpus-construction flow ----------
    flowA = {
        "records_identified_total": ingest,
        "records_by_source": per_source,
        "after_type_filter": typef,
        "after_year_filter": yearf,
        "unique_studies_after_dedup_pass1": dd1,
        "dedup_pass1_merges": merges,
        "canonical_after_enrich_and_dedup_pass2": dd2,
        "screen_ready_input_to_tick_relevance": tr_in,
        "tick_relevant": tr_out,
        "not_tick_relevant_removed": (tr_in - tr_out) if (tr_in and tr_out) else None,
        "topic_modeled": tm,
        "nontick_noise_topics_removed": noise_removed,
        "final_analysis_ready_corpus": final_corpus,
    }
    json.dump(flowA, open(os.path.join(WORK, "flow_corpus_construction.json"), "w"), indent=2)

    def fmt(n): return f"{n:,}" if isinstance(n, int) else "?"
    mmA = ["flowchart TD"]
    mmA.append(f'  A["Records identified across 7 sources<br/>n = {fmt(ingest)}"]')
    src = "<br/>".join(f"{k}: {fmt(v)}" for k, v in per_source.items())
    mmA.append(f'  A --> B["Document-type filter<br/>n = {fmt(typef)}"]')
    mmA.append(f'  B --> C["Year filter (&ge;2000)<br/>n = {fmt(yearf)}"]')
    mmA.append(f'  C --> D["Cross-source de-duplication<br/>unique studies n = {fmt(dd1)}"]')
    mmA.append(f'  D --> E["Enrichment + oa_id-aware merge<br/>canonical n = {fmt(dd2)}"]')
    mmA.append(f'  E --> F["Tick-relevance filter (rule-based)<br/>tick-relevant n = {fmt(tr_out)}"]')
    if tr_in and tr_out:
        mmA.append(f'  F -. "excluded not-tick<br/>n = {fmt(tr_in - tr_out)}" .-> X1[ ]')
    mmA.append(f'  F --> G["BERTopic topic modelling<br/>n = {fmt(tm)}"]')
    if noise_removed:
        mmA.append(f'  G -. "non-tick homonym topics<br/>n = {fmt(noise_removed)}" .-> X2[ ]')
    mmA.append(f'  G --> H["Analysis-ready tick corpus<br/>n = {fmt(final_corpus)}"]')
    open(os.path.join(WORK, "flow_corpus_construction.mmd"), "w").write("\n".join(mmA))

    # ---------- (B) PRISMA for the SR arm ----------
    screen = S.get("screen", {})
    sx = screen.get("extra", {}) if screen else {}
    prisma = {
        "identification_records": ingest,
        "records_by_source": per_source,
        "after_type_filter": typef,
        "after_year_filter": yearf,
        "records_after_duplicates_removed": dd2 or dd1,
        "records_screened": sx.get("include", 0) is not None and (dd2 or dd1) or None,
        "included": sx.get("include"),
        "excluded": sx.get("exclude"),
        "sent_to_human_review": sx.get("human_review"),
        "exclusion_reasons": sx.get("reasons"),
        "SCREENING_STATUS": "COMPLETE" if sx else "NOT RUN YET — screening produced no output",
    }
    json.dump(prisma, open(os.path.join(WORK, "flow_prisma_sr.json"), "w"), indent=2)

    print("=== (A) CORPUS CONSTRUCTION FLOW (bibliometric atlas) ===")
    for k, v in flowA.items():
        if k not in ("records_by_source", "dedup_pass1_merges"):
            print(f"  {k:42} {fmt(v) if isinstance(v,int) else v}")
    print("\n  Mermaid diagram -> flow_corpus_construction.mmd")
    print("\n=== (B) PRISMA (systematic review arm) ===")
    print(f"  screening status: {prisma['SCREENING_STATUS']}")
    print("  (include/exclude counts will populate once screening runs)")
    print("\n  Saved: flow_corpus_construction.json/.mmd, flow_prisma_sr.json")
    return flowA, prisma

if __name__ == "__main__":
    run()
