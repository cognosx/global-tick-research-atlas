"""Stage implementations: each run_<stage>(ctx) reads its input via ctx.find(),
does the work, writes via io.write_table to ctx.out(), and logs provenance."""
from __future__ import annotations
import os
import re
import pandas as pd
import numpy as np
from . import io as _io
from . import provenance as _prov
from .runner import F


def _norm_doi(d):
    d = str(d or "").strip().lower()
    for p in ("https://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/", "doi:"):
        d = d.replace(p, "")
    return d.strip() if d.startswith("10.") else ""


def _save(ctx, df, stem):
    return _io.write_table(df, ctx.out(stem), fmt=ctx.fmt,
                            also_xlsx=ctx.cfg.get("io", "also_write_xlsx", default=False),
                            xlsx_max_rows=ctx.cfg.get("io", "xlsx_max_rows", default=200000))


KEEP_TYPES = {"article", "journal article", "research article", "research-article", "review",
              "review article", "review-article", "systematic review", "scoping review",
              "meta analysis", "book", "book chapter", "chapter", "monograph", "edited book",
              "conference paper", "proceedings", "proceeding", "proceedings paper",
              "conference proceedings", "preprint", "posted content", "posted-content",
              "case reports", "case report", "case-report", "data paper", "clinical trial",
              "randomized controlled trial", "controlled clinical trial", "observational study",
              "comparative study", "evaluation study", "multicenter study", "validation study",
              "historical article", "practice guideline", "guideline", "consensus statement",
              "clinical trial protocol", "early access", "short survey"}
OVERRIDE_DROP = {"retracted", "retraction", "erratum", "published erratum", "corrigendum",
                  "correction", "expression of concern", "editorial", "comment", "news",
                  "news item", "letter", "book review", "biographical", "bibliography",
                  "interview", "discussion", "patent", "meeting abstract",
                  "introductory journal article", "study guide", "seminar", "congress"}
DROP_TYPES = {"grant", "awarded grant", "dataset", "data set", "data study", "patent",
              "dissertation", "thesis", "reprint", "software", "report", "note", "paratext",
              "meeting", "journal issue", "non conventional", "non-conventional", "conference review"}


def _norm_type(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", str(t or "").lower())).strip()


def _has(text, vocab):
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in vocab)


def _decide_type(t):
    raw = str(t or "").lower()
    parts = [_norm_type(p) for p in re.split(r"[;/|]", raw) if _norm_type(p)]
    if not parts:
        return "unknown"
    joined = " ; ".join(parts)
    if _has(joined, OVERRIDE_DROP):
        return "drop"
    if any(_has(p, KEEP_TYPES) for p in parts):
        return "keep"
    if _has(joined, DROP_TYPES):
        return "drop"
    return "drop"


def run_type_filter(ctx):
    inp = ctx.find(F["master"])
    df = _io.read_table(inp)
    print(f"loaded {len(df)} rows")
    if "document_type" not in df.columns:
        df["document_type"] = ""
    dt = df["document_type"].astype(str)
    for src_col in ["webofscience__Document_Type", "lens__Publication_Type",
                     "dimensions__Publication_Type", "scopus__Document_Type"]:
        if src_col in df.columns:
            dt = dt.where(dt.str.len() > 0, df[src_col].astype(str))
    df["_dtype"] = dt
    df["keep_decision"] = df["_dtype"].map(_decide_type)
    df.loc[df["keep_decision"].eq("unknown"), "keep_decision"] = "keep_unknown"
    keep_mask = df["keep_decision"].isin(["keep", "keep_unknown"])
    kept = df[keep_mask].drop(columns=["_dtype"]).copy()
    print(f"KEPT {len(kept)} / DROPPED {(~keep_mask).sum()}")
    _save(ctx, kept, F["typed"])
    _prov.log_stage(ctx.work, "type_filter", inputs=[inp], output=ctx.out(F["typed"]),
                     rows_in=len(df), rows_out=len(kept),
                     extra={"dropped": int((~keep_mask).sum())})
    return kept


def _parse_year(s):
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(s or ""))
    return int(m.group(0)) if m else None


def run_year_filter(ctx):
    inp = ctx.find(F["typed"])
    df = _io.read_table(inp)
    cutoff = int(ctx.cfg.get("year_filter", "min_year", default=2000))
    keep_missing = bool(ctx.cfg.get("year_filter", "keep_missing_year", default=True))
    yr = df["year"].map(_parse_year) if "year" in df.columns else pd.Series([None] * len(df))
    if "oa_year" in df.columns:
        yr = yr.where(yr.notna(), df["oa_year"].map(_parse_year))
    df["_yr"] = yr
    keep = (df["_yr"] >= cutoff) | (df["_yr"].isna()) if keep_missing else (df["_yr"] >= cutoff)
    kept = df[keep].drop(columns=["_yr"]).copy()
    print(f"year>={cutoff}: kept {len(kept)} / dropped {(~keep).sum()} (missing kept={keep_missing})")
    _save(ctx, kept, F["year"])
    _prov.log_stage(ctx.work, "year_filter", inputs=[inp], output=ctx.out(F["year"]),
                     rows_in=len(df), rows_out=len(kept),
                     params={"min_year": cutoff, "keep_missing": keep_missing})
    return kept


def _run_dedupe(ctx, in_stem, out_audit, out_canon, label):
    from . import dedupe as _dd
    inp = ctx.find(in_stem)
    df = _io.read_table(inp)
    out, merges = _dd.deduplicate(
        df, fuzzy_threshold=ctx.cfg.get("dedupe", "fuzzy_threshold", default=90),
        fuzzy_year_tol=ctx.cfg.get("dedupe", "fuzzy_year_tolerance", default=2),
        author_fp_mode=ctx.cfg.get("dedupe", "author_fp_mode", default="last_initial"))
    n_studies = out["study_id"].nunique()
    _save(ctx, out, out_audit)
    _save(ctx, out[out["is_canonical"]], out_canon)
    _prov.log_stage(ctx.work, label, inputs=[inp], output=ctx.out(out_audit),
                     rows_in=len(df), rows_out=n_studies, extra={"merges": merges})
    print(f"{label}: {len(df)} -> {n_studies} studies | merges={merges}")
    return out


def run_dedupe_pass1(ctx):
    return _run_dedupe(ctx, F["year"], F["dedupe1"], F["canon1"], "dedupe_pass1")


def run_dedupe_pass2(ctx):
    return _run_dedupe(ctx, F["enriched"], F["dedupe_audit"], F["canonical"], "dedupe")


def run_enrich(ctx):
    """Memory-safe chunked enrichment (free-tier Colab friendly). Delegates to
    enrich_chunked.run_chunked_enrich: reads only needed columns in row-chunks
    (using direct row-group access, NOT a from-zero rescan), writes self-
    contained chunk files (resumable), then streams them into the final file."""
    from . import enrich_chunked as _ec
    inp = ctx.find(F["canon1"], F["year"])
    out_path, n = _ec.run_chunked_enrich(ctx, inp, F["enriched"])
    try:
        _cov = pd.read_parquet(out_path, columns=["abstract", "oa_match_method"])
        abs_cov = round((_cov["abstract"].astype(str).str.len() > 0).mean() * 100, 1)
        matched = int((_cov["oa_match_method"].astype(str).str.len() > 0).sum())
    except Exception:
        abs_cov, matched = None, None
    _prov.log_stage(ctx.work, "enrich", inputs=[inp], output=out_path,
                     rows_out=n, extra={"oa_matched": matched, "abstract_cov": abs_cov})
    print(f"[enrich] done: {n} rows | matched {matched} | abstract cov {abs_cov}%")
    return out_path


def run_fix_funders(ctx):
    inp = ctx.find(F["enriched"])
    df = _io.read_table(inp)

    def nz(c):
        return df[c].astype(str).str.len() > 0 if c in df.columns else pd.Series(False, index=df.index)

    df["funders_unified"] = df["oa_funders"] if "oa_funders" in df.columns else ""
    if "cr_funders" in df.columns:
        m = df["funders_unified"].eq("") & df["cr_funders"].ne("")
        df.loc[m, "funders_unified"] = df.loc[m, "cr_funders"]
    matched = nz("oa_match_method") if "oa_match_method" in df.columns else nz("oa_id")
    funded = nz("funders_unified") | (df.get("oa_has_funding", "").eq("yes") if "oa_has_funding" in df else False) \
        | (df.get("cr_has_funding", "").eq("yes") if "cr_has_funding" in df else False)
    df["is_funded"] = np.where(funded, "yes", np.where(matched, "no", "unknown"))
    _save(ctx, df, F["enriched"])
    _prov.log_stage(ctx.work, "fix_funders", inputs=[inp], output=ctx.out(F["enriched"]),
                     rows_out=len(df), extra={"funded_pct": round((df['is_funded'].eq('yes')).mean() * 100, 1)})
    print(f"funders_unified cov {round(nz('funders_unified').mean()*100,1)}% | "
          f"funded {round((df['is_funded'].eq('yes')).mean()*100,1)}%")
    return df


def run_coverage(ctx):
    inp = ctx.find(F["dedupe_audit"])
    df = _io.read_table(inp)
    df["is_canonical"] = df["is_canonical"].astype(str).str.lower().isin(["true", "1", "yes"])
    canon = df[df["is_canonical"]]
    fields = ["oa_topic", "oa_field", "oa_countries", "oa_institutions", "funders_unified",
              "is_funded", "abstract", "year", "doi", "oa_id", "citation_count"]
    rows = [(f, round((canon[f].astype(str).str.len() > 0).mean() * 100, 1) if f in canon.columns else None)
            for f in fields]
    cov = pd.DataFrame(rows, columns=["field", "canonical_pct"])
    print(cov.to_string(index=False))
    _save(ctx, cov, "canonical_coverage")
    _prov.log_stage(ctx.work, "coverage", inputs=[inp], rows_out=len(canon), notes="coverage computed")
    return cov


def run_type_audit(ctx):
    inp = ctx.find(F["canonical"])
    canon = _io.read_table(inp)

    def slip(row):
        parts = []
        for c in ["document_type", "oa_type"]:
            if c in row and row[c]:
                parts += re.split(r"[;/|]", str(row[c]).lower())
        joined = _norm_type(" ; ".join(parts))
        title = _norm_type(row.get("title", ""))
        if _has(joined, OVERRIDE_DROP):
            return "override_drop"
        if title.startswith("book review"):
            return "title_book_review"
        return ""

    canon["_slip"] = canon.apply(slip, axis=1)
    clean = canon[canon["_slip"].eq("")].drop(columns=["_slip"])
    print(f"slipped {int(canon['_slip'].ne('').sum())} removed -> {len(clean)} clean")
    _save(ctx, clean, F["canon_clean"])
    _prov.log_stage(ctx.work, "type_audit", inputs=[inp], rows_in=len(canon), rows_out=len(clean))
    return clean


def run_reenrich(ctx):
    inp = ctx.find(F["canon_clean"], F["canonical"])
    df = _io.read_table(inp)
    _save(ctx, df, F["canon_enriched"])
    _prov.log_stage(ctx.work, "reenrich", inputs=[inp], rows_out=len(df),
                     notes="passthrough (enable country parse if coverage weak)")
    print(f"reenrich passthrough -> {len(df)} (enable country parse if needed)")
    return df


def _usable_abstract(s):
    s = str(s or "").strip()
    if len(s) < 100:
        return False
    if re.match(r'^(no abstract|not available|n/?a|none|\[.*\])\.?$', s, re.I):
        return False
    return sum(c.isalpha() for c in s) / max(len(s), 1) > 0.5


def run_readiness(ctx):
    inp = ctx.find(F["canon_enriched"], F["canon_clean"], F["canonical"])
    df = _io.read_table(inp)
    df["_ok_abs"] = df["abstract"].map(_usable_abstract) if "abstract" in df.columns else False
    strong = df["_ok_abs"].mean() * 100
    print(f"usable-abstract {strong:.1f}%")
    _prov.log_stage(ctx.work, "readiness", inputs=[inp], rows_out=len(df),
                     extra={"usable_abstract_pct": round(strong, 1)})
    return df


def run_backfill(ctx):
    from .apiutil import make_session, get_json, parallel_map
    import requests
    inp = ctx.find(F["canon_enriched"], F["canon_clean"], F["canonical"])
    df = _io.read_table(inp).reset_index(drop=True)
    for c in ["abstract", "doi", "abstract_source"]:
        if c not in df.columns:
            df[c] = ""
    df["_ok"] = df["abstract"].map(_usable_abstract)
    need = df[(~df["_ok"]) & (df["doi"].map(_norm_doi).ne(""))]
    idxs = need.index.tolist()
    print(f"backfill (parallel EuropePMC): {len(idxs)} records")
    sess = make_session(ctx.email)

    def fetch(idx):
        d = _norm_doi(df.at[idx, "doi"])
        q = f'DOI:"{d}"'
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search"
               f"?query={requests.utils.quote(q)}&resultType=core&format=json&pageSize=1")
        return idx, get_json(sess, url)

    def write(item, res):
        if not res:
            return
        idx, j = res
        if not j:
            return
        rl = (j or {}).get("resultList", {}).get("result", [])
        if rl:
            ab = rl[0].get("abstractText", "") or ""
            if _usable_abstract(ab):
                df.at[idx, "abstract"] = ab
                df.at[idx, "abstract_source"] = "europepmc_backfill"

    parallel_map(fetch, idxs, max_workers=6, on_result=write, progress_every=200)
    df["_ok"] = df["abstract"].map(_usable_abstract)
    df["screen_input"] = np.where(df["_ok"], "title+abstract", "title_only")
    df["eligibility_review_required"] = np.where(df["_ok"], "no", "yes")
    df = df.drop(columns=[c for c in ["_ok"] if c in df.columns])
    _save(ctx, df, F["screen_ready"])
    _prov.log_stage(ctx.work, "backfill", inputs=[inp], rows_out=len(df),
                     extra={"usable_abstract_pct": round(df['screen_input'].eq('title+abstract').mean() * 100, 1)})
    print(f"screen-ready -> {len(df)} | with abstract "
          f"{round(df['screen_input'].eq('title+abstract').mean()*100,1)}%")
    return df
