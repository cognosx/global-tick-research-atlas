"""Re-enrichment pass for STRAGGLERS — records that came out of the main
enrich stage unmatched (oa_match_method == "") purely because of API rate
limiting, not because the record genuinely isn't findable. Re-runs ONLY
those records, optionally under a different contact email (a fresh
OpenAlex "polite pool" identity), so you can:
  * retry immediately with a second email if you hit a daily/hourly limit
  * wait a day or two and retry with the SAME email once limits reset
  * do this as many times as needed before moving to screening/analysis

This is intentionally a separate, idempotent stage you can run repeatedly:
each run only touches records still unmatched, and merges newly-matched
ones back into the SAME output file in place.
"""
from __future__ import annotations
import os
import shutil
import pandas as pd
from . import io as _io
from . import provenance as _prov
from .enrich_chunked import (
    _norm_doi, _enrich_records, _coalesce, _mk_session, NEED_COLS,
)


def run_reenrich_stragglers(ctx, in_stem=None, email_override=None, max_records=None):
    """Find records with oa_match_method == "" (or missing) in the given
    enriched file, retry enrichment on just those, and merge results back
    in place. Safe to call repeatedly — it always re-evaluates "still
    unmatched" from the CURRENT state of the file.

    email_override: pass a different contact email for this run only (does
    not change ctx.email / your config), useful if you've hit OpenAlex's
    polite-pool rate limit under your usual address.
    """
    from .runner import F
    stem = in_stem or F["enriched"]
    inp = ctx.find(stem)
    df = _io.read_table(inp)
    print(f"Loaded {len(df)} rows from {os.path.basename(inp)}")

    if "oa_match_method" not in df.columns:
        df["oa_match_method"] = ""
    for c in NEED_COLS:
        if c not in df.columns:
            df[c] = ""
    if "doi_norm" not in df.columns:
        df["doi_norm"] = df["doi"].map(_norm_doi) if "doi" in df.columns else ""

    unmatched_mask = df["oa_match_method"].astype(str).str.strip().eq("")
    n_unmatched = int(unmatched_mask.sum())
    print(f"Still unmatched: {n_unmatched} / {len(df)} ({n_unmatched/len(df)*100:.1f}%)")
    if n_unmatched == 0:
        print("Nothing to re-enrich — every record already has an oa_match_method.")
        return df

    stragglers = df[unmatched_mask].copy()
    if max_records:
        stragglers = stragglers.head(max_records)
        print(f"Capped to first {max_records} stragglers for this run (set max_records=None for all).")

    # build the skinny frame the same way the main enrich stage does
    skin = stragglers[[c for c in NEED_COLS if c in stragglers.columns] + ["doi_norm"]].copy()
    if "record_id" not in skin.columns:
        raise SystemExit("record_id column missing — cannot safely merge results back.")

    def _pmid(row):
        import re
        for v in [row.get("source_record_id", ""), row.get("pubmed__PMID", ""), row.get("europepmc__EXTERNAL_ID", "")]:
            s = str(v)
            m = re.search(r'(?:PMID[:\s]*|MEDLINE:)(\d{5,9})', s)
            if m:
                return m.group(1)
            if re.fullmatch(r'\d{5,9}', s.strip()):
                return s.strip()
        return ""

    if "pmid" not in skin.columns:
        skin["pmid"] = skin.apply(_pmid, axis=1)

    email = email_override or ctx.email
    print(f"Retrying {len(skin)} stragglers using contact email: {email}")
    sess = _mk_session(email)
    res = _enrich_records(skin, sess, email, ctx.cfg)

    res_df = pd.DataFrame(list(res.values()))
    newly_matched = int((res_df["oa_match_method"].astype(str).str.strip() != "").sum())
    print(f"Newly matched this pass: {newly_matched} / {len(res_df)}")

    # merge results back into the FULL df, in place, by record_id
    df = df.set_index("record_id")
    res_df = res_df.set_index("record_id")
    df.update(res_df)
    df = df.reset_index()

    # re-coalesce core fields (title/abstract/etc.) only for rows that JUST matched,
    # so already-finalized rows are untouched
    just_matched_ids = set(res_df.index[res_df["oa_match_method"].astype(str).str.strip() != ""])
    if just_matched_ids:
        sub = df[df["record_id"].isin(just_matched_ids)].copy()
        sub = _coalesce(sub)
        df = df.set_index("record_id")
        sub = sub.set_index("record_id")
        df.update(sub)
        df = df.reset_index()

    # backup the previous file once per run before overwriting, so a bad retry
    # can be rolled back manually if needed
    backup_path = inp + ".before_reenrich.bak"
    if not os.path.exists(backup_path):
        shutil.copy(inp, backup_path)
        print(f"Backed up previous file -> {os.path.basename(backup_path)}")

    out_path = _io.write_table(df, ctx.out(stem), fmt=ctx.fmt)
    still_unmatched = int(df["oa_match_method"].astype(str).str.strip().eq("").sum())
    print(f"\nSaved {out_path} ({len(df)} rows). Still unmatched: {still_unmatched} "
          f"({still_unmatched/len(df)*100:.1f}%).")
    if still_unmatched > 0:
        print("Re-run this stage again (optionally with a different email_override, or after "
              "waiting for rate limits to reset) to retry the remainder.")

    _prov.log_stage(ctx.work, "reenrich_stragglers", inputs=[inp], output=out_path,
                     rows_in=len(df), rows_out=len(df),
                     extra={"attempted": len(skin), "newly_matched": newly_matched,
                            "still_unmatched": still_unmatched, "email_used": email})
    return df
