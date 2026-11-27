"""
Recall audit: why are records from the old human-verified set missing from the new harvest?

The old corpus was built with OpenAlex full-text search; the new one searches title, abstract
and keywords only. For every old-gold OpenAlex ID that is not in the new harvest, this module
asks OpenAlex directly and assigns ONE reason (first match wins):

  not_found        the ID no longer resolves (merged into another work or deleted)
  year_out         publication_year outside the harvest window
  type_out         OpenAlex type outside the harvest types (e.g. book-chapter, dissertation)
  query_match_now  the work DOES match Block A in title/abstract/keywords today -> a real
                   harvest gap (index changed since the harvest, or harvest incomplete)
  fulltext_only    no Block A term in title/abstract/keywords: only full text mentioned ticks

The reason uses OpenAlex itself (same search field and query as the harvest), not a regex,
so "fulltext_only" means exactly "the new query cannot see it".
"""
from __future__ import annotations

import time

import pandas as pd

from .openalex_harvester import OpenAlexHarvester, short_id

REASONS = ["not_found", "year_out", "type_out", "query_match_now", "fulltext_only"]


def _batches(ids: list[str], n: int):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def fetch_by_ids(h: OpenAlexHarvester, ids, search: str | None = None,
                 search_field: str = "title_abstract_keywords", batch: int = 50,
                 verbose: bool = True) -> list[dict]:
    """Raw works for these IDs (optionally only those matching `search`)."""
    ids = [short_id(i) for i in dict.fromkeys(ids) if short_id(i)]
    out: list[dict] = []
    for k, chunk in enumerate(_batches(ids, batch)):
        params = h._params({"openalex_id": chunk}, search, search_field, per_page=200)
        out += h._get(params).get("results") or []
        time.sleep(h.pause)
        if verbose and k % 40 == 0:
            print(f"  {min((k + 1) * batch, len(ids)):,}/{len(ids):,} ids -> {len(out):,} works"
                  + (" (query match)" if search else ""))
    return out


def audit(h: OpenAlexHarvester, ids, block_a: str, year_range: tuple[int, int], types: list[str],
          search_field: str = "title_abstract_keywords") -> pd.DataFrame:
    ids = [short_id(i) for i in dict.fromkeys(ids)]
    print(f"[audit] fetching {len(ids):,} works")
    raw = fetch_by_ids(h, ids)
    print("[audit] asking OpenAlex which of them match the harvest query today")
    matched = {short_id(w.get("id")) for w in fetch_by_ids(h, ids, block_a, search_field)}

    rows = [h.flatten_work(w) for w in raw]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["oa_id"])
    df["oa_id"] = df["oa_id"].map(short_id)
    found = set(df["oa_id"])
    y0, y1 = year_range
    yr = pd.to_numeric(df.get("oa_year"), errors="coerce")
    typ = df.get("oa_type", pd.Series("", index=df.index)).astype(str).str.lower()
    reason = []
    for i in range(len(df)):
        if not (y0 <= (yr.iloc[i] if pd.notna(yr.iloc[i]) else -1) <= y1):
            reason.append("year_out")
        elif typ.iloc[i] not in types:
            reason.append("type_out")
        elif df["oa_id"].iloc[i] in matched:
            reason.append("query_match_now")
        else:
            reason.append("fulltext_only")
    df["audit_reason"] = reason
    # IDs that did not come back: merged or deleted (OpenAlex merges old IDs into a new one)
    lost = [i for i in ids if i not in found]
    if lost:
        df = pd.concat([df, pd.DataFrame({"oa_id": lost, "audit_reason": "not_found"})], ignore_index=True)
    df["audit_reason"] = pd.Categorical(df["audit_reason"], categories=REASONS)
    print(df["audit_reason"].value_counts().reindex(REASONS).to_string())
    return df


def harvest_recall(present_layers: pd.Series, missing: pd.DataFrame,
                   layers=("core", "review")) -> pd.DataFrame:
    """Recall of the new harvest for tick-focused papers in the old gold set, overall and
    after removing misses that are out of scope by design (years, types).
    present_layers: gtra_layer of old-gold records that ARE in the new corpus.
    missing: audit() output with gtra_layer added for the screened misses."""
    hit = present_layers.isin(layers).sum()
    m = missing[missing["gtra_layer"].isin(layers)] if "gtra_layer" in missing.columns else missing.iloc[0:0]
    by = m["audit_reason"].value_counts().reindex(REASONS, fill_value=0)
    in_scope_miss = by["query_match_now"] + by["fulltext_only"]
    rows = [
        {"measure": "tick-focused old-gold papers found by the new harvest", "n": int(hit)},
        {"measure": "tick-focused misses, all reasons", "n": int(by.sum())},
        *[{"measure": f"  miss: {r}", "n": int(by[r])} for r in REASONS],
        {"measure": "recall, all misses", "n": round(hit / (hit + by.sum()), 4) if hit + by.sum() else None},
        {"measure": "recall within harvest scope (years and types)",
         "n": round(hit / (hit + in_scope_miss), 4) if hit + in_scope_miss else None},
    ]
    return pd.DataFrame(rows)
