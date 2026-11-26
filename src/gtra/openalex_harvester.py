"""
openalex_harvester.py — OpenAlex /works harvester for GTRA (and any science-of-science corpus)
=============================================================================================

Updated 2026-11-25 for the current OpenAlex schema ("Walden", Nov 2025, and the 2026 funding update):
  - `grants` was REMOVED from works -> funding now comes from `funders` + `awards`.
  - `is_paratext` is deprecated -> exclude paratext with the `type` filter instead.
  - New or newly useful fields captured: `awards`, `funders`, `study_designs`, `has_content`
    (pdf / grobid_xml), `has_fulltext`, `open_access.any_repository_has_fulltext`, `apc_list`, `apc_paid`,
    `countries_distinct_count`, `institutions_distinct_count`, `corresponding_author_ids`,
    `corresponding_institution_ids`, `cited_by_percentile_year`, `counts_by_year`, `related_works`,
    `locations_count`, `is_xpac`, `created_date`, `updated_date`, and `authorships[].affiliations`.
  - Scoped search parameters: `search.title`, `search.title_and_abstract`, `search.title_abstract_keywords`
    (the plain `search=` also matches full text).

Outputs, per harvest folder (all keyed on `oa_id`):
  raw/part_*.jsonl.gz      lossless API records -> new fields later need no re-harvest
  works.parquet            one row per work (flat `oa_*` columns)
  authorships.parquet      one row per work x author, in byline order (keeps per-author countries)
  awards.parquet           one row per work x award (funder, funder award id)
  _cursor.txt              resume point; "DONE" when finished

Requires: requests, pandas, pyarrow.   API key: required by OpenAlex (env OPENALEX_API_KEY or api_key=...).
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import quote_plus

import pandas as pd
import requests

API = "https://api.openalex.org/works"
SEARCH_FIELDS = {None: "search", "all": "search", "title": "search.title",
                 "title_and_abstract": "search.title_and_abstract",
                 "title_abstract_keywords": "search.title_abstract_keywords"}

# ----------------------------------------------------------------------------- small helpers

def short_id(url: str | None) -> str:
    """https://openalex.org/W123 -> W123; also strips doi.org / orcid.org / ror.org prefixes."""
    if not url:
        return ""
    return re.sub(r"^https?://(openalex\.org|doi\.org|orcid\.org|ror\.org)/", "", str(url))


def clean_pmid(url: str | None) -> str:
    m = re.search(r"(\d+)/?$", url or "")
    return m.group(1) if m else ""


def clean_pmcid(url: str | None) -> str:
    m = re.search(r"(PMC\d+)", url or "")
    return m.group(1) if m else ""


def reconstruct_abstract(inv: dict | None) -> str | None:
    """abstract_inverted_index {word: [positions]} -> text."""
    if not inv:
        return None
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos)) if pos else None


def dedupe(seq: Iterable) -> list:
    seen, out = set(), []
    for x in seq:
        if x not in (None, "") and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def boolean_or(terms: Iterable[str]) -> str:
    """['tick', 'hard tick'] -> '(tick OR "hard tick")'. Multi-word / hyphenated terms are quoted."""
    parts = []
    for t in dedupe((t or "").strip() for t in terms):
        parts.append(f'"{t}"' if (" " in t or "-" in t) else t)
    return "(" + " OR ".join(parts) + ")"


def chunk_terms(terms: list[str], budget: int = 3500) -> list[list[str]]:
    """Split terms so each OR-query stays under `budget` URL-encoded bytes (OpenAlex caps URLs ~4 KB)."""
    chunks, cur = [], []
    for t in terms:
        if cur and len(quote_plus(boolean_or(cur + [t]))) > budget:
            chunks.append(cur)
            cur = [t]
        else:
            cur.append(t)
    return chunks + ([cur] if cur else [])


def _usd(obj) -> float | None:
    return (obj or {}).get("value_usd") if isinstance(obj, dict) else None


# ----------------------------------------------------------------------------- harvester

class OpenAlexHarvester:
    def __init__(self, api_key: str | None = None, per_page: int = 200, include_xpac: bool = False,
                 sep: str = " | ", max_retries: int = 8, timeout: int = 60, pause: float = 0.12):
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        if not self.api_key:
            print("WARNING: no OpenAlex API key. Get a free key at https://openalex.org/settings/api")
        self.per_page = min(per_page, 200)
        self.include_xpac = include_xpac   # True adds the ~190M DataCite/repository "expansion" works
        self.sep = sep
        self.max_retries = max_retries
        self.timeout = timeout
        self.pause = pause                 # stays under ~10 requests/second
        self.session = requests.Session()

    # ---- request building ------------------------------------------------------------------
    @staticmethod
    def build_filter(filters: dict | str | None) -> str | None:
        """{'publication_year': '2000-2025', 'type': ['article', 'review']} ->
        'publication_year:2000-2025,type:article|review' (comma = AND, | = OR). Strings pass through."""
        if filters is None or isinstance(filters, str):
            return filters
        parts = []
        for k, v in filters.items():
            if isinstance(v, (list, tuple, set)):
                v = "|".join(str(x) for x in v)
            parts.append(f"{k}:{v}")
        return ",".join(parts)

    def _params(self, filters=None, search=None, search_field="title_abstract_keywords", **extra) -> dict:
        p = {k: v for k, v in extra.items() if v is not None}
        if "per_page" in p:                      # OpenAlex spells it per-page
            p["per-page"] = p.pop("per_page")
        f = self.build_filter(filters)
        if f:
            p["filter"] = f
        if search:
            p[SEARCH_FIELDS[search_field]] = search
        return p

    def _get(self, params: dict) -> dict:
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.include_xpac:
            params["include_xpac"] = "true"
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(API, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                wait = min(2 ** attempt, 60)
                print(f"  network error ({e.__class__.__name__}); retry in {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After") or min(2 ** attempt, 60))
                print(f"  HTTP {r.status_code}; retry in {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code in (401, 403, 409):
                raise RuntimeError(f"HTTP {r.status_code}: missing/invalid API key or credits used up. "
                                   f"{r.text[:300]}")
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError("max retries exceeded")

    # ---- size check ------------------------------------------------------------------------
    def count(self, filters=None, search=None, search_field="title_abstract_keywords") -> int:
        """One request returning meta.count. Run before any full harvest."""
        data = self._get(self._params(filters, search, search_field, per_page=1))
        return int(data.get("meta", {}).get("count") or 0)

    # ---- raw iteration ---------------------------------------------------------------------
    def iter_pages(self, filters=None, search=None, search_field="title_abstract_keywords",
                   cursor: str = "*") -> Iterator[tuple[list[dict], str | None]]:
        """Yields (results, next_cursor) with cursor paging (no 10,000-record limit)."""
        base = self._params(filters, search, search_field, per_page=self.per_page)
        while cursor:
            data = self._get(dict(base, cursor=cursor))
            results = data.get("results") or []
            cursor = data.get("meta", {}).get("next_cursor")
            if not results:
                return
            yield results, cursor
            time.sleep(self.pause)

    def sample(self, n: int, filters=None, search=None, search_field="title_abstract_keywords",
               seed: int = 2026) -> list[dict]:
        """Reproducible random sample (OpenAlex `sample` + `seed`, max 10,000)."""
        base = self._params(filters, search, search_field, sample=min(n, 10_000), seed=seed,
                            per_page=self.per_page)
        rows, page = [], 1
        while len(rows) < n:
            res = self._get(dict(base, page=page)).get("results") or []
            if not res:
                break
            rows.extend(res)
            page += 1
            time.sleep(self.pause)
        return list({w["id"]: w for w in rows}.values())[:n]

    # ---- flattening ------------------------------------------------------------------------
    def flatten_work(self, w: dict) -> dict:
        j = lambda seq: self.sep.join(dedupe(seq))
        ids = w.get("ids") or {}
        oa = w.get("open_access") or {}
        ploc = w.get("primary_location") or {}
        src = ploc.get("source") or {}
        bib = w.get("biblio") or {}
        pt = w.get("primary_topic") or {}
        cnp = w.get("citation_normalized_percentile") or {}
        cpy = w.get("cited_by_percentile_year") or {}
        hc = w.get("has_content") or {}
        auths = w.get("authorships") or []
        awards = w.get("awards") or []
        funders = w.get("funders") or []
        # per-author lists (same logic and names as the original harvester)
        authors, orcids, corr = [], [], []
        raw_affils, inst_names, inst_ids, inst_rors, inst_types, countries = [], [], [], [], [], []
        for a in auths:
            au = a.get("author") or {}
            name = au.get("display_name") or ""
            authors.append(name)
            orcids.append(short_id(au.get("orcid")))              # index-aligned with oa_authors
            if a.get("is_corresponding"):
                corr.append(name)
            raw_affils.extend(a.get("raw_affiliation_strings") or [])
            for inst in (a.get("institutions") or []):
                inst_names.append(inst.get("display_name") or "")
                inst_ids.append(short_id(inst.get("id")))
                inst_rors.append(short_id(inst.get("ror")))
                inst_types.append(inst.get("type") or "")
                if inst.get("country_code"):
                    countries.append(inst["country_code"])
            countries.extend(a.get("countries") or [])
        return {
            # identifiers
            "oa_id": short_id(w.get("id")), "oa_doi": short_id(w.get("doi") or ids.get("doi")),
            "oa_pmid": clean_pmid(ids.get("pmid")), "oa_pmcid": clean_pmcid(ids.get("pmcid")),
            "oa_mag": str(ids.get("mag") or ""),
            # text
            "oa_title": w.get("title") or w.get("display_name"),
            "oa_abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
            "oa_language": w.get("language"), "oa_type": w.get("type"),
            # dates
            "oa_year": w.get("publication_year"), "oa_publication_date": w.get("publication_date"),
            "oa_created_date": w.get("created_date"), "oa_updated_date": w.get("updated_date"),
            # venue
            "oa_source_id": short_id(src.get("id")), "oa_journal": src.get("display_name"),
            "oa_issn": src.get("issn_l") or (src.get("issn") or [None])[0],      # original column
            "oa_issn_l": src.get("issn_l"), "oa_issns": j(src.get("issn") or []),
            "oa_source_type": src.get("type"), "oa_publisher": src.get("host_organization_name"),
            "oa_source_in_doaj": src.get("is_in_doaj"), "oa_volume": bib.get("volume"),
            "oa_issue": bib.get("issue"), "oa_first_page": bib.get("first_page"),
            "oa_last_page": bib.get("last_page"), "oa_locations_count": w.get("locations_count"),
            "oa_indexed_in": j(w.get("indexed_in") or []),
            # people (order and duplicates kept so lists stay aligned; detail in authorships.parquet)
            "oa_authors": self.sep.join(authors),
            "oa_author_orcids": self.sep.join(orcids),
            "oa_corresponding_authors": j(corr),
            "oa_authors_count": len(authors),
            # affiliations and geography
            "oa_affiliations_raw": j(raw_affils),
            "oa_institutions": j(inst_names),
            "oa_institution_ids": j(inst_ids),
            "oa_institution_rors": j(inst_rors),
            "oa_institution_types": j(inst_types),
            "oa_countries": j(countries),
            "oa_first_author_countries": j((auths[0].get("countries") or []) if auths else []),
            "oa_last_author_countries": j((auths[-1].get("countries") or []) if auths else []),
            "oa_countries_distinct_count": w.get("countries_distinct_count"),
            "oa_institutions_distinct_count": w.get("institutions_distinct_count"),
            "oa_corresponding_author_ids": j(short_id(x) for x in (w.get("corresponding_author_ids") or [])),
            "oa_corresponding_institution_ids": j(short_id(x) for x in
                                                  (w.get("corresponding_institution_ids") or [])),
            # funding (new schema: funders + awards; `grants` no longer exists)
            # original names kept: oa_funders / oa_funder_ids / oa_award_ids (= the funder's award number)
            "oa_funders": j([f.get("display_name") for f in funders] +
                            [a.get("funder_display_name") for a in awards]),
            "oa_funder_ids": j([short_id(f.get("id")) for f in funders] +
                               [short_id(a.get("funder_id")) for a in awards]),
            "oa_award_ids": j(a.get("funder_award_id") for a in awards),
            "oa_openalex_award_ids": j(short_id(a.get("id")) for a in awards),   # new: G… ids
            "oa_n_awards": len(awards), "oa_has_funding": bool(funders or awards),
            # open access and content
            "oa_is_oa": oa.get("is_oa"), "oa_oa_status": oa.get("oa_status"), "oa_oa_url": oa.get("oa_url"),
            "oa_any_repository_has_fulltext": oa.get("any_repository_has_fulltext"),
            "oa_license": (w.get("best_oa_location") or {}).get("license") or ploc.get("license"),
            "oa_apc_list_usd": _usd(w.get("apc_list")), "oa_apc_paid_usd": _usd(w.get("apc_paid")),
            "oa_has_pdf": hc.get("pdf"), "oa_has_grobid_xml": hc.get("grobid_xml"),
            "oa_has_fulltext": w.get("has_fulltext"),
            # impact
            "oa_cited_by_count": w.get("cited_by_count"), "oa_fwci": w.get("fwci"),
            "oa_citation_pct": cnp.get("value") if isinstance(cnp, dict) else cnp,
            "oa_top1pct": cnp.get("is_in_top_1_percent") if isinstance(cnp, dict) else None,
            "oa_top10pct": cnp.get("is_in_top_10_percent") if isinstance(cnp, dict) else None,
            "oa_cited_pct_year_min": cpy.get("min"), "oa_cited_pct_year_max": cpy.get("max"),
            "oa_counts_by_year": json.dumps(w.get("counts_by_year") or []),
            "oa_referenced_works": j(short_id(x) for x in (w.get("referenced_works") or [])),
            "oa_referenced_works_count": w.get("referenced_works_count"),
            "oa_related_works": j(short_id(x) for x in (w.get("related_works") or [])),
            # classification
            "oa_topic_id": short_id(pt.get("id")), "oa_topic": pt.get("display_name"),
            "oa_subfield": (pt.get("subfield") or {}).get("display_name"),
            "oa_field": (pt.get("field") or {}).get("display_name"),
            "oa_domain": (pt.get("domain") or {}).get("display_name"),
            "oa_topics_all": j(f"{t.get('display_name')} ({round(t.get('score') or 0, 3)})"
                               for t in (w.get("topics") or [])),
            "oa_keywords": j(k.get("display_name") for k in (w.get("keywords") or [])),
            "oa_mesh": j(m.get("descriptor_name") for m in (w.get("mesh") or [])),
            "oa_mesh_major": j(m.get("descriptor_name") for m in (w.get("mesh") or [])
                               if m.get("is_major_topic")),
            "oa_study_designs": j(s.get("display_name") for s in (w.get("study_designs") or [])),
            "oa_sdgs": j(s.get("display_name") for s in (w.get("sustainable_development_goals") or [])),
            "oa_concepts": j(c.get("display_name") for c in (w.get("concepts") or [])),   # deprecated
            # integrity and corpus
            "oa_is_retracted": w.get("is_retracted"), "oa_is_xpac": w.get("is_xpac"),
        }

    @staticmethod
    def flatten_authorships(w: dict) -> list[dict]:
        rows = []
        for i, a in enumerate(w.get("authorships") or []):
            au = a.get("author") or {}
            insts = a.get("institutions") or []
            rows.append({
                "oa_id": short_id(w.get("id")), "author_order": i + 1,
                "author_position": a.get("author_position"),
                "author_id": short_id(au.get("id")), "author_name": au.get("display_name"),
                "raw_author_name": a.get("raw_author_name"), "orcid": short_id(au.get("orcid")),
                "is_corresponding": a.get("is_corresponding"),
                "countries": "|".join(a.get("countries") or []),
                "institution_ids": "|".join(short_id(x.get("id")) for x in insts),
                "institution_names": "|".join(x.get("display_name") or "" for x in insts),
                "institution_rors": "|".join(short_id(x.get("ror")) for x in insts),
                "institution_types": "|".join(x.get("type") or "" for x in insts),
                "institution_countries": "|".join(x.get("country_code") or "" for x in insts),
                "raw_affiliations": " || ".join(a.get("raw_affiliation_strings") or []),
            })
        return rows

    @staticmethod
    def flatten_awards(w: dict) -> list[dict]:
        return [{"oa_id": short_id(w.get("id")), "award_id": short_id(a.get("id")),
                 "award_name": a.get("display_name"), "funder_award_id": a.get("funder_award_id"),
                 "funder_id": short_id(a.get("funder_id")), "funder_name": a.get("funder_display_name")}
                for a in (w.get("awards") or [])]

    # ---- full harvest ----------------------------------------------------------------------
    def harvest(self, filters=None, search=None, search_field="title_abstract_keywords",
                outdir: str = "./openalex_out", max_records: int | None = None, resume: bool = True,
                flush_every_pages: int = 10) -> pd.DataFrame:
        """Cursor-paged harvest. Writes raw JSONL parts (lossless) and builds the three tables."""
        out = Path(outdir)
        (out / "raw").mkdir(parents=True, exist_ok=True)
        state = out / "_cursor.txt"
        cursor = "*"
        if resume and state.exists():
            saved = state.read_text().strip()
            if saved == "DONE":
                print("Already DONE: rebuilding tables from raw parts.")
                return self.build_tables(out)
            if saved:
                cursor = saved
                print("Resuming from saved cursor.")
        if max_records:
            print(f"Estimate: up to {math.ceil(max_records / self.per_page)} list calls.")

        part_no = len(glob.glob(str(out / "raw" / "part_*.jsonl.gz")))
        buffer, pages, total = [], 0, 0
        for results, nxt in self.iter_pages(filters, search, search_field, cursor):
            buffer.extend(results)
            total += len(results)
            pages += 1
            if pages % flush_every_pages == 0:
                self._flush(buffer, out, part_no)
                part_no += 1
                buffer = []
                state.write_text(nxt or "DONE")
                print(f"  ...{total:,} records this session")
            if max_records and total >= max_records:
                break
        if buffer:
            self._flush(buffer, out, part_no)
        state.write_text("DONE")
        print(f"Done: {total:,} records this session -> {outdir}")
        return self.build_tables(out)

    @staticmethod
    def _flush(buffer: list[dict], out: Path, part_no: int) -> None:
        with gzip.open(out / "raw" / f"part_{part_no:05d}.jsonl.gz", "wt", encoding="utf-8") as f:
            for w in buffer:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")

    @staticmethod
    def iter_raw(out: str | Path) -> Iterator[dict]:
        for p in sorted(glob.glob(str(Path(out) / "raw" / "part_*.jsonl.gz"))):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)

    def build_tables(self, out: str | Path) -> pd.DataFrame:
        """Raw parts -> works / authorships / awards parquet (deduplicated on oa_id)."""
        out = Path(out)
        works, auths, awards, seen = [], [], [], set()
        for w in self.iter_raw(out):
            wid = short_id(w.get("id"))
            if wid in seen:
                continue
            seen.add(wid)
            works.append(self.flatten_work(w))
            auths.extend(self.flatten_authorships(w))
            awards.extend(self.flatten_awards(w))
        df = pd.DataFrame(works)
        df.to_parquet(out / "works.parquet", index=False)
        pd.DataFrame(auths).to_parquet(out / "authorships.parquet", index=False)
        pd.DataFrame(awards, columns=["oa_id", "award_id", "award_name", "funder_award_id",
                                      "funder_id", "funder_name"]).to_parquet(out / "awards.parquet",
                                                                              index=False)
        print(f"Tables: works {len(df):,} | authorships {len(auths):,} | awards {len(awards):,}")
        return df
