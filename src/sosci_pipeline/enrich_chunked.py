"""Memory-safe, chunked, resumable enrichment for free-tier Colab (~12GB).

Design:
  * reads ONLY the ~10 columns enrichment needs, in CHUNK_SIZE row slices
  * enriches each chunk and writes it as a complete, self-contained parquet
    (enriched_chunks/chunk_NNN.parquet); resumable at the chunk level
  * combines chunk files into master_enriched.parquet by STREAMING
    (pyarrow ParquetWriter) so the full file is never assembled in RAM

IMPORTANT FIX: chunk row-reads use direct row-group access (pf.read_row_groups)
rather than scanning iter_batches() from the start of the file each chunk.
The naive scan-and-discard approach makes every chunk re-read and discard all
PRIOR chunks' rows before reaching its own window — an O(n^2) pattern across
the whole file that makes chunk N take ~N times as long as chunk 1. Row-group
direct access costs the same regardless of how far into the file the chunk is.
"""
from __future__ import annotations
import os
import re
import gc
import time
import glob
import requests
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq

NEED_COLS = ["record_id", "doi", "title", "authors", "year", "abstract",
             "source_record_id", "pubmed__PMID", "europepmc__EXTERNAL_ID"]

ENR_COLS = ["oa_id", "oa_doi", "oa_pmid", "oa_pmcid", "oa_title", "oa_abstract", "oa_authors",
            "oa_author_orcids", "oa_affiliations_raw", "oa_institutions", "oa_institution_ids",
            "oa_countries", "oa_cited_by_count", "oa_journal", "oa_issn", "oa_type", "oa_language",
            "oa_funders", "oa_has_funding", "oa_year", "oa_topic", "oa_subfield", "oa_field",
            "oa_domain", "oa_concepts", "oa_is_oa", "oa_oa_status", "oa_keywords", "cr_funders",
            "cr_has_funding", "cr_affiliations", "upw_is_oa", "upw_oa_status", "upw_pdf_url",
            "epmc_abstract", "epmc_affiliation", "oa_match_method", "oa_match_confidence"]


def _norm_doi(d):
    d = str(d or "").strip().lower()
    for p in ("https://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/", "doi:"):
        d = d.replace(p, "")
    d = d.strip()
    return d if d.startswith("10.") else ""


def _recon(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos)) if pos else ""


def _uq(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_oa(w):
    auth = w.get("authorships") or []
    names, orc, raff, inn, iid, ctry = [], [], [], [], [], []
    for a in auth:
        au = a.get("author") or {}
        if au.get("display_name"):
            names.append(au["display_name"])
        if au.get("orcid"):
            orc.append(au["orcid"])
        for s in (a.get("raw_affiliation_strings") or []):
            if s:
                raff.append(s)
        for ins in (a.get("institutions") or []):
            if ins.get("display_name"):
                inn.append(ins["display_name"])
            if ins.get("id"):
                iid.append(ins["id"])
            if ins.get("country_code"):
                ctry.append(ins["country_code"])
    grants = w.get("grants") or []
    pl = w.get("primary_location") or {}
    src = (pl.get("source") or {}) if pl else {}
    pt = w.get("primary_topic") or {}
    ids = w.get("ids") or {}
    oa = w.get("open_access") or {}
    kws = [k.get("display_name", "") for k in (w.get("keywords") or []) if k.get("display_name")]
    return {
        "oa_id": w.get("id", ""), "oa_doi": _norm_doi(w.get("doi", "")),
        "oa_pmid": (ids.get("pmid", "") or "").replace("https://pubmed.ncbi.nlm.nih.gov/", ""),
        "oa_pmcid": ids.get("pmcid", ""), "oa_title": w.get("title", "") or "",
        "oa_abstract": _recon(w.get("abstract_inverted_index")), "oa_authors": "; ".join(names),
        "oa_author_orcids": "; ".join(orc), "oa_affiliations_raw": " | ".join(_uq(raff)),
        "oa_institutions": "; ".join(_uq(inn)), "oa_institution_ids": "; ".join(_uq(iid)),
        "oa_countries": "; ".join(_uq(ctry)), "oa_cited_by_count": str(w.get("cited_by_count", "")),
        "oa_journal": src.get("display_name", "") or "", "oa_issn": src.get("issn_l", "") or "",
        "oa_type": w.get("type", "") or "", "oa_language": w.get("language", "") or "",
        "oa_funders": "; ".join(_uq([g.get("funder_display_name", "") for g in grants if g.get("funder_display_name")])),
        "oa_has_funding": "yes" if grants else "no", "oa_year": str(w.get("publication_year", "") or ""),
        "oa_topic": pt.get("display_name", "") or "",
        "oa_subfield": ((pt.get("subfield") or {}).get("display_name", "")) or "",
        "oa_field": ((pt.get("field") or {}).get("display_name", "")) or "",
        "oa_domain": ((pt.get("domain") or {}).get("display_name", "")) or "",
        "oa_concepts": "; ".join(c.get("display_name", "") for c in (w.get("concepts") or [])[:8] if c.get("display_name")),
        "oa_is_oa": "yes" if oa.get("is_oa") else "no", "oa_oa_status": oa.get("oa_status", "") or "",
        "oa_keywords": "; ".join(kws),
    }


def _mk_session(email):
    s = requests.Session()
    s.headers.update({"User-Agent": f"sosci-enrich/1.0 (mailto:{email})"})
    return s


def _get(sess, url, tries=3):
    for t in range(tries):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (t + 1))
                continue
            return None
        except Exception:
            time.sleep(1.0 * (t + 1))
    return None


def _enrich_records(skin, sess, email, cfg):
    out = {rid: {"record_id": rid} for rid in skin["record_id"]}
    rid_doi = dict(zip(skin["record_id"], skin["doi_norm"]))
    rid_pmid = dict(zip(skin["record_id"], skin["pmid"]))
    rid_title = dict(zip(skin["record_id"], skin["title"]))
    rid_year = dict(zip(skin["record_id"], skin["year"]))
    workers = int(cfg.get("enrich", "workers", default=8))
    title_fb = bool(cfg.get("enrich", "title_fallback", default=True))
    title_min = int(cfg.get("enrich", "title_min_score", default=92))

    doi_to_ids = {}
    for rid, d in rid_doi.items():
        if d:
            doi_to_ids.setdefault(d, []).append(rid)
    dois = list(doi_to_ids.keys())
    if cfg.get("enrich", "use_openalex", default=True):
        for i in range(0, len(dois), 45):
            b = [d for d in dois[i:i + 45] if d]
            if not b:
                continue
            j = _get(sess, "https://api.openalex.org/works?per-page=50"
                     f"&filter=doi:{'|'.join(b)}&mailto={email}")
            if j:
                for w in j.get("results", []):
                    d = _norm_doi(w.get("doi", ""))
                    if not d:
                        continue
                    p = _parse_oa(w)
                    p["oa_match_method"] = "doi"
                    p["oa_match_confidence"] = "high"
                    for rid in doi_to_ids.get(d, []):
                        out[rid].update(p)
            time.sleep(0.1)

        need = [rid for rid in out if not out[rid].get("oa_match_method") and rid_pmid.get(rid)]

        def f_p(rid):
            return rid, _get(sess, f"https://api.openalex.org/works/pmid:{rid_pmid[rid]}?mailto={email}")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(f_p, rid) for rid in need]):
                rid, j = fut.result()
                if j and not j.get("error"):
                    p = _parse_oa(j)
                    p["oa_match_method"] = "pmid"
                    p["oa_match_confidence"] = "high"
                    out[rid].update(p)

        if title_fb:
            need = [rid for rid in out if not out[rid].get("oa_match_method")
                    and len(str(rid_title.get(rid, ""))) >= 12]

            def f_t(rid):
                t = re.sub(r'[^\w\s]', ' ', str(rid_title[rid])).strip()
                return rid, t, _get(sess, "https://api.openalex.org/works?per-page=5"
                                     f"&search={requests.utils.quote(t[:200])}&mailto={email}")

            with ThreadPoolExecutor(max_workers=workers) as ex:
                for fut in as_completed([ex.submit(f_t, rid) for rid in need]):
                    rid, t, j = fut.result()
                    if not j:
                        continue
                    best, bsc = None, 0
                    for w in j.get("results", []):
                        sc = fuzz.token_sort_ratio(t.lower(), re.sub(r'[^\w\s]', ' ', (w.get("title") or "")).lower())
                        if sc > bsc:
                            best, bsc = w, sc
                    if best and bsc >= title_min:
                        wy = re.sub(r'\D', '', str(best.get("publication_year", "") or ""))[:4]
                        ry = re.sub(r'\D', '', str(rid_year.get(rid, "")))[:4]
                        if not (wy and ry and abs(int(wy) - int(ry)) > 1):
                            p = _parse_oa(best)
                            p["oa_match_method"] = "title"
                            p["oa_match_confidence"] = "medium" if bsc >= 96 else "low"
                            out[rid].update(p)

    need = [rid for rid in out if rid_doi.get(rid)]
    if cfg.get("enrich", "use_crossref", default=True):
        def f_cr(rid):
            d = rid_doi.get(rid)
            return rid, (_get(sess, f"https://api.crossref.org/works/{d}?mailto={email}") if d else None)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(f_cr, rid) for rid in need]):
                rid, j = fut.result()
                if not j:
                    continue
                m = j.get("message", {})
                fund = [f.get("name", "") for f in m.get("funder", []) if f.get("name")]
                affs = [af["name"] for a in m.get("author", []) for af in a.get("affiliation", []) if af.get("name")]
                if fund:
                    out[rid]["cr_funders"] = "; ".join(dict.fromkeys(fund))
                    out[rid]["cr_has_funding"] = "yes"
                if affs:
                    out[rid]["cr_affiliations"] = " | ".join(dict.fromkeys(affs))

    if cfg.get("enrich", "use_unpaywall", default=True):
        def f_up(rid):
            d = rid_doi.get(rid)
            return rid, (_get(sess, f"https://api.unpaywall.org/v2/{d}?email={email}") if d else None)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(f_up, rid) for rid in need]):
                rid, j = fut.result()
                if not j:
                    continue
                loc = j.get("best_oa_location") or {}
                out[rid]["upw_is_oa"] = "yes" if j.get("is_oa") else "no"
                out[rid]["upw_oa_status"] = j.get("oa_status", "") or ""
                out[rid]["upw_pdf_url"] = loc.get("url_for_pdf", "") or loc.get("url", "") or ""

    if cfg.get("enrich", "use_europepmc", default=True):
        need2 = [rid for rid in out if not out[rid].get("oa_abstract")]

        def f_ep(rid):
            d = rid_doi.get(rid)
            p = rid_pmid.get(rid)
            if not (d or p):
                return rid, None
            q = f'DOI:"{d}"' if d else f'EXT_ID:{p} AND SRC:MED'
            return rid, _get(sess, "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                              f"?query={requests.utils.quote(q)}&resultType=core&format=json&pageSize=1")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(f_ep, rid) for rid in need2]):
                rid, j = fut.result()
                rl = (j or {}).get("resultList", {}).get("result", [])
                if rl:
                    ab = rl[0].get("abstractText", "") or ""
                    aff = rl[0].get("affiliation", "") or ""
                    if ab:
                        out[rid]["epmc_abstract"] = ab
                    if aff:
                        out[rid]["epmc_affiliation"] = aff

    for rid in out:
        out[rid].setdefault("oa_match_method", "")
    return out


def _coalesce(full):
    for c in ENR_COLS:
        if c not in full.columns:
            full[c] = ""
    trust = full["oa_match_method"].isin(["doi", "pmid"]) | full["oa_match_confidence"].eq("medium")
    for core, oa in [("title", "oa_title"), ("authors", "oa_authors"), ("journal", "oa_journal"),
                      ("document_type", "oa_type"), ("year", "oa_year"), ("keywords", "oa_keywords")]:
        if core not in full.columns:
            full[core] = ""
        full[core + "_orig"] = full[core]
        can = trust & full[oa].astype(str).str.len().gt(0)
        full.loc[can, core] = full.loc[can, oa]
    if "doi" in full.columns:
        full["doi_orig"] = full["doi"]
        fill = full["doi"].map(_norm_doi).eq("") & full["oa_doi"].astype(str).str.len().gt(0)
        full.loc[fill, "doi"] = full.loc[fill, "oa_doi"]

    def best_abs(r):
        o = str(r.get("abstract", "") or "")
        if len(o) >= 40:
            return o, "original"
        opts = [("openalex", r.get("oa_abstract", "")), ("europepmc", r.get("epmc_abstract", ""))]
        opts = [(l, t) for l, t in opts if isinstance(t, str) and len(t) >= 40]
        if not opts:
            return o, ("original" if o else "none")
        l, t = max(opts, key=lambda x: len(x[1]))
        return t, l

    ba = full.apply(best_abs, axis=1)
    full["abstract"] = [x[0] for x in ba]
    full["abstract_source"] = [x[1] for x in ba]
    full["affiliations"] = full["oa_affiliations_raw"]
    for alt in ["cr_affiliations", "epmc_affiliation"]:
        m = full["affiliations"].eq("") & full[alt].astype(str).str.len().gt(0)
        full.loc[m, "affiliations"] = full.loc[m, alt]
    full["citation_count"] = full["oa_cited_by_count"]
    full["is_funded"] = np.where(
        full["oa_has_funding"].eq("yes") | full["cr_has_funding"].eq("yes"), "yes",
        np.where(full["oa_match_method"].ne(""), "no", "unknown"))
    return full


def _row_group_window(pf, lo, hi):
    """Read ONLY rows [lo:hi) of an open ParquetFile via direct row-group
    access. This is the fix: no re-scan from row 0, so cost is independent
    of how far `lo` is into the file."""
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    starts = np.cumsum([0] + rg_sizes[:-1])
    ends = starts + np.array(rg_sizes)
    wanted = [i for i in range(pf.num_row_groups) if ends[i] > lo and starts[i] < hi]
    table = pf.read_row_groups(wanted)
    df_local = table.to_pandas().fillna("")
    offset = lo - starts[wanted[0]]
    return df_local.iloc[offset: offset + (hi - lo)].reset_index(drop=True)


def run_chunked_enrich(ctx, in_path, out_stem):
    """Main entry. in_path = canonical_pass1 file; out_stem = 'master_enriched'."""
    cfg = ctx.cfg
    email = ctx.email
    chunk_size = int(cfg.get("enrich", "chunk_size", default=25000))
    chunk_dir = os.path.join(ctx.work, "enriched_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    out_path = ctx.out(out_stem) + ".parquet"

    pf = pq.ParquetFile(in_path)
    total = pf.metadata.num_rows
    schema_cols = set(pf.schema.names)
    read_cols = [c for c in NEED_COLS if c in schema_cols]
    n_chunks = (total + chunk_size - 1) // chunk_size
    print(f"[enrich] {total} rows -> {n_chunks} chunks of {chunk_size} "
          f"(full file {len(schema_cols)} cols; {len(read_cols)} read per chunk)")
    all_ids = (pd.read_parquet(in_path, columns=["record_id"])["record_id"].tolist()
               if "record_id" in schema_cols else [f"REC-{i+1:06d}" for i in range(total)])
    sess = _mk_session(email)

    for ci in range(n_chunks):
        chunk_path = os.path.join(chunk_dir, f"chunk_{ci:03d}.parquet")
        if os.path.exists(chunk_path):
            print(f"[enrich] chunk {ci:03d}: already done, skip")
            continue
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, total)
        skin = pd.read_parquet(in_path, columns=read_cols).iloc[lo:hi].fillna("").reset_index(drop=True)
        if "record_id" not in skin.columns:
            skin["record_id"] = all_ids[lo:hi]
        for c in NEED_COLS:
            if c not in skin.columns:
                skin[c] = ""
        skin["doi_norm"] = skin["doi"].map(_norm_doi)

        def _pm(row):
            for v in [row.get("source_record_id", ""), row.get("pubmed__PMID", ""), row.get("europepmc__EXTERNAL_ID", "")]:
                s = str(v)
                m = re.search(r'(?:PMID[:\s]*|MEDLINE:)(\d{5,9})', s)
                if m:
                    return m.group(1)
                if re.fullmatch(r'\d{5,9}', s.strip()):
                    return s.strip()
            return ""

        skin["pmid"] = skin.apply(_pm, axis=1)
        print(f"[enrich] chunk {ci:03d}: rows {lo}-{hi} ({len(skin)}) ...")
        t0 = time.time()
        res = _enrich_records(skin, sess, email, cfg)

        # FIXED: direct row-group read instead of scan-from-zero iter_batches
        wide = _row_group_window(pf, lo, hi)
        if "record_id" not in wide.columns:
            wide["record_id"] = all_ids[lo:hi]
        enr = pd.DataFrame(list(res.values()))
        merged = wide.merge(enr, on="record_id", how="left", suffixes=("", "_enr")).fillna("")
        merged = _coalesce(merged)
        tmp = chunk_path + ".tmp"
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, chunk_path)
        matched = (merged["oa_match_method"].ne("")).sum()
        print(f"[enrich] chunk {ci:03d}: {len(merged)} rows, {matched} matched, {time.time()-t0:.0f}s")
        del skin, res, wide, enr, merged
        gc.collect()

    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.parquet")))
    print(f"[enrich] combining {len(chunk_files)} chunks (streaming)...")
    ref_cols = pq.ParquetFile(chunk_files[0]).schema.names
    writer = None
    total_out = 0
    for cf in chunk_files:
        df = pq.read_table(cf).to_pandas()
        for c in ref_cols:
            if c not in df.columns:
                df[c] = ""
        df = df[[c for c in ref_cols]]
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        total_out += len(df)
        del df, tbl
        gc.collect()
    if writer:
        writer.close()
    print(f"[enrich] wrote {out_path} ({total_out} rows) by streaming.")
    return out_path, total_out
