"""
GTRA fast deduplication (v4) for OpenAlex-only harvests (works.parquet).

What changed vs the old DEDUPLICATION v3.1 notebook
---------------------------------------------------
* Full outputs are written as PARQUET only. The slow 334k-row openpyxl
  .xlsx exports are gone; Excel is used only for one small QC workbook
  (sampled, truncated, written with xlsxwriter -> seconds, not minutes).
* No global ``fillna("")``: numeric columns keep their dtypes, so no
  ``to_parquet_safe`` round-trip is needed.
* Union-find on numpy arrays (no per-row dict lookups on DataFrames).
* Fuzzy stage is vectorised with ``rapidfuzz.process.cdist`` (multi-core,
  score_cutoff) inside blocks, run twice (first and last content token)
  so a change in the first word does not hide a duplicate.
* Every merge is logged as an edge (pair, stage, score) -> merge_edges.parquet,
  so any cluster can be audited or undone.
* Preprint <-> published version pairs are NOT merged. They are linked in
  ``version_group`` (flag, keep both), as decided for GTRA.
* Correction / erratum / retraction notices are never merged on title.

Stages (all merges respect a cluster-level DOI conflict guard)
  0  same OpenAlex ID (repeated rows from overlapping queries)
  1  same DOI                     (always merge)
  1b same PMID                    (merge unless DOI conflict)
  2a same title + author fp + year
  2b same title + journal/ISSN + year
  3  same title + year, >=1 shared author (or no authors on one side)
  4  fuzzy title (token_sort_ratio >= threshold), year +/- tol,
     gated by >=1 shared author, or same journal at a stricter threshold
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    raise ImportError("pip install rapidfuzz")


# ---------------------------------------------------------------- config
@dataclass
class DedupeConfig:
    fuzzy_threshold: int = 90          # title only, token_sort_ratio
    fuzzy_journal_threshold: int = 95  # stricter when only the journal confirms
    fuzzy_year_tol: int = 1
    version_year_tol: int = 3          # preprint -> published gap allowed
    version_threshold: int = 92        # fuzzy score to link a preprint/VOR pair
    max_len_diff: int = 25
    min_title_tokens: int = 3
    min_title_chars: int = 15
    block_chunk: int = 2000            # rows per cdist chunk (memory cap)
    workers: int = -1                  # rapidfuzz cdist threads
    type_priority: dict = field(default_factory=lambda: {
        "article": 10, "review": 8, "letter": 4, "preprint": 3,
        "book-chapter": 5, "book": 5, "report": 4, "dataset": 3,
        "dissertation": 2, "editorial": 1, "erratum": 1, "retraction": 1,
        "paratext": 0, "other": 0})


# ---------------------------------------------------------------- normalisers
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_TAGS = re.compile(r"<[^>]+>")
NA_LIKE = {"", "none", "null", "nan", "n/a", "na", "[]", "{}", "<na>"}
LEAD_STOP = {"a", "an", "the", "on", "of", "in", "and", "for", "to", "la", "le",
             "les", "de", "des", "el", "los", "las", "der", "die", "das", "um", "uma"}
GENERIC_TITLES = frozenset({
    "front matter", "back matter", "summary", "editorial", "introduction", "preface",
    "contents", "table of contents", "index", "book review", "letter", "news",
    "abstract", "abstracts", "poster", "supplement", "no title", "retracted",
    "corrigendum", "erratum", "obituary", "in this issue", "highlights",
    "acknowledgements", "author index", "subject index", "meeting", "conference"})
NOTICE_RE = re.compile(
    r"^(correction|corrigendum|erratum|errata|retraction|retracted|expression of concern|"
    r"publisher s note|notice of retraction|withdrawn|addendum|correction to|reply|response to)\b")


def _strip_acc(x: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))


def clean_title(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = _strip_acc(_TAGS.sub(" ", str(s)).lower().strip())
    if s in NA_LIKE:
        return ""
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def norm_doi(d) -> str:
    d = str(d or "").strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d).strip()
    return d if d.startswith("10.") else ""


def norm_pmid(p) -> str:
    m = re.search(r"(\d{4,9})\s*$", str(p or ""))
    return m.group(1) if m else ""


def norm_year(y) -> int:
    m = re.search(r"(19|20)\d{2}", str(y or ""))
    return int(m.group(0)) if m else -1


def journal_key(j, issn_l=None) -> str:
    """ISSN-L first (most stable), else case/space-folded display name."""
    i = str(issn_l or "").strip().lower()
    if i and i not in NA_LIKE:
        return "issn:" + i
    j = _WS.sub(" ", str(j or "").strip().lower())
    return "" if j in NA_LIKE else "name:" + j


def author_fp(authors) -> str:
    """'Jane A. Doe | Li Wei' -> 'doe|j;wei|l' ; also handles 'Doe, Jane'."""
    a = str(authors or "").strip()
    if a.lower() in NA_LIKE:
        return ""
    out = set()
    for au in re.split(r"[;|]", a):
        au = _strip_acc(au.lower().strip())
        if not au or au in NA_LIKE:
            continue
        if "," in au:
            last, first = au.split(",", 1)
            lt = _PUNCT.sub(" ", last).split()
            ft = _PUNCT.sub(" ", first).split()
            if lt and ft:
                out.add(f"{lt[-1]}|{ft[0][0]}")
        else:
            t = _PUNCT.sub(" ", au).split()
            if len(t) >= 2:
                out.add(f"{t[-1]}|{t[0][0]}")
    return ";".join(sorted(out))


def content_tokens(t: str) -> list[str]:
    return [w for w in t.split() if w not in LEAD_STOP and len(w) > 1]


# ---------------------------------------------------------------- union-find
class UnionFind:
    def __init__(self, n: int, dois: list[str]):
        self.p = np.arange(n)
        # cluster-level DOI sets; only roots hold entries
        self.dois = {i: {d} for i, d in enumerate(dois) if d}

    def find(self, x: int) -> int:
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def conflict(self, a: int, b: int) -> bool:
        da = self.dois.get(self.find(a))
        db = self.dois.get(self.find(b))
        return bool(da and db and not (da & db))

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        if ra in self.dois:
            self.dois.setdefault(rb, set()).update(self.dois.pop(ra))
        return True


# ---------------------------------------------------------------- main class
class FastDeduper:
    def __init__(self, cfg: DedupeConfig | None = None, verbose: bool = True):
        self.cfg = cfg or DedupeConfig()
        self.verbose = verbose
        self.edges: list[tuple] = []        # (i, j, stage, score)
        self.version_edges: list[tuple] = []
        self.stage_counts: dict[str, int] = {}
        self.timings: dict[str, float] = {}

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ---------------- prepare
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.perf_counter()
        df = df.reset_index(drop=True).copy()
        g = lambda c: df[c] if c in df.columns else pd.Series([None] * len(df))
        df["_title"] = g("oa_title").map(clean_title)
        df["_is_notice"] = df["_title"].str.match(NOTICE_RE) | g("oa_type").astype(str).str.lower().isin(
            ["erratum", "retraction"])
        risky = (df["_title"].isin(GENERIC_TITLES)
                 | (df["_title"].str.split().str.len().fillna(0) < self.cfg.min_title_tokens)
                 | (df["_title"].str.len() < self.cfg.min_title_chars))
        df["_title_ok"] = ~risky & ~df["_is_notice"]
        df["_doi"] = g("oa_doi").map(norm_doi)
        df["_pmid"] = g("oa_pmid").map(norm_pmid)
        df["_year"] = g("oa_year").map(norm_year).astype(int)
        df["_afp"] = g("oa_authors").map(author_fp)
        df["_jkey"] = [journal_key(j, i) for j, i in zip(g("oa_journal"), g("oa_issn_l"))]
        df["_type"] = g("oa_type").astype(str).str.lower().str.strip()
        df["_is_preprint"] = df["_type"].eq("preprint")
        toks = df["_title"].map(content_tokens)
        df["_blk_first"] = toks.map(lambda t: t[0] if t else "")
        df["_blk_last"] = toks.map(lambda t: t[-1] if t else "")
        self.timings["prepare"] = time.perf_counter() - t0
        self.log(f"prepare: {len(df):,} rows in {self.timings['prepare']:.1f}s | "
                 f"risky/notice titles: {(~df['_title_ok']).sum():,}")
        return df

    # ---------------- helpers
    def _afp_sets(self, df):
        return [set(x.split(";")) if x else set() for x in df["_afp"]]

    def _is_version_pair(self, i, j) -> bool:
        pp = self._pre
        return pp[i] != pp[j]          # exactly one is a preprint

    def _try(self, i, j, stage, score=100.0, allow_version=True) -> None:
        uf = self.uf
        if uf.find(i) == uf.find(j):
            return
        if allow_version and self._is_version_pair(i, j):
            self.version_edges.append((i, j, stage, score))
            return
        if uf.conflict(i, j):
            return
        uf.union(i, j)
        self.edges.append((i, j, stage, score))
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def _group_pairs(self, df, mask, keycols, stage, gate=None):
        key = df.loc[mask, keycols[0]].astype(str)
        for c in keycols[1:]:
            key = key + "||" + df.loc[mask, c].astype(str)
        dup = key[key.duplicated(keep=False)]
        for _, idx in dup.groupby(dup).groups.items():
            ix = list(idx)
            for a in range(len(ix)):
                for b in range(a + 1, len(ix)):
                    if gate is None or gate(ix[a], ix[b]):
                        self._try(ix[a], ix[b], stage)

    # ---------------- run
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        df = self.prepare(df)
        n = len(df)
        self.uf = UnionFind(n, df["_doi"].tolist())
        self._pre = df["_is_preprint"].to_numpy()
        afp = self._afp_sets(df)
        years = df["_year"].to_numpy()
        T = time.perf_counter

        def shared_author(i, j, allow_empty=False):
            if not afp[i] or not afp[j]:
                return allow_empty
            return bool(afp[i] & afp[j])

        # Stage 0: same OpenAlex ID
        t = T()
        if "oa_id" in df.columns:
            ids = df["oa_id"].astype(str)
            self._group_pairs(df, ids.ne("") & ids.duplicated(keep=False), ["oa_id"], "0_oa_id")
        # Stage 1: DOI (merge even across preprint/article: identical DOI = identical record)
        m = df["_doi"].ne("")
        dup = df.loc[m, "_doi"]
        for _, idx in dup[dup.duplicated(keep=False)].groupby(dup).groups.items():
            ix = list(idx)
            for j in ix[1:]:
                if self.uf.union(ix[0], j):
                    self.edges.append((ix[0], j, "1_doi", 100.0))
                    self.stage_counts["1_doi"] = self.stage_counts.get("1_doi", 0) + 1
        # Stage 1b: PMID
        self._group_pairs(df, df["_pmid"].ne(""), ["_pmid"], "1b_pmid")
        self.timings["ids"] = T() - t

        # Stages 2-3: exact title
        t = T()
        ok = df["_title_ok"] & (df["_year"] > 0)
        self._group_pairs(df, ok & df["_afp"].ne(""), ["_title", "_afp", "_year"], "2a_title_auth_year")
        self._group_pairs(df, ok & df["_jkey"].ne(""), ["_title", "_jkey", "_year"], "2b_title_journal_year")
        self._group_pairs(df, ok, ["_title", "_year"], "3_title_year_authgate",
                          gate=lambda i, j: shared_author(i, j, allow_empty=True))
        self.timings["exact_title"] = T() - t

        # Stage 4: fuzzy title, vectorised with cdist.
        # Inside each token block, rows of year y are compared only with rows of
        # years y..y+tol (tol=1), and preprints with non-preprints of y-3..y+3.
        # This keeps the big "tick ..." block from becoming a full n^2 problem.
        t = T()
        titles = df["_title"].to_numpy()
        lens = df["_title"].str.len().to_numpy()
        jk = df["_jkey"].to_numpy()
        pre = self._pre
        cutoff = min(cfg.fuzzy_threshold, cfg.version_threshold)
        self._max_block, self._n_cmp = 0, 0

        def handle(i, j, sc):
            if abs(int(lens[i]) - int(lens[j])) > cfg.max_len_diff:
                return
            version = pre[i] != pre[j]
            yi, yj = years[i], years[j]
            tol = cfg.version_year_tol if version else cfg.fuzzy_year_tol
            if yi > 0 and yj > 0 and abs(yi - yj) > tol:
                return
            sa = shared_author(i, j)
            if version:
                if sc >= cfg.version_threshold and sa and self.uf.find(i) != self.uf.find(j):
                    self.version_edges.append((i, j, "4_fuzzy_version", float(sc)))
                return
            if sc < cfg.fuzzy_threshold:
                return
            same_j = bool(jk[i]) and jk[i] == jk[j]
            if sa or (same_j and sc >= cfg.fuzzy_journal_threshold):
                self._try(i, j, f"4_fuzzy_{self._blk}", float(sc), allow_version=False)

        def compare(rows, cols, triangle):
            if len(rows) == 0 or len(cols) == 0:
                return
            ct = titles[cols].tolist()
            for s0 in range(0, len(rows), cfg.block_chunk):
                rr = rows[s0:s0 + cfg.block_chunk]
                M = process.cdist(titles[rr].tolist(), ct, scorer=fuzz.token_sort_ratio,
                                  score_cutoff=cutoff, dtype=np.uint8, workers=cfg.workers)
                self._n_cmp += M.size
                r, c = np.nonzero(M)
                if triangle:                      # rows == cols: keep upper triangle
                    keep = c > (r + s0)
                    r, c = r[keep], c[keep]
                for a_, b_ in zip(r, c):
                    handle(int(rr[a_]), int(cols[b_]), int(M[a_, b_]))

        for blk_col in ("_blk_first", "_blk_last"):
            self._blk = blk_col[5:]
            cand = df.index[df["_title_ok"] & df[blk_col].ne("")]
            for tok, idx in df.loc[cand].groupby(blk_col).groups.items():
                ix = np.asarray(idx)
                if len(ix) < 2:
                    continue
                self._max_block = max(self._max_block, len(ix))
                y = years[ix]
                nopy = ix[y <= 0]
                for yr in np.unique(y[y > 0]):
                    same = ix[y == yr]
                    compare(same, same, triangle=True)
                    nxt = ix[(y > yr) & (y <= yr + cfg.fuzzy_year_tol)]
                    compare(same, nxt, triangle=False)
                    # preprint <-> published links, wider window, one direction only
                    p_rows = same[pre[same]]
                    wide = ix[(np.abs(y - yr) <= cfg.version_year_tol) & (y != yr) & ~pre[ix]]
                    far = wide[np.abs(years[wide] - yr) > cfg.fuzzy_year_tol]
                    compare(p_rows, far, triangle=False)
                if len(nopy):                     # unknown year: compare with whole block
                    compare(nopy, ix, triangle=False)
        max_block, n_cmp = self._max_block, self._n_cmp
        self.timings["fuzzy"] = T() - t
        self.log(f"fuzzy: {n_cmp:,} comparisons, largest block {max_block:,}, {self.timings['fuzzy']:.1f}s")

        return self._finalise(df)

    # ---------------- finalise
    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        uf = self.uf
        n = len(df)
        df["_cluster"] = [uf.find(i) for i in range(n)]

        # canonical: DOI > not-preprint > type priority > metadata completeness > abstract > citations
        meta_cols = [c for c in ["oa_title", "oa_authors", "oa_year", "oa_doi", "oa_journal",
                                 "oa_abstract", "oa_keywords", "oa_pmid", "oa_countries"] if c in df.columns]
        filled = sum(df[c].notna() & df[c].astype(str).str.strip().ne("") for c in meta_cols)
        zero = pd.Series(0, index=df.index)
        has_abs = (df["oa_abstract"].fillna("").astype(str).str.len() > 50) if "oa_abstract" in df.columns else zero
        cites = pd.to_numeric(df["oa_cited_by_count"], errors="coerce").fillna(0) \
            if "oa_cited_by_count" in df.columns else zero
        df["_score"] = (df["_doi"].ne("").astype(int) * 1000
                        + (~df["_is_preprint"]).astype(int) * 500
                        + df["_type"].map(cfg.type_priority).fillna(0).astype(int) * 30
                        + filled * 5 + has_abs.astype(int) * 6
                        + np.log1p(cites))
        order = df.sort_values(["_cluster", "_score", "oa_id" if "oa_id" in df.columns else "_title"],
                               ascending=[True, False, True])
        canon = order.drop_duplicates("_cluster").index
        df["is_canonical"] = df.index.isin(canon)

        # stable study ids: ordered by first appearance
        first = df.drop_duplicates("_cluster")["_cluster"]
        cl2id = {c: f"STU-{k + 1:06d}" for k, c in enumerate(first)}
        df["study_id"] = df["_cluster"].map(cl2id)
        df["n_records"] = df.groupby("study_id")["study_id"].transform("size")
        e = {}
        for i, j, st, _ in self.edges:
            e.setdefault(i, set()).add(st)
            e.setdefault(j, set()).add(st)
        df["merge_stages"] = [";".join(sorted(e.get(i, ()))) for i in range(n)]

        # version groups: union of study clusters linked by preprint/VOR edges
        vuf = UnionFind(n, [""] * n)
        for i, j, *_ in self.version_edges:
            vuf.union(int(df.at[i, "_cluster"]), int(df.at[j, "_cluster"]))
        vroot = df["_cluster"].map(lambda c: vuf.find(int(c)))
        multi = df.groupby(vroot)["study_id"].transform("nunique") > 1
        df["version_group"] = np.where(multi, "VG-" + vroot.astype(str), "")
        df["has_other_version"] = multi
        # within a version group, the published (non-preprint) canonical is the "primary"
        vg = df[df["is_canonical"] & multi].sort_values("_score", ascending=False)
        prim = set(vg.drop_duplicates("version_group")["study_id"])
        df["is_primary_version"] = df["is_canonical"] & (~multi | df["study_id"].isin(prim))
        return df

    # ---------------- reporting
    def report(self, df: pd.DataFrame) -> dict:
        nr = len(df)
        ns = df["study_id"].nunique()
        rep = {
            "records_in": nr,
            "unique_studies": ns,
            "duplicates_removed": nr - ns,
            "merges_by_stage": dict(sorted(self.stage_counts.items())),
            "largest_cluster": int(df["n_records"].max()),
            "version_links": len(self.version_edges),
            "studies_with_other_version": int(df.loc[df["is_canonical"], "has_other_version"].sum()),
            "primary_unique_studies": int(df["is_primary_version"].sum()),
            "timings_s": {k: round(v, 1) for k, v in self.timings.items()},
        }
        return rep

    def edges_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["oa_id", "oa_title", "oa_year", "oa_type", "oa_doi"]
        cols = [c for c in cols if c in df.columns]
        rows = [(i, j, st, sc, "merge") for i, j, st, sc in self.edges] + \
               [(i, j, st, sc, "version_link") for i, j, st, sc in self.version_edges]
        e = pd.DataFrame(rows, columns=["i", "j", "stage", "score", "kind"])
        if e.empty:
            return e
        a = df.loc[e["i"], cols].reset_index(drop=True).add_suffix("_a")
        b = df.loc[e["j"], cols].reset_index(drop=True).add_suffix("_b")
        return pd.concat([e[["stage", "score", "kind"]], a, b], axis=1)


# ---------------------------------------------------------------- IO
HELPER_COLS = ["_title", "_is_notice", "_title_ok", "_doi", "_pmid", "_year", "_afp", "_jkey",
               "_type", "_is_preprint", "_blk_first", "_blk_last", "_cluster", "_score"]
DEDUPE_COLS = ["study_id", "is_canonical", "n_records", "merge_stages",
               "version_group", "has_other_version", "is_primary_version"]

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f﷐-﷯￾￿\ud800-\udfff]")


def write_qc_workbook(sheets: dict[str, pd.DataFrame], path: str, max_rows: int = 500,
                      max_chars: int = 300, seed: int = 42) -> None:
    """Small human-review workbook. Sampled + truncated + xlsxwriter = fast."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        for name, f in sheets.items():
            f = f.sample(max_rows, random_state=seed).sort_index() if len(f) > max_rows else f
            f = f.copy()
            for c in f.columns:
                if f[c].dtype == object or str(f[c].dtype).startswith("string"):
                    f[c] = f[c].astype(str).str.slice(0, max_chars).str.replace(_ILLEGAL, " ", regex=True)
            f.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(f), 1), max(len(f.columns) - 1, 0))


def save_outputs(df: pd.DataFrame, dd: FastDeduper, out_dir: str, qc_rows: int = 500) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    keep = [c for c in df.columns if c not in HELPER_COLS]
    front = [c for c in DEDUPE_COLS if c in keep]
    full = df[front + [c for c in keep if c not in front]]

    paths = {
        "dedupe_map": os.path.join(out_dir, "dedupe_map.parquet"),
        "works_canonical": os.path.join(out_dir, "works_canonical.parquet"),
        "merge_edges": os.path.join(out_dir, "merge_edges.parquet"),
        "qc": os.path.join(out_dir, "qc_dedupe_review.xlsx"),
    }
    # light audit map: one row per input record (no abstracts) -> small and fast
    map_cols = front + [c for c in ["oa_id", "oa_doi", "oa_pmid", "oa_title", "oa_year", "oa_type",
                                    "oa_journal"] if c in df.columns]
    df[map_cols].to_parquet(paths["dedupe_map"], index=False)
    full[full["is_canonical"]].to_parquet(paths["works_canonical"], index=False)
    edges = dd.edges_frame(df)
    edges.to_parquet(paths["merge_edges"], index=False)

    show = [c for c in ["study_id", "n_records", "merge_stages", "version_group", "oa_id", "oa_type",
                        "oa_year", "oa_title", "oa_authors", "oa_journal", "oa_doi"] if c in df.columns]
    multi = df[df["n_records"] > 1].sort_values(["study_id", "is_canonical"], ascending=[True, False])
    fuzzy = edges[edges["stage"].str.startswith("4_") & edges["kind"].eq("merge")] if len(edges) else edges
    vers = df[df["has_other_version"]].sort_values(["version_group", "study_id"])
    study_has_doi = df.groupby("study_id")["_doi"].transform(lambda s: s.ne("").any())
    no_doi = df[df["is_canonical"] & df["_doi"].eq("") & study_has_doi]
    write_qc_workbook({
        "summary": pd.DataFrame([(k, str(v)) for k, v in dd.report(df).items()], columns=["metric", "value"]),
        "fuzzy_merges": fuzzy,
        "multi_record_studies": multi[show],
        "version_links": vers[show + ["is_primary_version"]],
        "big_clusters": df[df["n_records"] >= 5][show],
        "canon_missing_doi": no_doi[show],
    }, paths["qc"], max_rows=qc_rows)
    dt = time.perf_counter() - t0
    if dd.verbose:
        print(f"saved in {dt:.1f}s:")
        for k, p in paths.items():
            print(f"  {k:16s} {os.path.getsize(p) / 1e6:7.1f} MB  {p}")
    return paths
