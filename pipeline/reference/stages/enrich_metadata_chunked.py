# ============================================================
#  METADATA ENRICHMENT for science-of-science analysis (Colab)
#  Tick-Borne Diseases / Climate / Sub-Saharan Africa review
#  ---- CHUNKED EDITION ----
#
#  WHY THIS VERSION EXISTS:
#  The original script loads ALL records and runs every pass over the
#  WHOLE file each time, which crashes free-tier Colab's RAM at ~114k
#  records. This version processes ONE CHUNK_SIZE-sized slice per run,
#  through ALL 7 passes, then saves THAT CHUNK ONLY and stops. Each run
#  costs roughly the SAME time (one chunk's worth of work), instead of
#  the whole file growing slower every time you re-run.
#
#  HOW TO USE (re-run this exact cell repeatedly):
#    Run 1: processes rows      0 .. CHUNK_SIZE   -> saves chunk_000.parquet
#    Run 2: processes rows CHUNK_SIZE .. 2*CHUNK_SIZE -> saves chunk_001.parquet
#    ... and so on. A finished chunk file is auto-skipped, so re-running
#    the SAME cell just advances to the next unfinished chunk.
#  When ALL chunks exist, set MODE = "combine" and run once more to
#  stitch them into master_enriched.parquet (streamed, low memory).
#
#  RUN ORDER:  type_filter  ->  THIS  ->  dedup  ->  analysis
#  Enriches each chunk's records against OpenAlex (primary),
#  Crossref (funder/affiliation gap-fill), Unpaywall (OA status),
#  matching by DOI -> PMID -> title(+year+author) fallback.
#  Same 7 passes, same logic, same columns as the original script —
#  only the OUTER LOOP (which rows, when to stop and save) changed.
#
#  Input : master_filtered_by_type.xlsx   (from the type filter)
#  Output: enriched_chunks/chunk_NNN.parquet  (one per run, self-contained)
#          master_enriched.parquet            (after MODE="combine")
#          enrichment_coverage.xlsx           (field-by-field coverage, on combine)
# ============================================================

## ---- 1. setup ----
from google.colab import drive
drive.mount('/content/drive')
!pip -q install requests rapidfuzz pandas openpyxl pyarrow

import os, re, time, json, math, gc, glob
import pandas as pd, numpy as np
import requests
from rapidfuzz import fuzz
import pyarrow as pa, pyarrow.parquet as pq

## ---- 2. config ----
FOLDER   = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR  = os.path.join(FOLDER, "preprocessing_data")
IN_PATH  = os.path.join(OUT_DIR, "master_filtered_by_type.xlsx")
CHUNK_DIR = os.path.join(OUT_DIR, "enriched_chunks"); os.makedirs(CHUNK_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "master_enriched.parquet")
COV_PATH = os.path.join(OUT_DIR, "enrichment_coverage.xlsx")

POLITE_EMAIL    = "cognosxx@gmail.com"
USE_CROSSREF    = True     # funder / affiliation gap-fill
USE_UNPAYWALL   = True     # OA status + best free PDF (set False to skip ~half the time)
USE_EUROPEPMC   = True     # PASS 7: best free ABSTRACT source (biomedical) — high impact
USE_S2          = True     # Semantic Scholar — abstract gap-fill (+ citations, references, TLDR)
S2_API_KEY      = ""       # optional; leave "" for the free (rate-limited) pool. A key raises limits.
S2_GET_REFS     = False    # also pull reference/citation IDs for citation-network analysis (slower)
TITLE_FALLBACK  = True     # match DOI-less records by title (+year+author gate)
TITLE_MIN_SCORE = 92       # token_sort_ratio threshold for a title match
CHECKPOINT_EVERY = 500     # save progress every N processed records (within-chunk, not used
                            # for crash-resume here since the chunk itself IS the save unit;
                            # kept only to preserve the original passes' print cadence)
MAX_RECORDS     = None     # cap for a test run (e.g. 300); None = all

CHUNK_SIZE = 25000          # <-- the new control: rows processed per run. Lower this
                             #     (e.g. 10000 or 5000) if a single chunk still strains RAM.
MODE = "enrich"             # "enrich" = process the next unfinished chunk.
                             # "combine" = stitch all finished chunks into master_enriched.parquet.

# ---- columns needed to DRIVE matching (read once, cheaply, to know total size + ids) ----
NEED_COLS_FOR_ID = ["record_id"]  # just to count rows / get ids if record_id pre-exists

## ============================================================
##  ALL HELPER FUNCTIONS BELOW ARE UNCHANGED FROM THE ORIGINAL SCRIPT.
##  Only the driving loop (section 3 onward) is restructured to chunk.
## ============================================================

def norm_doi(d):
    d = str(d or "").strip().lower()
    for p in ("https://doi.org/","http://dx.doi.org/","https://dx.doi.org/","doi:"):
        d = d.replace(p,"")
    d = d.strip()
    return d if d.startswith("10.") else ""

def extract_pmid(row):
    for v in [row.get("source_record_id",""), row.get("pubmed__PMID",""), row.get("europepmc__EXTERNAL_ID","")]:
        m = re.search(r'(?:PMID[:\s]*|MEDLINE:)(\d{5,9})', str(v))
        if m: return m.group(1)
        if re.fullmatch(r'\d{5,9}', str(v).strip()): return str(v).strip()
    return ""

def reconstruct_abstract(inv):
    if not inv: return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs: pos[i] = word
    if not pos: return ""
    return " ".join(pos[i] for i in sorted(pos))

def parse_openalex(w):
    """Map an OpenAlex work object -> flat dict of oa_* fields."""
    auth = w.get("authorships") or []
    names, orcids, raw_affs, inst_names, inst_ids, countries = [], [], [], [], [], []
    for a in auth:
        au = a.get("author") or {}
        if au.get("display_name"): names.append(au["display_name"])
        if au.get("orcid"): orcids.append(au["orcid"])
        for s in (a.get("raw_affiliation_strings") or []):
            if s: raw_affs.append(s)
        for ins in (a.get("institutions") or []):
            if ins.get("display_name"): inst_names.append(ins["display_name"])
            if ins.get("id"): inst_ids.append(ins["id"])
            if ins.get("country_code"): countries.append(ins["country_code"])
    grants = w.get("grants") or []
    funders = [g.get("funder_display_name","") for g in grants if g.get("funder_display_name")]
    funder_ids = [g.get("funder","") for g in grants if g.get("funder")]
    pl = w.get("primary_location") or {}
    src = (pl.get("source") or {}) if pl else {}
    pt = w.get("primary_topic") or {}
    ids = w.get("ids") or {}
    oa = w.get("open_access") or {}
    kws = [k.get("display_name","") for k in (w.get("keywords") or []) if k.get("display_name")]
    def uniq(seq):
        seen=set(); out=[]
        for x in seq:
            if x and x not in seen: seen.add(x); out.append(x)
        return out
    return {
        "oa_id": w.get("id",""),
        "oa_doi": norm_doi(w.get("doi","")),
        "oa_pmid": (ids.get("pmid","") or "").replace("https://pubmed.ncbi.nlm.nih.gov/",""),
        "oa_pmcid": ids.get("pmcid",""),
        "oa_mag": str(ids.get("mag","")),
        "oa_title": w.get("title","") or "",
        "oa_abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "oa_authors": "; ".join(names),
        "oa_author_orcids": "; ".join(orcids),
        "oa_affiliations_raw": " | ".join(uniq(raw_affs)),
        "oa_institutions": "; ".join(uniq(inst_names)),
        "oa_institution_ids": "; ".join(uniq(inst_ids)),
        "oa_countries": "; ".join(uniq(countries)),
        "oa_cited_by_count": str(w.get("cited_by_count","")),
        "oa_journal": src.get("display_name","") or "",
        "oa_issn": src.get("issn_l","") or "",
        "oa_type": w.get("type","") or "",
        "oa_language": w.get("language","") or "",
        "oa_funders": "; ".join(uniq(funders)),
        "oa_funder_ids": "; ".join(uniq(funder_ids)),
        "oa_has_funding": "yes" if grants else "no",
        "oa_year": str(w.get("publication_year","") or ""),
        "oa_topic": pt.get("display_name","") or "",
        "oa_subfield": ((pt.get("subfield") or {}).get("display_name","")) or "",
        "oa_field": ((pt.get("field") or {}).get("display_name","")) or "",
        "oa_domain": ((pt.get("domain") or {}).get("display_name","")) or "",
        "oa_concepts": "; ".join(c.get("display_name","") for c in (w.get("concepts") or [])[:8] if c.get("display_name")),
        "oa_sdgs": "; ".join(s.get("display_name","") for s in (w.get("sustainable_development_goals") or []) if s.get("display_name")),
        "oa_is_oa": "yes" if oa.get("is_oa") else "no",
        "oa_oa_status": oa.get("oa_status","") or "",
        "oa_keywords": "; ".join(kws),
    }

sess = requests.Session()
sess.headers.update({"User-Agent": f"sysrev-enrich/1.0 (mailto:{POLITE_EMAIL})"})

def oa_get(url):
    try:
        r = sess.get(url, timeout=30)
        if r.status_code == 200: return r.json()
    except Exception:
        return None
    return None

def fetch_openalex_by_doi_batch(dois):
    """Batch up to 50 DOIs; returns {doi_norm: work}."""
    out = {}
    B = 45
    for i in range(0, len(dois), B):
        batch = [d for d in dois[i:i+B] if d]
        if not batch: continue
        url = ("https://api.openalex.org/works?per-page=50"
               f"&filter=doi:{'|'.join(batch)}&mailto={POLITE_EMAIL}")
        j = oa_get(url)
        if j:
            for w in j.get("results", []):
                d = norm_doi(w.get("doi",""))
                if d: out[d] = w
        time.sleep(0.12)
    return out

def fetch_openalex_by_pmid(pmid):
    j = oa_get(f"https://api.openalex.org/works/pmid:{pmid}?mailto={POLITE_EMAIL}")
    time.sleep(0.12); return j

def fetch_openalex_by_title(title, year, author):
    t = re.sub(r'[^\w\s]',' ', str(title or '')).strip()
    if len(t) < 12: return None, 0
    url = ("https://api.openalex.org/works?per-page=5"
           f"&search={requests.utils.quote(t[:200])}&mailto={POLITE_EMAIL}")
    j = oa_get(url); time.sleep(0.15)
    if not j: return None, 0
    best, best_sc = None, 0
    for w in j.get("results", []):
        sc = fuzz.token_sort_ratio(t.lower(), re.sub(r'[^\w\s]',' ',(w.get("title") or "")).lower())
        if sc > best_sc: best, best_sc = w, sc
    return best, best_sc

def safe_xlsx(frame, path, sheet):
    ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out = frame.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: ILLEGAL.sub(" ", str(v))[:32000])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name=sheet, index=False)


## ============================================================
##  SECTION 3 — CHUNK-DRIVING LOGIC (this is the new part)
## ============================================================

def enrich_chunk(df):
    """Run ALL 7 passes + the final coalesce on a chunk-sized DataFrame.
    This is the ORIGINAL script's body, verbatim in logic, just wrapped as
    a function operating on `df` = one chunk instead of the whole file."""
    if "record_id" not in df.columns:
        df["record_id"] = [f"REC-{i+1:06d}" for i in range(len(df))]

    OA_FIELDS = ["oa_id","oa_doi","oa_pmid","oa_pmcid","oa_mag","oa_title","oa_abstract",
        "oa_authors","oa_author_orcids","oa_affiliations_raw","oa_institutions",
        "oa_institution_ids","oa_countries","oa_cited_by_count","oa_journal","oa_issn",
        "oa_type","oa_language","oa_funders","oa_funder_ids","oa_has_funding","oa_year",
        "oa_topic","oa_subfield","oa_field","oa_domain","oa_concepts","oa_sdgs",
        "oa_is_oa","oa_oa_status","oa_keywords"]
    for c in OA_FIELDS:
        if c not in df.columns: df[c] = ""
    for c in ["cr_funders","cr_has_funding","cr_affiliations","upw_is_oa","upw_oa_status","upw_pdf_url",
              "s2_id","s2_abstract","s2_tldr","s2_cited_by_count","s2_reference_count",
              "s2_fields_of_study","s2_reference_ids","s2_citation_ids","epmc_abstract","epmc_affiliation"]:
        if c not in df.columns: df[c] = ""
    df["oa_match_method"]     = df.get("oa_match_method","")
    df["oa_match_confidence"] = df.get("oa_match_confidence","")
    df["doi_norm"] = df["doi"].map(norm_doi) if "doi" in df.columns else ""
    if "doi" not in df.columns: df["doi"] = ""
    if "abstract" not in df.columns: df["abstract"] = ""
    if "title" not in df.columns: df["title"] = ""
    if "year" not in df.columns: df["year"] = ""
    df["pmid"] = df.apply(extract_pmid, axis=1)

    ## ---- PASS 1: OpenAlex by DOI (batched) ----
    doi_rows = df[df["doi_norm"].ne("")]
    uniq_dois = sorted(set(doi_rows["doi_norm"]))
    print(f"  PASS 1: OpenAlex by DOI — {len(uniq_dois)} unique DOIs")
    processed = 0
    for i in range(0, len(uniq_dois), 45):
        chunk = uniq_dois[i:i+45]
        found = fetch_openalex_by_doi_batch(chunk)
        for d, w in found.items():
            parsed = parse_openalex(w)
            sel = df["doi_norm"].eq(d)
            for k, v in parsed.items(): df.loc[sel, k] = v
            df.loc[sel, "oa_match_method"] = "doi"
            df.loc[sel, "oa_match_confidence"] = "high"
        processed += len(chunk)
        if processed % 4500 < 45:
            print(f"    ...{processed}/{len(uniq_dois)} dois")

    ## ---- PASS 2: OpenAlex by PMID ----
    unm = df[df["oa_match_method"].eq("") & df["pmid"].ne("")]
    print(f"  PASS 2: OpenAlex by PMID — {len(unm)} records")
    for n,(idx,row) in enumerate(unm.iterrows()):
        w = fetch_openalex_by_pmid(row["pmid"])
        if w and not w.get("error"):
            parsed = parse_openalex(w)
            for k,v in parsed.items(): df.at[idx,k]=v
            df.at[idx,"oa_match_method"]="pmid"; df.at[idx,"oa_match_confidence"]="high"
        if n and n % 500 == 0: print(f"    ...{n}/{len(unm)}")

    ## ---- PASS 3: OpenAlex by title (+year+author gate) ----
    if TITLE_FALLBACK:
        unm = df[df["oa_match_method"].eq("") & df["title"].str.len().ge(12)]
        print(f"  PASS 3: OpenAlex by title — {len(unm)} records")
        for n,(idx,row) in enumerate(unm.iterrows()):
            w, sc = fetch_openalex_by_title(row["title"], row["year"], row.get("authors",""))
            if w and sc >= TITLE_MIN_SCORE:
                yr_ok = True
                wy = re.sub(r'\D','',str(w.get("publication_year","") or ""))[:4]
                ry = re.sub(r'\D','',str(row["year"]))[:4]
                if wy and ry and abs(int(wy)-int(ry))>1: yr_ok=False
                if yr_ok:
                    parsed = parse_openalex(w)
                    for k,v in parsed.items(): df.at[idx,k]=v
                    df.at[idx,"oa_match_method"]="title"
                    df.at[idx,"oa_match_confidence"]= "medium" if sc>=96 else "low"
            if n and n % 500 == 0: print(f"    ...{n}/{len(unm)}")

    ## ---- PASS 4: Crossref funder/affiliation gap-fill ----
    if USE_CROSSREF:
        need = df[df["doi_norm"].ne("") & (df["oa_funders"].eq("") | df["oa_affiliations_raw"].eq(""))]
        print(f"  PASS 4: Crossref gap-fill — {len(need)} records")
        for n,(idx,row) in enumerate(need.iterrows()):
            j = oa_get(f"https://api.crossref.org/works/{row['doi_norm']}?mailto={POLITE_EMAIL}")
            time.sleep(0.05)
            if not j: continue
            m = j.get("message",{})
            fund = [f.get("name","") for f in m.get("funder",[]) if f.get("name")]
            affs = []
            for a in m.get("author",[]):
                for af in a.get("affiliation",[]):
                    if af.get("name"): affs.append(af["name"])
            if fund:
                df.at[idx,"cr_funders"]="; ".join(dict.fromkeys(fund))
                df.at[idx,"cr_has_funding"]="yes"
            if affs:
                df.at[idx,"cr_affiliations"]=" | ".join(dict.fromkeys(affs))
            if n and n % 500 == 0: print(f"    ...{n}/{len(need)}")

    ## ---- PASS 5: Unpaywall OA status ----
    if USE_UNPAYWALL:
        need = df[df["doi_norm"].ne("") & df["upw_is_oa"].eq("")]
        print(f"  PASS 5: Unpaywall OA — {len(need)} records")
        for n,(idx,row) in enumerate(need.iterrows()):
            j = oa_get(f"https://api.unpaywall.org/v2/{row['doi_norm']}?email={POLITE_EMAIL}")
            time.sleep(0.05)
            if not j: continue
            df.at[idx,"upw_is_oa"]="yes" if j.get("is_oa") else "no"
            df.at[idx,"upw_oa_status"]=j.get("oa_status","") or ""
            loc = j.get("best_oa_location") or {}
            df.at[idx,"upw_pdf_url"]=loc.get("url_for_pdf","") or loc.get("url","") or ""
            if n and n % 500 == 0: print(f"    ...{n}/{len(need)}")

    ## ---- PASS 6: Semantic Scholar (abstract gap-fill + citations/TLDR) ----
    if USE_S2:
        s2_headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
        s2_fields = "externalIds,title,abstract,tldr,citationCount,referenceCount,fieldsOfStudy"
        if S2_GET_REFS: s2_fields += ",references.externalIds,citations.externalIds"
        need_abs = df[df["doi_norm"].ne("") & df["abstract"].astype(str).str.len().eq(0)]
        other    = df[df["doi_norm"].ne("") & df["abstract"].astype(str).str.len().gt(0) & df["s2_id"].eq("")]
        target = pd.concat([need_abs, other])
        dois = list(dict.fromkeys(target["doi_norm"].tolist()))
        print(f"  PASS 6: Semantic Scholar — {len(dois)} DOIs ({len(need_abs)} missing abstracts, prioritized)")

        def s2_parse(p):
            tldr = (p.get("tldr") or {}).get("text","") if p.get("tldr") else ""
            out = {
                "s2_id": p.get("paperId",""),
                "s2_abstract": p.get("abstract","") or "",
                "s2_tldr": tldr,
                "s2_cited_by_count": str(p.get("citationCount","") or ""),
                "s2_reference_count": str(p.get("referenceCount","") or ""),
                "s2_fields_of_study": "; ".join(p.get("fieldsOfStudy") or []),
            }
            if S2_GET_REFS:
                refs = [ (r.get("externalIds") or {}).get("DOI","") for r in (p.get("references") or []) ]
                cits = [ (c.get("externalIds") or {}).get("DOI","") for c in (p.get("citations") or []) ]
                out["s2_reference_ids"] = "; ".join([r for r in refs if r][:200])
                out["s2_citation_ids"]  = "; ".join([c for c in cits if c][:200])
            return out

        B = 100
        for i in range(0, len(dois), B):
            batch = dois[i:i+B]
            ids = [f"DOI:{d}" for d in batch]
            try:
                r = sess.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                              params={"fields": s2_fields}, json={"ids": ids},
                              headers=s2_headers, timeout=60)
                if r.status_code == 200:
                    results = r.json()
                    for d, p in zip(batch, results):
                        if not p: continue
                        parsed = s2_parse(p)
                        sel = df["doi_norm"].eq(d)
                        for k,v in parsed.items(): df.loc[sel, k] = v
                elif r.status_code == 429:
                    time.sleep(3); continue
            except Exception as e:
                print("    S2 batch error:", e)
            time.sleep(1.1 if not S2_API_KEY else 0.3)
            if (i//B) % 5 == 0: print(f"    ...{i+len(batch)}/{len(dois)}")

    ## ---- PASS 7: Europe PMC abstract gap-fill ----
    if USE_EUROPEPMC:
        need = df[df["abstract"].astype(str).str.len().lt(40) &
                  (df["doi_norm"].ne("") | df["pmid"].ne(""))]
        print(f"  PASS 7: Europe PMC abstracts — {len(need)} records missing abstract")
        for n,(idx,row) in enumerate(need.iterrows()):
            if row["doi_norm"]:
                q = f'DOI:"{row["doi_norm"]}"'
            else:
                q = f'EXT_ID:{row["pmid"]} AND SRC:MED'
            url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                   f"?query={requests.utils.quote(q)}&resultType=core&format=json&pageSize=1")
            j = oa_get(url); time.sleep(0.1)
            try:
                res = (j or {}).get("resultList",{}).get("result",[])
                if res:
                    r0 = res[0]
                    ab = r0.get("abstractText","") or ""
                    if ab and len(ab) > len(str(row["abstract"])):
                        df.at[idx,"epmc_abstract"] = ab
                    aff = r0.get("affiliation","")
                    if aff: df.at[idx,"epmc_affiliation"] = aff
            except Exception:
                pass
            if n and n % 500 == 0: print(f"    ...{n}/{len(need)}")

    ## ---- OVERWRITE core columns (OpenAlex primary; S2 abstract as fallback) ----
    trust = df["oa_match_method"].isin(["doi","pmid"]) | df["oa_match_confidence"].eq("medium")
    overwrite_map = {
        "title":"oa_title", "authors":"oa_authors",
        "journal":"oa_journal", "document_type":"oa_type", "year":"oa_year",
        "keywords":"oa_keywords",
    }
    for core_col, oa_col in overwrite_map.items():
        if core_col not in df.columns: df[core_col]=""
        df[core_col+"_orig"] = df[core_col]
        can = trust & df[oa_col].ne("")
        df.loc[can, core_col] = df.loc[can, oa_col]
    df["doi_orig"]=df["doi"]
    fill_doi = df["doi_norm"].eq("") & df["oa_doi"].ne("")
    df.loc[fill_doi,"doi"]=df.loc[fill_doi,"oa_doi"]; df.loc[fill_doi,"doi_norm"]=df.loc[fill_doi,"oa_doi"]

    df["abstract_orig"] = df["abstract"]
    def _best_abstract_src(r):
        orig = r.get("abstract_orig","") or ""
        if len(orig) >= 40: return orig, "original"
        options = [("openalex", r.get("oa_abstract","")), ("europepmc", r.get("epmc_abstract","")),
                   ("semantic_scholar", r.get("s2_abstract","")), ("s2_tldr", r.get("s2_tldr",""))]
        options = [(lbl,t) for lbl,t in options if isinstance(t,str) and len(t) >= 40]
        if not options: return orig, ("original" if orig else "none")
        lbl, txt = max(options, key=lambda x: len(x[1]))
        return txt, lbl
    _res = df.apply(_best_abstract_src, axis=1)
    df["abstract"]        = [x[0] for x in _res]
    df["abstract_source"] = [x[1] for x in _res]
    df["affiliations"] = df["oa_affiliations_raw"]
    _m = df["affiliations"].eq("") & df["cr_affiliations"].ne(""); df.loc[_m,"affiliations"] = df.loc[_m,"cr_affiliations"]
    _m = df["affiliations"].eq("") & df["epmc_affiliation"].ne(""); df.loc[_m,"affiliations"] = df.loc[_m,"epmc_affiliation"]
    df["citation_count"] = df["oa_cited_by_count"]
    _m = df["citation_count"].eq("") & df["s2_cited_by_count"].ne(""); df.loc[_m,"citation_count"] = df.loc[_m,"s2_cited_by_count"]
    df["has_funding"]=np.where(df["oa_has_funding"].eq("yes")|df["cr_has_funding"].eq("yes"),"yes",
                      np.where(df["oa_match_method"].ne(""),"no",""))
    df["cited_by_count"] = np.where(df["oa_cited_by_count"].ne(""), df["oa_cited_by_count"], df["s2_cited_by_count"])
    return df


def get_total_rows_and_groups(path):
    """Read the parquet/xlsx schema cheaply to know total row count, and (for
    parquet) the row-group boundaries so we can SEEK directly to a chunk's
    rows instead of scanning the whole file from the start every run (that
    re-scan-from-zero pattern is what made earlier chunked runs grow slower
    each time: chunk 2 re-read+discarded chunk 1's rows, chunk 3 re-read+
    discarded chunks 1 AND 2, etc. Direct row-group access avoids that)."""
    if path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        total = pf.metadata.num_rows
        return total, pf
    else:
        # xlsx has no row-group concept; we must load it once. To still avoid
        # repeated full xlsx reads on every run, we convert it to parquet ONCE
        # the first time this script runs, then always read the parquet copy.
        cache_path = path.rsplit(".",1)[0] + "_asparquet.parquet"
        if not os.path.exists(cache_path):
            print(f"One-time conversion: {os.path.basename(path)} -> parquet cache "
                  f"(so subsequent chunk runs don't reload the whole xlsx every time)")
            tmp = pd.read_excel(path, sheet_name="filtered_master", dtype=str).fillna("")
            tmp.to_parquet(cache_path, index=False)
            del tmp; gc.collect()
        pf = pq.ParquetFile(cache_path)
        return pf.metadata.num_rows, pf


def read_chunk_rows(pf, lo, hi):
    """Read ONLY rows [lo:hi) directly via row-group access (no re-scan)."""
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    starts = np.cumsum([0]+rg_sizes[:-1])
    ends = starts + np.array(rg_sizes)
    wanted = [i for i in range(pf.num_row_groups) if ends[i] > lo and starts[i] < hi]
    table = pf.read_row_groups(wanted)
    df_local = table.to_pandas().fillna("")
    offset = lo - starts[wanted[0]]
    return df_local.iloc[offset: offset + (hi - lo)].reset_index(drop=True)


def run_enrich_one_chunk():
    total, pf = get_total_rows_and_groups(IN_PATH)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total records: {total} -> {n_chunks} chunks of {CHUNK_SIZE}")

    # find the FIRST unfinished chunk (so re-running the cell auto-advances)
    target_ci = None
    for ci in range(n_chunks):
        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{ci:03d}.parquet")
        if not os.path.exists(chunk_path):
            target_ci = ci
            break
    if target_ci is None:
        print("\nAll chunks already enriched. Set MODE='combine' and re-run to build "
              "master_enriched.parquet.")
        return

    lo, hi = target_ci*CHUNK_SIZE, min((target_ci+1)*CHUNK_SIZE, total)
    print(f"\n=== Processing chunk {target_ci+1}/{n_chunks} (rows {lo}-{hi}) ===")
    t0 = time.time()
    chunk_df = read_chunk_rows(pf, lo, hi)
    if MAX_RECORDS: chunk_df = chunk_df.head(MAX_RECORDS)

    enriched = enrich_chunk(chunk_df)

    chunk_path = os.path.join(CHUNK_DIR, f"chunk_{target_ci:03d}.parquet")
    tmp = chunk_path + ".tmp"
    enriched.to_parquet(tmp, index=False)
    os.replace(tmp, chunk_path)   # atomic: a crash mid-write can't corrupt resume state

    matched = (enriched["oa_match_method"].ne("")).sum()
    elapsed = time.time() - t0
    print(f"\n=== Chunk {target_ci+1}/{n_chunks} done: {len(enriched)} records, "
          f"{matched} matched, {elapsed/60:.1f} min ===")
    remaining = n_chunks - target_ci - 1
    print(f"Remaining chunks: {remaining}. Re-run this SAME cell to process the next one "
          f"(it will take roughly the same time as this one — {elapsed/60:.1f} min).")
    if remaining == 0:
        print("\nALL CHUNKS DONE. Set MODE='combine' above and re-run to build the final file.")
    del chunk_df, enriched; gc.collect()


def run_combine():
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.parquet")))
    if not chunk_files:
        raise SystemExit("No chunk files found — run with MODE='enrich' first.")
    print(f"Combining {len(chunk_files)} chunks (streamed, low memory)...")
    ref_cols = pq.ParquetFile(chunk_files[0]).schema.names
    writer = None
    total = 0
    for cf in chunk_files:
        d = pq.read_table(cf).to_pandas()
        for c in ref_cols:
            if c not in d.columns: d[c] = ""
        d = d[ref_cols]
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PATH, tbl.schema)
        writer.write_table(tbl)
        total += len(d)
        print(f"  + {os.path.basename(cf)} ({len(d)} rows, total {total})")
        del d, tbl; gc.collect()
    if writer: writer.close()
    print(f"\nSaved {OUT_PATH} ({total} rows) by streaming.")

    # coverage report, computed via a second light pass (only reads needed columns)
    final = pq.read_table(OUT_PATH).to_pandas()
    def cov(col): return round((final[col].astype(str).str.len()>0).mean()*100,1) if col in final.columns else None
    sci_fields = {
     "title":"title","abstract":"abstract","keywords":"keywords","authors":"authors",
     "author_orcids":"oa_author_orcids","affiliations_raw":"oa_affiliations_raw",
     "institutions":"oa_institutions","institution_ids":"oa_institution_ids",
     "countries":"oa_countries","cited_by_count":"cited_by_count","journal":"journal",
     "issn":"oa_issn","document_type":"document_type","language":"oa_language",
     "funders":"oa_funders","has_funding":"has_funding","year":"year","doi":"doi",
     "openalex_id":"oa_id","pmid":"oa_pmid","topic":"oa_topic","subfield":"oa_subfield",
     "field":"oa_field","domain":"oa_domain","concepts":"oa_concepts","sdgs":"oa_sdgs",
     "oa_status":"upw_oa_status" if USE_UNPAYWALL else "oa_oa_status",
     "s2_abstract":"s2_abstract","s2_tldr":"s2_tldr","s2_fields_of_study":"s2_fields_of_study",
    }
    covtbl = pd.DataFrame(
        [(name, col, cov(col)) for name,col in sci_fields.items() if col in final.columns],
        columns=["field","column","coverage_pct"]).sort_values("coverage_pct",ascending=False)
    abs_src_tbl = final["abstract_source"].value_counts().rename("n").reset_index()
    abs_src_tbl.columns = ["abstract_source","n"]
    match_tbl = final["oa_match_method"].replace("","(unmatched)").value_counts().rename("n").reset_index()
    match_tbl.columns=["match_method","n"]

    print("\n=== ENRICHMENT COVERAGE (% non-empty across all records) ===")
    print(covtbl.to_string(index=False))
    print("\n=== OpenAlex match method ===")
    print(match_tbl.to_string(index=False))
    print("\n=== Final abstract source (what filled each abstract) ===")
    print(abs_src_tbl.to_string(index=False))

    with pd.ExcelWriter(COV_PATH, engine="openpyxl") as xw:
        covtbl.to_excel(xw, sheet_name="coverage", index=False)
        match_tbl.to_excel(xw, sheet_name="match_method", index=False)
        abs_src_tbl.to_excel(xw, sheet_name="abstract_source", index=False)
    print(f"\nSaved {COV_PATH}")
    print("\nNEXT: point 07_dedupe at master_enriched.parquet, and in the canonical")
    print("scorer add metadata-completeness weighting so the richest record wins.")


## ============================================================
##  ENTRY POINT
## ============================================================
if MODE == "enrich":
    run_enrich_one_chunk()
elif MODE == "combine":
    run_combine()
else:
    raise SystemExit(f"Unknown MODE={MODE!r}; use 'enrich' or 'combine'.")
