# ============================================================
#  MASTER SEARCH-RESULTS BUILDER  (Google Colab) — ROBUST EDITION
#  Climate change / Tick-Borne Diseases / Sub-Saharan Africa review
#
#  Hardened to tolerate many real-world scenarios:
#   * ANY number of files per source (1 or many), read & concatenated
#   * MANY formats: .csv .tsv .txt .xls .xlsx .xlsm .csv.gz .zip (multi-member),
#     multi-sheet Excel, semicolon/tab/pipe delimiters, BOM, latin-1/cp1252,
#     preamble lines before the header row
#   * VARIED folder names: case/punctuation/extra words/year suffixes, and
#     one level of nesting (e.g. data/openalex). Falls back to flat files whose
#     NAME matches a source if there is no dedicated subfolder.
#   * SCHEMA DRIFT: if a source renames columns, generic regex patterns still
#     find title/doi/year/abstract/etc. instead of silently returning blanks.
#   * Robust IDs, within-source dedupe of overlapping exports, full diagnostics.
#
#  Output: preprocessing_data/master_search_results_with_ids.xlsx
# ============================================================

## ---- 1. Mount Drive & install deps ----
from google.colab import drive
drive.mount('/content/drive')
!pip -q install openpyxl xlrd==2.0.1 pandas pyarrow

import os, re, csv, sys, glob, zipfile, gzip, io, hashlib
import pandas as pd
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

## ---- 2. CONFIG ----
FOLDER = "/content/drive/MyDrive/ticks"   # <-- EDIT
DEDUPE_WITHIN_SOURCE = True   # remove overlapping rows within a source's multiple files
VERBOSE = True               # print per-file diagnostics
COVERAGE_MIN = 0.15          # drop DB-specific columns below 15% coverage WITHIN their own source
SAVE_PARQUET = True          # primary master format: compact, fast, low-RAM (recommended at scale)
SAVE_XLSX    = False         # also write a spreadsheet? only safe for smaller datasets
XLSX_MAX_ROWS= 200000        # skip xlsx above this many rows to avoid RAM blowups

# search roots: the base folder AND a 'data' subfolder if present (covers both layouts)
SEARCH_ROOTS = [FOLDER]
_data_sub = os.path.join(FOLDER, "data")
if os.path.isdir(_data_sub): SEARCH_ROOTS.append(_data_sub)
OUT  = os.path.join(FOLDER, "preprocessing_data/master_search_results_with_ids.xlsx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
print("Base folder  :", FOLDER)
print("Search roots :", SEARCH_ROOTS)
print("Will write   :", OUT)

## ---- 3. CORE schema + helpers ----
CORE = ['source_database','source_record_id','title','authors','year','doi',
        'journal','abstract','keywords','url','document_type','funding','citation_count']
TABULAR_EXTS  = ('.csv','.tsv','.txt','.tab','.xls','.xlsx','.xlsm')
CONTAINER_EXTS= ('.gz','.zip')

def clean(v):
    if v is None: return ''
    s=str(v).strip()
    return '' if s.lower() in ('nan','none','null','n/a','na') else s

def norm_doi(d):
    d=str(d or '').strip().lower()
    for p in ('https://doi.org/','http://dx.doi.org/','https://dx.doi.org/','doi:'):
        d=d.replace(p,'')
    d=d.strip()
    return d if d.startswith('10.') else ''

def core_row(source, **kw):
    r={c:'' for c in CORE}; r['source_database']=source
    for k in ('source_record_id','title','authors','year','journal','abstract',
              'keywords','url','document_type','funding','citation_count'):
        r[k]=clean(kw.get(k,''))
    r['doi']=norm_doi(kw.get('doi',''))
    return r

def ns(source, d):
    pref=re.sub(r'[^a-z0-9]','',source.lower()); out={}
    for k,v in d.items():
        c=clean(v)
        if c: out[f"{pref}__{re.sub(r'[^0-9A-Za-z]+','_',str(k)).strip('_')}"]=c
    return out

def _norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def _norm_title(t): return re.sub(r"[^a-z0-9]+"," ",str(t or "").lower()).strip()
def _norm_year(y):
    m=re.search(r"(19|20)\d{2}",str(y or "")); return m.group(0) if m else ""
def _h(*parts):
    return hashlib.sha1("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]

## ---- 4. UNIVERSAL FILE READER ----
def _decode(raw):
    for enc in ("utf-8-sig","utf-8","latin-1","cp1252"):
        try: return raw.decode(enc)
        except Exception: continue
    return raw.decode("utf-8","replace")

def _sniff_delimiter(sample):
    try: return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        counts={d:sample.count(d) for d in [",",";","\t","|"]}
        return max(counts,key=counts.get) if max(counts.values())>0 else ","

def _find_header_line(lines, max_scan=15):
    HINTS=('title','doi','author','year','abstract','pmid','eid','lens','source','publication','type')
    best_i,best=0,-1
    for i,ln in enumerate(lines[:max_scan]):
        low=ln.lower()
        delims=sum(low.count(d) for d in [",",";","\t","|"])
        score=sum(1 for h in HINTS if h in low)*10 + (1 if delims>=2 else 0)
        if score>best: best,best_i=score,i
    return best_i if best>0 else 0

def _df_from_text(text, sep_hint=None):
    text=text.replace("\r\n","\n").replace("\r","\n")
    lines=text.split("\n")
    hi=_find_header_line(lines)
    body="\n".join(lines[hi:])
    sep=sep_hint or _sniff_delimiter("\n".join(lines[hi:hi+20]))
    try:
        return pd.read_csv(io.StringIO(body),sep=sep,dtype=str,engine="python",on_bad_lines="skip").fillna("")
    except Exception:
        return pd.read_csv(io.StringIO(body),sep=None,dtype=str,engine="python",on_bad_lines="skip").fillna("")

def read_tabular_any(path):
    """Return list of DataFrames from any supported file (a container may yield several)."""
    low=path.lower()
    if low.endswith(('.xls','.xlsx','.xlsm')):
        engine='xlrd' if low.endswith('.xls') else None
        try:
            xl=pd.read_excel(path,dtype=str,sheet_name=None,engine=engine)
            return [d.fillna("") for d in xl.values() if d is not None and len(d)]
        except Exception:
            try: return [_df_from_text(_decode(open(path,'rb').read()))]   # mislabelled csv/html
            except Exception as e:
                print(f"    [skip] {os.path.basename(path)}: {e}"); return []
    if low.endswith('.gz'):
        try:
            with gzip.open(path,'rb') as f: return [_df_from_text(_decode(f.read()))]
        except Exception as e:
            print(f"    [skip] {os.path.basename(path)}: {e}"); return []
    if low.endswith('.zip'):
        out=[]
        try:
            with zipfile.ZipFile(path) as z:
                for nm in z.namelist():
                    if nm.endswith('/') or not nm.lower().endswith(TABULAR_EXTS): continue
                    raw=z.read(nm)
                    if nm.lower().endswith(('.xls','.xlsx','.xlsm')):
                        try:
                            xl=pd.read_excel(io.BytesIO(raw),dtype=str,sheet_name=None)
                            out+=[d.fillna("") for d in xl.values() if len(d)]
                        except Exception: pass
                    else:
                        out.append(_df_from_text(_decode(raw)))
        except Exception as e:
            print(f"    [skip zip] {os.path.basename(path)}: {e}")
        return out
    try:
        raw=open(path,'rb').read()
        sep='\t' if low.endswith(('.tsv','.tab')) else None
        return [_df_from_text(_decode(raw),sep_hint=sep)]
    except Exception as e:
        print(f"    [skip] {os.path.basename(path)}: {e}"); return []

## ---- 5. ROBUST DISCOVERY ----
SOURCE_ALIASES = {
    'PubMed':              ['pubmed','pub_med','pub med','medline','ncbi'],
    'Europe PMC':          ['europepmc','europe_pmc','europe pmc','epmc','europepubmedcentral'],
    'OpenAlex':            ['openalex','open_alex','open alex'],
    'Lens.org':            ['lens','lens.org','lensorg','lens_org'],
    'Global Index Medicus':['global_index_medicus','global index','gim','index medicus','globalindexmedicus','globalindex'],
    'Scopus':              ['scopus','scop'],
    'Web of Science':      ['wos','web of science','web_of_science','webofscience','woscc','core collection'],
    'Dimensions':          ['dimensions','dimension','digital science'],
}

def discover_source_dir(roots, aliases):
    norm_aliases=[_norm(a) for a in aliases]; cands=[]
    globs=[]
    for base in roots:
        globs += [os.path.join(base,'*'), os.path.join(base,'*','*')]
    for g in globs:
        for p in glob.glob(g):
            if os.path.isdir(p):
                nb=_norm(os.path.basename(p))
                for na in norm_aliases:
                    if na and (na in nb or nb in na):
                        score=3 if nb==na else (2 if nb.startswith(na) or na.startswith(nb) else 1)
                        cands.append((score,-len(nb),p))
    if cands: cands.sort(reverse=True); return cands[0][2]
    return None

def gather_source_files(source_dir, roots=None, aliases=None):
    files=[]
    if source_dir and os.path.isdir(source_dir):
        for ext in TABULAR_EXTS+CONTAINER_EXTS:
            files+=glob.glob(os.path.join(source_dir,'**',f'*{ext}'),recursive=True)
    elif roots and aliases:
        norm_aliases=[_norm(a) for a in aliases]
        for base in roots:
            for ext in TABULAR_EXTS+CONTAINER_EXTS:
                for p in glob.glob(os.path.join(base,f'*{ext}')):
                    nb=_norm(os.path.basename(p))
                    if any(na and na in nb for na in norm_aliases): files.append(p)
    seen=set(); out=[]
    for f in sorted(files):
        bn=os.path.basename(f)
        if bn.startswith(('~$','.')) or bn in seen: continue
        seen.add(bn); out.append(f)
    return out

## ---- 6. RESILIENT COLUMN RESOLUTION ----
GENERIC_FIELD_PATTERNS = {
    'title':         [r'^title$', r'article.?title', r'document.?title', r'display.?name', r'\btitle\b'],
    'doi':           [r'^doi$', r'\bdoi\b', r'digital.?object'],
    'authors':       [r'^authors?$', r'author.?full.?names', r'author.?\(s\)', r'author/s', r'authorships', r'\bauthors?\b', r'creator'],
    'year':          [r'publication.?year', r'^year$', r'pub.?year', r'first.?pub.*date', r'\byear\b'],
    'journal':       [r'source.?title', r'^journal', r'journal.?title', r'\bsource\b', r'host.?venue', r'publisher', r'anthology'],
    'abstract':      [r'^abstract', r'\babstract\b', r'summary'],
    'keywords':      [r'author.?keywords', r'index.?keywords', r'\bkeywords?\b', r'keyword\(s\)', r'descriptor', r'mesh', r'concepts', r'fields.?of.?study'],
    'doctype':       [r'document.?type', r'publication.?type', r'^type$', r'\btype\b', r'resource.?type'],
    'url':           [r'^link$', r'^url$', r'\burl\b', r'landing.?page', r'fulltext.?url', r'external.?url', r'dimensions.?url', r'source.?urls?'],
    'funding':       [r'funding.?details', r'funding.?text', r'funding.?orgs?', r'^funding$', r'\bfunding\b', r'\bgrants?\b', r'grant.?number', r'funder'],
    'citation_count':[r'cited.?by', r'times.?cited', r'citation.?count', r'cited_by_count', r'citing.?works', r'scholarly.?citation'],
    'id':            [r'^eid$', r'lens.?id', r'publication.?id', r'^id$', r'external.?id', r'accession', r'\bpmid\b', r'^ut', r'unique.?id'],
}

def resolve_columns(df_columns, field_aliases):
    cols=list(df_columns); low={c.lower().strip():c for c in cols}
    resolved={}; used=set()
    for field, aliases in field_aliases.items():
        for a in aliases:
            if a.lower() in low: resolved[field]=low[a.lower()]; used.add(low[a.lower()]); break
    for field, patterns in GENERIC_FIELD_PATTERNS.items():
        if resolved.get(field): continue
        for pat in patterns:
            rx=re.compile(pat,re.I)
            cand=[c for c in cols if c not in used and rx.search(c.lower().strip())]
            if cand:
                cand.sort(key=len); resolved[field]=cand[0]; used.add(cand[0]); break
    return resolved, used

def _make_source_deduper():
    seen_id,seen_doi,seen_ti=set(),set(),set()
    def is_dup(sid,doi,title,year):
        i=clean(sid).lower(); d=norm_doi(doi); nt=_norm_title(title); ny=_norm_year(year)
        t=(nt+'|'+ny) if len(nt)>=12 else ''
        dup=(i and i in seen_id) or (d and d in seen_doi) or (t and t in seen_ti)
        if i: seen_id.add(i)
        if d: seen_doi.add(d)
        if t: seen_ti.add(t)
        return bool(dup)
    return is_dup

records=[]
diagnostics=[]   # (source, status, files, rows, note)

## ---- 7. GENERIC SOURCE INGESTION (covers all CSV-like sources) ----
def ingest_source(source_label, field_aliases, special=None):
    aliases=SOURCE_ALIASES.get(source_label,[source_label])
    d=discover_source_dir(SEARCH_ROOTS, aliases)
    files=gather_source_files(d, SEARCH_ROOTS, aliases)
    if not files:
        print(f"{source_label:22s}: no files found (folder aliases tried: {aliases[:3]}...)")
        diagnostics.append((source_label,'MISSING',0,0,'no files')); return
    is_dup=_make_source_deduper() if DEDUPE_WITHIN_SOURCE else None
    n_total=n_skipped=0; per_file=[]
    for path in files:
        frames=read_tabular_any(path)
        if not frames:
            per_file.append((os.path.basename(path),0)); continue
        for df in frames:
            if not len(df): continue
            res,used_cols=resolve_columns(df.columns, field_aliases)
            cmap_used={df.columns.get_loc(c) for c in used_cols if c in df.columns}
            nf=0
            for _,r in df.iterrows():
                g=lambda f: clean(r.get(res[f],'')) if res.get(f) else ''
                sid_native,doi_val,title_val,year_val=g('id'),g('doi'),g('title'),g('year')
                if not (title_val or doi_val or sid_native):     # skip empty rows
                    continue
                if is_dup and is_dup(sid_native,doi_val,title_val,year_val):
                    n_skipped+=1; continue
                row=core_row(source_label,
                    source_record_id=sid_native,title=title_val,authors=g('authors'),
                    year=year_val,doi=doi_val,journal=g('journal'),abstract=g('abstract'),
                    keywords=g('keywords'),document_type=g('doctype'),url=g('url'),
                    funding=g('funding'),citation_count=g('citation_count'))
                if special: special(row, r, res)   # hook for source-specific extras
                extra={c:r[c] for c in df.columns if c not in used_cols and clean(r.get(c,''))}
                row.update(ns(source_label,extra))
                records.append(row); nf+=1; n_total+=1
            per_file.append((os.path.basename(path),nf))
    dnote=f", {n_skipped} within-source dupes removed" if DEDUPE_WITHIN_SOURCE and n_skipped else ""
    locrel=(os.path.relpath(d,FOLDER) if d else 'FLAT')
    print(f"{source_label:22s}: {n_total} rows from {len(files)} file(s) in '{locrel}'{dnote}")
    if VERBOSE:
        for nm,c in per_file: print(f"      {nm}: {c}")
    diagnostics.append((source_label,'OK',len(files),n_total,dnote.strip(', ') or 'concat'))

## ---- 8. PubMed special handling (MEDLINE .txt enrichment merged on PMID) ----
MEDLINE_TAG_NAMES={'PMID':'PMID','PT':'PublicationType','MH':'MeSHTerms','OT':'OtherTerms',
    'GR':'GrantList','AB':'Abstract','TI':'Title','FAU':'FullAuthorNames','AU':'Authors',
    'JT':'JournalTitle','PMC':'PMCID','AID':'ArticleID','LID':'LocationID','LA':'Language',
    'AD':'Affiliation','DP':'PubDate'}
MEDLINE_MULTI={'IS','FAU','AU','AUID','AD','GR','PT','MH','OT','AID','PHST','LID','RN','NM','EC'}

def parse_medline_full(path):
    text=_decode(open(path,'rb').read()); recs={}; cur={}; last=None
    def commit(rec):
        pmid=rec.get('PMID')
        if pmid:
            k=pmid[0] if isinstance(pmid,list) else pmid
            recs[k]={kk:('; '.join(vv) if isinstance(vv,list) else vv) for kk,vv in rec.items()}
    for raw in text.splitlines():
        if not raw.strip(): continue
        m=re.match(r'^([A-Z]{2,4})\s*-\s?(.*)$',raw)
        if m:
            tag,val=m.group(1),m.group(2).rstrip(); last=tag
            if tag=='PMID' and cur: commit(cur); cur={}
            if tag in MEDLINE_MULTI: cur.setdefault(tag,[]).append(val)
            elif tag in cur: cur[tag]=(cur[tag]+[val]) if isinstance(cur[tag],list) else [cur[tag],val]
            else: cur[tag]=val
        else:
            c=raw.strip()
            if last and last in cur:
                if isinstance(cur[last],list): cur[last][-1]+=' '+c
                else: cur[last]+=' '+c
    if cur: commit(cur)
    return recs

def parse_pubmed_abstract_format(path):
    text=_decode(open(path,'rb').read()).replace('\r',''); out={}
    for b in re.split(r'\n\s*\n(?=\d+\.\s)',text):
        pm=re.search(r'^PMID:\s*(\d+)',b,re.M)
        if not pm: continue
        dm=re.search(r'\bDOI:\s*(\S+)',b)
        paras=re.split(r'\n\s*\n',b)
        cand=[p.replace('\n',' ').strip() for p in paras if len(p)>200 and 'Author information' not in p]
        out[pm.group(1)]={'AB':max(cand,key=len) if cand else '','LID':(dm.group(1) if dm else '')}
    return out

def parse_pubmed_txt(path):
    head=_decode(open(path,'rb').read()[:2000])
    if re.search(r'^PMID-\s',head,re.M) or re.search(r'^AB\s+-\s',head,re.M):
        return parse_medline_full(path)
    return parse_pubmed_abstract_format(path)

def med_doi(rec):
    for fld in ('LID','AID'):
        for piece in str(rec.get(fld,'')).split(';'):
            mm=re.search(r'(10\.\S+?)\s*\[doi\]',piece)
            if mm: return mm.group(1)
    return ''

def ingest_pubmed():
    aliases=SOURCE_ALIASES['PubMed']
    d=discover_source_dir(SEARCH_ROOTS,aliases)
    files=gather_source_files(d,SEARCH_ROOTS,aliases)
    if not files:
        print(f"{'PubMed':22s}: no files found"); diagnostics.append(('PubMed','MISSING',0,0,'no files')); return
    txts=[f for f in files if f.lower().endswith('.txt')]
    csvs=[f for f in files if f.lower().endswith(('.csv','.tsv'))]
    med={}
    for t in txts:
        try: med.update(parse_pubmed_txt(t))
        except Exception as e: print(f"    [pubmed txt skip] {os.path.basename(t)}: {e}")
    seen=set(); n=0; per_file=[]
    pm_aliases=dict(id=['pmid'],title=['title'],authors=['authors'],year=['publication year','year'],
        doi=['doi'],journal=['journal/book','journal'])
    for cpath in csvs:
        for df in read_tabular_any(cpath):
            res,_=resolve_columns(df.columns, pm_aliases); nf=0
            for _,r in df.iterrows():
                pmid=clean(r.get(res.get('id',''),''))
                if pmid and pmid in seen: continue
                if pmid: seen.add(pmid)
                rec=med.get(pmid,{})
                kw='; '.join(x for x in [rec.get('MH',''),rec.get('OT','')] if x)
                row=core_row('PubMed',source_record_id=pmid,
                    title=clean(r.get(res.get('title',''),'')) or rec.get('TI',''),
                    authors=clean(r.get(res.get('authors',''),'')) or rec.get('FAU','') or rec.get('AU',''),
                    year=clean(r.get(res.get('year',''),'')) or _norm_year(rec.get('DP','')),
                    doi=clean(r.get(res.get('doi',''),'')) or med_doi(rec),
                    journal=clean(r.get(res.get('journal',''),'')) or rec.get('JT',''),
                    abstract=rec.get('AB',''),keywords=kw,document_type=rec.get('PT',''),
                    funding=rec.get('GR',''),url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else '')
                extra={MEDLINE_TAG_NAMES.get(t2,t2):v for t2,v in rec.items() if t2 not in ('AB','TI')}
                extra['PMCID']=rec.get('PMC','')
                row.update(ns('PubMed',extra)); records.append(row); n+=1; nf+=1
            per_file.append((os.path.basename(cpath),nf))
    med_only=0
    for pmid,rec in med.items():
        if pmid in seen: continue
        kw='; '.join(x for x in [rec.get('MH',''),rec.get('OT','')] if x)
        row=core_row('PubMed',source_record_id=pmid,title=rec.get('TI',''),
            authors=rec.get('FAU','') or rec.get('AU',''),doi=med_doi(rec),journal=rec.get('JT',''),
            abstract=rec.get('AB',''),keywords=kw,document_type=rec.get('PT',''),funding=rec.get('GR',''),
            year=_norm_year(rec.get('DP','')),url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        extra={MEDLINE_TAG_NAMES.get(t2,t2):v for t2,v in rec.items() if t2 not in ('AB','TI')}
        extra['PMCID']=rec.get('PMC',''); row.update(ns('PubMed',extra)); records.append(row); med_only+=1
    print(f"{'PubMed':22s}: {len(seen)} PMIDs from {len(csvs)} CSV(s) + {med_only} MEDLINE-only "
          f"({len(txts)} txt, {sum(1 for p in med.values() if p.get('AB'))} abstracts)")
    if VERBOSE:
        for nm,c in per_file: print(f"      {nm}: {c}")
    diagnostics.append(('PubMed','OK',len(files),len(seen)+med_only,f'{len(txts)} MEDLINE txt merged'))

## ---- 9. RUN ALL SOURCES ----
ingest_pubmed()
ingest_source('Europe PMC', dict(id=['external_id','pmid'],title=['title'],authors=['authors','author string'],
    year=['first_pub_date (dd/mm/yyyy)','publication_year'],doi=['doi'],journal=['journal','source'],
    abstract=['abstract'],doctype=['publication type'],citation_count=['times_cited'],funding=['grants']))
ingest_source('OpenAlex', dict(id=['id'],title=['title','display_name'],authors=['authors','authorships'],
    year=['publication_year'],doi=['doi'],journal=['host_venue','source'],abstract=['abstract'],
    keywords=['concepts','keywords'],doctype=['type'],url=['url'],citation_count=['cited_by_count'],funding=['funder']))
ingest_source('Lens.org', dict(id=['lens id','lens_id'],title=['title'],authors=['author/s','authors'],
    year=['publication year'],doi=['doi'],journal=['source title','publisher'],abstract=['abstract'],
    keywords=['keywords','fields of study'],doctype=['publication type'],url=['external url'],
    citation_count=['scholarly citation count','citing works count'],funding=['funding details']))
ingest_source('Global Index Medicus', dict(id=['id','accession'],title=['title'],authors=['authors'],
    year=['publication year'],doi=['doi'],journal=['journal'],abstract=['abstract'],
    keywords=['keyword(s)','descriptor(s)'],doctype=['type'],url=['fulltext url']))
ingest_source('Scopus', dict(id=['eid'],title=['title'],authors=['author full names','authors'],
    year=['year'],doi=['doi'],journal=['source title'],abstract=['abstract'],
    keywords=['author keywords','index keywords'],doctype=['document type'],url=['link'],
    citation_count=['cited by'],funding=['funding details','funding text 1']))
ingest_source('Web of Science', dict(id=['ut (unique wos id)','ut (unique id)','ut'],
    title=['article title'],authors=['authors','author full names'],year=['publication year'],
    doi=['doi'],journal=['source title'],abstract=['abstract'],
    keywords=['author keywords','keywords plus'],doctype=['document type'],
    citation_count=['times cited, all databases','times cited, wos core'],
    funding=['funding orgs','funding name preferred','grant number']))
ingest_source('Dimensions', dict(id=['publication id'],title=['title'],authors=['authors'],
    year=['pubyear','publication year'],doi=['doi'],journal=['source title','anthology title'],
    abstract=['abstract'],keywords=['mesh terms'],doctype=['publication type'],
    url=['dimensions url'],citation_count=['times cited'],funding=['funding']))

## ---- 10. Assemble + IDs ----
if not records:
    raise SystemExit("No records ingested from ANY source. Check FOLDER path and that "
                     "exports exist. See the per-source 'no files found' messages above.")

df=pd.DataFrame(records).fillna('')
core_present=[c for c in CORE if c in df.columns]
extra_cols=sorted(c for c in df.columns if c not in core_present)
df=df[core_present+extra_cols]

# IDs
seen,src_ids={}, []
DB_CODE={"PubMed":"PMID","Europe PMC":"EPMC","OpenAlex":"OPENALEX","Lens.org":"LENS",
    "Global Index Medicus":"GIM","Scopus":"SCOPUS","Web of Science":"WOS","Dimensions":"DIM"}
for i,row in df.iterrows():
    code=DB_CODE.get(row.get("source_database",""),
        re.sub(r"[^A-Z0-9]","",str(row.get("source_database","")).upper())[:8] or "SRC")
    native=re.sub(r"^https?://\S+/","",str(row.get("source_record_id","")).strip())
    if not native or native.lower() in ("nan","none"):
        native="h"+_h(row.get("source_database",""),row.get("title",""),row.get("doi",""),row.get("year",""),i)
    sid=f"{code}:{native}"; seen[sid]=seen.get(sid,-1)+1
    if seen[sid]: sid=f"{sid}#{seen[sid]}"
    src_ids.append(sid)
df["source_record_id"]=src_ids
key_to_stu,stu_ids,c={}, [],0
for _,row in df.iterrows():
    doi=norm_doi(row.get("doi",""))
    if doi: key=("doi",doi)
    else:
        nt,ny=_norm_title(row.get("title","")),_norm_year(row.get("year",""))
        key=("ti",f"{nt}|{ny}") if len(nt)>=12 else None
    if key is None: c+=1; stu_ids.append(f"STU-{c:06d}")
    else:
        if key not in key_to_stu: c+=1; key_to_stu[key]=f"STU-{c:06d}"
        stu_ids.append(key_to_stu[key])
df["study_id"]=stu_ids
df.insert(0,"record_id",[f"REC-{i+1:06d}" for i in range(len(df))])
front=["record_id","study_id","source_record_id","source_database"]
df=df[front+[c for c in df.columns if c not in front]]

n_rows=len(df); n_studies=df['study_id'].nunique()
ILLEGAL=re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
df=df.map(lambda v: ILLEGAL.sub(' ',str(v))[:32000])

## ---- 11. Diagnostics + prune + memory-safe save ----
def prune_low_coverage_columns(df, core_cols, coverage_min=COVERAGE_MIN, verbose=True):
    keep=list(core_cols); dropped=[]
    src_to_prefix={s:re.sub(r'[^a-z0-9]','',s.lower()) for s in df['source_database'].unique()}
    prefix_to_mask={pre:(df['source_database']==src) for src,pre in src_to_prefix.items()}
    for col in df.columns:
        if col in keep: continue
        if '__' not in col: keep.append(col); continue
        prefix=col.split('__',1)[0]; mask=prefix_to_mask.get(prefix)
        if mask is None or mask.sum()==0:
            cov=(df[col].astype(str).str.len()>0).mean()
        else:
            cov=(df.loc[mask,col].astype(str).str.len()>0).mean()
        (keep.append(col) if cov>=coverage_min else dropped.append((col,round(cov*100,2))))
    report=pd.DataFrame(dropped,columns=['column','coverage_pct_within_source']).sort_values('coverage_pct_within_source')
    if verbose:
        print(f"\nColumn pruning (<{coverage_min*100:.0f}% within-source coverage): "
              f"kept {len(keep)}, dropped {len(dropped)} (core always kept)")
    return df[keep], report

## ---- 11. Assemble, IDs already added above; now diagnostics ----
print("\n=== INGESTION DIAGNOSTICS ===")
print(f"{'SOURCE':22s} {'STATUS':8s} {'FILES':>5s} {'ROWS':>7s}  NOTE")
for s,st,f,r,note in diagnostics:
    print(f"{s:22s} {st:8s} {f:5d} {r:7d}  {note}")

# free memory: downcast object columns already strings; drop helper cols if any
import gc
df = df.copy()
gc.collect()

# ---- prune low-coverage DB-specific columns ----
_front = ["record_id","study_id","source_record_id","source_database"]
core_keep = list(dict.fromkeys(_front + [c for c in CORE if c in df.columns]))  # dedupe, keep order
# also guard against any duplicate columns in df itself
df = df.loc[:, ~df.columns.duplicated()]
df, prune_report = prune_low_coverage_columns(df, core_keep, COVERAGE_MIN)
print(f"Final shape: {len(df)} rows x {len(df.columns)} columns")

# sanitise illegal chars + cap cell length (cheap, in place per column to limit RAM spikes)
ILLEGAL=re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
for c in df.columns:
    df[c]=df[c].map(lambda v: ILLEGAL.sub(' ',str(v))[:32000])

n_rows=len(df); n_studies=df['study_id'].nunique()

## ---- 12. SAVE (Parquet primary, optional slim XLSX) ----
OUT_PARQUET = os.path.join(os.path.dirname(OUT), "master_search_results_with_ids.parquet")
OUT_DIAG    = os.path.join(os.path.dirname(OUT), "build_diagnostics.csv")
OUT_PRUNE   = os.path.join(os.path.dirname(OUT), "pruned_columns.csv")

# always write the small companion CSVs (tiny)
pd.DataFrame(diagnostics,columns=['source','status','files','rows','note']).to_csv(OUT_DIAG,index=False)
prune_report.to_csv(OUT_PRUNE,index=False)

if SAVE_PARQUET:
    try:
        df.to_parquet(OUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
        sz=os.path.getsize(OUT_PARQUET)/1e6
        print(f"\nSaved PARQUET master: {OUT_PARQUET}  ({sz:.1f} MB, {n_rows} rows x {len(df.columns)} cols)")
    except Exception as e:
        print("Parquet save failed:", e)
        # fallback: write compressed CSV in chunks (very low RAM)
        OUT_CSVGZ=OUT_PARQUET.replace('.parquet','.csv.gz')
        df.to_csv(OUT_CSVGZ, index=False, compression='gzip')
        print(f"Fell back to gzipped CSV: {OUT_CSVGZ}")

if SAVE_XLSX:
    if n_rows>XLSX_MAX_ROWS:
        print(f"\nSkipping XLSX: {n_rows} rows exceeds XLSX_MAX_ROWS={XLSX_MAX_ROWS} (would risk RAM). "
              f"Use the Parquet file; convert a subset to xlsx later if needed.")
    else:
        with pd.ExcelWriter(OUT, engine='openpyxl') as xw:
            counts=df['source_database'].value_counts()
            readme=pd.DataFrame({'README':[
                'MASTER (robust builder, pruned). Parquet is the primary file.',
                f'Low-coverage DB-specific columns (<{COVERAGE_MIN*100:.0f}% within source) dropped.',
                '','Counts:'] + [f'  {s}: {c}' for s,c in counts.items()] +
                ['',f'TOTAL rows: {n_rows}',f'UNIQUE studies: {n_studies}',f'COLUMNS: {len(df.columns)}']})
            readme.to_excel(xw,sheet_name='README',index=False)
            df.to_excel(xw,sheet_name='master_data',index=False)
        print(f"\nSaved XLSX master: {OUT}")

print(f"\nDONE. rows={n_rows} studies={n_studies} cols={len(df.columns)}")
print(f"Diagnostics: {OUT_DIAG}\nPruned columns log: {OUT_PRUNE}")
print("\nDOWNSTREAM: change pd.read_excel(...,'master_data') -> pd.read_parquet(OUT_PARQUET) in the next scripts.")
