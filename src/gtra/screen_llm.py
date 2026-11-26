"""
GTRA LLM screening  (gtra-screen-v1)
====================================

Decides, for every canonical record, WHAT was studied in relation to ticks, then
derives the GTRA inclusion decision in code.

Built from three earlier pipelines:
  * GenAI facet-1 engine (facet1_annotate v5): OpenAI-compatible adapter, cache keyed
    (oa_id, provider, prompt_version), failures never cached, the model echoes the
    record number and misaligned batches are retried then split, verbatim-evidence
    check, two strong models + tie-break (no adjudicator), agree/split accuracy.
  * genai_screen v3: the model names a TYPE; the binary decision is derived in code,
    so moving a line later is a free replay over the cache; rule flags never exclude.
  * tick_classify (old tick pipeline): the object-of-study taxonomy, which already
    matches the GTRA scope, plus the facets (tick_examined, field collection, setting,
    taxa).

Scope (decided 2026-11-25): include research where ticks, tick parts or tick-derived
material, or treatments / repellents on ticks, are studied. Distribution or climate
models built from tick records are included. Reviews are included and flagged.
Disease-only studies are excluded but kept as a flagged layer.

Colab:
    import gtra.screen_llm as G
    G.check_providers()
    df = G.prepare(works_canonical)            # rule flags + coding_abstract
    sheet = G.write_pilot_sheet(G.draw_pilot(df, 300), OUT)
    res = G.screen(df_sample)                  # cached, resumable
    G.score(sheet, res)                        # after you coded the sheet
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================================== #
# CONFIG                                                                      #
# =========================================================================== #
PROMPT_VERSION = "gtra-screen-v1"   # BUMP when SYSTEM_PROMPT or CLASSES change
LOCAL_CACHE = "/content/gtra_screen_cache.jsonl"
DRIVE_CACHE = None                  # set to a Drive path in the notebook

BATCH = 6                # records per call (reasoning models: keep small)
WORKERS = 5
TITLE_CHARS = 500
ABS_CHARS = 3500
EVIDENCE_WORDS = 30

PROVIDERS = {
    "gpt_6_luna": {"env": "OPENAI_API_KEY", "base_url": None, "model": "gpt-6-luna",
                   "max_tokens": None, "temperature": None},
    "deepseek": {"env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com",
                 "model": "deepseek-v4-flash", "max_tokens": 6000, "temperature": 0},
    # spare, low-cost options for a sensitivity run; not used by default
    "gpt_5_nano": {"env": "OPENAI_API_KEY", "base_url": None, "model": "gpt-5-nano",
                   "max_tokens": None, "temperature": None},
    "gpt": {"env": "OPENAI_API_KEY", "base_url": None, "model": "gpt-4o-mini",
            "max_tokens": 6000, "temperature": 0},
}
# GenAI facet-1 pilot 2 (n=500): gpt_6_luna kappa 0.885, deepseek 0.840, cheaper models ~0.59;
# an adjudicator was worse than the tie-break. Same design here.
PRIMARY = ["gpt_6_luna", "deepseek"]
TIEBREAK = "gpt_6_luna"

CLASSES = ["tick_organism", "tick_derived", "pathogen_in_tick", "tick_control",
           "host_infestation", "tick_distribution_model", "disease_only", "not_tick", "uncertain"]
TICK_CLASSES = CLASSES[:6]          # -> gtra_include
SETTINGS = ["field", "laboratory", "clinical", "modelling", "review", "mixed", "none"]
CONFIDENCE = {"high", "medium", "low"}


# =========================================================================== #
# THE PROMPT                                                                  #
# =========================================================================== #
SYSTEM_PROMPT = """You screen scientific records for the Global Tick Research Atlas, a science-of-science study of TICK research worldwide. The corpus was built by keyword search, so it contains disease papers with no tick, papers about mites, and other meanings of the word "tick". For each record decide what was actually STUDIED in relation to ticks. Records may be in any language; read them as they are.

=== CLASSES: assign EXACTLY ONE. If several fit, the LOWEST NUMBER wins ===
1 tick_organism: a tick (any stage) is the direct object: taxonomy, new species, morphology, genome, physiology, behaviour, ecology, phenology, field surveys and collections, tick microbiome, colonies, feeding experiments, xenodiagnosis.
2 tick_derived: material or molecules FROM ticks: saliva and salivary proteins, salivary glands, midgut, cuticle, haemolymph, tick genes or proteins, tick cell lines (IDE8, ISE6, BME26), tick antigens and anti-tick vaccines (Bm86, subolesin).
3 pathogen_in_tick: a pathogen or microbe is detected, isolated or characterised IN or FROM ticks (the tick is the sample): PCR screening of ticks, infection prevalence in ticks, isolation from ticks, vector competence with real ticks.
4 tick_control: treatments aimed at ticks: acaricides and acaricide resistance, repellents, plant extracts, biological control (fungi, nematodes), vaccines or treatments judged by tick burden, integrated tick management. Includes studies of tick acaricides (residues, toxicity, application) when ticks are the declared target.
5 host_infestation: the host (animal or human) is the unit AND ticks on it are examined, counted or identified: tick burden on cattle, infestation prevalence in dogs, tick-bite cases where the tick was identified.
6 tick_distribution_model: ticks are the object but no ticks were handled in this study: distribution, niche or suitability models built from tick records, climate projections for ticks, risk maps of tick presence, meta-analyses of tick occurrence.
7 disease_only: a tick-borne disease or pathogen studied with NO tick examined, derived or modelled: patients, clinical trials, diagnostics, serosurveys of humans or animals, outbreak reports, pathogen work in culture or lab animals without ticks, pathogen vaccines tested in hosts, alpha-gal clinical studies without tick material.
8 not_tick: not about ticks: mites only (Varroa, spider mites, dust mites, Demodex, chiggers), acaricides used only against mites, or another meaning of "tick".
9 uncertain: the title and abstract are genuinely not enough to decide.

=== OTHER MEANINGS OF "TICK" (class not_tick) ===
"tick size", "tick data", "tick-by-tick" and "uptick" in finance and trading; "tick trefoil" (Desmodium, a plant); a tick box, tick mark or checklist; a clock tick; "ticking" fabric; "tic" (a movement disorder); a person or place named Tick.

=== BOUNDARY RULES ===
- Pathogen sampled FROM ticks -> pathogen_in_tick. Only hosts sampled, ticks counted -> host_infestation. Neither -> disease_only.
- Ticks collected from hosts and tested for pathogens -> pathogen_in_tick.
- Tick cell lines or tick tissue in vitro -> tick_derived.
- Antigen FROM a tick -> tick_derived. Vaccine against the PATHOGEN, tested in hosts -> disease_only.
- Tick-bite case WITH the tick identified -> host_infestation; without -> disease_only.
- A model of DISEASE cases (human incidence) with no tick records -> disease_only. A model built on tick records -> tick_distribution_model.
- Reviews: classify by the OBJECT reviewed (a review of acaricide resistance -> tick_control; a review of Lyme disease treatment -> disease_only) and set "review": true.
- A paper about mites AND ticks -> classify by the tick part.
- Do not infer beyond the text. When nothing says ticks were collected, field_collection is "unclear".

=== OUTPUT ===
Return ONLY a JSON array, one object per input record, in the SAME order:
[{"i":1,"class":"pathogen_in_tick","review":false,"tick_examined":true,"field_collection":"yes","setting":"field","tick_taxa":["Ixodes ricinus"],"evidence":"...","confidence":"high","rationale":"..."}]
- "i": the record number exactly as given. Required.
- "class": one value from the list.
- "review": true for any review, systematic review, meta-analysis or book chapter that summarises literature; else false.
- "tick_examined": true only if physical ticks or tick material were handled or analysed in this study.
- "field_collection": "yes", "no" or "unclear".
- "setting": one of field, laboratory, clinical, modelling, review, mixed, none.
- "tick_taxa": tick genus or species names written in the record (may be empty).
- "evidence": a VERBATIM span copied from the title or abstract (max 30 words) that decided the class. Copy exactly.
- "confidence": "high", "medium" or "low". Use low honestly.
- "rationale": max 25 words.
No prose, no markdown fences, no extra keys."""


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
def S(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x)


def short_id(x) -> str:
    return S(x).rsplit("/", 1)[-1]


def _norm_text(t: str) -> str:
    t = S(t).lower()
    t = re.sub(r"[‘’“”]", "'", t)
    t = re.sub(r"[‐-―]", "-", t)
    return re.sub(r"\s+", " ", t).strip()


def evidence_ok(evidence: str, title: str, abstract: str) -> bool:
    """True when the quote is verbatim (pieces between '...' must appear in order)."""
    ev = _norm_text(evidence).replace("…", "...")
    ev = re.sub(r"[()\[\]]", "", ev)
    pieces = [p.strip(" .\"'") for p in re.split(r"\.{3,}", ev)]
    pieces = [p for p in pieces if p]
    if not pieces or sum(len(p) for p in pieces) < 8:
        return False
    text = re.sub(r"[()\[\]]", "", _norm_text(f"{title} {abstract}"))
    pos = 0
    for p in pieces:
        i = text.find(p, pos)
        if i == -1:
            return False
        pos = i + len(p)
    return True


def _abstract_of(rec) -> str:
    for c in ("coding_abstract", "oa_abstract", "abstract"):
        v = S(rec.get(c))
        if v:
            return v
    return ""


# =========================================================================== #
# RULE LAYER: flags only, nothing is ever excluded                            #
# =========================================================================== #
GENERA = ["ixodes", "amblyomma", "rhipicephalus", "boophilus", "dermacentor", "haemaphysalis",
          "hyalomma", "ornithodoros", "argas", "otobius", "carios", "nuttalliella", "rhipicentor",
          "margaropus", "anomalohimalaya", "cosmiomma", "nosomma", "bothriocroton", "antricola",
          "nothoaspis", "aponomma"]
FAMILIES = ["ixodidae", "argasidae", "ixodida", "nuttalliellidae", "ixodid", "argasid"]
WORDS = ["ticks?", "tick[- ]?borne", "tique?s?", "carrapatos?", "garrapatas?", "zecche", "zecca",
         "zecken?", "acaricid\\w*", "kleszcz\\w*", "клещ\\w*", "蜱"]
_TICK_RE = re.compile(r"(?<!\w)(?:" + "|".join(GENERA + FAMILIES + WORDS) + r")(?!\w)", re.I)
_TAXON_RE = re.compile(r"(?<!\w)(?:" + "|".join(GENERA + FAMILIES) + r")(?!\w)", re.I)

NOISE_RULES = {
    "finance": r"\b(tick size|tick data|tick[- ]by[- ]tick|uptick|high[- ]frequency trad\w*|order book|"
               r"stock market|limit order|bid[- ]ask|nasdaq)\b",
    "plant_trefoil": r"\b(tick[- ]trefoil|desmodium|tick[- ]?clover)\b",
    "checkbox": r"\b(tick[- ]box\w*|tick mark|checklist tick|tick the box)\b",
    "mites_only": r"\b(varroa|tetranychus|spider mites?|dust mites?|demodex|sarcoptes|psoroptes|"
                  r"chiggers?|dermanyssus|panonychus|eriophy\w*)\b",
    "tic_disorder": r"\b(tic disorders?|tourette)\b",
    "archaeology": r"\b(burial|archaeolog\w*|excavation|pottery|potsherd)\b",
}
_NOISE = {k: re.compile(v, re.I) for k, v in NOISE_RULES.items()}


def rule_flags(rec) -> dict:
    """Where the tick signal sits + noise flags. Used to stratify and audit, never to drop."""
    t, a = S(rec.get("oa_title")), _abstract_of(rec)
    kw = " ".join(S(rec.get(c)) for c in ("oa_keywords", "oa_mesh", "oa_topic", "oa_concepts"))
    if _TICK_RE.search(t):
        where = "title"
    elif _TICK_RE.search(a):
        where = "abstract"
    elif _TICK_RE.search(kw):
        where = "keywords_only"
    else:
        where = "none"
    text = f"{t} {a}"
    noise = [k for k, rx in _NOISE.items() if rx.search(text)]
    taxon = bool(_TAXON_RE.search(text))
    if taxon:   # a tick genus overrides mites/archaeology hints; finance & plant still flagged
        noise = [n for n in noise if n not in ("mites_only", "archaeology")]
    return {"rule_tick_where": where, "rule_taxon": taxon, "rule_noise": "|".join(noise)}


def prepare(df: pd.DataFrame, abstract_extra: dict | None = None) -> pd.DataFrame:
    """Adds coding_abstract (+ source), input level, and rule flags."""
    out = df.copy()
    out["oa_id"] = out["oa_id"].map(short_id)
    ab = out.get("oa_abstract", pd.Series("", index=out.index)).fillna("").astype(str)
    src = np.where(ab.str.len() >= 50, "openalex", "")
    if abstract_extra:
        pm = out["oa_id"].map(abstract_extra).fillna("")
        use = (ab.str.len() < 50) & (pm.str.len() >= 50)
        ab = ab.where(~use, pm)
        src = np.where(use, "pubmed", src)
    out["coding_abstract"] = ab.where(ab.str.len() >= 50, "")
    out["coding_abstract_from"] = src
    out["screen_input"] = np.where(out["coding_abstract"].str.len() > 0, "title_abstract", "title_only")
    flags = pd.DataFrame([rule_flags(r) for r in out[["oa_title", "coding_abstract"] +
                          [c for c in ("oa_keywords", "oa_mesh", "oa_topic", "oa_concepts")
                           if c in out.columns]].to_dict("records")], index=out.index)
    return pd.concat([out, flags], axis=1)


# =========================================================================== #
# PubMed abstract fill (records without an OpenAlex abstract)                 #
# =========================================================================== #
def fetch_pubmed_abstracts(pmids, api_key: str | None = None, email: str = "cognosxx@gmail.com",
                           chunk: int = 200, pause: float = 0.12) -> dict[str, str]:
    """{pmid: abstract} via NCBI E-utilities efetch. ~10 req/s with an API key, 3 without."""
    import requests
    import xml.etree.ElementTree as ET
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    pmids = [p for p in dict.fromkeys(S(x).strip() for x in pmids) if p.isdigit()]
    out: dict[str, str] = {}
    for i in range(0, len(pmids), chunk):
        ids = pmids[i:i + chunk]
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "gtra", "email": email}
        if api_key:
            params["api_key"] = api_key
        for attempt in range(5):
            try:
                r = requests.post(url, data=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    continue
                r.raise_for_status()
                root = ET.fromstring(r.content)
                for art in root.iter("PubmedArticle"):
                    pid = S(art.findtext(".//MedlineCitation/PMID"))
                    parts = []
                    for t in art.iter("AbstractText"):
                        txt = "".join(t.itertext()).strip()
                        lab = t.get("Label")
                        parts.append(f"{lab}: {txt}" if lab else txt)
                    if pid and parts:
                        out[pid] = " ".join(parts)
                break
            except Exception as e:  # pragma: no cover - network
                if attempt == 4:
                    print(f"[pubmed] chunk {i}: {e}", file=sys.stderr)
                time.sleep(1 + attempt)
        time.sleep(pause if api_key else 0.35)
        if (i // chunk) % 20 == 0:
            print(f"[pubmed] {min(i + chunk, len(pmids)):,}/{len(pmids):,} -> {len(out):,} abstracts",
                  file=sys.stderr)
    return out


# =========================================================================== #
# provider adapter                                                            #
# =========================================================================== #
_CLIENTS: dict[str, object] = {}
_LOCK = threading.Lock()


def client_for(provider: str):
    with _LOCK:
        if provider not in _CLIENTS:
            from openai import OpenAI
            cfg = PROVIDERS[provider]
            key = os.environ.get(cfg["env"])
            if not key:
                raise RuntimeError(f"provider {provider!r}: {cfg['env']} is not set")
            kw = {"api_key": key}
            if cfg.get("base_url"):
                kw["base_url"] = cfg["base_url"]
            _CLIENTS[provider] = OpenAI(**kw)
        return _CLIENTS[provider]


def check_providers(providers=None) -> dict[str, str]:
    out = {}
    for p in providers or PRIMARY:
        model = PROVIDERS[p]["model"]
        try:
            r = client_for(p).chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with the single word OK."}])
            body = (r.choices[0].message.content or "").strip()
            out[p] = f"OK ({model}) -> {body[:30]!r}" if body else f"EMPTY ({model})"
        except Exception as e:
            out[p] = f"FAIL ({model}): {type(e).__name__}: {e}"
        print(f"[provider] {p:<11} {out[p]}", file=sys.stderr)
    return out


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


def _parse_array(txt: str):
    txt = _FENCE.sub("", S(txt).strip())
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        a, b = txt.find("["), txt.rfind("]")
        if a != -1 and b > a:
            try:
                return json.loads(txt[a:b + 1])
            except json.JSONDecodeError:
                return None
    return None


def _bool(x) -> bool | None:
    if isinstance(x, bool):
        return x
    s = S(x).strip().lower()
    return True if s in {"true", "yes", "1"} else False if s in {"false", "no", "0"} else None


def _norm_one(o, rec) -> dict | None:
    if not isinstance(o, dict):
        return None
    cls = S(o.get("class")).strip().lower().replace(" ", "_").replace("-", "_")
    if cls not in CLASSES:
        return None
    fc = S(o.get("field_collection")).strip().lower()
    fc = fc if fc in {"yes", "no", "unclear"} else "unclear"
    st = S(o.get("setting")).strip().lower()
    st = st if st in SETTINGS else "none"
    taxa = o.get("tick_taxa")
    taxa = [S(t).strip() for t in taxa if S(t).strip()] if isinstance(taxa, list) else []
    conf = S(o.get("confidence")).strip().lower()
    ev = " ".join(S(o.get("evidence")).split()[:EVIDENCE_WORDS])
    return {
        "class": cls,
        "review": bool(_bool(o.get("review"))),
        "tick_examined": _bool(o.get("tick_examined")),
        "field_collection": fc,
        "setting": st,
        "tick_taxa": taxa[:10],
        "evidence": ev,
        "evidence_ok": evidence_ok(ev, rec.get("oa_title"), _abstract_of(rec)),
        "confidence": conf if conf in CONFIDENCE else "low",
        "rationale": S(o.get("rationale"))[:300],
    }


def _user_msg(batch) -> str:
    parts = []
    for i, r in enumerate(batch, 1):
        ab = _abstract_of(r)[:ABS_CHARS] or "[no abstract available]"
        kw = S(r.get("oa_keywords"))[:300]
        parts.append(f"{i}. TITLE: {S(r.get('oa_title'))[:TITLE_CHARS]}\nABSTRACT: {ab}"
                     + (f"\nKEYWORDS: {kw}" if kw else ""))
    return "Screen each record:\n\n" + "\n\n".join(parts)


_COMPLETE = None   # test hook: function(provider, system, user) -> str


def _complete(provider: str, user: str) -> str:
    if _COMPLETE is not None:
        return _COMPLETE(provider, SYSTEM_PROMPT, user)
    cfg = PROVIDERS[provider]
    kw = {}
    if cfg.get("temperature") is not None:
        kw["temperature"] = cfg["temperature"]
    if cfg.get("max_tokens") is not None:
        kw["max_tokens"] = cfg["max_tokens"]
    resp = client_for(provider).chat.completions.create(
        model=cfg["model"], messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": user}], **kw)
    return resp.choices[0].message.content or ""


def call_batch(provider: str, batch, max_retries: int = 4) -> list[dict | None]:
    """Same length as batch; None = failure (never cached, retried on the next run)."""
    n = len(batch)
    user = _user_msg(batch)
    for attempt in range(max_retries):
        try:
            arr = _parse_array(_complete(provider, user))
            if not isinstance(arr, list):
                time.sleep(1 + attempt)
                continue
            out: list[dict | None] = [None] * n
            for pos, obj in enumerate(arr):
                if not isinstance(obj, dict):
                    continue
                try:
                    idx = int(obj.get("i", pos + 1)) - 1
                except (TypeError, ValueError):
                    idx = pos
                if 0 <= idx < n and out[idx] is None:
                    out[idx] = _norm_one(obj, batch[idx])
            got = sum(o is not None for o in out)
            if got == n or (got >= n - 1 and attempt >= 1):
                return out
            time.sleep(1 + attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [{provider}] giving up on batch: {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr)
            time.sleep(min(2 ** attempt, 30))
    if n > 1:
        return [call_batch(provider, [r], max_retries=2)[0] for r in batch]
    return [None]


# =========================================================================== #
# cache                                                                       #
# =========================================================================== #
def cache_key(oa_id: str, provider: str, pv: str | None = None) -> str:
    return f"{short_id(oa_id)}|{provider}|{pv or PROMPT_VERSION}"


BACKUP_SECONDS = 60      # copy new answers to Drive at least this often


def _seg_dir() -> Path | None:
    return Path(str(DRIVE_CACHE) + ".parts") if DRIVE_CACHE else None


def restore_cache() -> dict[str, dict]:
    """Rebuild the cache from Drive (old single file + incremental segments) and local disk.
    A half-written last line (crash mid-write) is skipped."""
    cache: dict[str, dict] = {}
    files = []
    if DRIVE_CACHE and Path(DRIVE_CACHE).exists():
        files.append(Path(DRIVE_CACHE))
    if _seg_dir() and _seg_dir().exists():
        files += sorted(_seg_dir().glob("seg_*.jsonl"))
    if Path(LOCAL_CACHE).exists():
        files.append(Path(LOCAL_CACHE))
    on_drive: set[str] = set()
    for f in files:
        is_local = f == Path(LOCAL_CACHE)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    cache[d["key"]] = d
                    if not is_local:
                        on_drive.add(d["key"])
                except Exception:
                    pass
    _BK["offset"] = Path(LOCAL_CACHE).stat().st_size if Path(LOCAL_CACHE).exists() else 0
    # answers that reached the local disk but not Drive (e.g. "Restart session"): push them now
    missing = [d for k, d in cache.items() if k not in on_drive]
    if DRIVE_CACHE and missing:
        seg = _seg_dir()
        seg.mkdir(parents=True, exist_ok=True)
        name = seg / f"seg_{time.strftime('%Y%m%d_%H%M%S')}_recovered_{len(missing):08d}.jsonl"
        tmp = name.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in missing), encoding="utf-8")
        tmp.rename(name)
        print(f"[cache] {len(missing):,} local answers were not on Drive yet -> saved", file=sys.stderr)
    print(f"[cache] {len(cache):,} answers restored", file=sys.stderr)
    return cache


_BK = {"offset": 0, "t": 0.0}


def backup_cache(force: bool = True) -> None:
    """Write only the NEW local lines to a small segment file on Drive (no big re-copy)."""
    if not DRIVE_CACHE or not Path(LOCAL_CACHE).exists():
        return
    if not force and time.time() - _BK["t"] < BACKUP_SECONDS:
        return
    with _WRITE:
        size = Path(LOCAL_CACHE).stat().st_size
        if size <= _BK["offset"]:
            _BK["t"] = time.time()
            return
        with open(LOCAL_CACHE, "rb") as fh:
            fh.seek(_BK["offset"])
            chunk = fh.read(size - _BK["offset"])
        cut = chunk.rfind(b"\n") + 1            # only complete lines
        if cut == 0:
            return
        seg = _seg_dir()
        seg.mkdir(parents=True, exist_ok=True)
        name = seg / f"seg_{time.strftime('%Y%m%d_%H%M%S')}_{_BK['offset']:012d}.jsonl"
        tmp = name.with_suffix(".tmp")
        tmp.write_bytes(chunk[:cut])
        tmp.rename(name)                         # a segment is either complete or absent
        _BK["offset"] += cut
        _BK["t"] = time.time()


def cache_status(n_expected: int | None = None, providers=None) -> dict:
    """Read-only check of what is safely on Drive (+ local disk): answers per model,
    broken lines, gaps between segments, and what has not reached Drive yet."""
    import re as _re
    seg = sorted(_seg_dir().glob("seg_*.jsonl")) if _seg_dir() and _seg_dir().exists() else []
    drive_files = ([Path(DRIVE_CACHE)] if DRIVE_CACHE and Path(DRIVE_CACHE).exists() else []) + seg
    keys, bad = {}, 0
    for f in drive_files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    keys[d["key"]] = d["provider"]
                except Exception:
                    bad += 1
    gaps = []
    for a, b in zip(seg, seg[1:]):
        ma = _re.search(r"_(\d{12})\.jsonl$", a.name)
        mb = _re.search(r"_(\d{12})\.jsonl$", b.name)
        if ma and mb:
            end_a = int(ma.group(1)) + a.stat().st_size
            if int(mb.group(1)) not in (end_a, 0):
                gaps.append((a.name, b.name))
    local = Path(LOCAL_CACHE).stat().st_size if Path(LOCAL_CACHE).exists() else 0
    per = Counter(v for k, v in keys.items() if k.endswith("|" + PROMPT_VERSION))
    print(f"[cache] Drive: {len(seg)} segments + {'1 main file' if DRIVE_CACHE and Path(DRIVE_CACHE).exists() else 'no main file'}"
          f" | {len(keys):,} answers | {bad} broken lines | {len(gaps)} gaps")
    for p in providers or PRIMARY:
        k = per.get(p, 0)
        print(f"  {p:<11} {k:,}" + (f"/{n_expected:,} ({k / n_expected:.1%})" if n_expected else ""))
    print(f"  not yet on Drive: {max(local - _BK['offset'], 0) / 1e3:.0f} KB of this runtime's answers")
    for g in gaps[:5]:
        print("  gap:", *g)
    return {"answers": len(keys), "per_model": dict(per), "broken": bad, "gaps": gaps, "segments": len(seg)}


def compact_cache() -> None:
    """Merge all Drive segments into the main cache file, then remove them.
    Run only when NO screening is running (end of a session). Safe: the main file is
    replaced atomically and segments are deleted only after it is written."""
    seg = sorted(_seg_dir().glob("seg_*.jsonl")) if _seg_dir() and _seg_dir().exists() else []
    if not seg:
        print("[cache] nothing to compact")
        return
    cache = {}
    for f in ([Path(DRIVE_CACHE)] if Path(DRIVE_CACHE).exists() else []) + seg:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    cache[d["key"]] = d
                except Exception:
                    pass
    tmp = Path(str(DRIVE_CACHE) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for d in cache.values():
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    tmp.replace(DRIVE_CACHE)
    for f in seg:
        f.unlink()
    print(f"[cache] compacted {len(seg)} segments -> {len(cache):,} answers in {DRIVE_CACHE}")


_WRITE = threading.Lock()


def _persist(rows: list[dict]) -> None:
    with _WRITE:
        Path(LOCAL_CACHE).parent.mkdir(parents=True, exist_ok=True)
        with open(LOCAL_CACHE, "a", encoding="utf-8") as fh:
            for d in rows:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def _run_provider(records: list[dict], provider: str, cache: dict) -> None:
    todo = [r for r in records if cache_key(r["oa_id"], provider) not in cache]
    if not todo:
        return
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"[{provider}] {len(todo):,} records in {len(batches)} batches", file=sys.stderr)

    def job(b):
        rows = []
        for rec, ans in zip(b, call_batch(provider, b)):
            if ans is None:
                continue
            rows.append({"key": cache_key(rec["oa_id"], provider), "oa_id": short_id(rec["oa_id"]),
                         "provider": provider, "model": PROVIDERS[provider]["model"],
                         "pv": PROMPT_VERSION, **ans})
        _persist(rows)
        return rows

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for fut in as_completed([ex.submit(job, b) for b in batches]):
                for d in fut.result():
                    cache[d["key"]] = d
                backup_cache(force=False)          # at most once a minute
    finally:
        backup_cache()                             # also on stop button / error


# =========================================================================== #
# decision: class -> GTRA layer (derived in code, replayable)                 #
# =========================================================================== #
def layer_of(cls: str | None, review: bool) -> str:
    if cls in TICK_CLASSES:
        return "review" if review else "core"
    if cls == "disease_only":
        return "disease_only"
    if cls == "not_tick":
        return "excluded"
    return "uncertain"


def combine(ans: dict[str, dict | None]) -> dict:
    got = {p: a for p, a in ans.items() if a}
    if not got:
        return {"gtra_class": None, "gtra_status": "failed"}
    votes = Counter(a["class"] for a in got.values())
    top, n_top = votes.most_common(1)[0]
    leaders = [k for k, v in votes.items() if v == n_top]
    tb = got.get(TIEBREAK)
    if len(leaders) > 1 and tb and tb["class"] in leaders:
        final, status = tb["class"], "tiebreak"
    elif len(votes) == 1:
        final, status = top, ("agree" if len(got) == len(ans) else "single")
    else:
        final, status = top, "majority"
    src = next(a for a in ([tb] if tb else []) + list(got.values()) if a and a["class"] == final)
    revs = {a["review"] for a in got.values()}
    review = revs.pop() if len(revs) == 1 else (tb["review"] if tb else src["review"])
    incl = [a["class"] in TICK_CLASSES for a in got.values()]
    agree_incl = len(got) == len(ans) and len(set(incl)) == 1
    taxa = sorted({t for a in got.values() for t in a["tick_taxa"]})
    return {
        "gtra_class": final,
        "gtra_review": bool(review),
        "gtra_layer": layer_of(final, bool(review)),
        "gtra_include": final in TICK_CLASSES,
        "gtra_status": status,
        "gtra_models_agree": len(got) == len(ans) and len(votes) == 1,
        "gtra_agree_include": agree_incl,
        "gtra_low_conf": any(a["confidence"] == "low" for a in got.values()),
        "gtra_tick_examined": src["tick_examined"],
        "gtra_field_collection": src["field_collection"],
        "gtra_setting": src["setting"],
        "gtra_tick_taxa": "|".join(taxa),
        "gtra_evidence": src["evidence"],
        "gtra_evidence_ok": src["evidence_ok"],
    }


def needs_human(row: pd.Series) -> bool:
    """Review queue: include/exclude splits, uncertain, failed, or low confidence on a split."""
    return (row.get("gtra_status") == "failed" or row.get("gtra_class") == "uncertain"
            or not row.get("gtra_agree_include", True)
            or (row.get("gtra_low_conf") and not row.get("gtra_models_agree")))


def screen(df: pd.DataFrame, providers=None, cache: dict | None = None,
           min_answer_share: float = 0.5) -> pd.DataFrame:
    """df needs oa_id, oa_title, coding_abstract (prepare() adds it)."""
    providers = list(providers or PRIMARY)
    cache = restore_cache() if cache is None else cache
    records = df.to_dict("records")
    for p in providers:
        _run_provider(records, p, cache)
    for p in providers:
        got = sum(cache_key(r["oa_id"], p) in cache for r in records)
        if records and got < min_answer_share * len(records):
            raise RuntimeError(f"{p} answered only {got} of {len(records)} records. Check its model id "
                               f"({PROVIDERS[p]['model']!r}) and key with check_providers().")
    rows = []
    for r in records:
        ans = {p: cache.get(cache_key(r["oa_id"], p)) for p in providers}
        row = {"oa_id": short_id(r["oa_id"])}
        for p, a in ans.items():
            a = a or {}
            row[f"{p}_class"] = a.get("class")
            row[f"{p}_review"] = a.get("review")
            row[f"{p}_conf"] = a.get("confidence")
        row.update(combine(ans))
        row["gtra_prompt_version"] = PROMPT_VERSION
        rows.append(row)
    out = pd.DataFrame(rows)
    out["gtra_needs_human"] = out.apply(needs_human, axis=1)
    print(out["gtra_status"].value_counts().to_string(), file=sys.stderr)
    bad = (out["gtra_evidence_ok"] == False).sum()  # noqa: E712
    if bad:
        print(f"!! {bad} records whose evidence is not a verbatim span: read a few", file=sys.stderr)
    return out


def replay(out: pd.DataFrame) -> pd.DataFrame:
    """Re-derive layers after you change layer_of(); costs nothing."""
    out = out.copy()
    out["gtra_layer"] = [layer_of(c, bool(r)) for c, r in zip(out["gtra_class"], out["gtra_review"])]
    out["gtra_include"] = out["gtra_class"].isin(TICK_CLASSES)
    return out


# =========================================================================== #
# pilot sample + blind coding sheet + scoring                                  #
# =========================================================================== #
def draw_pilot(df: pd.DataFrame, n: int = 300, seed: int = 20261125,
               gold_ids: set | None = None, exclude_ids: set | None = None) -> pd.DataFrame:
    """Stratified by where the tick signal sits, noise flags and (optionally) old gold membership.
    Hard strata are over-sampled on purpose; weights let you re-weight to the corpus."""
    d = df.copy()
    if exclude_ids:
        d = d[~d["oa_id"].isin(exclude_ids)]
    d["_gold"] = d["oa_id"].isin(gold_ids) if gold_ids else False
    d["_stratum"] = (d["rule_tick_where"] + "|" + np.where(d["rule_noise"].ne(""), "noise", "clean")
                     + "|" + np.where(d["_gold"], "oldgold", "notgold") + "|" + d["screen_input"])
    sizes = d["_stratum"].value_counts()
    # square-root allocation: big strata still dominate, small hard strata get enough rows
    alloc = np.sqrt(sizes)
    alloc = (alloc / alloc.sum() * n).round().astype(int).clip(lower=3)
    alloc = alloc.clip(upper=sizes)
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1
    parts = [d[d["_stratum"] == s].sample(k, random_state=seed) for s, k in alloc.items() if k > 0]
    out = pd.concat(parts).sample(frac=1, random_state=seed)
    out["_weight"] = out["_stratum"].map(sizes / alloc)
    out["record_no"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def write_pilot_sheet(sample: pd.DataFrame, outdir: str, name: str = "gtra_pilot_coding.xlsx") -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    wb = Workbook()
    ws = wb.active
    ws.title = "coding"
    ws.append(["record_no", "oa_id", "year", "title", "abstract", "class", "review",
               "evidence (verbatim span)", "confidence", "memo (hard case? why?)"])
    for _, r in sample.iterrows():
        ws.append([int(r["record_no"]), r["oa_id"], S(r.get("oa_year")), S(r.get("oa_title")),
                   S(r.get("coding_abstract")) or "[no abstract]", None, None, None, None, None])
    for i, w in enumerate([8, 13, 6, 40, 90, 22, 8, 40, 11, 40], start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DFE9F6")
    wrap = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = wrap
    ws.freeze_panes = "F2"
    n = len(sample) + 1
    for formula, col in ((",".join(CLASSES), "F"), ("TRUE,FALSE", "G"), ("high,medium,low", "I")):
        dv = DataValidation(type="list", formula1=f'"{formula}"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")
    key_cols = [c for c in ["record_no", "oa_id", "_stratum", "_weight", "rule_tick_where", "rule_noise",
                            "screen_input", "oa_type", "oa_language"] if c in sample.columns]
    key = wb.create_sheet("key_do_not_open")
    key.append(key_cols)
    for _, r in sample[key_cols].iterrows():
        key.append([None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r])
    key.sheet_state = "hidden"
    leg = wb.create_sheet("classes")
    leg.append(["class", "GTRA layer", "meaning"])
    for c, m in [("tick_organism", "the tick itself (taxonomy, ecology, surveys, genome, microbiome)"),
                 ("tick_derived", "tick material (saliva, proteins, genes, cell lines, anti-tick vaccines)"),
                 ("pathogen_in_tick", "pathogen detected in or isolated from ticks"),
                 ("tick_control", "acaricides, resistance, repellents, biocontrol aimed at ticks"),
                 ("host_infestation", "ticks counted or identified on hosts"),
                 ("tick_distribution_model", "models or maps built from tick records, no ticks handled"),
                 ("disease_only", "tick-borne disease with no tick examined (excluded, kept as a layer)"),
                 ("not_tick", "mites, other meanings of tick, unrelated"),
                 ("uncertain", "not enough text to decide")]:
        leg.append([c, layer_of(c, False), m])
    leg.column_dimensions["A"].width = 24
    leg.column_dimensions["C"].width = 80
    wb.save(path)
    print(f"coding sheet -> {path}")
    return path


def cohen_kappa(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(sheet_path: str, llm: pd.DataFrame) -> pd.DataFrame:
    """Compare model output with your hand codes: class kappa, include/exclude metrics
    (sensitivity is the number that matters), agree vs split accuracy, disagreements."""
    h = pd.read_excel(sheet_path, sheet_name="coding")
    h["h_class"] = h["class"].astype(str).str.strip().str.lower()
    h = h[h["h_class"].isin(CLASSES)]
    m = h.merge(llm, on="oa_id", how="inner")
    if m.empty:
        print("nothing to score: code the sheet first (class column)")
        return m
    m["h_incl"] = m["h_class"].isin(TICK_CLASSES)
    rows = []
    for name, col in [("consensus", "gtra_class")] + [(c[:-6], c) for c in m.columns
                                                      if c.endswith("_class") and c not in
                                                      {"gtra_class", "h_class"}]:
        ok = m[col].notna()
        a, b = m.loc[ok, "h_class"].tolist(), m.loc[ok, col].tolist()
        hi = m.loc[ok, "h_incl"]
        mi = m.loc[ok, col].isin(TICK_CLASSES)
        tp, fn = (hi & mi).sum(), (hi & ~mi).sum()
        fp, tn = (~hi & mi).sum(), (~hi & ~mi).sum()
        rows.append({"coder": name, "n": len(a),
                     "class_acc": round(np.mean([x == y for x, y in zip(a, b)]), 3),
                     "class_kappa": round(cohen_kappa(a, b), 3),
                     "incl_kappa": round(cohen_kappa(hi.map({True: "in", False: "out"}).tolist(),
                                                     mi.map({True: "in", False: "out"}).tolist()), 3),
                     "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
                     "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
                     "precision": round(tp / (tp + fp), 3) if tp + fp else None})
    tab = pd.DataFrame(rows)
    print(f"{len(m)} records coded by you and the models\n{tab.to_string(index=False)}")
    for flag, lab in ((True, "models agree"), (False, "models split")):
        sub = m[m["gtra_models_agree"] == flag]
        if len(sub):
            print(f"{lab}: {len(sub)} ({len(sub)/len(m):.0%}), class accuracy "
                  f"{(sub['h_class'] == sub['gtra_class']).mean():.3f}")
    print("\nconfusion (rows = you, columns = consensus):")
    print(pd.crosstab(m["h_class"], m["gtra_class"]).to_string())
    dis = m[m["h_incl"] != m["gtra_class"].isin(TICK_CLASSES)]
    if len(dis):
        print("\ninclude/exclude disagreements (these drive the prompt revision):")
        with pd.option_context("display.max_colwidth", 80):
            print(dis[[c for c in ["record_no", "h_class", "gtra_class", "gtra_status", "gtra_evidence",
                                   "memo (hard case? why?)"] if c in dis.columns]].to_string(index=False))
    return tab
