"""
record_kind: tells real studies apart from records that are not studies.

OpenAlex types many repository items as "article". In the 2026-11-26 harvest this matters:
~800 Zenodo records are single taxonomic treatments ("Amblyomma ovale Koch 1844", one per
name, mostly from a 2025 world tick catalogue). The LLM screen correctly says they are about
ticks, but they are not studies and would create a false jump in 2025.

Kinds (first match wins)
  notice                correction / erratum / retraction / expression of concern
  figure_or_supplement  "Figure 3 ...", "Table 2 ...", "MOESM4 of ...", "Additional file 1 ..."
  taxonomic_treatment   repository DOI + title that is only a taxon name and its authority
  repository_other      other Zenodo / figshare / Dryad / Mendeley Data records (kept, flagged)
  research              everything else

count_as_study = gtra_include and kind in {research, repository_other}
Nothing is deleted: the flag lets every analysis choose.
"""
from __future__ import annotations

import re

import pandas as pd

REPO_DOI = re.compile(r"10\.5281/zenodo|10\.6084/m9\.figshare|10\.5061/dryad|10\.17632/|10\.15468/")
NOTICE = re.compile(r"(?i)^\s*(correction|corrigendum|erratum|errata|retraction|retracted|"
                    r"expression of concern|notice of retraction|withdrawn|addendum|publisher'?s note|deprecated)\b")
FIGURE = re.compile(r"(?i)^\s*(fig(ure)?s?\.?\s*\d|table\s*\d|plate\s*\d|moesm\d|"
                    r"additional file\s*\d|supplementa(ry|l) (file|table|figure|material|data)|"
                    r"s\d+\s+(fig|table|file|data)\b)")
FUNCTION_WORDS = {"the", "of", "in", "and", "a", "an", "on", "for", "with", "from", "to", "by", "at",
                  "is", "are", "as", "de", "la", "le", "des", "du", "et", "en", "y", "da", "do",
                  "und", "der", "die", "das", "von", "new", "first", "report", "study", "analysis",
                  "registro", "record", "parasitando", "parasitizing"}
# Genus [(Subgenus)] [species [var./subsp. infra]] [authority] [year [short tail: sp. nov., s. l., N. V.]]
TAXON = re.compile(
    r"^[A-Z][a-z]+(?:\s+\([A-Z][a-z]+\))?"
    r"(?:\s+(?:(?:var|subsp|ssp|f)\.\s+)?[a-z][a-z-]+){0,3}"
    r"(?:\s*,?\s+(?:[A-Z][A-Za-z'\-]*\.?|&|and|de|da|dos|van|von|in|et al\.?)){0,10}"
    r"(?:\s*,?\s*\(?\d{4}\)?(?:\s*,\s*[A-Za-zÀ-ÿ.\s]{1,25})?)?\s*\.?$")


TAIL = re.compile(r"(?i)\s*,?\s*(s\.\s*l\.|s\.\s*str\.|nov\.?\s*spec\.?|new species|sp\.\s*nov\.?|n\.\s*sp\.|"
                  r"sp\.\s*n\.|ssp\.\s*n\.|subsp\.\s*n\.|nova esp[ée]cie|n\.\s*v\.|stat\.\s*nov\.?|"
                  r"comb\.\s*nov\.?|syn\.\s*nov\.?)\s*\.?$")


def is_taxon_title(title: str) -> bool:
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    t = TAIL.sub("", t).strip(" ,")
    if not t or len(t.split()) > 14 or any(c in t for c in ":?!;\""):
        return False
    lower_words = {w.strip(",.()") for w in t.split() if w[:1].islower()}
    if lower_words & FUNCTION_WORDS:
        return False
    return bool(TAXON.match(t))


def record_kind(title, doi) -> str:
    t, d = str(title or ""), str(doi or "").lower()
    if NOTICE.match(t):
        return "notice"
    if FIGURE.match(t):
        return "figure_or_supplement"
    if REPO_DOI.search(d):
        return "taxonomic_treatment" if is_taxon_title(t) else "repository_other"
    return "research"


def add_record_kind(df: pd.DataFrame, title_col: str = "oa_title", doi_col: str = "oa_doi") -> pd.DataFrame:
    out = df.copy()
    out["record_kind"] = [record_kind(t, d) for t, d in zip(out[title_col], out[doi_col])]
    if "gtra_include" in out.columns:
        out["count_as_study"] = out["gtra_include"].fillna(False).astype(bool) & \
            out["record_kind"].isin(["research", "repository_other"])
    return out
