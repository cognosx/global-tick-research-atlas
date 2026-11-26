import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from gtra.dedupe_fast import FastDeduper, author_fp, norm_doi


def _df():
    T = "Acaricide resistance in Rhipicephalus microplus from cattle farms in Cameroon"
    return pd.DataFrame([
        dict(oa_id="W1", oa_doi="https://doi.org/10.1/a", oa_title=T, oa_authors="Ana Ngu | Eli Kim", oa_year=2020, oa_journal="J", oa_type="article"),
        dict(oa_id="W2", oa_doi="10.1/A", oa_title=T, oa_authors="Ana Ngu", oa_year=2020, oa_journal="J", oa_type="article"),           # same DOI
        dict(oa_id="W3", oa_doi="", oa_title=T.upper() + ".", oa_authors="Ngu, Ana", oa_year=2021, oa_journal="K", oa_type="article"),   # fuzzy/title
        dict(oa_id="W4", oa_doi="10.1101/x", oa_title=T, oa_authors="Ana Ngu", oa_year=2019, oa_journal="bioRxiv", oa_type="preprint"),  # version
        dict(oa_id="W5", oa_doi="", oa_title="Correction to: " + T, oa_authors="Ana Ngu", oa_year=2020, oa_journal="J", oa_type="article"),
        dict(oa_id="W6", oa_doi="10.1/b", oa_title="Ixodes ricinus abundance and climate across Europe", oa_authors="Mo Rossi", oa_year=2020, oa_journal="J", oa_type="article"),
    ])


def test_helpers():
    assert norm_doi("https://doi.org/10.1/X") == "10.1/x"
    assert author_fp("Ana Ngu | Ngu, Ana") == "ngu|a"


def test_ladder():
    dd = FastDeduper(verbose=False)
    out = dd.run(_df()).set_index("oa_id")
    s = out["study_id"]
    assert s["W1"] == s["W2"] == s["W3"]
    assert s["W4"] != s["W1"] and out.at["W4", "version_group"] == out.at["W1", "version_group"] != ""
    assert s["W5"] != s["W1"] and s["W6"] != s["W1"]
    assert out["is_canonical"].sum() == 4
    assert out["is_primary_version"].sum() == 3
