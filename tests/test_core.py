"""Core invariant tests. Run with: pytest tests/"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sosci_pipeline import dedupe, io, provenance


def test_norm_doi():
    assert dedupe.norm_doi("https://doi.org/10.1/ABC") == "10.1/abc"
    assert dedupe.norm_doi("") == ""


def test_norm_oaid():
    assert dedupe.norm_oaid("https://openalex.org/W2755950973") == "w2755950973"
    assert dedupe.norm_oaid("short") == ""


def test_ensure_core():
    df = io.ensure_core(pd.DataFrame({"title": ["x"]}))
    assert all(c in df.columns for c in io.CORE_COLUMNS)


def test_dedupe_invariants():
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "title": ["Climate change effects on Ixodes ricinus tick distribution in Europe",
                  "Climate change effects on Ixodes ricinus tick distribution in Europe",
                  "Rainfall and Amblyomma abundance patterns across savanna ecosystems",
                  "A totally unrelated study on malaria mosquito breeding sites"],
        "authors": ["Smith, John; Doe, Alice"] * 2 + ["Brown, Bob", "Green, Gwen"],
        "year": ["2020", "2020", "2019", "2018"],
        "doi": ["10.1/abc", "10.1/abc", "10.2/def", "10.3/ghi"],
        "source_database": ["Scopus", "PubMed", "WoS", "Lens"],
    })
    out, merges = dedupe.deduplicate(df)
    assert out["study_id"].nunique() == 3
    assert out[out.record_id == "R1"]["study_id"].iloc[0] == out[out.record_id == "R2"]["study_id"].iloc[0]
    assert merges["doi"] == 1


def test_provenance_handles_unsafe_values(tmp_path):
    work = str(tmp_path)
    provenance.log_stage(work, "test_stage", rows_out=5, extra={"a_set": {1, 2, 3}, "ellipsis": ...})
    P, path = provenance.write_prisma(work)
    assert os.path.exists(path)
