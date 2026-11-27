import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from gtra.record_kind import add_record_kind, is_taxon_title, record_kind

TREATMENTS = ["Amblyomma ovale Koch 1844", "Haemaphysalis , C. L. Koch 1844", "Rhipicephalus",
              "Haemaphysalis kolonini Du, Sun, Xu & Shao 2018, sp. nov.",
              "Hyalomma Robbins, Nava, Ronai, Chong & Guglielmone, 2025, N. V.",
              "Dermacentor compactus var. tricuspis Schulze 1933, s. l.",
              "Rhipicephalus tendeiroi Santos Dias 1950, nova espécie", "Acarus aureolatus , Pallas 1772",
              "Dermacentor rhinocerotis De Geer 1778", "Haemaphysalis shimoga Trapido & Hoogstraal 1964, subsp. n.",
              "Ixodes moschiferi Nemenz 1968, nov. spec.", "Rhipicephalus tricuspis, s. l.",
              "Amblyomma komodoense, nov. spec.", "Ixodes stromi Filippova 1957, new species"]
NOT = ["Registro de Amblyomma dissimile Koch, 1844 (Acari: Ixodidae) parasitando Kinosternon",
       "Survival of larvae and nymphs of Ixodes scapularis", "Tick Control Strategies in Dairy Animals",
       "Prevalence of ixodid ticks in post acaricide treated cattle"]


def test_taxon_titles():
    for t in TREATMENTS:
        assert is_taxon_title(t), t
    for t in NOT:
        assert not is_taxon_title(t), t


def test_kinds():
    z = "10.5281/zenodo.1"
    assert record_kind("Amblyomma ovale Koch 1844", z) == "taxonomic_treatment"
    assert record_kind("Amblyomma ovale Koch 1844", "10.1/x") == "research"      # journal DOI: keep
    assert record_kind("MOESM8 of Environmental drivers of Ixodes", "10.6084/m9.figshare.5") == "figure_or_supplement"
    assert record_kind("Correction to: Ticks in Kenya", "10.1/y") == "notice"
    assert record_kind("DEPRECATED", "10.5281/zenodo.9") == "notice"
    assert record_kind("Tick Control Strategies in Dairy Animals", z) == "repository_other"
    d = add_record_kind(pd.DataFrame({"oa_title": ["Amblyomma ovale Koch 1844", "Ticks in Kenya"],
                                      "oa_doi": [z, "10.1/k"], "gtra_include": [True, True]}))
    assert d["count_as_study"].tolist() == [False, True]
