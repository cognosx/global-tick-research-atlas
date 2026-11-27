import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from gtra.openalex_harvester import OpenAlexHarvester
from gtra.recall_audit import audit, harvest_recall

WORKS = {
    "W1": dict(publication_year=2015, type="article", title="Ixodes ricinus in Norway", match=True),
    "W2": dict(publication_year=1995, type="article", title="Old tick paper", match=True),
    "W3": dict(publication_year=2015, type="book-chapter", title="Ticks chapter", match=True),
    "W4": dict(publication_year=2015, type="article", title="Lyme disease treatment", match=False),
}


class FakeH(OpenAlexHarvester):
    def __init__(self):
        super().__init__(api_key=None, pause=0)

    def _get(self, params):
        ids = params["filter"].split("openalex_id:")[1].split("|")
        res = []
        for i in ids:
            w = WORKS.get(i)
            if not w or ("search.title_abstract_keywords" in params and not w["match"]):
                continue
            res.append({"id": f"https://openalex.org/{i}", "title": w["title"], "display_name": w["title"],
                        "publication_year": w["publication_year"], "type": w["type"]})
        return {"results": res}


def test_audit_reasons():
    df = audit(FakeH(), ["W1", "W2", "W3", "W4", "W9"], "tick", (2000, 2027),
               ["article", "review", "letter", "preprint"]).set_index("oa_id")
    assert df.loc["W1", "audit_reason"] == "query_match_now"
    assert df.loc["W2", "audit_reason"] == "year_out"
    assert df.loc["W3", "audit_reason"] == "type_out"
    assert df.loc["W4", "audit_reason"] == "fulltext_only"
    assert df.loc["W9", "audit_reason"] == "not_found"
    df["gtra_layer"] = ["core", "core", "core", "disease_only", None]
    r = harvest_recall(pd.Series(["core"] * 97 + ["disease_only"] * 10), df.reset_index()).set_index("measure")["n"]
    assert r["recall, all misses"] == round(97 / 100, 4)
    assert r["recall within harvest scope (years and types)"] == round(97 / 98, 4)
