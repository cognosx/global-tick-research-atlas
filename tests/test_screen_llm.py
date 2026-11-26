import json, os, re, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import gtra.screen_llm as G

RECS = pd.DataFrame([
    dict(oa_id="https://openalex.org/W1", oa_title="Rickettsia detected in Ixodes ricinus ticks collected in Norway",
         oa_abstract="We collected 500 questing ticks by flagging and tested them by PCR for Rickettsia helvetica."),
    dict(oa_id="W2", oa_title="Doxycycline for early Lyme disease: a randomised trial",
         oa_abstract="Patients with erythema migrans were randomised to 10 or 14 days of doxycycline therapy."),
    dict(oa_id="W3", oa_title="Tick size reduction and liquidity on the NASDAQ exchange",
         oa_abstract="We study how a smaller tick size affects bid-ask spreads and high-frequency trading."),
    dict(oa_id="W4", oa_title="Acaricide resistance in Rhipicephalus microplus: a review",
         oa_abstract=""),
])

ANS = {"W1": "pathogen_in_tick", "W2": "disease_only", "W3": "not_tick", "W4": "tick_control"}


def fake(provider, system, user):
    out = []
    for i, block in enumerate(user.split("\n\n")[1:], 1):
        title = re.search(r"TITLE: (.*)", block).group(1)
        oid = next(r.oa_id for r in RECS.itertuples() if r.oa_title == title)
        cls = ANS[G.short_id(oid)]
        if provider == "deepseek" and G.short_id(oid) == "W4":
            cls = "disease_only"                     # a split, luna breaks it
        out.append({"i": i, "class": cls, "review": G.short_id(oid) == "W4", "tick_examined": cls == "pathogen_in_tick",
                    "field_collection": "yes" if cls == "pathogen_in_tick" else "unclear", "setting": "field",
                    "tick_taxa": ["Ixodes ricinus"] if "Ixodes" in title else [],
                    "evidence": title[:40], "confidence": "high", "rationale": "x"})
    return json.dumps(out[::-1])                      # reversed order: index echo must realign


def test_rules_and_prepare():
    d = G.prepare(RECS).set_index("oa_id")
    assert d.loc["W1", "rule_tick_where"] == "title" and d.loc["W1", "rule_taxon"]
    assert "finance" in d.loc["W3", "rule_noise"]
    assert d.loc["W4", "screen_input"] == "title_only"
    d2 = G.prepare(RECS, abstract_extra={"W4": "Resistance to amitraz and pyrethroids is widespread in cattle ticks worldwide."})
    assert d2.set_index("oa_id").loc["W4", "coding_abstract_from"] == "pubmed"


def test_screen_end_to_end():
    G._COMPLETE = fake
    G.LOCAL_CACHE = tempfile.mktemp(suffix=".jsonl")
    out = G.screen(G.prepare(RECS)).set_index("oa_id")
    G._COMPLETE = None
    assert out.loc["W1", "gtra_layer"] == "core" and out.loc["W1", "gtra_models_agree"]
    assert out.loc["W2", "gtra_layer"] == "disease_only" and not out.loc["W2", "gtra_include"]
    assert out.loc["W3", "gtra_layer"] == "excluded"
    assert out.loc["W4", "gtra_status"] == "tiebreak" and out.loc["W4", "gtra_layer"] == "review"
    assert out.loc["W4", "gtra_needs_human"]           # include/exclude split -> human queue
    assert out.loc["W1", "gtra_evidence_ok"]
    # cache: a second run makes no calls
    G._COMPLETE = lambda *a: (_ for _ in ()).throw(AssertionError("called"))
    G.screen(G.prepare(RECS))
    G._COMPLETE = None


def test_pilot_sheet(tmp_path):
    big = pd.concat([RECS.assign(oa_id=RECS.oa_id + f"_{k}") for k in range(30)], ignore_index=True)
    s = G.draw_pilot(G.prepare(big), n=40)
    assert 30 <= len(s) <= 40 and s["_weight"].gt(0).all()
    p = G.write_pilot_sheet(s, tmp_path)
    assert pd.read_excel(p, sheet_name="coding").shape[0] == len(s)


def test_resume_after_lost_runtime(tmp_path):
    G._COMPLETE = fake
    G.LOCAL_CACHE = str(tmp_path / "local.jsonl")
    G.DRIVE_CACHE = str(tmp_path / "drive" / "cache.jsonl")
    G.screen(G.prepare(RECS))
    os.remove(G.LOCAL_CACHE)                       # Colab runtime lost: local disk gone
    calls = []
    G._COMPLETE = lambda *a: calls.append(1) or "[]"
    c = G.restore_cache()
    assert len(c) == 8                             # 4 records x 2 models, from Drive segments
    G.screen(G.prepare(RECS), cache=c)
    assert not calls
    G._COMPLETE, G.DRIVE_CACHE = None, None
