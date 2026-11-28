import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import gtra.validation as V


def _m():
    rows = []
    for i in range(40):
        h = "tick_organism" if i < 20 else "not_tick"
        g = h if i not in (0, 25) else ("not_tick" if i == 0 else "tick_control")
        rows.append(dict(record_no=i + 1, oa_id=f"W{i}", h0=h, h=h, gtra_class=g, gpt_6_luna_class=g,
                         deepseek_class=g, _stratum="a" if i % 2 else "b", _weight=10.0 if i % 2 else 2.0))
    return pd.DataFrame(rows)


def test_weighted_and_adjudication(tmp_path):
    m = _m()
    r = V.weighted_metrics(m, B=50).set_index("measure")["value"]
    assert r["errors_in_sample"] == 2 and 0.8 < r["sensitivity"] < 1
    d = V.disagreements(m, ref="h0")
    assert set(d["record_no"]) == {1, 26}
    adj = pd.DataFrame({"record_no": [1, 26], "final_class": ["not_tick", "tick_control"]})   # models were right
    m2 = V.apply_adjudication(m, adj)
    r2 = V.weighted_metrics(m2, B=50).set_index("measure")["value"]
    assert r2["errors_in_sample"] == 0 and r2["sensitivity"] == 1.0
    p = V.save_labels(m2, tmp_path / "lab.parquet")
    assert pd.read_parquet(p)["final_include"].sum() == 20
