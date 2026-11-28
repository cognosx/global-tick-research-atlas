"""
GTRA screening validation: adjudication sheet + corpus-weighted metrics.

This is the exact code used on 2026-11-28 for the v1 validation (598 records):
  1. score_raw()          compare your blind codes with the models (unweighted, quick look)
  2. write_adjudication() list the include/exclude disagreements for a second look
  3. apply_adjudication() final reference = your code, replaced by final_class where adjudicated
  4. weighted_metrics()   sensitivity / specificity / precision / accuracy, weighted to the corpus
     with the sampling weights, + stratified bootstrap 95% intervals
  5. save_labels()        the label table for the repo (no abstracts)

Inputs
  sheet     gtra_validation_coding.xlsx  (sheet "coding" = your codes, hidden "key_do_not_open"
            = record_no, oa_id, _stratum, _weight written by notebook 03, cell 14)
  codes     the model decisions: screening/full/<run>/parts/part_*.parquet concatenated
            (or works_screened.parquet), one row per oa_id with gtra_* and <model>_class columns

Why weighted: the sample over-represents hard strata (model splits, uncertain, reviews).
Each record stands for _weight records of its stratum (stratum size / sampled), so weighted
estimates describe the whole corpus. Intervals: resample records within each stratum
(stratified bootstrap), recompute, take the 2.5% and 97.5% quantiles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .screen_llm import CLASSES, TICK_CLASSES, cohen_kappa, short_id


# --------------------------------------------------------------------------- load
def load_validation(sheet: str | Path, codes: pd.DataFrame) -> pd.DataFrame:
    """One row per coded record: your code + stratum/weight + model decisions."""
    c = pd.read_excel(sheet, sheet_name="coding")
    k = pd.read_excel(sheet, sheet_name="key_do_not_open")
    codes = codes.copy()
    codes["oa_id"] = codes["oa_id"].map(short_id)
    c["oa_id"] = c["oa_id"].map(short_id)
    m = (c.merge(k[["record_no", "_stratum", "_weight"]], on="record_no")
          .merge(codes, on="oa_id", how="left"))
    m["h0"] = m["class"].astype(str).str.strip().str.lower()
    m = m[m["h0"].isin(CLASSES)].copy()
    m["h"] = m["h0"]                                  # reference; replaced after adjudication
    return m


# --------------------------------------------------------------------------- 1. quick look
def score_raw(m: pd.DataFrame, ref: str = "h") -> pd.DataFrame:
    """Unweighted agreement per coder (consensus and each model)."""
    rows = []
    model_cols = ["gtra_class"] + [c for c in m.columns if c.endswith("_class")
                                   and c not in ("gtra_class",)]
    for col in dict.fromkeys(model_cols):
        ok = m[col].notna()
        a, b = m.loc[ok, ref], m.loc[ok, col]
        hi, mi = a.isin(TICK_CLASSES), b.isin(TICK_CLASSES)
        tp, fn, fp, tn = (hi & mi).sum(), (hi & ~mi).sum(), (~hi & mi).sum(), (~hi & ~mi).sum()
        rows.append({"coder": "consensus" if col == "gtra_class" else col[:-6], "n": int(ok.sum()),
                     "class_acc": round((a == b).mean(), 3),
                     "class_kappa": round(cohen_kappa(a.tolist(), b.tolist()), 3),
                     "incl_kappa": round(cohen_kappa(hi.map(str).tolist(), mi.map(str).tolist()), 3),
                     "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
                     "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
                     "precision": round(tp / (tp + fp), 3) if tp + fp else None})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. adjudication sheet
def disagreements(m: pd.DataFrame, ref: str = "h") -> pd.DataFrame:
    """Records where you and the consensus disagree on include/exclude."""
    d = m[m[ref].isin(TICK_CLASSES) != m["gtra_class"].isin(TICK_CLASSES)].copy()
    d["direction"] = np.where(d[ref].isin(TICK_CLASSES), "you INCLUDE, models exclude",
                              "you EXCLUDE, models include")
    return d.sort_values(["direction", "record_no"])


def write_adjudication(m: pd.DataFrame, path: str | Path) -> Path:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    d = disagreements(m, ref="h0")
    models = [c for c in d.columns if c.endswith("_class") and c != "gtra_class"]
    cols = ["record_no", "oa_id", "title", "abstract", "h0", *models, "gtra_class", "gtra_status",
            "gtra_evidence", "direction"]
    out = d[[c for c in cols if c in d.columns]].rename(columns={"h0": "your_code", "gtra_class": "consensus"})
    out["final_class"], out["note"] = "", ""
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="adjudicate")
    wb = load_workbook(path)
    ws = wb.active
    widths = [8, 12, 45, 80, 18] + [18] * len(models) + [18, 10, 45, 26, 20, 30]
    for i, wd in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = wd
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DFE9F6")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "E2"
    fc = [c.column_letter for c in ws[1] if c.value == "final_class"][0]
    dv = DataValidation(type="list", formula1='"' + ",".join(CLASSES) + '"')
    ws.add_data_validation(dv)
    dv.add(f"{fc}2:{fc}{len(out) + 1}")
    wb.save(path)
    print(f"{len(out)} disagreements -> {path}  {out['direction'].value_counts().to_dict()}")
    return path


# --------------------------------------------------------------------------- 3. apply adjudication
def apply_adjudication(m: pd.DataFrame, adjudicated: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Reference = your code, replaced by final_class for the adjudicated records.
    Accepts the adjudication .xlsx, a .csv export (e.g. from Google Sheets) or a DataFrame."""
    if isinstance(adjudicated, pd.DataFrame):
        a = adjudicated
    elif str(adjudicated).endswith(".xlsx"):
        a = pd.read_excel(adjudicated)
    else:
        a = pd.read_csv(adjudicated)
    a = a[a["final_class"].notna()]
    fin = dict(zip(a["record_no"], a["final_class"].astype(str).str.strip().str.lower()))
    bad = {r: v for r, v in fin.items() if v not in CLASSES}
    if bad:
        raise ValueError(f"unknown final_class values: {bad}")
    m = m.copy()
    m["adjudicated"] = m["record_no"].isin(fin)
    m["h"] = [fin.get(r, h) for r, h in zip(m["record_no"], m["h0"])]
    flips = (m["h0"].isin(TICK_CLASSES) != m["h"].isin(TICK_CLASSES)).sum()
    print(f"{len(fin)} adjudicated, {flips} include/exclude decisions changed")
    return m


# --------------------------------------------------------------------------- 4. weighted metrics
def _metrics(d: pd.DataFrame, model_col: str, ref: str) -> dict:
    w = d["_weight"]
    hi, mi = d[ref].isin(TICK_CLASSES), d[model_col].isin(TICK_CLASSES)
    tp, fn = (w * (hi & mi)).sum(), (w * (hi & ~mi)).sum()
    fp, tn = (w * (~hi & mi)).sum(), (w * (~hi & ~mi)).sum()
    n = tp + fn + fp + tn
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "precision": tp / (tp + fp),
            "accuracy": (tp + tn) / n, "share_tick_ref": (tp + fn) / n, "share_tick_model": (tp + fp) / n,
            "class_accuracy": float(np.average(d[ref] == d[model_col], weights=w)),
            "errors_in_sample": int((hi != mi).sum()), "n": len(d)}


def weighted_metrics(m: pd.DataFrame, model_col: str = "gtra_class", ref: str = "h",
                     drop_uncertain: bool = False, B: int = 1000, seed: int = 7) -> pd.DataFrame:
    """drop_uncertain=False: the model's "uncertain" counts as exclude (strict).
    drop_uncertain=True: "uncertain" records are left out, i.e. assumed settled by human review."""
    d = m[m[model_col] != "uncertain"] if drop_uncertain else m
    point = _metrics(d, model_col, ref)
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby("_stratum")]
    boots = []
    for _ in range(B):
        s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1_000_000_000)))
                       for g in groups])
        boots.append(_metrics(s, model_col, ref))
    q = pd.DataFrame(boots).quantile([0.025, 0.975])
    rows = []
    for k in ["sensitivity", "specificity", "precision", "accuracy", "class_accuracy",
              "share_tick_ref", "share_tick_model"]:
        rows.append({"measure": k, "value": round(point[k], 3),
                     "ci_low": round(q.loc[0.025, k], 3), "ci_high": round(q.loc[0.975, k], 3)})
    rows.append({"measure": "errors_in_sample", "value": point["errors_in_sample"], "ci_low": None, "ci_high": None})
    rows.append({"measure": "n", "value": point["n"], "ci_low": None, "ci_high": None})
    return pd.DataFrame(rows)


def report(m: pd.DataFrame, B: int = 1000) -> dict[str, pd.DataFrame]:
    out = {
        "unweighted": score_raw(m),
        "weighted_strict": weighted_metrics(m, B=B),
        "weighted_after_review": weighted_metrics(m, drop_uncertain=True, B=B),
    }
    for p in [c[:-6] for c in m.columns if c.endswith("_class") and c != "gtra_class"]:
        out[f"weighted_{p}_alone"] = weighted_metrics(m, model_col=f"{p}_class", B=B)
    return out


# --------------------------------------------------------------------------- 5. labels for the repo
def save_labels(m: pd.DataFrame, path: str | Path, prompt_version: str = "gtra-screen-v1") -> Path:
    keep = ["record_no", "oa_id", "class", "review", "_stratum", "_weight", "adjudicated", "h",
            *[c for c in m.columns if c.endswith("_class")], "gtra_status", "gtra_layer", "gtra_review"]
    lab = m[[c for c in dict.fromkeys(keep) if c in m.columns]].rename(columns={
        "class": "coder1_class", "review": "coder1_review", "_stratum": "stratum", "_weight": "weight",
        "h": "final_class"})
    lab["final_include"] = lab["final_class"].isin(TICK_CLASSES)
    lab["prompt_version"] = prompt_version
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(path, index=False)
    return Path(path)
