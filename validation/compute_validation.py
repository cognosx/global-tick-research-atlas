#!/usr/bin/env python3
"""
Score the blind validation once both reviewers have filled their sheets.

Steps
 1. Agreement between reviewers A and B (Cohen's kappa, raw agreement) for Q1 and Q2.
 2. Reference label = the two reviewers' label when they agree; otherwise the
    'adjudicated' column of adjudication.csv (written on the first run, fill it, re-run).
 3. For the final verdict, each model and the earlier human label:
    sensitivity, specificity, precision, F1 vs the reference, design-weighted to the
    whole corpus, with 95% bootstrap confidence intervals, overall and per bloc.

Usage:
  python validation/compute_validation.py --dir "/content/drive/MyDrive/ticks/validation_v1"
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

B = 2000
SEED = 7


def norm(s, yes=("include", "yes"), no=("exclude", "no")):
    t = s.astype(str).str.strip().str.lower()
    return pd.Series(np.where(t.isin(yes), 1, np.where(t.isin(no), 0, np.nan)), index=s.index)


def kappa(a, b):
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    if len(a) < 2:
        return np.nan, np.nan, 0
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan, po, len(a)


def metrics(pred, ref, w):
    m = pred.notna() & ref.notna()
    p, r, w = pred[m].values, ref[m].values, w[m].values
    tp = w[(p == 1) & (r == 1)].sum()
    fp = w[(p == 1) & (r == 0)].sum()
    fn = w[(p == 0) & (r == 1)].sum()
    tn = w[(p == 0) & (r == 0)].sum()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and sens else np.nan
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


def boot(pred, ref, w, rng):
    idx = np.arange(len(pred))
    draws = []
    for _ in range(B):
        s = rng.choice(idx, len(idx), replace=True)
        draws.append(metrics(pred.iloc[s].reset_index(drop=True),
                             ref.iloc[s].reset_index(drop=True),
                             w.iloc[s].reset_index(drop=True)))
    d = pd.DataFrame(draws)
    return d.quantile(0.025), d.quantile(0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    key = pd.read_csv(os.path.join(a.dir, "sample_key.csv"))
    ra = pd.read_excel(os.path.join(a.dir, "reviewer_A.xlsx"), sheet_name="review")
    rb = pd.read_excel(os.path.join(a.dir, "reviewer_B.xlsx"), sheet_name="review")
    df = key.merge(ra[["review_id", "Q1_about_ticks", "Q2_climate_effect"]], on="review_id") \
            .merge(rb[["review_id", "Q1_about_ticks", "Q2_climate_effect"]], on="review_id",
                   suffixes=("_A", "_B"))

    out = []
    for q in ("Q1_about_ticks", "Q2_climate_effect"):
        k, po, n = kappa(norm(df[f"{q}_A"]), norm(df[f"{q}_B"]))
        out.append(f"{q}: kappa={k:.3f} agreement={po:.3f} (n={n})")
    print("\n".join(out))

    a1, b1 = norm(df["Q1_about_ticks_A"]), norm(df["Q1_about_ticks_B"])
    df["reference"] = np.where(a1 == b1, a1, np.nan)
    adj_path = os.path.join(a.dir, "adjudication.csv")
    todo = df[df["reference"].isna()][["review_id", "Q1_about_ticks_A", "Q1_about_ticks_B"]]
    if os.path.exists(adj_path):
        adj = pd.read_csv(adj_path)
        adj["adj"] = norm(adj["adjudicated"])
        df = df.merge(adj[["review_id", "adj"]], on="review_id", how="left")
        df["reference"] = df["reference"].fillna(df["adj"])
    elif len(todo):
        todo.assign(adjudicated="").to_csv(adj_path, index=False)
        print(f"\n{len(todo)} disagreements written to adjudication.csv - fill 'adjudicated' "
              "(include/exclude) and re-run for final metrics.")
    still = int(df["reference"].isna().sum())
    if still:
        print(f"note: {still} records still have no reference label and are left out")

    preds = ["final_verdict", "prior_human"] + [c for c in key.columns if c.endswith(("_label", "_verdict"))
                                                 and c != "final_verdict"]
    rng = np.random.default_rng(SEED)
    rows = []
    for scope, part in [("ALL", df)] + [(b, g) for b, g in df.groupby("bloc")]:
        part = part.reset_index(drop=True)
        for p in preds:
            if p not in part or part[p].isna().all():
                continue
            est = metrics(part[p], part["reference"], part["weight"])
            lo, hi = boot(part[p], part["reference"], part["weight"], rng)
            for mname, v in est.items():
                rows.append({"scope": scope, "screener": p, "metric": mname, "estimate": round(v, 3),
                             "ci_low": round(lo[mname], 3), "ci_high": round(hi[mname], 3),
                             "n": int(part[p].notna().sum())})
    res = pd.DataFrame(rows)
    path = os.path.join(a.dir, "validation_metrics.csv")
    res.to_csv(path, index=False)
    print("\n" + res[res.scope == "ALL"].to_string(index=False))
    print("\nwrote", path)
    if not len(res):
        sys.exit("no metrics computed - check labels")


if __name__ == "__main__":
    main()
