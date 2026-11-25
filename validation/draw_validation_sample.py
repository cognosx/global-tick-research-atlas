#!/usr/bin/env python3
"""
Draw a BLIND, STRATIFIED validation sample for the GTRA screening.

Why: earlier metrics compared the LLMs with labels that came mostly from
reviewing LLM disagreements. Records where all models agreed were never checked
by a person, so shared model errors were invisible. This sample checks them.

Design (default n = 600):
  strata = region bloc (ssa / maghreb / other)
         x agreement (all models agree / models disagree)
         x LLM verdict (include / exclude)
  Equal quotas per bloc (200 each) so Africa is not drowned out by "other".
  Each record carries a design weight = stratum population / stratum sample,
  so metrics can be weighted back to the whole corpus.

Outputs (in --out):
  reviewer_A.xlsx, reviewer_B.xlsx  -> BLIND sheets: no verdicts, no bloc, own order
  sample_key.csv                    -> PRIVATE: review_id -> oa_id, stratum, weight, verdicts
  sample_design.csv                 -> population and sample size per stratum

Usage (Colab):
  python validation/draw_validation_sample.py \
      --input "/content/drive/MyDrive/ticks/preprocessing_data/tickclass2/screened_llm_v3.parquet" \
      --out   "/content/drive/MyDrive/ticks/validation_v1"
Column names are auto-detected; override with --id-col, --title-col, etc.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SEED = 20261125

CANDIDATES = {
    "id": ["oa_id", "openalex_id", "id", "record_id", "study_id"],
    "title": ["oa_title", "title"],
    "abstract": ["oa_abstract", "abstract", "abstract_text"],
    "year": ["oa_year", "year", "publication_year"],
    "venue": ["oa_journal", "journal", "venue", "source_title"],
    "language": ["oa_language", "language", "lang"],
    "doi": ["doi", "oa_doi"],
    "bloc": ["_bloc", "bloc", "region_bloc"],
    "final": ["llm_label", "screen_verdict", "final_verdict", "consensus"],
    "human": ["final_include", "human_label", "gold_class", "human_class"],
}
MODEL_HINTS = ("gpt_label", "deepseek_label", "claude_label", "gemini_label",
               "openai_verdict", "anthropic_verdict", "google_verdict")


def pick(df, key, override=None):
    if override:
        if override not in df.columns:
            sys.exit(f"--{key}-col '{override}' not in columns")
        return override
    for c in CANDIDATES[key]:
        if c in df.columns:
            return c
    return None


def to_binary(s):
    """Map many label spellings to 1 (include), 0 (exclude), NaN (other)."""
    t = s.astype(str).str.strip().str.lower()
    inc = t.isin(["1", "1.0", "true", "yes", "include", "included", "tick", "relevant", "y"])
    exc = t.isin(["0", "0.0", "false", "no", "exclude", "excluded", "not_tick", "irrelevant", "n"])
    out = pd.Series(np.nan, index=s.index)
    out[inc] = 1
    out[exc] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--disagree-share", type=float, default=0.2,
                    help="share of each bloc quota given to model-disagreement records")
    for k in CANDIDATES:
        ap.add_argument(f"--{k}-col")
    ap.add_argument("--model-cols", help="comma-separated per-model verdict columns")
    a = ap.parse_args()

    df = pd.read_parquet(a.input) if a.input.endswith(".parquet") else pd.read_csv(a.input, low_memory=False)
    print(f"loaded {len(df):,} rows, {df.shape[1]} columns")

    col = {k: pick(df, k, getattr(a, f"{k}_col")) for k in CANDIDATES}
    for k in ("id", "title", "final"):
        if not col[k]:
            sys.exit(f"could not find a '{k}' column; pass --{k}-col")
    models = [c.strip() for c in a.model_cols.split(",")] if a.model_cols else \
             [c for c in df.columns if c in MODEL_HINTS]
    print("columns used:", {k: v for k, v in col.items() if v}, "| models:", models)

    df = df.drop_duplicates(col["id"]).reset_index(drop=True)
    df["_final"] = to_binary(df[col["final"]])
    df = df[df["_final"].notna()].copy()
    df["_bloc"] = df[col["bloc"]].astype(str).str.lower() if col["bloc"] else "all"

    if models:
        mv = pd.DataFrame({m: to_binary(df[m]) for m in models})
        valid = mv.notna().sum(axis=1)
        agree = (mv.nunique(axis=1, dropna=True) <= 1) & (valid >= 2)
        df["_agree"] = np.where(agree, "agree", "disagree")
    else:
        df["_agree"] = "unknown"
    df["_stratum"] = df["_bloc"] + "|" + df["_agree"] + "|" + df["_final"].map({1: "inc", 0: "exc"})

    blocs = sorted(df["_bloc"].unique())
    per_bloc = a.n // len(blocs)
    rng = np.random.default_rng(SEED)
    parts, design = [], []
    for b in blocs:
        sub = df[df["_bloc"] == b]
        quotas = {}
        for agr, share in (("agree", 1 - a.disagree_share), ("disagree", a.disagree_share),
                           ("unknown", 1.0)):
            for v in ("inc", "exc"):
                quotas[f"{b}|{agr}|{v}"] = int(round(per_bloc * share / 2))
        pops = {s: int((sub["_stratum"] == s).sum()) for s in quotas}
        present = {s: min(q, pops[s]) for s, q in quotas.items() if pops[s]}
        # move quota that small or empty strata cannot fill to the strata with most room
        short = per_bloc - sum(present.values())
        while short > 0:
            room = {s: pops[s] - present[s] for s in present if pops[s] > present[s]}
            if not room:
                break
            present[max(room, key=room.get)] += 1
            short -= 1
        for s, q in present.items():
            pop = sub[sub["_stratum"] == s]
            k = min(q, len(pop))
            take = pop.sample(k, random_state=int(rng.integers(1e9)))
            take = take.assign(_weight=len(pop) / k if k else np.nan)
            parts.append(take)
            design.append({"stratum": s, "population": len(pop), "sampled": k,
                           "weight": round(len(pop) / k, 3) if k else None})

    samp = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    samp["review_id"] = [f"R{i + 1:04d}" for i in range(len(samp))]
    os.makedirs(a.out, exist_ok=True)

    def g(c):
        return samp[col[c]] if col[c] else ""

    blind = pd.DataFrame({
        "review_id": samp["review_id"], "title": g("title"), "abstract": g("abstract"),
        "year": g("year"), "venue": g("venue"), "language": g("language"), "doi": g("doi"),
        "Q1_tick_focused": "", "Q2_climate_effect": "", "confidence": "", "note": "",
    })
    for r, seed in (("A", 1), ("B", 2)):  # each reviewer sees a different order
        sheet = blind.sample(frac=1, random_state=SEED + seed)
        path = os.path.join(a.out, f"reviewer_{r}.xlsx")
        with pd.ExcelWriter(path) as xw:
            sheet.to_excel(xw, index=False, sheet_name="review")
            pd.DataFrame({"field": ["Q1_tick_focused", "Q2_climate_effect", "confidence"],
                          "allowed values": ["include / exclude / unsure",
                                             "yes / no / unsure",
                                             "high / medium / low"]}
                         ).to_excel(xw, index=False, sheet_name="how_to_fill")
        print("wrote", path)

    key = pd.DataFrame({
        "review_id": samp["review_id"], "oa_id": samp[col["id"]], "bloc": samp["_bloc"],
        "agreement": samp["_agree"], "stratum": samp["_stratum"], "weight": samp["_weight"],
        "final_verdict": samp["_final"].astype(int),
        # in the verified set an empty final_include means exclude
        "prior_human": (to_binary(samp[col["human"]]).fillna(0) if col["human"] == "final_include"
                        else to_binary(samp[col["human"]])) if col["human"] else np.nan,
    })
    for m in models:
        key[m] = to_binary(samp[m])
    key.to_csv(os.path.join(a.out, "sample_key.csv"), index=False)
    pd.DataFrame(design).to_csv(os.path.join(a.out, "sample_design.csv"), index=False)
    print(f"sample: {len(samp)} records across {len(design)} strata")
    print(pd.DataFrame(design).to_string(index=False))


if __name__ == "__main__":
    main()
