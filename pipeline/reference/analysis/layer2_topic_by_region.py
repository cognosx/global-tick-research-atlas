# ============================================================
#  LAYER 2 — Topical composition by region (SSA vs. global)
#  Tests whether SSA's tick research is topically narrower/different
#  from the global field, and identifies WHICH topics drive the gap.
#
#  Method: chi-square test of independence (topic x region), then
#  per-topic log-odds ratios + standardized residuals to localize
#  which specific topics SSA is over/under-represented in.
#
#  Input : master_topic_modeled.parquet (has topic_id/topic_label)
#          country_output_table.parquet (from layer1, for SSA flag)
#  Output: topic_region_crosstab.csv, topic_region_oddsratios.csv
# ============================================================
import os
import pandas as pd, numpy as np
from scipy import stats

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET   = os.path.join(WORK, "master_topic_modeled.parquet")
COUNTRY_TBL  = os.path.join(WORK, "country_output_table.parquet")
OUT_CROSSTAB = os.path.join(WORK, "topic_region_crosstab.csv")
OUT_ODDS     = os.path.join(WORK, "topic_region_oddsratios.csv")
OUT_FIGDATA  = os.path.join(WORK, "topic_region_figure_data.csv")

SSA_ISO3 = {
 "AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COG","COD","CIV",
 "DJI","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR",
 "MDG","MWI","MLI","MRT","MUS","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC",
 "SLE","SOM","ZAF","SSD","SDN","TZA","TGO","UGA","ZMB","ZWE",
}

import pycountry
def iso2_to_iso3(code):
    try:
        return pycountry.countries.get(alpha_2=str(code).strip().upper()).alpha_3
    except Exception:
        return None

def assign_region(oa_countries_str):
    """A paper may have multiple author countries. Region assignment rule
    (stated explicitly, since this choice matters for interpretation):
    a paper counts as SSA-affiliated if ANY author affiliation is in SSA.
    This is the standard 'any African co-author' convention used in equity
    bibliometrics (e.g. Confraria & Godinho 2015) — it credits African
    institutions' participation even in international collaborations,
    rather than requiring SSA to be the ONLY or FIRST affiliation."""
    codes = [c.strip() for c in str(oa_countries_str or "").split(";") if c.strip()]
    iso3s = [iso2_to_iso3(c) for c in codes]
    if any(i in SSA_ISO3 for i in iso3s if i):
        return "SSA"
    if iso3s and any(i for i in iso3s):
        return "non-SSA"
    return "unknown"

def run():
    df = pd.read_parquet(IN_PARQUET)
    print(f"Loaded {len(df)} records")
    if "topic_label" not in df.columns:
        raise SystemExit("No topic_label column — run the BERTopic stage first.")

    df["region"] = df["oa_countries"].map(assign_region) if "oa_countries" in df.columns else "unknown"
    print("\nRegion assignment:")
    print(df["region"].value_counts().to_string())

    known = df[df["region"].isin(["SSA","non-SSA"])].copy()
    # drop the BERTopic outlier/noise bucket if any remain (topic_id == -1)
    known = known[known["topic_id"] != -1]
    print(f"\nUsable for topic x region test: {len(known)} "
          f"({known['region'].eq('SSA').sum()} SSA, {known['region'].eq('non-SSA').sum()} non-SSA)")

    # ---- crosstab: topic x region, counts ----
    ct = pd.crosstab(known["topic_label"], known["region"])
    ct.to_csv(OUT_CROSSTAB)
    print(f"\nSaved crosstab -> {OUT_CROSSTAB}")

    # ---- chi-square test of independence: does topic distribution differ by region? ----
    chi2, p, dof, expected = stats.chi2_contingency(ct[["SSA","non-SSA"]])
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))  # effect size, 0-1
    print(f"\n=== CHI-SQUARE TEST (topic x region independence) ===")
    print(f"  chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4g}")
    print(f"  Cramer's V (effect size) = {cramers_v:.3f}  "
          f"({'small' if cramers_v<0.1 else 'medium' if cramers_v<0.3 else 'large'} effect)")
    if p < 0.05:
        print("  -> SSA's topic distribution DIFFERS significantly from non-SSA's.")
    else:
        print("  -> No significant difference detected (check N; may be underpowered).")

    # ---- per-topic log-odds ratio: which topics drive the difference? ----
    rows = []
    ssa_total = ct["SSA"].sum()
    other_total = ct["non-SSA"].sum()
    for topic in ct.index:
        a = ct.loc[topic, "SSA"]                 # SSA papers in this topic
        b = ssa_total - a                         # SSA papers NOT in this topic
        c = ct.loc[topic, "non-SSA"]              # non-SSA papers in this topic
        d = other_total - c                       # non-SSA papers NOT in this topic
        # Haldane-Anscombe correction (+0.5) to handle zero cells safely
        odds_ratio = ((a+0.5)*(d+0.5)) / ((b+0.5)*(c+0.5))
        log_or = np.log(odds_ratio)
        se = np.sqrt(1/(a+0.5) + 1/(b+0.5) + 1/(c+0.5) + 1/(d+0.5))
        ci_low, ci_high = np.exp(log_or - 1.96*se), np.exp(log_or + 1.96*se)
        ssa_pct = a/ssa_total*100 if ssa_total else 0
        other_pct = c/other_total*100 if other_total else 0
        rows.append({
            "topic": topic, "ssa_n": a, "ssa_pct": round(ssa_pct,2),
            "non_ssa_n": c, "non_ssa_pct": round(other_pct,2),
            "odds_ratio": round(odds_ratio,3), "log_odds_ratio": round(log_or,3),
            "ci_low": round(ci_low,3), "ci_high": round(ci_high,3),
            "direction": "SSA over-represented" if odds_ratio>1 else "SSA under-represented",
            "significant": not (ci_low < 1 < ci_high),  # CI excludes 1 -> significant
        })
    odds = pd.DataFrame(rows).sort_values("log_odds_ratio")
    odds.to_csv(OUT_ODDS, index=False)

    print(f"\n=== TOPIC-LEVEL ODDS RATIOS (SSA vs non-SSA) ===")
    print("Most UNDER-represented topics in SSA (significant only):")
    sig_under = odds[(odds.significant) & (odds.odds_ratio<1)].head(8)
    print(sig_under[["topic","ssa_pct","non_ssa_pct","odds_ratio"]].to_string(index=False))
    print("\nMost OVER-represented topics in SSA (significant only):")
    sig_over = odds[(odds.significant) & (odds.odds_ratio>1)].sort_values("log_odds_ratio",ascending=False).head(8)
    print(sig_over[["topic","ssa_pct","non_ssa_pct","odds_ratio"]].to_string(index=False))

    # ---- figure-ready long format: % composition per region per topic ----
    fig = ct.copy()
    fig["SSA_pct"] = fig["SSA"]/fig["SSA"].sum()*100
    fig["non-SSA_pct"] = fig["non-SSA"]/fig["non-SSA"].sum()*100
    fig.reset_index().to_csv(OUT_FIGDATA, index=False)
    print(f"\nSaved figure-ready composition data -> {OUT_FIGDATA}")
    print(f"Saved odds ratios -> {OUT_ODDS}")

    return ct, odds

if __name__ == "__main__":
    run()
