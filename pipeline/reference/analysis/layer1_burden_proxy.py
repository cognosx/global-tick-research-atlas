# ============================================================
#  LAYER 1 — Research output vs. tick-borne disease relevance (equity)
#  Builds country-level publication shares from the enriched corpus,
#  joins Sambado & Ryan (2026) covariates as predictors, and computes
#  output-normalized BARI-style residuals.
#
#  Positioning (cite explicitly in the paper):
#  Sambado & Ryan (2026, Commun Med) modeled whether countries REPORT
#  tick-borne VIRUSES (ZOVER presence/absence), using PubMed citations
#  only as a control. We instead model research OUTPUT SHARE itself as
#  the outcome, across the FULL tick literature (not just 14 viruses),
#  and add a topical-composition layer they did not examine.
#
#  Input : master_topic_modeled.parquet (post Stage A+B+C)
#          sambado_global_trait_matrix.csv (download from Dryad, see NOTE)
#  Output: country_output_table.parquet, bari_residuals.csv
# ============================================================
import os, re
import pandas as pd, numpy as np
from scipy import stats

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET = os.path.join(WORK, "master_topic_modeled.parquet")
# NOTE: download from https://datadryad.org/dataset/doi:10.5061/dryad.1g1jwsv9w
#       (Sambado & Ryan 2026 supplementary data) and place here:
TRAITS_CSV = os.path.join(WORK, "sambado_global_trait_matrix.csv")
OUT_TABLE  = os.path.join(WORK, "country_output_table.parquet")
OUT_BARI   = os.path.join(WORK, "bari_residuals.csv")

# ---- ISO3 normalization (oa_countries is ISO2 from OpenAlex) ----
import pycountry
def iso2_to_iso3(code):
    try:
        return pycountry.countries.get(alpha_2=code.strip().upper()).alpha_3
    except Exception:
        return None

SSA_ISO3 = {  # standard UN SSA list (World Bank definition), for grouping
 "AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COG","COD","CIV",
 "DJI","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR",
 "MDG","MWI","MLI","MRT","MUS","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC",
 "SLE","SOM","ZAF","SSD","SDN","TZA","TGO","UGA","ZMB","ZWE",
}

def explode_countries(df, col="oa_countries", sep=";"):
    """Each record may list multiple author-affiliation countries (oa_countries
    is semicolon-joined ISO2 codes from OpenAlex authorships). Explode to one
    row per (record, country) for fractional-or-whole counting."""
    if col not in df.columns: df[col] = ""
    rows = []
    for _, r in df.iterrows():
        codes = [c.strip() for c in str(r[col]).split(sep) if c.strip()]
        codes = list(dict.fromkeys(codes))  # de-dup within a record
        if not codes:
            continue
        for c in codes:
            rows.append({
                "record_id": r.get("record_id"),
                "iso2": c,
                "iso3": iso2_to_iso3(c),
                "year": r.get("year"),
                "topic_id": r.get("topic_id"),
                "topic_label": r.get("topic_label"),
                "n_countries_on_paper": len(codes),  # for fractional weighting
            })
    return pd.DataFrame(rows)

def run():
    df = pd.read_parquet(IN_PARQUET)
    print(f"Loaded {len(df)} records")
    exploded = explode_countries(df)
    exploded = exploded.dropna(subset=["iso3"])
    print(f"Exploded to {len(exploded)} (record, country) pairs across "
          f"{exploded['iso3'].nunique()} countries")

    # fractional credit: a paper with N author-countries gives 1/N credit to each
    # (standard bibliometric practice — avoids inflating multi-country papers)
    exploded["frac_credit"] = 1.0 / exploded["n_countries_on_paper"]

    country_tbl = (exploded.groupby("iso3")
                   .agg(whole_count=("record_id","nunique"),
                        frac_count=("frac_credit","sum"))
                   .reset_index())
    country_tbl["is_ssa"] = country_tbl["iso3"].isin(SSA_ISO3)
    country_tbl["world_share_whole"] = country_tbl["whole_count"] / country_tbl["whole_count"].sum()
    country_tbl["world_share_frac"]  = country_tbl["frac_count"]  / country_tbl["frac_count"].sum()

    print(f"\nSSA countries represented: {country_tbl[country_tbl.is_ssa]['iso3'].nunique()} of {len(SSA_ISO3)}")
    print(f"SSA share of whole-count tick output: "
          f"{country_tbl[country_tbl.is_ssa]['whole_count'].sum()/country_tbl['whole_count'].sum()*100:.2f}%")

    country_tbl.to_parquet(OUT_TABLE, index=False)
    print(f"\nSaved {OUT_TABLE}")

    # ---- join Sambado covariates if available, compute BARI-style residual ----
    if os.path.exists(TRAITS_CSV):
        traits = pd.read_csv(TRAITS_CSV)
        # NOTE: confirm the actual country-code column name after downloading;
        # Dryad supplementary tables commonly use 'iso_a3' or 'country_iso3'.
        id_col = next((c for c in traits.columns if c.lower() in
                       ("iso_a3","iso3","country_iso3","adm0_a3")), None)
        if id_col is None:
            print(f"\nWARNING: couldn't auto-detect ISO3 column in {TRAITS_CSV}. "
                  f"Columns found: {list(traits.columns)[:10]}... -> set id_col manually.")
        else:
            merged = country_tbl.merge(traits, left_on="iso3", right_on=id_col, how="left")
            # BARI-style: regress log(output) on log(population) [+ available covariates],
            # rank countries by standardized residual (positive = over-performing
            # relative to size; negative = under-performing / research gap)
            pop_col = next((c for c in merged.columns if "population" in c.lower()), None)
            if pop_col:
                m = merged[(merged["whole_count"]>0) & (merged[pop_col]>0)].copy()
                m["log_output"] = np.log10(m["whole_count"])
                m["log_pop"] = np.log10(m[pop_col])
                slope, intercept, r, p, se = stats.linregress(m["log_pop"], m["log_output"])
                m["predicted"] = intercept + slope*m["log_pop"]
                m["residual"] = m["log_output"] - m["predicted"]
                m["bari_z"] = (m["residual"] - m["residual"].mean()) / m["residual"].std()
                m.sort_values("bari_z").to_csv(OUT_BARI, index=False)
                print(f"\nRegression: log(output) ~ log(population): R={r:.3f}, p={p:.4g}")
                print(f"Saved ranked BARI residuals -> {OUT_BARI}")
                print("\nMost UNDER-performing (research gap) countries:")
                print(m.sort_values("bari_z")[["iso3","is_ssa","whole_count","bari_z"]].head(10).to_string(index=False))
                print("\nMost OVER-performing countries:")
                print(m.sort_values("bari_z",ascending=False)[["iso3","is_ssa","whole_count","bari_z"]].head(10).to_string(index=False))
            else:
                print("WARNING: no population column found in traits file for the regression.")
    else:
        print(f"\nNOTE: {TRAITS_CSV} not found. Download Sambado & Ryan (2026) supplementary "
              f"data from Dryad (doi:10.5061/dryad.1g1jwsv9w) and place it there to run "
              f"the BARI regression. Country output table is saved regardless.")
    return country_tbl

if __name__ == "__main__":
    run()
