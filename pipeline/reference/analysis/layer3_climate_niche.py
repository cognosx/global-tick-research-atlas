# ============================================================
#  LAYER 3 — Climate-tick research as an emerging niche
#  Quantifies what share of the GENERAL tick literature engages with
#  climate, and whether that share is growing significantly over time.
#  This directly motivates the climate-tick SR: it's a small, recent
#  niche within a much larger field, not yet a mainstream sub-theme.
#
#  TWO climate signals, used together (this is a strength, not
#  redundancy — they answer different questions and a reviewer will
#  want to see they agree):
#    (a) screen_verdict == 'include' from the strict LLM screening
#        (your 3-part test: ticks AND climate AND assessed impact)
#    (b) BERTopic topic_label containing a climate-related term
#        (looser thematic signal — catches papers that touch climate
#         as part of a broader theme, e.g. "climate-driven range
#         expansion" clustered with vector ecology)
#
#  Method: annual fraction (climate-tick / all tick-relevant), then
#  Mann-Kendall trend test (robust, standard for environmental time
#  series, doesn't assume normality or linearity) for significance,
#  Sen's slope for the trend MAGNITUDE, and a simple change-point
#  scan to identify WHEN the niche took off.
#
#  Input : master_topic_modeled.parquet (needs topic_label; screen_verdict
#          optional — only present if you've run screening on this set)
#  Output: climate_fraction_by_year.csv, climate_trend_test.json
# ============================================================
import os, re, json
import pandas as pd, numpy as np
import pymannkendall as mk

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET = os.path.join(WORK, "master_topic_modeled.parquet")
OUT_SERIES = os.path.join(WORK, "climate_fraction_by_year.csv")
OUT_TEST   = os.path.join(WORK, "climate_trend_test.json")

MIN_YEAR_FOR_TREND = 2000   # exclude sparse early years from the trend test
                              # (your year_filter cutoff; keeps consistent with the rest of the pipeline)
MIN_N_PER_YEAR = 20          # years with fewer than this many tick-relevant papers
                              # are too noisy for a stable fraction; flagged, not silently dropped

# NOTE: BERTopic labels are underscore-joined (e.g. "0_genomics_climate_vector"),
# so  word-boundary anchors FAIL between an underscore and a letter (both count
# as word characters in regex). Match the bare terms without boundary anchors,
# which is safe here since these are distinctive enough roots not to false-match
# inside unrelated words.
CLIMATE_TERMS = re.compile(
    r"climat|weather|temperatur|rainfall|precipitat|"
    r"warming|seasonal|drought|humidity", re.I)

def parse_year(s):
    m = re.search(r"(19|20)\d{2}", str(s or ""))
    return int(m.group(0)) if m else None

def run():
    df = pd.read_parquet(IN_PARQUET)
    print(f"Loaded {len(df)} records")
    df["_year"] = df["year"].map(parse_year) if "year" in df.columns else None
    df = df[df["_year"].notna()].copy()
    df["_year"] = df["_year"].astype(int)

    # ---- signal (b): topic label mentions a climate-related term ----
    df["_topic_climate"] = df.get("topic_label","").astype(str).str.contains(CLIMATE_TERMS, na=False)

    # ---- signal (a): strict screening verdict, if present ----
    has_screen = "screen_verdict" in df.columns and df["screen_verdict"].notna().any()
    if has_screen:
        df["_screen_climate"] = df["screen_verdict"].eq("include")
        print(f"Screening signal available: {df['_screen_climate'].sum()} climate-tick includes")
    else:
        df["_screen_climate"] = np.nan
        print("NOTE: no screen_verdict column found — using topic-based signal only. "
              "Re-run after screening completes for the dual-signal comparison.")

    yearly = df.groupby("_year").agg(
        n_total=("record_id","count"),
        n_climate_topic=("_topic_climate","sum"),
        n_climate_screen=("_screen_climate","sum") if has_screen else ("_topic_climate","sum"),
    ).reset_index().rename(columns={"_year":"year"})
    yearly["frac_topic"] = yearly["n_climate_topic"] / yearly["n_total"]
    if has_screen:
        yearly["frac_screen"] = yearly["n_climate_screen"] / yearly["n_total"]
    yearly["reliable"] = yearly["n_total"] >= MIN_N_PER_YEAR

    yearly.to_csv(OUT_SERIES, index=False)
    print(f"\nSaved annual series -> {OUT_SERIES}")
    print(yearly[["year","n_total","frac_topic"] + (["frac_screen"] if has_screen else [])
                 + ["reliable"]].to_string(index=False))

    n_unreliable = (~yearly["reliable"]).sum()
    if n_unreliable:
        print(f"\nNOTE: {n_unreliable} year(s) have <{MIN_N_PER_YEAR} tick-relevant papers — "
              f"their fraction is noisy; consider excluding from the plotted trend line "
              f"or widening to multi-year bins for those years.")

    # ---- trend test on the reliable, >=MIN_YEAR_FOR_TREND window ----
    trend_window = yearly[(yearly.year >= MIN_YEAR_FOR_TREND) & (yearly.reliable)].sort_values("year")
    results = {}
    for sig, col in [("topic", "frac_topic")] + ([("screen", "frac_screen")] if has_screen else []):
        if len(trend_window) < 8:
            print(f"\n[{sig}] Too few reliable years ({len(trend_window)}) for a trend test (need >=8).")
            continue
        series = trend_window[col].values
        mk_result = mk.original_test(series)
        results[sig] = {
            "trend": mk_result.trend, "p_value": float(mk_result.p),
            "tau": float(mk_result.Tau), "sens_slope_per_year": float(mk_result.slope),
            "n_years": len(series), "years": [int(y) for y in trend_window["year"]],
        }
        print(f"\n=== MANN-KENDALL TREND TEST [{sig}-based fraction] ===")
        print(f"  trend: {mk_result.trend}  (p={mk_result.p:.4g}, Kendall's tau={mk_result.Tau:.3f})")
        print(f"  Sen's slope: {mk_result.slope:+.5f} fraction-points/year "
              f"({mk_result.slope*100:+.3f} pct-points/year)")
        if mk_result.p < 0.05:
            direction = "increasing" if mk_result.slope > 0 else "decreasing"
            print(f"  -> SIGNIFICANT {direction} trend.")
        else:
            print(f"  -> No significant trend detected at alpha=0.05.")

    # ---- simple change-point scan: which year shows the steepest jump? ----
    # (descriptive, not a formal Chow test — flag candidate inflection years for
    #  visual confirmation against known field events, e.g. IPCC AR4 2007, Paris 2015)
    if len(trend_window) >= 6:
        diffs = trend_window.set_index("year")["frac_topic"].diff().dropna()
        candidate_year = int(diffs.idxmax())
        print(f"\nLargest single-year jump in climate-topic fraction: {candidate_year} "
              f"(+{diffs.max()*100:.2f} pct-points vs prior year).")
        print("  CAVEAT: this finds the single biggest year-over-year jump, which is a poor")
        print("  detector for a GRADUAL acceleration (a smooth ramp has no one obvious jump")
        print("  year). Treat this as a rough pointer, not a formal changepoint estimate.")
        print("  For a defensible inflection-point claim, fit a piecewise/segmented regression")
        print("  (e.g. the ruptures package, or a Chow test at candidate years like 2007/2015)")
        print("  and report THAT instead, especially if this heuristic looks unstable.")
        results["candidate_inflection_year"] = candidate_year

    with open(OUT_TEST, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved trend test results -> {OUT_TEST}")
    return yearly, results

if __name__ == "__main__":
    run()
