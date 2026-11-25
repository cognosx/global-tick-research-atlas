"""Post-screening analysis stages for the bibliometric / science-of-science
study: tick-relevance gatekeeping, BERTopic classification, and the three
analysis layers (research-output-vs-burden equity, topic composition by
region, climate-as-emerging-niche)."""
from __future__ import annotations
import os
import re
import json
import pandas as pd
import numpy as np
from . import io as _io
from . import provenance as _prov
from .runner import F

# ----------------------------------------------------------------------
# STAGE A — tick relevance gatekeeping (rule-based)
# ----------------------------------------------------------------------
TICK_GENERA = [
    "ixodes", "amblyomma", "rhipicephalus", "boophilus", "dermacentor", "haemaphysalis",
    "hyalomma", "ornithodoros", "argas", "otobius", "carios", "nuttalliella",
    "rhipicentor", "margaropus", "anomalohimalaya", "cosmiomma", "nosomma",
]
TICK_FAMILY_TERMS = ["ixodidae", "argasidae", "ixodida", "nuttalliellidae", "acari"]
TICK_GENERIC = [r"\btick\b", r"\bticks\b", r"\btick-borne\b", r"\btick borne\b"]
TICK_BORNE_DISEASE_TERMS = [
    "lyme disease", "borreliosis", "anaplasmosis", "babesiosis", "ehrlichiosis",
    "rickettsiosis", "tick-borne encephalitis", "crimean-congo", "theileriosis",
    "tularemia", "relapsing fever", "q fever", "tick paralysis", "heartland virus",
    "bourbon virus", "powassan virus", "alpha-gal syndrome",
]


def _compile(terms, word_boundary=True):
    if word_boundary:
        return re.compile("|".join(r"\b" + re.escape(t) + r"\b" for t in terms), re.I)
    return re.compile("|".join(re.escape(t) for t in terms), re.I)


_genera_re = _compile(TICK_GENERA)
_family_re = _compile(TICK_FAMILY_TERMS)
_generic_re = re.compile("|".join(TICK_GENERIC), re.I)
_disease_re = _compile(TICK_BORNE_DISEASE_TERMS, word_boundary=False)


def _classify_relevance(title, abstract, keywords=""):
    text = f"{title or ''} {abstract or ''} {keywords or ''}"
    if _generic_re.search(text):
        return True, "tick_word", _generic_re.search(text).group(0)
    if _genera_re.search(text):
        return True, "genus", _genera_re.search(text).group(0)
    if _family_re.search(text):
        return True, "family", _family_re.search(text).group(0)
    if _disease_re.search(text):
        return True, "tick_borne_disease_only", _disease_re.search(text).group(0)
    return False, "none", ""


def run_tick_relevance(ctx):
    inp = ctx.find(F["screen_ready"], F["canon_enriched"], F["canon_clean"], F["canonical"], F["enriched"])
    df = _io.read_table(inp)
    print(f"Loaded {len(df)} rows")
    for c in ["title", "abstract", "keywords"]:
        if c not in df.columns:
            df[c] = ""
    results = df.apply(lambda r: _classify_relevance(r["title"], r["abstract"], r.get("keywords", "")), axis=1)
    df["tick_relevant"] = [x[0] for x in results]
    df["tick_match_type"] = [x[1] for x in results]
    df["tick_match_term"] = [x[2] for x in results]
    n = len(df)
    n_rel = df["tick_relevant"].sum()
    print(f"relevant: {n_rel} ({n_rel/n*100:.1f}%) | not relevant: {n-n_rel} ({(n-n_rel)/n*100:.1f}%)")
    irrelevant = df[~df["tick_relevant"]]
    audit_cols = [c for c in ["record_id", "study_id", "title", "abstract", "source_database"] if c in df.columns]
    _io.write_table(irrelevant[audit_cols], ctx.out("qc_not_tick_relevant"), fmt=ctx.fmt)
    _io.write_table(df, ctx.out(F["tick_checked"]), fmt=ctx.fmt)
    _prov.log_stage(ctx.work, "tick_relevance", inputs=[inp], output=ctx.out(F["tick_checked"]),
                     rows_in=n, rows_out=int(n_rel),
                     extra={"match_types": df[df.tick_relevant]["tick_match_type"].value_counts().to_dict()})
    return df


# ----------------------------------------------------------------------
# STAGE B/C — BERTopic classification
# ----------------------------------------------------------------------
def run_topic_model(ctx):
    import time
    inp = ctx.find(F["tick_checked"])
    df = _io.read_table(inp)
    only_relevant = ctx.cfg.get("topic_model", "only_tick_relevant", default=True)
    if only_relevant and "tick_relevant" in df.columns:
        before = len(df)
        df = df[df["tick_relevant"]].reset_index(drop=True)
        print(f"Filtered to tick_relevant: {len(df)}/{before}")
    for c in ["title", "abstract"]:
        if c not in df.columns:
            df[c] = ""

    def usable_abs(s):
        s = str(s).strip()
        return len(s) >= 40 and not re.match(r'^(no abstract|not available|n/?a|none)\.?$', s, re.I)

    df["_has_abs"] = df["abstract"].map(usable_abs)
    df["topic_text"] = np.where(df["_has_abs"], df["title"].astype(str) + ". " + df["abstract"].astype(str), df["title"])
    df["topic_text"] = df["topic_text"].map(lambda s: s if len(str(s).strip()) >= 8 else "untitled record")
    docs = df["topic_text"].tolist()
    print(f"Corpus for topic modeling: {len(df)} docs ({df['_has_abs'].mean()*100:.1f}% with usable abstract)")

    embed_model = ctx.cfg.get("topic_model", "embed_model",
                              default="paraphrase-multilingual-mpnet-base-v2")
    min_topic_size = int(ctx.cfg.get("topic_model", "min_topic_size", default=150))
    reduce_outliers = ctx.cfg.get("topic_model", "reduce_outliers", default=True)

    emb_cache = ctx.out("topic_embeddings.npy")
    embeddings = None
    if os.path.exists(emb_cache):
        cached = np.load(emb_cache)
        if len(cached) == len(docs):
            print(f"Loading cached embeddings from {emb_cache}")
            embeddings = cached

    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction import text as sk_text

    embedder = None
    if embeddings is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embed_model)
        t0 = time.time()
        embeddings = embedder.encode(docs, batch_size=256, show_progress_bar=True,
                                     normalize_embeddings=True)
        np.save(emb_cache, embeddings)
        print(f"Embedding done in {time.time()-t0:.0f}s -> cached to {emb_cache}")

    domain_stop = ["study", "studies", "result", "results", "using", "based", "data",
                   "analysis", "article", "paper", "review", "present", "presented",
                   "show", "showed", "shown", "found", "conclusion", "conclusions",
                   "background", "objective", "objectives", "method", "methods",
                   "significant", "significantly"]
    combined_stop = list(sk_text.ENGLISH_STOP_WORDS.union(domain_stop))
    vectorizer_model = CountVectorizer(stop_words=combined_stop, ngram_range=(1, 2), min_df=1, max_df=0.95)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                       random_state=42, low_memory=True)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                             cluster_selection_method="eom", prediction_data=True)
    topic_model = BERTopic(embedding_model=embedder, umap_model=umap_model, hdbscan_model=hdbscan_model,
                            vectorizer_model=vectorizer_model, nr_topics="auto",
                            calculate_probabilities=False, verbose=True)

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    if reduce_outliers and -1 in topics:
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"Reassigning {n_outliers} outlier docs ({n_outliers/len(topics)*100:.1f}%)...")
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
        topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)
        topics = new_topics

    df["topic_id"] = topics
    info = topic_model.get_topic_info()
    label_map = dict(zip(info["Topic"], info["Name"]))
    df["topic_label"] = df["topic_id"].map(label_map)
    print(info.sort_values("Count", ascending=False).head(20)[["Topic", "Count", "Name"]].to_string(index=False))

    info.to_csv(ctx.out("topic_summary.csv"), index=False)
    try:
        topic_model.save(ctx.out("bertopic_model"), serialization="safetensors",
                          save_ctfidf=True, save_embedding_model=embed_model)
    except Exception as e:
        print(f"NOTE: could not save full model ({e}); topic_id/topic_label still saved in output.")
    out_df = df.drop(columns=["_has_abs", "topic_text"])
    _io.write_table(out_df, ctx.out(F["topic_modeled"]), fmt=ctx.fmt)
    _prov.log_stage(ctx.work, "topic_model", inputs=[inp], output=ctx.out(F["topic_modeled"]),
                     rows_out=len(out_df), extra={"n_topics": int(info["Topic"].nunique())})
    return out_df


# ----------------------------------------------------------------------
# LAYER 1 — research output vs. tick-borne disease relevance (equity)
# ----------------------------------------------------------------------
SSA_ISO3 = {
    "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "COD", "CIV",
    "DJI", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR",
    "MDG", "MWI", "MLI", "MRT", "MUS", "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC",
    "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "UGA", "ZMB", "ZWE",
}


def _iso2_to_iso3(code):
    import pycountry
    try:
        return pycountry.countries.get(alpha_2=str(code).strip().upper()).alpha_3
    except Exception:
        return None


def _explode_countries(df, col="oa_countries", sep=";"):
    rows = []
    for _, r in df.iterrows():
        codes = [c.strip() for c in str(r.get(col, "")).split(sep) if c.strip()]
        codes = list(dict.fromkeys(codes))
        if not codes:
            continue
        for c in codes:
            rows.append({"record_id": r.get("record_id"), "iso2": c, "iso3": _iso2_to_iso3(c),
                         "year": r.get("year"), "topic_id": r.get("topic_id"),
                         "topic_label": r.get("topic_label"), "n_countries_on_paper": len(codes)})
    return pd.DataFrame(rows)


def run_layer1_burden(ctx):
    from scipy import stats
    inp = ctx.find(F["analysis_ready"], F["topic_modeled"], F["canon_enriched"], F["canonical"])
    df = _io.read_table(inp)
    print(f"Loaded {len(df)} records")
    exploded = _explode_countries(df)
    exploded = exploded.dropna(subset=["iso3"])
    print(f"Exploded to {len(exploded)} (record, country) pairs across {exploded['iso3'].nunique()} countries")
    exploded["frac_credit"] = 1.0 / exploded["n_countries_on_paper"]
    country_tbl = (exploded.groupby("iso3").agg(whole_count=("record_id", "nunique"),
                                                  frac_count=("frac_credit", "sum")).reset_index())
    country_tbl["is_ssa"] = country_tbl["iso3"].isin(SSA_ISO3)
    country_tbl["world_share_whole"] = country_tbl["whole_count"] / country_tbl["whole_count"].sum()
    country_tbl["world_share_frac"] = country_tbl["frac_count"] / country_tbl["frac_count"].sum()
    _io.write_table(country_tbl, ctx.out("country_output_table"), fmt=ctx.fmt)

    ssa_share = country_tbl[country_tbl.is_ssa]['whole_count'].sum() / country_tbl['whole_count'].sum() * 100
    print(f"SSA share of tick output: {ssa_share:.2f}%")

    traits_csv = ctx.out("sambado_global_trait_matrix.csv")
    result_summary = {"ssa_countries_represented": int(country_tbl[country_tbl.is_ssa]['iso3'].nunique()),
                       "ssa_share_pct": round(ssa_share, 2)}
    if os.path.exists(traits_csv):
        traits = pd.read_csv(traits_csv)
        id_col = next((c for c in traits.columns if c.lower() in
                       ("iso_a3", "iso3", "country_iso3", "adm0_a3")), None)
        pop_col = next((c for c in traits.columns if "population" in c.lower()), None)
        if id_col and pop_col:
            merged = country_tbl.merge(traits, left_on="iso3", right_on=id_col, how="left")
            m = merged[(merged["whole_count"] > 0) & (merged[pop_col] > 0)].copy()
            m["log_output"] = np.log10(m["whole_count"])
            m["log_pop"] = np.log10(m[pop_col])
            slope, intercept, r, p, se = stats.linregress(m["log_pop"], m["log_output"])
            m["predicted"] = intercept + slope * m["log_pop"]
            m["residual"] = m["log_output"] - m["predicted"]
            m["bari_z"] = (m["residual"] - m["residual"].mean()) / m["residual"].std()
            _io.write_table(m.sort_values("bari_z"), ctx.out("bari_residuals"), fmt="csv")
            result_summary.update({"regression_r": round(r, 3), "regression_p": round(p, 4)})
            print(f"Regression: log(output) ~ log(population): R={r:.3f}, p={p:.4g}")
        else:
            print(f"NOTE: couldn't find ISO3/population columns in {traits_csv}; skipping regression.")
    else:
        print(f"NOTE: {traits_csv} not found (Sambado & Ryan 2026 supplementary data). "
              f"Country output table saved regardless; download from Dryad to run the BARI regression.")
    _prov.log_stage(ctx.work, "layer1_burden", inputs=[inp], rows_out=len(country_tbl), extra=result_summary)
    return country_tbl


# ----------------------------------------------------------------------
# LAYER 2 — topical composition by region
# ----------------------------------------------------------------------
def _assign_region(oa_countries_str):
    codes = [c.strip() for c in str(oa_countries_str or "").split(";") if c.strip()]
    iso3s = [_iso2_to_iso3(c) for c in codes]
    if any(i in SSA_ISO3 for i in iso3s if i):
        return "SSA"
    if iso3s and any(i for i in iso3s):
        return "non-SSA"
    return "unknown"


def run_layer2_topic_region(ctx):
    from scipy import stats
    inp = ctx.find(F["analysis_ready"], F["topic_modeled"])
    df = _io.read_table(inp)
    print(f"Loaded {len(df)} records")
    if "topic_label" not in df.columns:
        raise SystemExit("No topic_label column — run topic_model stage first.")
    df["region"] = df["oa_countries"].map(_assign_region) if "oa_countries" in df.columns else "unknown"
    known = df[df["region"].isin(["SSA", "non-SSA"])].copy()
    known = known[known["topic_id"] != -1] if "topic_id" in known.columns else known
    print(f"Usable for topic x region test: {len(known)} "
          f"({known['region'].eq('SSA').sum()} SSA, {known['region'].eq('non-SSA').sum()} non-SSA)")

    ct = pd.crosstab(known["topic_label"], known["region"])
    _io.write_table(ct.reset_index(), ctx.out("topic_region_crosstab"), fmt="csv")

    chi2, p, dof, expected = stats.chi2_contingency(ct[["SSA", "non-SSA"]])
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    print(f"chi2={chi2:.2f} dof={dof} p={p:.4g} Cramer's V={cramers_v:.3f}")

    rows = []
    ssa_total, other_total = ct["SSA"].sum(), ct["non-SSA"].sum()
    for topic in ct.index:
        a = ct.loc[topic, "SSA"]
        b = ssa_total - a
        c = ct.loc[topic, "non-SSA"]
        d = other_total - c
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        log_or = np.log(odds_ratio)
        se = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))
        ci_low, ci_high = np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se)
        rows.append({"topic": topic, "ssa_n": a, "ssa_pct": round(a / ssa_total * 100, 2) if ssa_total else 0,
                     "non_ssa_n": c, "non_ssa_pct": round(c / other_total * 100, 2) if other_total else 0,
                     "odds_ratio": round(odds_ratio, 3), "log_odds_ratio": round(log_or, 3),
                     "ci_low": round(ci_low, 3), "ci_high": round(ci_high, 3),
                     "direction": "SSA over-represented" if odds_ratio > 1 else "SSA under-represented",
                     "significant": not (ci_low < 1 < ci_high)})
    odds = pd.DataFrame(rows).sort_values("log_odds_ratio")
    _io.write_table(odds, ctx.out("topic_region_oddsratios"), fmt="csv")

    fig = ct.copy()
    fig["SSA_pct"] = fig["SSA"] / fig["SSA"].sum() * 100
    fig["non-SSA_pct"] = fig["non-SSA"] / fig["non-SSA"].sum() * 100
    _io.write_table(fig.reset_index(), ctx.out("topic_region_figure_data"), fmt="csv")

    _prov.log_stage(ctx.work, "layer2_topic_region", inputs=[inp], rows_out=len(known),
                     extra={"chi2": round(chi2, 2), "p_value": round(p, 6), "cramers_v": round(cramers_v, 3)})
    return ct, odds


# ----------------------------------------------------------------------
# LAYER 3 — climate-tick research as an emerging niche
# ----------------------------------------------------------------------
CLIMATE_TERMS = re.compile(
    r"climat|weather|temperatur|rainfall|precipitat|warming|seasonal|drought|humidity", re.I)


def run_layer3_climate_niche(ctx):
    import pymannkendall as mk
    inp = ctx.find(F["analysis_ready"], F["topic_modeled"])
    df = _io.read_table(inp)
    print(f"Loaded {len(df)} records")
    df["_year"] = df["year"].map(_parse_year_l3) if "year" in df.columns else None
    df = df[df["_year"].notna()].copy()
    df["_year"] = df["_year"].astype(int)
    df["_topic_climate"] = df.get("topic_label", "").astype(str).str.contains(CLIMATE_TERMS, na=False)
    has_screen = "screen_verdict" in df.columns and df["screen_verdict"].notna().any()
    if has_screen:
        df["_screen_climate"] = df["screen_verdict"].eq("include")
    else:
        df["_screen_climate"] = np.nan
        print("NOTE: no screen_verdict column — using topic-based signal only.")

    agg = {"n_total": ("record_id", "count"), "n_climate_topic": ("_topic_climate", "sum")}
    if has_screen:
        agg["n_climate_screen"] = ("_screen_climate", "sum")
    yearly = df.groupby("_year").agg(**agg).reset_index().rename(columns={"_year": "year"})
    yearly["frac_topic"] = yearly["n_climate_topic"] / yearly["n_total"]
    if has_screen:
        yearly["frac_screen"] = yearly["n_climate_screen"] / yearly["n_total"]
    min_n = int(ctx.cfg.get("layer3", "min_n_per_year", default=20))
    yearly["reliable"] = yearly["n_total"] >= min_n
    _io.write_table(yearly, ctx.out("climate_fraction_by_year"), fmt="csv")
    print(yearly.to_string(index=False))

    min_year = int(ctx.cfg.get("year_filter", "min_year", default=2000))
    trend_window = yearly[(yearly.year >= min_year) & (yearly.reliable)].sort_values("year")
    results = {}
    for sig, col in [("topic", "frac_topic")] + ([("screen", "frac_screen")] if has_screen else []):
        if len(trend_window) < 8:
            print(f"[{sig}] Too few reliable years for a trend test.")
            continue
        series = trend_window[col].values
        r = mk.original_test(series)
        results[sig] = {"trend": r.trend, "p_value": float(r.p), "tau": float(r.Tau),
                         "sens_slope_per_year": float(r.slope), "n_years": len(series)}
        print(f"[{sig}] trend={r.trend} p={r.p:.4g} tau={r.Tau:.3f} slope={r.slope:+.5f}/yr")

    if len(trend_window) >= 6:
        diffs = trend_window.set_index("year")["frac_topic"].diff().dropna()
        candidate_year = int(diffs.idxmax())
        print(f"Largest single-year jump: {candidate_year} (+{diffs.max()*100:.2f} pct-points). "
              f"CAVEAT: poor detector for a gradual ramp — consider segmented regression for a "
              f"defensible inflection-point claim.")
        results["candidate_inflection_year"] = candidate_year

    with open(ctx.out("climate_trend_test.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    _prov.log_stage(ctx.work, "layer3_climate_niche", inputs=[inp], rows_out=len(df), extra=results)
    return yearly, results


def _parse_year_l3(s):
    m = re.search(r"(19|20)\d{2}", str(s or ""))
    return int(m.group(0)) if m else None


# ----------------------------------------------------------------------
# STAGE — finalize corpus (drop non-tick noise topics + re-apply year)
# ----------------------------------------------------------------------
def run_finalize_corpus(ctx):
    """Produce the clean analysis-ready corpus:
      (1) drop NON-TICK homonym topics flagged in the project taxonomy
          (topic modelling doubles as a QC filter — 'tick box', 'tick
          size', 'tick-tock', clock/quantum, etc. self-cluster and are
          removed here);
      (2) re-apply the year window on the ENRICHMENT-CORRECTED year
          (the original year_filter ran before OpenAlex revised years,
          so a few pre-min-year and future-dated records slipped through);
      (3) attach clean labels + theme groups from the taxonomy.
    Taxonomy path comes from config: analysis.topic_taxonomy_csv (a path,
    absolute or relative to WORK). If absent, falls back to a configured
    drop-id list analysis.noise_topic_ids, else keeps all topics."""
    inp = ctx.find(F["topic_modeled"])
    df = _io.read_table(inp)
    n0 = len(df)
    print(f"loaded {n0} topic-modeled records")

    min_year = int(ctx.cfg.get("year_filter", "min_year", default=2000))
    max_year = int(ctx.cfg.get("analysis", "max_year", default=2026))

    # resolve taxonomy
    tax_path = ctx.cfg.get("analysis", "topic_taxonomy_csv", default=None)
    drop_ids, keep_map, theme_map = set(), {}, {}
    if tax_path:
        if not os.path.isabs(tax_path):
            tax_path = os.path.join(ctx.work, tax_path)
        if os.path.exists(tax_path):
            tax = pd.read_csv(tax_path)
            drop_ids = set(tax[tax["keep_or_drop"].str.upper() == "DROP"]["topic_id"])
            keep_map = dict(zip(tax["topic_id"], tax.get("clean_label", tax["topic_id"])))
            theme_map = dict(zip(tax["topic_id"], tax.get("theme_group", "")))
            print(f"taxonomy: {tax_path} ({len(drop_ids)} topics flagged DROP)")
        else:
            print(f"NOTE: taxonomy csv not found at {tax_path}; no topic drops applied.")
    else:
        noise = ctx.cfg.get("analysis", "noise_topic_ids", default=[])
        drop_ids = set(noise)
        if drop_ids:
            print(f"using config noise_topic_ids: {sorted(drop_ids)}")
        else:
            print("no taxonomy and no noise_topic_ids configured; keeping all topics.")

    before = len(df)
    if drop_ids and "topic_id" in df.columns:
        df = df[~df["topic_id"].isin(drop_ids)].copy()
    n_noise = before - len(df)
    if keep_map:
        df["topic_clean_label"] = df["topic_id"].map(keep_map)
        df["theme_group"] = df["topic_id"].map(theme_map)
    print(f"(1) dropped {n_noise} non-tick-topic records -> {len(df)}")

    df["_year"] = df["year"].map(_parse_year_l3) if "year" in df.columns else None
    before = len(df)
    n_pre = int((df["_year"] < min_year).sum())
    n_fut = int((df["_year"] > max_year).sum())
    n_noyr = int(df["_year"].isna().sum())
    df = df[df["_year"].between(min_year, max_year)].drop(columns=["_year"]).copy()
    print(f"(2) year window [{min_year},{max_year}]: dropped {n_pre} pre-min, {n_fut} future, "
          f"{n_noyr} no-year -> {len(df)}")

    _io.write_table(df, ctx.out(F["analysis_ready"]), fmt=ctx.fmt,
                    also_xlsx=ctx.cfg.get("io", "also_write_xlsx", default=False),
                    xlsx_max_rows=ctx.cfg.get("io", "xlsx_max_rows", default=200000))
    _prov.log_stage(ctx.work, "finalize_corpus", inputs=[inp], output=ctx.out(F["analysis_ready"]),
                     rows_in=n0, rows_out=len(df),
                     extra={"dropped_noise": n_noise, "dropped_pre_year": n_pre,
                            "dropped_future": n_fut, "dropped_no_year": n_noyr})
    print(f"\nsaved {ctx.out(F['analysis_ready'])} ({len(df)} rows) = analysis-ready corpus")
    return df


# ----------------------------------------------------------------------
# STAGE — optional translation of non-English title/abstract
# ----------------------------------------------------------------------
def run_translate(ctx, keys=None, batch_size=20, target_langs=None):
    """Translate non-English title+abstract to English (title_en/abstract_en)
    for the analysis-ready corpus. Optional: use ONLY if you want permanent
    English text (dashboards, LLM screening on non-English records). For
    better topic clustering alone, prefer a multilingual embed_model in the
    topic_model config instead (no API cost). Resumable via checkpoint."""
    import json, time, re, requests  # noqa
    keys = keys or {k: os.environ.get(k, "") for k in ["ANTHROPIC_API_KEY"]}
    inp = ctx.find(F["analysis_ready"], F["topic_modeled"])
    df = _io.read_table(inp)
    for c in ["title", "abstract", "oa_language"]:
        if c not in df.columns:
            df[c] = ""

    _CYR = re.compile(r'[\u0400-\u04ff]'); _CJK = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff]')
    _ARA = re.compile(r'[\u0600-\u06ff]'); _LAT = re.compile(r'[àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿœ]', re.I)

    def is_en(r):
        lang = str(r.get("oa_language", "") or "").strip().lower()
        if lang:
            return lang in ("en", "eng", "english")
        t = str(r.get("title", "") or "") + " " + str(r.get("abstract", "") or "")
        if _CYR.search(t) or _CJK.search(t) or _ARA.search(t) or _LAT.search(t):
            return False
        return True

    df["_is_en"] = df.apply(is_en, axis=1)
    non_en = df[~df["_is_en"]]
    if target_langs:
        non_en = non_en[non_en["oa_language"].str.lower().isin(target_langs)]
    print(f"non-English records to translate: {len(non_en)}/{len(df)} "
          f"({len(non_en)/max(len(df),1)*100:.1f}%)")
    if len(non_en) == 0:
        df["title_en"] = df["title"]; df["abstract_en"] = df["abstract"]; df["translated"] = False
        _io.write_table(df.drop(columns=["_is_en"]), ctx.out(F["translated"]), fmt=ctx.fmt)
        return df

    import anthropic
    client = anthropic.Anthropic(api_key=keys.get("ANTHROPIC_API_KEY", ""))
    ckpt = ctx.out("translation_checkpoint.json")
    results = {}
    if os.path.exists(ckpt):
        results = {d["record_id"]: d for d in json.load(open(ckpt)).get("results", [])}
    todo = non_en[~non_en["record_id"].isin(results.keys())].to_dict("records")
    print(f"remaining: {len(todo)}")

    def prompt(recs):
        items = "".join(f'[{r["record_id"]}]\nTITLE: {(r.get("title") or "")[:300]}\n'
                        f'ABSTRACT: {(r.get("abstract") or "")[:1200]}\n\n' for r in recs)
        return ("Translate ONLY non-English text to English. Return a JSON array of "
                '{"record_id","title_en","abstract_en"}. Already-English fields: reproduce '
                "unchanged. Preserve scientific terms. No commentary.\n\n" + items)

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        for att in range(3):
            try:
                msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=8000,
                                              messages=[{"role": "user", "content": prompt(batch)}])
                m = re.search(r'\[.*\]', "".join(b.text for b in msg.content if b.type == "text"), re.S)
                for r in (json.loads(m.group(0)) if m else []):
                    if r.get("record_id"):
                        results[r["record_id"]] = r
                break
            except Exception as e:
                print(f"  batch {i//batch_size} attempt {att+1}: {e}"); time.sleep(2 ** att)
        json.dump({"results": list(results.values())}, open(ckpt, "w"))
        if (i // batch_size) % 5 == 0:
            print(f"  ...{i+len(batch)}/{len(todo)}")

    df = df.set_index("record_id")
    for rid, r in results.items():
        if rid in df.index:
            df.at[rid, "title_en"] = r.get("title_en", df.at[rid, "title"])
            df.at[rid, "abstract_en"] = r.get("abstract_en", df.at[rid, "abstract"])
    df = df.reset_index()
    m_en = df["_is_en"]
    for c in ["title_en", "abstract_en"]:
        if c not in df.columns:
            df[c] = ""
    df.loc[m_en, "title_en"] = df.loc[m_en, "title"]
    df.loc[m_en, "abstract_en"] = df.loc[m_en, "abstract"]
    df["translated"] = ~df["_is_en"]
    df = df.drop(columns=["_is_en"])
    _io.write_table(df, ctx.out(F["translated"]), fmt=ctx.fmt)
    _prov.log_stage(ctx.work, "translate", inputs=[inp], output=ctx.out(F["translated"]),
                     rows_out=len(df), extra={"translated": int(df["translated"].sum())})
    print(f"saved {ctx.out(F['translated'])} ({int(df['translated'].sum())} translated)")
    return df
