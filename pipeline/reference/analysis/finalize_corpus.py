# ============================================================
#  FINALIZE the analysis corpus:
#   (1) drop NON-TICK homonym topics revealed by BERTopic
#   (2) re-apply year >= MIN_YEAR on the ENRICHMENT-CORRECTED year
#       (the original year_filter ran before OpenAlex revised years,
#        so a few pre-2000 and future-dated records slipped through)
#   (3) attach clean labels + theme groups from topic_taxonomy.csv
#   writes master_analysis_ready.parquet  (+ a corpus-flow json)
# ============================================================
import os, re, json
import pandas as pd

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET  = os.path.join(WORK, "master_topic_modeled.parquet")
TAXONOMY    = os.path.join(WORK, "topic_taxonomy.csv")   # upload the file we built
OUT_PARQUET = os.path.join(WORK, "master_analysis_ready.parquet")
FLOW_JSON   = os.path.join(WORK, "corpus_construction_flow.json")

MIN_YEAR = 2000
MAX_YEAR = 2026          # drop future-dated metadata errors (e.g. a stray 2027)

# fallback noise list if taxonomy csv not present (topic_ids to drop)
NOISE_TOPIC_IDS = [3, 5, 8, 13, 21, 23, 35, 38]

def _year(s):
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(s or ""))
    return int(m.group(0)) if m else None

def run():
    df = pd.read_parquet(IN_PARQUET)
    n0 = len(df)
    print(f"loaded {n0:,} topic-modeled records")

    # (1) drop noise topics
    if os.path.exists(TAXONOMY):
        tax = pd.read_csv(TAXONOMY)
        drop_ids = set(tax[tax.keep_or_drop == "DROP"]["topic_id"])
        keep_map = dict(zip(tax.topic_id, tax.clean_label))
        theme_map = dict(zip(tax.topic_id, tax.theme_group))
    else:
        drop_ids = set(NOISE_TOPIC_IDS)
        keep_map, theme_map = {}, {}
        print("  (taxonomy csv not found — using built-in noise id list)")
    before = len(df)
    df = df[~df["topic_id"].isin(drop_ids)].copy()
    n_noise = before - len(df)
    print(f"(1) dropped {n_noise:,} NON-TICK homonym records ({len(drop_ids)} topics) -> {len(df):,}")

    # attach clean labels/themes
    if keep_map:
        df["topic_clean_label"] = df["topic_id"].map(keep_map)
        df["theme_group"] = df["topic_id"].map(theme_map)

    # (2) re-apply year window on corrected year
    df["_year"] = df["year"].map(_year)
    before = len(df)
    keep = df["_year"].between(MIN_YEAR, MAX_YEAR)
    n_preyr = int((df["_year"] < MIN_YEAR).sum())
    n_future = int((df["_year"] > MAX_YEAR).sum())
    n_noyr = int(df["_year"].isna().sum())
    df = df[keep].copy()
    print(f"(2) year window [{MIN_YEAR},{MAX_YEAR}]: dropped {n_preyr} pre-{MIN_YEAR}, "
          f"{n_future} future, kept {len(df):,} (records with no year dropped: {n_noyr})")

    df = df.drop(columns=["_year"])
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nsaved {OUT_PARQUET}  ({len(df):,} rows) = clean analysis-ready corpus")

    flow = {
        "topic_modeled_input": int(n0),
        "removed_nontick_noise": int(n_noise),
        "removed_pre_min_year": int(n_preyr),
        "removed_future_year": int(n_future),
        "removed_missing_year": int(n_noyr),
        "final_analysis_corpus": int(len(df)),
        "min_year": MIN_YEAR, "max_year": MAX_YEAR,
    }
    json.dump(flow, open(FLOW_JSON, "w"), indent=2)
    print(f"saved {FLOW_JSON}")
    return df

if __name__ == "__main__":
    run()
