# ============================================================
#  STAGE C (v2) — BERTopic with multilingual embeddings
#  Replaces all-MiniLM-L6-v2 with paraphrase-multilingual-mpnet-base-v2
#  so French/Portuguese/Spanish/Russian/Chinese/Arabic papers cluster
#  by scientific THEME rather than by language.
#
#  Drop-in replacement for the original topic_model stage — identical
#  interface, same output columns (topic_id, topic_label), same caching
#  of embeddings to avoid re-running if you tweak UMAP/HDBSCAN params.
#
#  Input : master_analysis_ready.parquet  (cleaned, noise topics removed)
#  Output: master_topic_modeled_v2.parquet
#          topic_summary_v2.csv
#          bertopic_model_v2/
#          topic_embeddings_v2.npy  (cached; ~278M model, takes ~30-40min
#                                   on Colab GPU for 70k docs vs ~10min
#                                   for MiniLM — worth it for accuracy)
# ============================================================
import os, re, time, json
import numpy as np, pandas as pd

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET   = os.path.join(WORK, "master_analysis_ready.parquet")
OUT_PARQUET  = os.path.join(WORK, "master_topic_modeled_v2.parquet")
SUMMARY_CSV  = os.path.join(WORK, "topic_summary_v2.csv")
MODEL_DIR    = os.path.join(WORK, "bertopic_model_v2")
EMB_CACHE    = os.path.join(WORK, "topic_embeddings_v2.npy")

# ---- key change from v1 ----
EMBED_MODEL   = "paraphrase-multilingual-mpnet-base-v2"   # 50+ languages, ~278MB
# alternative if memory is very tight: "paraphrase-multilingual-MiniLM-L12-v2" (~100MB)
MIN_TOPIC_SIZE = 100   # slightly lower than v1 (150) since corpus is smaller after cleanup
REDUCE_OUTLIERS = True

def _usable(s):
    s = str(s or "").strip()
    return len(s) >= 40 and not re.match(r'^(no abstract|not available|n/?a|none)\.?$', s, re.I)

def run():
    df = pd.read_parquet(IN_PARQUET)
    print(f"Loaded {len(df)} analysis-ready records")
    for c in ["title","abstract"]:
        if c not in df.columns: df[c] = ""
    df["title"]    = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["_has_abs"] = df["abstract"].map(_usable)
    df["topic_text"] = np.where(
        df["_has_abs"],
        df["title"] + ". " + df["abstract"],
        df["title"]
    )
    df["topic_text"] = df["topic_text"].map(
        lambda s: s if len(str(s).strip()) >= 8 else "untitled record")
    docs = df["topic_text"].tolist()
    print(f"Corpus: {len(docs)} docs "
          f"({df['_has_abs'].mean()*100:.1f}% with usable abstract, "
          f"{(df.get('oa_language','en') != 'en').sum() if 'oa_language' in df.columns else '?'} non-English)")

    # ---- load or generate embeddings ----
    embeddings = None
    if os.path.exists(EMB_CACHE):
        cached = np.load(EMB_CACHE)
        if len(cached) == len(docs):
            print(f"Loading cached multilingual embeddings from {EMB_CACHE}")
            embeddings = cached
        else:
            print(f"Cache size mismatch ({len(cached)} vs {len(docs)}) -> re-embedding")

    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction import text as sk_text

    embedder = SentenceTransformer(EMBED_MODEL)
    if embeddings is None:
        t0 = time.time()
        print(f"\nEmbedding {len(docs)} docs with {EMBED_MODEL} "
              f"(~278MB model, GPU strongly recommended)...")
        embeddings = embedder.encode(
            docs, batch_size=128, show_progress_bar=True,
            normalize_embeddings=True)   # cosine-ready
        np.save(EMB_CACHE, embeddings)
        print(f"Embedding done in {(time.time()-t0)/60:.1f} min -> cached to {EMB_CACHE}")

    domain_stop = ["study","studies","result","results","using","based","data","analysis",
                   "article","paper","review","present","presented","show","showed","shown",
                   "found","conclusion","conclusions","background","objective","method","methods",
                   "significant","significantly","infection","infected","detected","samples",
                   "collected","total","number","used","reported","including","also","due"]
    combined_stop = list(sk_text.ENGLISH_STOP_WORDS.union(domain_stop))
    vectorizer = CountVectorizer(
        stop_words=combined_stop, ngram_range=(1, 2), min_df=1, max_df=0.95)
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=42, low_memory=True)
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True)

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=True,
    )
    t0 = time.time()
    print("\nFitting BERTopic (UMAP + HDBSCAN)...")
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    print(f"Fit done in {(time.time()-t0)/60:.1f} min | "
          f"raw topics: {len(set(topics)) - (1 if -1 in topics else 0)}")

    if REDUCE_OUTLIERS and -1 in topics:
        n_out = sum(1 for t in topics if t == -1)
        print(f"Reassigning {n_out} outlier docs ({n_out/len(topics)*100:.1f}%)...")
        topics = topic_model.reduce_outliers(
            docs, topics, strategy="embeddings", embeddings=embeddings)
        topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer)

    df["topic_id"] = topics
    info = topic_model.get_topic_info()
    label_map = dict(zip(info["Topic"], info["Name"]))
    df["topic_label"] = df["topic_id"].map(label_map)

    print("\n=== TOP 25 TOPICS BY SIZE ===")
    print(info.sort_values("Count", ascending=False).head(25)
          [["Topic","Count","Name"]].to_string(index=False))

    info.to_csv(SUMMARY_CSV, index=False)
    df.drop(columns=["_has_abs","topic_text"]).to_parquet(OUT_PARQUET, index=False)
    try:
        topic_model.save(MODEL_DIR, serialization="safetensors",
                          save_ctfidf=True, save_embedding_model=EMBED_MODEL)
        print(f"\nSaved model -> {MODEL_DIR}")
    except Exception as e:
        print(f"NOTE: model save failed ({e}); topic_id/topic_label still in {OUT_PARQUET}")

    print(f"\nSaved: {OUT_PARQUET}  ({len(df)} rows)")
    print(f"       {SUMMARY_CSV}")
    return df, topic_model

if __name__ == "__main__":
    run()
