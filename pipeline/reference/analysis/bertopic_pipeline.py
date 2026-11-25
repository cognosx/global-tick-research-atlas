# ============================================================
#  STAGE C — BERTopic classification for tick research corpus
#  Tick-Borne Diseases / Climate / Sub-Saharan Africa project
#
#  Discovers corpus-specific topics (vector ecology, pathogen biology,
#  control/acaricides, vaccines, epidemiology, climate drivers, etc.)
#  from title+abstract text. Complements OpenAlex's oa_topic/oa_field
#  (broader, pre-existing categories) with a tick-research-specific
#  breakdown that OpenAlex's general taxonomy won't surface.
#
#  Input : master_tick_checked.parquet (tick_relevant == True rows)
#  Output: master_topic_modeled.parquet (+ topic_id, topic_label, topic_prob)
#          bertopic_model/ (saved model, for inspection/reuse)
#          topic_summary.csv (topic -> top terms, doc count)
# ============================================================
import os, re, time
import pandas as pd, numpy as np

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET   = os.path.join(WORK, "master_tick_checked.parquet")
OUT_PARQUET  = os.path.join(WORK, "master_topic_modeled.parquet")
MODEL_DIR    = os.path.join(WORK, "bertopic_model")
SUMMARY_CSV  = os.path.join(WORK, "topic_summary.csv")

# ---- config ----
ONLY_TICK_RELEVANT = True       # model topics only on records that passed Stage A
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, solid quality. For scientific text upgrade
                                   # to "allenai-specter" (slower, often better for papers).
MIN_TOPIC_SIZE = 150              # smaller -> more, finer topics; larger -> fewer, broader
NR_TOPICS = "auto"                # int to force a target count, or "auto" for BERTopic's own reduction
REDUCE_OUTLIERS = True             # reassign -1 (noise) documents to their nearest real topic
RANDOM_STATE = 42
BATCH_SIZE_EMBED = 256             # embedding batch size (GPU-friendly)
CALC_PROBABILITIES = False        # True = soft-membership probabilities (slower, more RAM at 114k)

def load_corpus():
    df = pd.read_parquet(IN_PARQUET)
    if ONLY_TICK_RELEVANT and "tick_relevant" in df.columns:
        before = len(df)
        df = df[df["tick_relevant"]].reset_index(drop=True)
        print(f"Filtered to tick_relevant: {len(df)}/{before}")
    for c in ["title","abstract"]:
        if c not in df.columns: df[c] = ""
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    # build the modeling text: title + abstract; fall back to title-only if no abstract
    def usable_abs(s):
        s = s.strip()
        return len(s) >= 40 and not re.match(r'^(no abstract|not available|n/?a|none)\.?$', s, re.I)
    df["_has_abs"] = df["abstract"].map(usable_abs)
    df["topic_text"] = np.where(df["_has_abs"], df["title"] + ". " + df["abstract"], df["title"])
    # guard against empty/too-short text (BERTopic chokes on empty strings)
    df["topic_text"] = df["topic_text"].map(lambda s: s if len(s.strip()) >= 8 else "untitled record")
    print(f"Corpus for topic modeling: {len(df)} docs "
          f"({df['_has_abs'].mean()*100:.1f}% with usable abstract)")
    return df

def build_model(load_embedder=True):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    embedder = None
    if load_embedder:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL)

    # UMAP: low_memory=True is important at 100k+ docs on a constrained runtime.
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                       metric="cosine", random_state=RANDOM_STATE, low_memory=True)

    # HDBSCAN: min_cluster_size controls topic granularity directly.
    hdbscan_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, metric="euclidean",
                             cluster_selection_method="eom", prediction_data=True)

    # domain stopwords on top of English: generic scientific boilerplate that
    # would otherwise dominate every topic's top-terms and make them unreadable.
    domain_stop = ["study","studies","result","results","using","based","data",
                   "analysis","article","paper","review","present","presented",
                   "show","showed","shown","found","conclusion","conclusions",
                   "background","objective","objectives","method","methods",
                   "significant","significantly","p value","et al"]
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1,2),
                                        min_df=5, max_df=0.6)
    # extend the stopword list inside the vectorizer's vocabulary filtering
    # (CountVectorizer stop_words is a fixed list; combine via a wrapper if needed)
    from sklearn.feature_extraction import text as sk_text
    combined_stop = list(sk_text.ENGLISH_STOP_WORDS.union(domain_stop))
    # min_df as an ABSOLUTE count can conflict with max_df (a fraction) on small
    # topic-document groups during c-TF-IDF re-vectorization inside BERTopic
    # (max_df share of a small group < min_df count -> sklearn raises ValueError).
    # Using min_df=1 with a generous max_df avoids this at any corpus/topic size;
    # rare/noisy terms are still suppressed via the stopword list + top_n_words.
    vectorizer_model = CountVectorizer(stop_words=combined_stop, ngram_range=(1,2),
                                        min_df=1, max_df=0.95)

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=NR_TOPICS,
        calculate_probabilities=CALC_PROBABILITIES,
        verbose=True,
    )
    return topic_model, embedder

def run():
    df = load_corpus()
    docs = df["topic_text"].tolist()

    # Cache embeddings to disk: embedding 100k+ docs is the slowest step (GPU-bound).
    # If you need to re-tune UMAP/HDBSCAN params, re-run without re-embedding.
    emb_cache = os.path.join(WORK, "topic_embeddings.npy")
    embeddings = None
    if os.path.exists(emb_cache):
        cached = np.load(emb_cache)
        if len(cached) == len(docs):
            print(f"Loading cached embeddings from {emb_cache} (skipping embed model load)")
            embeddings = cached
        else:
            print("Cache size mismatch with current corpus -> re-embedding.")
    topic_model, embedder = build_model(load_embedder=(embeddings is None))
    if embeddings is None:
        t0 = time.time()
        print(f"\nEmbedding {len(docs)} documents with {EMBED_MODEL}...")
        embeddings = embedder.encode(docs, batch_size=BATCH_SIZE_EMBED, show_progress_bar=True)
        np.save(emb_cache, embeddings)
        print(f"Embedding done in {time.time()-t0:.0f}s -> cached to {emb_cache}")

    t0 = time.time()
    print("\nFitting BERTopic (UMAP + HDBSCAN)...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    print(f"Fit done in {time.time()-t0:.0f}s. Topics found: {len(set(topics)) - (1 if -1 in topics else 0)}")

    if REDUCE_OUTLIERS and -1 in topics:
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"\nReassigning {n_outliers} outlier docs ({n_outliers/len(topics)*100:.1f}%) "
              f"to nearest topic...")
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings",
                                                   embeddings=embeddings)
        topic_model.update_topics(docs, topics=new_topics, vectorizer_model=topic_model.vectorizer_model)
        topics = new_topics

    df["topic_id"] = topics
    info = topic_model.get_topic_info()
    label_map = dict(zip(info["Topic"], info["Name"]))
    df["topic_label"] = df["topic_id"].map(label_map)

    print("\n=== TOPIC SUMMARY (top 20 by size) ===")
    print(info.sort_values("Count", ascending=False).head(20)[["Topic","Count","Name"]].to_string(index=False))

    info.to_csv(SUMMARY_CSV, index=False)
    topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True, save_embedding_model=EMBED_MODEL)
    df.drop(columns=["_has_abs","topic_text"]).to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved:\n  {OUT_PARQUET} (+ topic_id, topic_label)\n  {SUMMARY_CSV}\n  {MODEL_DIR}/ (reusable model)")
    return df, topic_model

if __name__ == "__main__":
    run()
