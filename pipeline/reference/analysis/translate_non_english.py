# ============================================================
#  OPTIONAL: Translate non-English title+abstract to English
#  before topic modeling (or for any downstream use).
#
#  WHY THIS EXISTS: if you want the translated text as a
#  permanent column in the corpus (e.g. for manual review,
#  LLM screening, or dashboards that show English text for
#  non-English papers), translate FIRST, then run topic_model.
#  If you only want better topic clustering, use
#  topic_model_multilingual.py instead — same result, no cost.
#
#  APPROACH: translate ONLY records where oa_language != 'en'
#  (or heuristically detected as non-English). This is a much
#  smaller job (~5-10% of corpus, ~5-7k records) than
#  translating everything. Uses the Anthropic API in batches
#  (free with your existing API key; no extra cost beyond
#  normal token usage).
#
#  Input : master_analysis_ready.parquet
#  Output: adds columns title_en, abstract_en, translated (bool)
#          writes master_analysis_ready_translated.parquet
#  Resumable: a checkpoint tracks already-translated record_ids.
# ============================================================
import os, re, json, time
import pandas as pd, numpy as np
import anthropic

try:
    WORK
except NameError:
    BASE = "/content/drive/MyDrive/ticks"
    WORK = os.path.join(BASE, "preprocessing_data_v2")

IN_PARQUET   = os.path.join(WORK, "master_analysis_ready.parquet")
OUT_PARQUET  = os.path.join(WORK, "master_analysis_ready_translated.parquet")
CKPT         = os.path.join(WORK, "translation_checkpoint.json")

BATCH_SIZE   = 20       # records per API call (fits in one context window easily)
MAX_TOKENS   = 8000     # per API call; 20 short abstracts fit comfortably
TARGET_LANGS = None     # None = translate ALL non-English; or e.g. ['fr','pt','ru','zh','de','es','ar']

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---- language detection heuristic (cheap, no API needed) ----
# Uses oa_language from OpenAlex if available, else character-based detection.
_LATIN_EXT  = re.compile(r'[àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿœ]', re.I)
_CYRILLIC   = re.compile(r'[\u0400-\u04ff]')
_CJK        = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]')
_ARABIC     = re.compile(r'[\u0600-\u06ff]')

def _is_english(row):
    lang = str(row.get("oa_language","") or "").strip().lower()
    if lang:
        return lang in ("en","eng","english")
    # fallback: check text
    text = str(row.get("title","") or "") + " " + str(row.get("abstract","") or "")
    if _CYRILLIC.search(text) or _CJK.search(text) or _ARABIC.search(text):
        return False
    if _LATIN_EXT.search(text):
        return False   # Latin-extended = likely Romance/Germanic non-English
    return True        # default to English if uncertain

def _build_prompt(records):
    """Single prompt to translate a batch of records."""
    items = ""
    for r in records:
        items += (f'[{r["record_id"]}]\n'
                  f'TITLE: {(r.get("title") or "")[:300]}\n'
                  f'ABSTRACT: {(r.get("abstract") or "")[:1200]}\n\n')
    return (
        "You are a scientific translation assistant. Translate ONLY the text that is "
        "NOT already in English. Return a JSON array where each element has:\n"
        '  {"record_id": "...", "title_en": "...", "abstract_en": "..."}\n'
        "If a field is already in English, reproduce it unchanged. "
        "Preserve scientific terminology exactly. No extra commentary.\n\n"
        "Records to translate:\n\n" + items
    )

def _parse_response(text):
    m = re.search(r'\[.*\]', text, re.S)
    if not m: return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []

def run():
    df = pd.read_parquet(IN_PARQUET)
    print(f"Loaded {len(df)} records")
    for c in ["title","abstract","oa_language"]:
        if c not in df.columns: df[c] = ""

    df["_is_en"] = df.apply(_is_english, axis=1)
    non_en = df[~df["_is_en"]]
    print(f"Detected {len(non_en)} non-English records out of {len(df)} "
          f"({len(non_en)/len(df)*100:.1f}%) — only these will be translated")

    if TARGET_LANGS:
        non_en = non_en[non_en["oa_language"].str.lower().isin(TARGET_LANGS)]
        print(f"  Filtered to target languages {TARGET_LANGS}: {len(non_en)} records")

    if len(non_en) == 0:
        print("Nothing to translate — corpus is already fully in English.")
        df["title_en"] = df["title"]; df["abstract_en"] = df["abstract"]; df["translated"] = False
        df.drop(columns=["_is_en"]).to_parquet(OUT_PARQUET, index=False)
        return df

    # load checkpoint
    done = set()
    results = {}
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        done = set(ck.get("done", []))
        results = {d["record_id"]: d for d in ck.get("results", [])}
        print(f"Checkpoint found: {len(done)} already translated")

    todo = non_en[~non_en["record_id"].isin(done)]
    print(f"Remaining to translate: {len(todo)}")

    if not ANTHROPIC_KEY:
        raise SystemExit("ANTHROPIC_API_KEY not set. Add it via Colab Secrets.")
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    records = todo.to_dict("records")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]
        prompt = _build_prompt(batch)
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}]
                )
                parsed = _parse_response(msg.content[0].text)
                for r in parsed:
                    rid = r.get("record_id","")
                    if rid:
                        results[rid] = r
                        done.add(rid)
                break
            except Exception as e:
                print(f"  batch {i//BATCH_SIZE}: error attempt {attempt+1}: {e}")
                time.sleep(2**attempt)
        # save checkpoint after every batch
        json.dump({"done": list(done), "results": list(results.values())},
                  open(CKPT,"w"))
        if (i//BATCH_SIZE) % 5 == 0:
            print(f"  ...{i+len(batch)}/{len(records)} translated")
        time.sleep(0.3)

    # merge translations back into the full dataframe
    df = df.set_index("record_id")
    for rid, r in results.items():
        if rid in df.index:
            df.at[rid, "title_en"]   = r.get("title_en", df.at[rid, "title"])
            df.at[rid, "abstract_en"] = r.get("abstract_en", df.at[rid, "abstract"])
    df = df.reset_index()
    # English records: title_en = title, abstract_en = abstract
    mask_en = df["_is_en"]
    df.loc[mask_en, "title_en"]    = df.loc[mask_en, "title"]
    df.loc[mask_en, "abstract_en"] = df.loc[mask_en, "abstract"]
    for c in ["title_en","abstract_en"]:
        if c not in df.columns: df[c] = df[c.replace("_en","")]
    df["translated"] = ~df["_is_en"]
    df = df.drop(columns=["_is_en"])

    df.to_parquet(OUT_PARQUET, index=False)
    n_translated = df["translated"].sum()
    print(f"\nSaved {OUT_PARQUET} ({len(df)} rows)")
    print(f"  {n_translated} records have translated title_en/abstract_en")
    print(f"  {len(df)-n_translated} records are in English (title_en = title)")
    return df

if __name__ == "__main__":
    run()
