# ============================================================
#  SINGLE-MODEL SCREENING PASS  (Colab)  — run once per model
#  Each model screens ALL records independently, writing ONLY its own
#  columns ({model}_verdict / _reason / _conf) into a shared results file.
#  Run GPT first (fast) -> start reviewing -> run Claude & Gemini later,
#  each UPDATING the same file. Then run 14b_consensus to combine.
#
#  HOW TO USE: set MODEL_TO_RUN below, run the cell. Repeat for each model.
#  Safe to run the three in separate Colab sessions (different checkpoints).
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install anthropic openai google-genai pandas openpyxl

import os, re, json, time
import pandas as pd, numpy as np
from google.colab import userdata

# ============ CHOOSE THE MODEL FOR THIS PASS ============
MODEL_TO_RUN = "gpt"          # "gpt"  ->  run first;  then "claude",  then "gemini"
# ========================================================

CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL    = "gpt-4o-mini"           # fast, confirmed working
GEMINI_MODEL = "gemini-2.5-flash"      # fast, confirmed working
CHECKPOINT_EVERY = 100
MAX_RECORDS  = None     # e.g. 50 for a test

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
IN_PATH = os.path.join(OUT_DIR, "master_canonical_screen_ready.xlsx")
RESULTS = os.path.join(OUT_DIR, "screening_results.xlsx")     # shared file, updated by each pass
CKPT    = os.path.join(OUT_DIR, f"screen_ckpt_{MODEL_TO_RUN}.parquet")   # per-model checkpoint

ANTHROPIC_KEY = userdata.get("ANTHROPIC_API_KEY") if MODEL_TO_RUN=="claude" else ""
OPENAI_KEY    = userdata.get("OPENAI_API_KEY")    if MODEL_TO_RUN=="gpt"    else ""
GEMINI_KEY    = userdata.get("GEMINI_API_KEY")    if MODEL_TO_RUN=="gemini" else ""

CRITERIA = """You are screening titles and abstracts for a systematic review.

INCLUSION QUESTION: Does this study EVALUATE THE IMPACT OF CLIMATE (or weather, temperature, humidity, rainfall, seasonality, or climate change) ON TICKS (or tick populations, distribution, abundance, activity, phenology, or tick-borne disease transmission driven by climate)?

To INCLUDE, ALL THREE must be true:
 1. TICKS: the study concerns ticks (Ixodes, Amblyomma, Rhipicephalus, etc.) or tick-borne disease vectors.
 2. CLIMATE: the study involves a climate/weather/environmental/seasonal variable.
 3. IMPACT/RELATIONSHIP: the study actually ASSESSES a relationship, effect, or association between the climate variable and ticks. Merely MENTIONING both topics is NOT enough.

Apply STRICT interpretation: if the climate-tick relationship is only incidental, background, or speculative, EXCLUDE.
If you are genuinely UNSURE or the abstract is too thin to judge, mark "uncertain" (do NOT guess).

REASON CODES (use exactly one): INCLUDE, EXCLUDE_NOT_TICKS, EXCLUDE_NO_CLIMATE, EXCLUDE_NO_IMPACT, EXCLUDE_WRONG_TYPE, UNCERTAIN

Respond with ONLY a JSON object, no other text:
{"verdict":"include|exclude|uncertain","reason_code":"<one code>","confidence":"high|medium|low","justification":"<one sentence>"}"""

def build_user_msg(title, abstract, keywords, screen_input):
    body=f"TITLE: {title}\n\n"
    body+=f"ABSTRACT: {abstract}\n\n" if (screen_input=="title+abstract" and abstract) else "ABSTRACT: [not available — screen on title only; if insufficient, mark uncertain]\n\n"
    if keywords: body+=f"KEYWORDS: {keywords}\n"
    return body

def parse_json(txt):
    if not txt: return None
    m=re.search(r'\{.*\}',txt,re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def call_claude(sys,usr):
    import anthropic
    c=anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    r=c.messages.create(model=CLAUDE_MODEL,max_tokens=300,system=sys,messages=[{"role":"user","content":usr}])
    return parse_json("".join(b.text for b in r.content if b.type=="text"))

def call_gpt(sys,usr):
    from openai import OpenAI
    c=OpenAI(api_key=OPENAI_KEY)
    msgs=[{"role":"system","content":sys},{"role":"user","content":usr}]
    try: r=c.chat.completions.create(model=GPT_MODEL,max_completion_tokens=300,messages=msgs)
    except Exception: r=c.chat.completions.create(model=GPT_MODEL,temperature=0,max_tokens=300,messages=msgs)
    return parse_json(r.choices[0].message.content)

def call_gemini(sys,usr):
    from google import genai
    from google.genai import types
    client=genai.Client(api_key=GEMINI_KEY)
    r=client.models.generate_content(model=GEMINI_MODEL,contents=usr,
        config=types.GenerateContentConfig(system_instruction=sys,temperature=0,max_output_tokens=300))
    return parse_json(r.text)

CALLERS={"claude":call_claude,"gpt":call_gpt,"gemini":call_gemini}
def call_retry(fn,sys,usr,tries=3):
    for t in range(tries):
        try:
            o=fn(sys,usr)
            if o: return o
        except Exception:
            time.sleep(2**t)
    return None

# ---- self-test the one model ----
fn=CALLERS[MODEL_TO_RUN]
print(f"Self-test {MODEL_TO_RUN}...")
_t=call_retry(fn,CRITERIA,build_user_msg("Climate change alters Ixodes ricinus tick distribution",
    "We modelled how rising temperature affects tick abundance and range expansion.","ticks; climate","title+abstract"),tries=2)
if not _t or not _t.get("verdict"):
    raise SystemExit(f"{MODEL_TO_RUN} self-test FAILED — check key/model name. Nothing screened.")
print(f"  OK -> {_t.get('verdict')}/{_t.get('reason_code','')}\n")

# ---- load shared results file if it exists, else the screen-ready input ----
vcol,rcol,ccol=f"{MODEL_TO_RUN}_verdict",f"{MODEL_TO_RUN}_reason",f"{MODEL_TO_RUN}_conf"
if os.path.exists(RESULTS):
    df=pd.read_excel(RESULTS,sheet_name="screening",dtype=str).fillna("")
    print("Loaded shared results file:",len(df))
else:
    df=pd.read_excel(IN_PATH,sheet_name="canonical",dtype=str).fillna("")
    print("Started from screen-ready input:",len(df))
if MAX_RECORDS: df=df.head(MAX_RECORDS)
for c in [vcol,rcol,ccol]:
    if c not in df.columns: df[c]=""

# resume THIS model's pass (skip rows already having a real verdict for this model)
if os.path.exists(CKPT):
    prev=pd.read_parquet(CKPT).set_index("record_id")
    df=df.set_index("record_id"); df.update(prev); df=df.reset_index()
todo=df[~df[vcol].isin(["include","exclude","uncertain"])]
print(f"{MODEL_TO_RUN}: {len(todo)} records to screen ({len(df)-len(todo)} already done)\n")

t0=time.time()
for n,(idx,row) in enumerate(todo.iterrows()):
    usr=build_user_msg(row["title"],row.get("abstract",""),row.get("keywords",""),row.get("screen_input","title+abstract"))
    o=call_retry(fn,CRITERIA,usr)
    if o:
        df.at[idx,vcol]=o.get("verdict",""); df.at[idx,rcol]=o.get("reason_code",""); df.at[idx,ccol]=o.get("confidence","")
    else:
        df.at[idx,vcol]="error"
    if n and n%CHECKPOINT_EVERY==0:
        df.to_parquet(CKPT,index=False)
        el=time.time()-t0; rate=(n+1)/el; eta=(len(todo)-n-1)/rate if rate>0 else 0
        print(f"  {n+1}/{len(todo)} ({(n+1)/len(todo)*100:.1f}%) | {rate*60:.1f}/min | ETA ~{int(eta//3600)}h{int((eta%3600)//60):02d}m | {df.loc[df[vcol].ne(''),vcol].value_counts().to_dict()}")

df.to_parquet(CKPT,index=False)
# write/refresh the shared results file with THIS model's columns merged in
def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)
safe_xlsx(df,RESULTS,"screening")
print(f"\n{MODEL_TO_RUN} pass complete. Verdicts: {df[vcol].value_counts().to_dict()}")
print(f"Updated shared file: {RESULTS}")
print("Run the other models the same way (change MODEL_TO_RUN), then run 14b_consensus.")
