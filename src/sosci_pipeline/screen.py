"""Screening stages: per-model LLM pass + consensus + validation. Config-driven
(criteria, models, consensus rule all from YAML)."""
from __future__ import annotations
import os
import re
import json
import time
import pandas as pd
import numpy as np
from . import io as _io
from . import provenance as _prov
from .runner import F


def _parse_json(txt):
    if not txt:
        return None
    m = re.search(r'\{.*\}', txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _build_prompt(cfg):
    q = cfg.get("screening", "question", default="")
    crit = cfg.get("screening", "criteria", default="")
    codes = cfg.get("screening", "reason_codes", default=["INCLUDE", "EXCLUDE", "UNCERTAIN"])
    return (f"You are screening titles/abstracts for a systematic review.\n\n"
            f"INCLUSION QUESTION: {q}\n\n{crit}\n\n"
            f"REASON CODES (use exactly one): {', '.join(codes)}\n\n"
            'Respond ONLY with JSON: {"verdict":"include|exclude|uncertain",'
            '"reason_code":"<one code>","confidence":"high|medium|low",'
            '"justification":"<one sentence>"}')


def _call(provider, model, sys, usr, keys):
    if provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=keys.get("ANTHROPIC_API_KEY", ""))
        r = c.messages.create(model=model, max_tokens=300, system=sys,
                               messages=[{"role": "user", "content": usr}])
        return _parse_json("".join(b.text for b in r.content if b.type == "text"))
    if provider == "openai":
        from openai import OpenAI
        c = OpenAI(api_key=keys.get("OPENAI_API_KEY", ""))
        try:
            r = c.chat.completions.create(model=model, max_completion_tokens=300,
                                           messages=[{"role": "system", "content": sys},
                                                     {"role": "user", "content": usr}])
        except Exception:
            r = c.chat.completions.create(model=model, temperature=0, max_tokens=300,
                                           messages=[{"role": "system", "content": sys},
                                                     {"role": "user", "content": usr}])
        return _parse_json(r.choices[0].message.content)
    if provider == "google":
        from google import genai
        from google.genai import types
        cl = genai.Client(api_key=keys.get("GEMINI_API_KEY", ""))
        r = cl.models.generate_content(model=model, contents=usr,
                                        config=types.GenerateContentConfig(
                                            system_instruction=sys, temperature=0, max_output_tokens=300))
        return _parse_json(r.text)
    return None


def run_screen(ctx, model_label=None, keys=None, max_records=None):
    keys = keys or {k: os.environ.get(k, "") for k in
                    ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]}
    models = ctx.cfg.get("screening", "models", default=[])
    if model_label is None:
        model_label = models[0]["provider"]
    spec = next((m for m in models if m["provider"] == model_label or m["model"] == model_label), None)
    if not spec:
        raise ValueError(f"model {model_label} not in config")
    provider, model = spec["provider"], spec["model"]
    tag = provider
    sysmsg = _build_prompt(ctx.cfg)

    results_path = ctx.find(F["screen_results"], required=False)
    df = _io.read_table(results_path) if results_path else _io.read_table(ctx.find(F["screen_ready"]))
    if max_records:
        df = df.head(max_records)
    vcol, rcol, ccol = f"{tag}_verdict", f"{tag}_reason", f"{tag}_conf"
    for c in (vcol, rcol, ccol):
        if c not in df.columns:
            df[c] = ""
    ckpt = ctx.out(f"screen_ckpt_{tag}.parquet")
    if os.path.exists(ckpt):
        prev = pd.read_parquet(ckpt).set_index("record_id")
        df = df.set_index("record_id")
        df.update(prev)
        df = df.reset_index()
    todo = df[~df[vcol].isin(["include", "exclude", "uncertain"])]
    print(f"[screen:{tag}] {len(todo)} to screen, {len(df)-len(todo)} done")

    ck = int(ctx.cfg.get("screening", "checkpoint_every", default=100))
    for n, (idx, row) in enumerate(todo.iterrows()):
        body = f"TITLE: {row.get('title','')}\n\n"
        if row.get("screen_input", "title+abstract") == "title+abstract" and row.get("abstract", ""):
            body += f"ABSTRACT: {row['abstract']}\n\n"
        else:
            body += "ABSTRACT: [not available — screen on title]\n\n"
        o = None
        for t in range(3):
            try:
                o = _call(provider, model, sysmsg, body, keys)
                if o:
                    break
            except Exception:
                time.sleep(2 ** t)
        if o:
            df.at[idx, vcol] = o.get("verdict", "")
            df.at[idx, rcol] = o.get("reason_code", "")
            df.at[idx, ccol] = o.get("confidence", "")
        else:
            df.at[idx, vcol] = "error"
        if n and n % ck == 0:
            df.to_parquet(ckpt, index=False)
    df.to_parquet(ckpt, index=False)
    _save_results(ctx, df)
    _prov.log_stage(ctx.work, f"screen_{tag}", rows_out=len(df), extra={tag: df[vcol].value_counts().to_dict()})
    print(f"[screen:{tag}] done: {df[vcol].value_counts().to_dict()}")
    return df


def _save_results(ctx, df):
    _io.write_table(df, ctx.out(F["screen_results"]), fmt=ctx.fmt)


def run_consensus(ctx):
    df = _io.read_table(ctx.find(F["screen_results"]))
    models = [m["provider"] for m in ctx.cfg.get("screening", "models", default=[])]
    present = [m for m in models if f"{m}_verdict" in df.columns and df[f"{m}_verdict"].ne("").any()]
    rule = ctx.cfg.get("screening", "consensus_rule", default="any_include")
    print(f"consensus on {present} (rule={rule})")

    def cons(row):
        vs = [row[f"{m}_verdict"].lower() for m in present if row[f"{m}_verdict"] in ("include", "exclude", "uncertain")]
        if not vs:
            return pd.Series(["pending", "none", "no"])
        inc, exc, unc = vs.count("include"), vs.count("exclude"), vs.count("uncertain")
        if unc > 0 and ctx.cfg.get("screening", "uncertain_to_human", default=True):
            return pd.Series(["human_review", "uncertain", "yes"])
        if inc > 0 and exc > 0 and ctx.cfg.get("screening", "disagreement_to_human", default=True):
            return pd.Series(["human_review", "disagreement", "yes"])
        if rule == "any_include":
            v = "include" if inc > 0 else "exclude"
        elif rule == "unanimous":
            v = "include" if inc == len(vs) else "exclude"
        else:
            v = "include" if inc > exc else "exclude"
        return pd.Series([v, "consensus", "no"])

    df[["screen_verdict", "screen_detail", "screen_disagreement"]] = df.apply(cons, axis=1)

    def maj(row):
        rs = [row[f"{m}_reason"] for m in present if row.get(f"{m}_reason", "")]
        return max(set(rs), key=rs.count) if rs else ""

    df["screen_reason"] = df.apply(maj, axis=1)
    _save_results(ctx, df)
    vc = df["screen_verdict"].value_counts().to_dict()
    _prov.log_stage(ctx.work, "screen", rows_in=len(df),
                     extra={"include": vc.get("include", 0), "exclude": vc.get("exclude", 0),
                            "human_review": vc.get("human_review", 0),
                            "reasons": df[df.screen_verdict == "exclude"]["screen_reason"].value_counts().to_dict()})
    print("consensus:", vc)
    return df


def draw_validation_sample(ctx, n=None):
    df = _io.read_table(ctx.find(F["screen_results"]))
    n = n or int(ctx.cfg.get("validation", "sample_size", default=250))
    rng = np.random.RandomState(2024)
    parts = []
    for v, g in df.groupby("screen_verdict"):
        k = min(len(g), max(20, int(n * len(g) / len(df))))
        parts.append(g.sample(k, random_state=rng))
    samp = pd.concat(parts).sample(frac=1, random_state=rng)
    samp["human_label"] = ""
    samp["human_notes"] = ""
    cols = [c for c in ["study_id", "record_id", "title", "abstract", "screen_verdict",
                         "screen_reason", "human_label", "human_notes"] if c in samp.columns]
    _io.write_table(samp[cols], ctx.out("validation_sample"), fmt=ctx.fmt, also_xlsx=True)
    print(f"drew {len(samp)} -> validation_sample (fill human_label, then compute_validation)")
    return samp


def compute_validation(ctx):
    from sklearn.metrics import cohen_kappa_score
    s = _io.read_table(ctx.find("validation_sample"))
    s["human_label"] = s["human_label"].astype(str).str.strip().str.lower()
    s = s[s["human_label"].isin(["include", "exclude"])]
    s["llm_inc"] = s["screen_verdict"].isin(["include", "human_review"]).astype(int)
    s["hum_inc"] = s["human_label"].eq("include").astype(int)
    tp = int(((s.llm_inc == 1) & (s.hum_inc == 1)).sum())
    fp = int(((s.llm_inc == 1) & (s.hum_inc == 0)).sum())
    fn = int(((s.llm_inc == 0) & (s.hum_inc == 1)).sum())
    tn = int(((s.llm_inc == 0) & (s.hum_inc == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    kappa = cohen_kappa_score(s["hum_inc"], s["llm_inc"]) if len(s) > 1 else float('nan')
    print(f"sens={sens:.3f} spec={spec:.3f} kappa={kappa:.3f} (TP{tp} FP{fp} FN{fn} TN{tn})")
    _prov.log_stage(ctx.work, "validate", rows_out=len(s),
                     extra={"sensitivity": round(sens, 3), "specificity": round(spec, 3),
                            "kappa": round(kappa, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return {"sensitivity": sens, "specificity": spec, "kappa": kappa}
