"""Shared HTTP utilities for enrichment/backfill stages, including a bounded
thread pool for per-record API passes (Crossref, Unpaywall, EuropePMC)."""
from __future__ import annotations
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def make_session(email: str = "") -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": f"sosci-pipeline/1.0 (mailto:{email})"})
    return s


def get_json(session, url, timeout=30, tries=3):
    for t in range(tries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (t + 1))
                continue
            return None
        except Exception:
            time.sleep(1.0 * (t + 1))
    return None


def parallel_map(fn, items, max_workers=8, jitter=0.02, on_result=None, progress_every=500):
    """Apply fn to each item concurrently with a bounded pool. on_result(item,
    result) runs in the MAIN thread (via as_completed), so DataFrame writes
    stay safe. Designed for I/O-bound API calls, not CPU work."""
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for it in items:
            if jitter:
                time.sleep(random.uniform(0, jitter))
            futures[ex.submit(fn, it)] = it
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if on_result:
                on_result(it, res)
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"    ...{done}/{len(items)}")
    return done
