"""Runtime context shared by all stage scripts. Set once by the runner (from
the YAML config); every stage imports get_ctx() instead of hardcoding paths.
This removes the need for any path/format monkey-patching."""
from __future__ import annotations
import os


class Ctx:
    def __init__(self, base, data, work, fmt="parquet", email="", cfg=None):
        self.base, self.data, self.work = base, data, work
        self.fmt, self.email, self.cfg = fmt, email, cfg
        os.makedirs(self.work, exist_ok=True)

    def find(self, *stems, required=True):
        """Resolve a stage input by stem, regardless of extension, in WORK."""
        for stem in stems:
            stem = os.path.splitext(os.path.basename(stem))[0]
            for ext in (".parquet", ".csv.gz", ".xlsx", ".csv"):
                p = os.path.join(self.work, stem + ext)
                if os.path.exists(p):
                    return p
        if required:
            raise FileNotFoundError(
                f"None of {stems} found in {self.work}. Run the previous stage first.")
        return None

    def out(self, stem):
        return os.path.join(self.work, stem)


_CTX = None


def set_ctx(ctx):
    global _CTX
    _CTX = ctx


def get_ctx():
    if _CTX is None:
        raise RuntimeError("Context not set. The runner must call set_ctx() first.")
    return _CTX
