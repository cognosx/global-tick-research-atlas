"""YAML config loader. A Config object exposes dotted-path lookups via
.get(section, key, default=...) and a few convenience properties used
throughout the package (base_dir, data_dir, work_dir, email, name)."""
from __future__ import annotations
import os
import yaml


class Config:
    def __init__(self, raw: dict):
        self.raw = raw

    def get(self, *path, default=None):
        node = self.raw
        for p in path:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    @property
    def name(self):
        return self.get("project", "name", default="sosci_project")

    @property
    def base_dir(self):
        return self.get("paths", "base_dir", default=".")

    @property
    def data_dir(self):
        return os.path.join(self.base_dir, self.get("paths", "data_subdir", default="data"))

    @property
    def work_dir(self):
        return os.path.join(self.base_dir, self.get("paths", "work_subdir", default="preprocessing_data"))

    @property
    def email(self):
        return self.get("project", "contact_email", default="")


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
