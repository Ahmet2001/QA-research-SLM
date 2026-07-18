#!/usr/bin/env python3
from __future__ import annotations
import json

CANDIDATES = [
    ("pubmed_qa", "pqa_labeled"),
    ("pubmed_qa", "pqa_artificial"),
    ("qasper", None),
    ("allenai/qasper", None),
    ("allenai/scifact", "claims"),
    ("allenai/scifact", "corpus"),
    ("sciq", None),
    ("allenai/sciq", None),
    ("hotpot_qa", "distractor"),
    ("hotpot_qa", "fullwiki"),
]


def short(x, n=500):
    s = repr(x)
    return s[:n]


def main():
    from datasets import load_dataset
    out = []
    for name, config in CANDIDATES:
        rec = {"name": name, "config": config}
        try:
            ds = load_dataset(name, config) if config else load_dataset(name)
            rec["ok"] = True
            rec["splits"] = {k: len(v) for k, v in ds.items()}
            first_split = next(iter(ds.keys()))
            rec["features"] = str(ds[first_split].features)
            rec["sample"] = short(ds[first_split][0])
        except Exception as e:
            rec["ok"] = False
            rec["error"] = repr(e)
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    print(json.dumps({"ok": True, "results": out}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
