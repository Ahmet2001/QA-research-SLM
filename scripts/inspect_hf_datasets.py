#!/usr/bin/env python3
from __future__ import annotations

import json
from datasets import load_dataset

CANDIDATES = [
    ("allenai/scifact", "claims", "train"),
    ("allenai/scifact", "corpus", "train"),
    ("allenai/qasper", None, "train"),
    ("fever", "v1.0", "train"),
]

for name, config, split in CANDIDATES:
    print("=== TRY", name, config, split)
    try:
        if config is None:
            ds = load_dataset(name, split=split)
        else:
            ds = load_dataset(name, config, split=split)
        print("OK", name, config, "len", len(ds))
        print("features", ds.features)
        for i in range(min(2, len(ds))):
            row = ds[i]
            # truncate big values
            small = {}
            for k, v in row.items():
                s = repr(v)
                small[k] = s[:700]
            print("sample", i, json.dumps(small, ensure_ascii=False, indent=2))
    except Exception as e:
        print("FAIL", name, config, repr(e))
