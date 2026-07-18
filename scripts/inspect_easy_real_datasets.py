#!/usr/bin/env python3
from __future__ import annotations
import json
from datasets import load_dataset

candidates = [
    ('rajpurkar/squad', None, 'train'),
    ('squad', None, 'train'),
    ('hotpotqa/hotpot_qa', 'distractor', 'train'),
    ('mteb/scifact', 'corpus', 'corpus'),
    ('mteb/scifact', 'queries', 'queries'),
    ('mteb/scifact', 'default', 'test'),
]
for name, config, split in candidates:
    print('=== TRY', name, config, split)
    try:
        if config is None:
            ds = load_dataset(name, split=split)
        else:
            ds = load_dataset(name, config, split=split)
        print('OK', len(ds), ds.features)
        for i in range(min(1, len(ds))):
            row = {k: repr(v)[:1000] for k, v in ds[i].items()}
            print(json.dumps(row, ensure_ascii=False, indent=2))
    except Exception as e:
        print('FAIL', repr(e))
