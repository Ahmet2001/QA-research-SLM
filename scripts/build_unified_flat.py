#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        print(json.dumps({'warning': 'missing', 'path': str(path)}))
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
            n += 1
    return n


def tag_rows(rows: List[Dict], tag: str) -> List[Dict]:
    out = []
    for i, r in enumerate(rows):
        rr = dict(r)
        rr['mix_source'] = tag
        rr['id'] = f"{tag}_{rr.get('id', i)}"
        out.append(rr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/processed/sft_unified_flat.jsonl')
    ap.add_argument('--seed', type=int, default=777)
    args = ap.parse_args()

    parts = []
    stats = {}
    sources = [
        ('real_v1_x2', 'data/processed/sft_stage1_real_compact_flat.jsonl', 2),
        ('v2_stage1_x1', 'data/processed/sft_v2_stage1_flat.jsonl', 1),
        ('v2_hard_x1', 'data/processed/sft_v2_stage2_hard_flat.jsonl', 1),
        ('rl_lite_selected_x3', 'data/processed/rl_rejection_sft_v2_flat.jsonl', 3),
    ]
    for tag, path, weight in sources:
        rows = tag_rows(read_jsonl(Path(path)), tag)
        stats[tag] = {'base_rows': len(rows), 'weight': weight, 'used_rows': len(rows) * weight}
        for _ in range(weight):
            parts.extend(rows)
    rng = random.Random(args.seed)
    rng.shuffle(parts)
    n = write_jsonl(Path(args.out), parts)
    print(json.dumps({'ok': True, 'out': args.out, 'rows': n, 'stats': stats}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
