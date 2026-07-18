#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def score_breakdown(run: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        'json_validity','required_fields','avoid_bad_sources','evidence_id_validity',
        'citation_precision','supported_claim_ratio','conflict_detection',
        'uncertainty_calibration','brevity_efficiency','total'
    ]
    acc = {k: [] for k in keys}
    for r in run.get('results', []):
        scores = r.get('scores', {})
        for k in keys:
            if k in scores and isinstance(scores[k], (int, float)):
                acc[k].append(float(scores[k]))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', required=True, help='label=path items')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows: List[Dict[str, Any]] = []
    for item in args.items:
        label, path = item.split('=', 1)
        data = load(Path(path))
        for run in data.get('runs', []):
            bd = score_breakdown(run)
            rows.append({
                'benchmark': label,
                'run': run.get('name'),
                'adapter': run.get('adapter'),
                'n': run.get('n'),
                **bd,
            })
    out = {'ok': True, 'rows': rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
