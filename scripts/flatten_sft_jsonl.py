#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Iterable


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            n += 1
    return n


def flat(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'id': str(row.get('id', '')),
        'task_family': str(row.get('task_family', '')),
        'difficulty': str(row.get('difficulty', '')),
        'language': str(row.get('language', '')),
        'messages_json': json.dumps(row.get('messages', []), ensure_ascii=False, separators=(',', ':')),
        'sources_json': json.dumps(row.get('sources', []), ensure_ascii=False, separators=(',', ':')),
        'target_json': json.dumps(row.get('target', {}), ensure_ascii=False, separators=(',', ':')),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    n = write_jsonl(Path(args.output), (flat(r) for r in read_jsonl(Path(args.input))))
    print(json.dumps({'ok': True, 'input': args.input, 'output': args.output, 'rows': n}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
