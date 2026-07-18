#!/usr/bin/env python3
"""Build real ResearchReasoner traces from SQuAD.

SQuAD gives document-grounded QA with real paragraphs and answer spans. We map it
into ResearchReasoner JSON: document_qa + supported claim/evidence source.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            n += 1
    return n


def trace_from_squad(row: Dict[str, Any], split: str, idx: int) -> Dict[str, Any] | None:
    q = str(row.get('question') or '').strip()
    ctx = str(row.get('context') or '').strip()
    title = str(row.get('title') or 'SQuAD document')
    ans = row.get('answers') or {}
    texts = ans.get('text') if isinstance(ans, dict) else []
    answer = str(texts[0]).strip() if texts else ''
    if not q or not ctx or not answer:
        return None
    source_id = f"squad_{split}_{row.get('id', idx)}"
    claim = f"The answer to the question is: {answer}"
    return {
        'id': f"squad_{split}_{row.get('id', idx)}",
        'task_family': 'document_qa',
        'difficulty': 'medium' if len(ctx) < 1800 else 'hard',
        'language': 'en',
        'messages': [
            {'role': 'system', 'content': 'You are ResearchReasoner. Output only valid compact JSON.'},
            {'role': 'user', 'content': f"Answer the question using only the provided source. Question: {q}"},
        ],
        'sources': [{
            'source_id': source_id,
            'title': title,
            'source_type': 'document_paragraph',
            'date': '2016',
            'reliability': 0.78,
            'status': 'usable',
            'text': ctx[:5000],
        }],
        'target': {
            'task_type': 'document_qa',
            'research_plan': ['Read the question', 'Find the answer span in the source', 'Cite the source id', 'Return compact JSON'],
            'evidence_needed': ['Relevant source paragraph', 'Answer span'],
            'selected_sources': [source_id],
            'claims': [{
                'claim': claim,
                'status': 'supported',
                'confidence': 0.9,
                'evidence': [source_id],
                'importance': 'high',
            }],
            'conflicts': [],
            'uncertainties': [],
            'answer': answer,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/real/squad_research_traces.jsonl')
    ap.add_argument('--limit-train', type=int, default=800)
    ap.add_argument('--limit-validation', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset
    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []
    for split, limit in [('train', args.limit_train), ('validation', args.limit_validation)]:
        ds = load_dataset('rajpurkar/squad', split=split)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        used = 0
        for i in idxs:
            tr = trace_from_squad(ds[i], split, i)
            if tr is None:
                continue
            rows.append(tr)
            used += 1
            if used >= limit:
                break
    n = write_jsonl(Path(args.out), rows)
    print(json.dumps({'ok': True, 'traces': n, 'out': args.out}, indent=2))

if __name__ == '__main__':
    main()
