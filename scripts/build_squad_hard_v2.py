#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def get_answer(trace: Dict[str, Any]) -> str:
    return str(trace.get('target', {}).get('answer', '')).strip()


def make_hard(a: Dict[str, Any], b: Dict[str, Any], idx: int) -> Dict[str, Any] | None:
    if not a.get('sources') or not b.get('sources'):
        return None
    answer_a = get_answer(a)
    answer_b = get_answer(b)
    if not answer_a or not answer_b or answer_a.lower() == answer_b.lower():
        return None
    correct = copy.deepcopy(a['sources'][0])
    wrong = copy.deepcopy(b['sources'][0])
    correct_id = str(correct['source_id'])
    wrong_id = f"distractor_{wrong.get('source_id', idx)}"
    wrong['source_id'] = wrong_id
    wrong['status'] = 'avoid'
    wrong['title'] = str(wrong.get('title', 'Distractor source')) + ' distractor'

    q = ''
    for m in a.get('messages', []):
        if m.get('role') == 'user':
            q = m.get('content', '')
            break
    if not q:
        q = 'Answer the question using only the relevant source.'

    return {
        'id': f"squad_hard_v2_{idx}_{a.get('id','a')}",
        'task_family': 'claim_verification',
        'difficulty': 'hard',
        'language': 'en',
        'messages': [
            {'role': 'system', 'content': 'You are ResearchReasoner. Output only valid compact JSON.'},
            {'role': 'user', 'content': q + ' One source is relevant and one is a distractor. Select only the relevant source_id.'},
        ],
        'sources': [correct, wrong],
        'target': {
            'task_type': 'claim_verification',
            'research_plan': ['Read the question', 'Compare both sources', 'Reject distractor sources', 'Cite only selected source_id', 'Return compact JSON'],
            'evidence_needed': ['Relevant source paragraph', 'Distractor check', 'Answer span'],
            'selected_sources': [correct_id],
            'claims': [
                {'claim': f'The answer to the question is: {answer_a}', 'status': 'supported', 'confidence': 0.9, 'evidence': [correct_id], 'importance': 'high'},
                {'claim': f'The answer to the question is: {answer_b}', 'status': 'insufficient', 'confidence': 0.72, 'evidence': [], 'importance': 'medium'},
            ],
            'conflicts': [{'issue': 'One source is relevant and the other is a distractor for this question.', 'source_ids': [correct_id, wrong_id], 'resolution': f'use {correct_id}; exclude {wrong_id}', 'confidence': 0.86}],
            'uncertainties': ['The distractor source may contain true facts, but it does not answer this question.'],
            'answer': answer_a,
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/real/squad_research_traces.jsonl')
    ap.add_argument('--out', default='data/real/squad_hard_v2_traces.jsonl')
    ap.add_argument('--limit', type=int, default=320)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    rows = list(read_jsonl(Path(args.input)))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < args.limit and attempts < args.limit * 20:
        a = rng.choice(rows)
        b = rng.choice(rows)
        tr = make_hard(a, b, attempts)
        attempts += 1
        if tr is not None:
            out.append(tr)
    n = write_jsonl(Path(args.out), out)
    print(json.dumps({'ok': True, 'hard_traces': n, 'out': args.out}, indent=2))

if __name__ == '__main__':
    main()
