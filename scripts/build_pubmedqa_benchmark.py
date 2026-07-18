#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any, Dict, List

SYSTEM = (
    "You are ResearchReasoner. Output ONLY valid minified JSON. No markdown. No <think>. "
    "Strict schema: task_type string; research_plan array of short strings; evidence_needed array of short strings; "
    "selected_sources array of source_id strings only; claims array of objects with claim,status,confidence,evidence,importance; "
    "conflicts array; uncertainties array of strings; answer string only. Evidence values must be source_id strings from SOURCES. "
    "For biomedical yes/no/maybe questions, put the final label at the start of answer: yes, no, or maybe."
)


def dumpj(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def get_context_text(ctx: Any) -> str:
    if isinstance(ctx, dict):
        parts = ctx.get('contexts') or ctx.get('context') or []
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if str(p).strip())
        return str(parts)
    if isinstance(ctx, list):
        return "\n".join(str(p) for p in ctx if str(p).strip())
    return str(ctx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/domain_benchmarks/pubmedqa_labeled_flat.jsonl')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset('pubmed_qa', 'pqa_labeled')
    split = 'train' if 'train' in ds else next(iter(ds.keys()))
    rows = list(ds[split])
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.limit]

    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        q = str(r.get('question', '')).strip()
        ctx_text = get_context_text(r.get('context', ''))
        long_answer = str(r.get('long_answer', '')).strip()
        label = str(r.get('final_decision', '')).strip().lower()
        if label not in {'yes', 'no', 'maybe'}:
            continue
        source_id = f"pmqa_{i:04d}_abstract"
        sources = [{
            'source_id': source_id,
            'title': 'PubMed abstract context',
            'text': ctx_text[:6000],
            'status': 'usable'
        }]
        user = (
            "Answer the biomedical research question using only the provided source. "
            "Return the required JSON schema. The answer string must start with one of: yes, no, maybe.\n\n"
            f"Question: {q}\n\nSOURCES:\n[{source_id}] {ctx_text[:6000]}"
        )
        target = {
            'task_type': 'biomedical_research_qa',
            'research_plan': ['read the biomedical question', 'inspect the abstract evidence', 'decide yes no or maybe'],
            'evidence_needed': ['abstract evidence that supports the yes/no/maybe decision'],
            'selected_sources': [source_id],
            'claims': [{
                'claim': f"The answer to the biomedical research question is {label}.",
                'status': 'supported',
                'confidence': 0.85,
                'evidence': [source_id],
                'importance': 'high'
            }],
            'conflicts': [],
            'uncertainties': ['Biomedical abstracts may require cautious interpretation.'] if label == 'maybe' else [],
            'answer': f"{label}: {long_answer[:500]}" if long_answer else label,
        }
        out_rows.append({
            'id': f'pubmedqa_labeled_{i:04d}',
            'task_family': 'biomedical_research_qa',
            'difficulty': 'domain_benchmark',
            'language': 'en',
            'messages_json': dumpj([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': dumpj(target)}]),
            'sources_json': dumpj(sources),
            'target_json': dumpj(target),
            'gold_label': label,
            'question': q,
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in out_rows:
            f.write(dumpj(row) + '\n')
    print(json.dumps({'ok': True, 'out': args.out, 'rows': len(out_rows), 'split': split}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
