#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from typing import Any, Dict, List

SYSTEM = (
    "You are ResearchReasoner. Output ONLY valid minified JSON. No markdown. No <think>. "
    "Strict schema: task_type string; research_plan array of short strings; evidence_needed array of short strings; "
    "selected_sources array of source_id strings only; claims array of objects with claim,status,confidence,evidence,importance; "
    "conflicts array; uncertainties array of strings; answer string only. Evidence values must be source_id strings from SOURCES."
)

def dumpj(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip()).lower()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/domain_benchmarks/hotpotqa_distractor_flat.jsonl')
    ap.add_argument('--limit', type=int, default=120)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset('hotpot_qa', 'distractor')
    split = 'validation' if 'validation' in ds else next(iter(ds.keys()))
    rows = list(ds[split])
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.limit]

    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        question = str(r.get('question', '')).strip()
        answer = str(r.get('answer', '')).strip()
        context = r.get('context', {})
        titles = context.get('title', []) if isinstance(context, dict) else []
        sentences = context.get('sentences', []) if isinstance(context, dict) else []
        sf = r.get('supporting_facts', {})
        sf_titles = {norm_title(t) for t in sf.get('title', [])} if isinstance(sf, dict) else set()
        sources = []
        selected = []
        for j, title in enumerate(titles[:10]):
            sent_list = sentences[j] if j < len(sentences) else []
            text = " ".join(str(s) for s in sent_list)[:3000]
            sid = f"hp_{i:04d}_{j:02d}"
            status = 'usable'
            if norm_title(title) in sf_titles:
                selected.append(sid)
            sources.append({'source_id': sid, 'title': str(title), 'text': text, 'status': status})
        if not selected and sources:
            selected = [sources[0]['source_id']]
        src_block = "\n".join(f"[{s['source_id']}] {s['title']}: {s['text']}" for s in sources)
        user = (
            "Answer the multi-hop question using only the provided sources. Return the required JSON schema. "
            "Use selected_sources and evidence fields to cite the sources that support the answer.\n\n"
            f"Question: {question}\n\nSOURCES:\n{src_block}"
        )
        target = {
            'task_type': 'multi_hop_document_qa',
            'research_plan': ['identify relevant sources', 'connect evidence across sources', 'answer concisely'],
            'evidence_needed': ['supporting facts needed to answer the multi-hop question'],
            'selected_sources': selected,
            'claims': [{
                'claim': f"The answer is {answer}.",
                'status': 'supported',
                'confidence': 0.85,
                'evidence': selected,
                'importance': 'high'
            }],
            'conflicts': [],
            'uncertainties': [],
            'answer': answer,
        }
        out_rows.append({
            'id': f'hotpotqa_{i:04d}',
            'task_family': 'multi_hop_document_qa',
            'difficulty': 'domain_benchmark',
            'language': 'en',
            'messages_json': dumpj([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': dumpj(target)}]),
            'sources_json': dumpj(sources),
            'target_json': dumpj(target),
            'gold_answer': answer,
            'question': question,
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in out_rows:
            f.write(dumpj(row) + '\n')
    print(json.dumps({'ok': True, 'out': args.out, 'rows': len(out_rows), 'split': split}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
