#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Iterable, List

SYSTEM = (
    'You are ResearchReasoner. Output ONLY valid minified JSON. No markdown. No <think>. '
    'Strict schema: task_type string; research_plan array of short strings; evidence_needed array of short strings; '
    'selected_sources array of source_id strings only; claims array of objects with claim,status,confidence,evidence,importance; '
    'conflicts array; uncertainties array of strings; answer string only. '
    'Never put JSON inside answer. Never use content instead of claim. Evidence values must be source_id strings from SOURCES.'
)
REQ_KEYS = ['task_type','research_plan','evidence_needed','selected_sources','claims','conflicts','uncertainties','answer']
VALID_STATUS = {'supported','contradicted','insufficient'}

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
    return n

def sid(x: Any) -> str:
    if isinstance(x, str):
        m = re.search(r'source_id=([^,\]\s]+)', x)
        return m.group(1) if m else x
    if isinstance(x, dict): return str(x.get('source_id') or x.get('id') or x.get('source') or '')
    return str(x)

def source_ids(sources: List[Dict[str, Any]]) -> List[str]:
    return [str(s.get('source_id')) for s in sources if s.get('source_id')]

def source_block(sources: List[Dict[str, Any]]) -> str:
    if not sources: return ''
    parts = []
    for s in sources:
        parts.append(f"[source_id={s.get('source_id','')} status={s.get('status','unknown')} type={s.get('source_type','')} reliability={s.get('reliability','')}]\ntitle: {s.get('title','')}\ntext: {str(s.get('text',''))[:5000]}")
    return '\n\nSOURCES:\n' + '\n\n'.join(parts)

def as_list_str(x: Any, default: List[str]) -> List[str]:
    if isinstance(x, list): return [str(i) for i in x if not isinstance(i, (dict, list))][:6] or default
    if isinstance(x, str) and x.strip(): return [x.strip()]
    return default

def normalize_target(t: Dict[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_ids = set(source_ids(sources))
    selected = []
    for x in t.get('selected_sources', []):
        sx = sid(x)
        if sx in valid_ids and sx not in selected:
            # Do not select avoid/contradicted sources.
            st = next((s.get('status','unknown') for s in sources if s.get('source_id') == sx), 'unknown')
            if st not in {'avoid','contradicted'}:
                selected.append(sx)
    claims = []
    for c in t.get('claims', [])[:8]:
        if not isinstance(c, dict): continue
        claim_text = str(c.get('claim') or c.get('content') or c.get('reasoning') or t.get('answer') or '').strip()
        if not claim_text: claim_text = 'The answer is supported by the selected source.'
        status = str(c.get('status') or 'supported').lower()
        if status not in VALID_STATUS: status = 'insufficient'
        ev = []
        for e in c.get('evidence', []):
            se = sid(e)
            if se in valid_ids and se not in ev:
                ev.append(se)
        if status == 'supported' and not ev and selected:
            ev = selected[:1]
        claims.append({'claim': claim_text[:500], 'status': status, 'confidence': float(c.get('confidence', 0.85) or 0.85), 'evidence': ev[:4], 'importance': str(c.get('importance') or 'high')})
    if not claims:
        ans = str(t.get('answer','')).strip() or 'Insufficient evidence.'
        claims = [{'claim': f'The answer is: {ans}'[:500], 'status': 'supported' if selected else 'insufficient', 'confidence': 0.8, 'evidence': selected[:1], 'importance': 'high'}]
    conflicts = []
    for cf in t.get('conflicts', [])[:4]:
        if isinstance(cf, dict):
            ids = [sid(x) for x in cf.get('source_ids', []) if sid(x) in valid_ids]
            conflicts.append({'issue': str(cf.get('issue') or cf.get('resolution') or 'Source conflict')[:250], 'source_ids': ids, 'resolution': str(cf.get('resolution') or 'prefer selected_sources')[:250], 'confidence': float(cf.get('confidence', 0.75) or 0.75)})
        elif isinstance(cf, str):
            conflicts.append({'issue': cf[:250], 'source_ids': selected[:2], 'resolution': 'prefer selected_sources', 'confidence': 0.75})
    uncertainties = as_list_str(t.get('uncertainties', []), [])[:5]
    ans = t.get('answer', '')
    if isinstance(ans, (dict, list)): ans = json.dumps(ans, ensure_ascii=False, separators=(',', ':'))
    ans = str(ans).strip()
    # Prevent nested JSON answer patterns.
    if ans.startswith('{') or ans.startswith('['):
        ans = 'See claims and selected_sources for the supported answer.'
    return {
        'task_type': str(t.get('task_type') or 'research')[:80],
        'research_plan': as_list_str(t.get('research_plan'), ['Read the task','Inspect sources','Cite source IDs','Return compact JSON'])[:6],
        'evidence_needed': as_list_str(t.get('evidence_needed'), ['Relevant sources','Valid source IDs'])[:6],
        'selected_sources': selected[:8],
        'claims': claims,
        'conflicts': conflicts,
        'uncertainties': uncertainties,
        'answer': ans[:700],
    }

def to_row(trace: Dict[str, Any]) -> Dict[str, Any]:
    sources = trace.get('sources', [])
    target = normalize_target(trace.get('target', {}), sources)
    user_msgs = [m.get('content','') for m in trace.get('messages', []) if m.get('role') == 'user']
    user = '\n'.join(user_msgs).strip() + source_block(sources)
    messages = [{'role':'system','content':SYSTEM}, {'role':'user','content':user.strip()}, {'role':'assistant','content':json.dumps(target, ensure_ascii=False, separators=(',', ':'))}]
    return {'id': str(trace.get('id','')), 'task_family': str(trace.get('task_family','unknown')), 'difficulty': str(trace.get('difficulty','medium')), 'language': str(trace.get('language','unknown')), 'messages': messages, 'sources': sources, 'target': target}

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', nargs='+', required=True); ap.add_argument('--out-stage1', required=True); ap.add_argument('--out-stage2', required=True); ap.add_argument('--out-eval', required=True); ap.add_argument('--eval-count', type=int, default=80)
    args = ap.parse_args()
    traces = []
    for p in args.inputs:
        pp = Path(p)
        if pp.exists(): traces.extend(read_jsonl(pp))
    rows = [to_row(t) for t in traces]
    # Put some seed/hard rows in eval but mostly train.
    eval_count = min(args.eval_count, max(1, len(rows)//8))
    eval_rows, train_rows = rows[:eval_count], rows[eval_count:]
    hard_rows = [r for r in train_rows if r.get('difficulty') == 'hard' or r.get('target', {}).get('conflicts') or r.get('target', {}).get('uncertainties')]
    if not hard_rows: hard_rows = train_rows
    print(json.dumps({'input_traces': len(traces), 'train': write_jsonl(Path(args.out_stage1), train_rows), 'hard': write_jsonl(Path(args.out_stage2), hard_rows), 'eval': write_jsonl(Path(args.out_eval), eval_rows)}, ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
