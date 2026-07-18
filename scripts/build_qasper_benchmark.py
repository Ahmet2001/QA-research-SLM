#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, tarfile, urllib.request, random, re
from pathlib import Path
from typing import Any, Dict, List

SYSTEM = (
    "You are ResearchReasoner. Output ONLY valid minified JSON. No markdown. No <think>. "
    "Strict schema: task_type string; research_plan array of short strings; evidence_needed array of short strings; "
    "selected_sources array of source_id strings only; claims array of objects with claim,status,confidence,evidence,importance; "
    "conflicts array; uncertainties array of strings; answer string only. Evidence values must be source_id strings from SOURCES."
)
URL_TRAIN_DEV = 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz'
URL_TEST = 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz'

def dumpj(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1000:
        return
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r, out.open('wb') as f:
        while True:
            b = r.read(1024 * 1024)
            if not b: break
            f.write(b)

def extract_member(tgz: Path, wanted: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, 'r:gz') as tar:
        names = tar.getnames()
        pick = None
        for n in names:
            if n.endswith(wanted):
                pick = n; break
        if pick is None:
            raise FileNotFoundError({'wanted': wanted, 'members': names[:50]})
        member = tar.getmember(pick)
        f = tar.extractfile(member)
        if f is None: raise RuntimeError('extractfile failed')
        out_path = out_dir / wanted
        out_path.write_bytes(f.read())
        return out_path

def flatten_sections(paper: Dict[str, Any], max_sections: int = 8) -> List[Dict[str,str]]:
    sources = []
    abstract = str(paper.get('abstract') or '').strip()
    if abstract:
        sources.append({'source_id':'qasper_abstract', 'title':'Abstract', 'text':abstract[:3500], 'status':'usable'})
    full = paper.get('full_text') or []
    for i, sec in enumerate(full[:max_sections]):
        name = str(sec.get('section_name') or f'Section {i+1}')
        paras = sec.get('paragraphs') or []
        text = '\n'.join(str(p) for p in paras if str(p).strip())[:3500]
        if text:
            sid = f'qasper_sec_{i:02d}'
            sources.append({'source_id':sid, 'title':name, 'text':text, 'status':'usable'})
    return sources[:10]

def answer_text(ans: Dict[str,Any]) -> str:
    a = ans.get('answer', ans)
    if a.get('unanswerable'):
        return 'unanswerable'
    spans = a.get('extractive_spans') or []
    if spans:
        return '; '.join(str(x) for x in spans if str(x).strip())[:500]
    ff = str(a.get('free_form_answer') or '').strip()
    if ff:
        return ff[:500]
    yn = a.get('yes_no')
    if isinstance(yn, bool):
        return 'yes' if yn else 'no'
    return ''

def evidence_texts(ans: Dict[str,Any]) -> List[str]:
    a = ans.get('answer', ans)
    ev = a.get('highlighted_evidence') or a.get('evidence') or []
    return [str(x).strip() for x in ev if str(x).strip()]

def choose_evidence_sources(evidence: List[str], sources: List[Dict[str,str]]) -> List[str]:
    selected = []
    for ev in evidence[:4]:
        ev_norm = re.sub(r'\s+', ' ', ev.lower())[:180]
        best = None
        for s in sources:
            txt = re.sub(r'\s+', ' ', s.get('text','').lower())
            if ev_norm and ev_norm[:80] in txt:
                best = s['source_id']; break
        if best and best not in selected:
            selected.append(best)
    if not selected and sources:
        selected = [sources[0]['source_id']]
    return selected

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev', choices=['train','dev','test'])
    ap.add_argument('--limit', type=int, default=80)
    ap.add_argument('--out', default='data/domain_benchmarks/qasper_dev_flat.jsonl')
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    raw_dir = Path('data/raw/qasper')
    tgz_train = raw_dir / 'qasper-train-dev-v0.3.tgz'
    tgz_test = raw_dir / 'qasper-test-and-evaluator-v0.3.tgz'
    download(URL_TRAIN_DEV, tgz_train)
    download(URL_TEST, tgz_test)
    tgz = tgz_test if args.split == 'test' else tgz_train
    json_path = extract_member(tgz, f'qasper-{args.split}-v0.3.json', raw_dir / 'extracted')
    data = load_json(json_path)
    papers = list(data.items()) if isinstance(data, dict) else [(str(i), p) for i, p in enumerate(data)]
    rng = random.Random(args.seed); rng.shuffle(papers)
    rows = []
    for pid, paper in papers:
        sources = flatten_sections(paper)
        if not sources: continue
        qas = paper.get('qas') or []
        for qa in qas:
            q = str(qa.get('question') or '').strip()
            answers = qa.get('answers') or []
            if not q or not answers: continue
            ans = answers[0]
            gold = answer_text(ans)
            if not gold: continue
            ev = evidence_texts(ans)
            selected = choose_evidence_sources(ev, sources)
            src_block = '\n'.join(f"[{s['source_id']}] {s['title']}: {s['text']}" for s in sources)
            user = (
                "Answer the question about the research paper using only the provided sources. "
                "Return the required JSON schema and cite source IDs in selected_sources/evidence.\n\n"
                f"Paper title: {paper.get('title','')}\nQuestion: {q}\n\nSOURCES:\n{src_block}"
            )
            target = {
                'task_type':'scientific_paper_qa',
                'research_plan':['read the question','identify relevant paper sections','answer with cited evidence'],
                'evidence_needed':['paper evidence that directly answers the question'],
                'selected_sources':selected,
                'claims':[{'claim':f'The answer is: {gold}', 'status':'supported', 'confidence':0.8, 'evidence':selected, 'importance':'high'}],
                'conflicts':[],
                'uncertainties':['The answer is based on the provided paper sections.'] if gold == 'unanswerable' else [],
                'answer':gold,
            }
            rows.append({
                'id':f'qasper_{args.split}_{len(rows):04d}',
                'task_family':'scientific_paper_qa',
                'difficulty':'domain_benchmark',
                'language':'en',
                'messages_json':dumpj([{'role':'system','content':SYSTEM},{'role':'user','content':user},{'role':'assistant','content':dumpj(target)}]),
                'sources_json':dumpj(sources),
                'target_json':dumpj(target),
                'gold_answer':gold,
                'question':q,
                'paper_id':pid,
            })
            if len(rows) >= args.limit: break
        if len(rows) >= args.limit: break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows: f.write(dumpj(r)+'\n')
    print(json.dumps({'ok':True,'out':args.out,'rows':len(rows),'split':args.split,'raw_json':str(json_path)}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
