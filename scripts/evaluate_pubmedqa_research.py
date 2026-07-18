#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from research_rewards import score_output, parse_json_output

LABELS = {'yes', 'no', 'maybe'}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def j(row, key, fallback):
    v = row.get(key)
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return fallback
    return v if v is not None else fallback


def chat_text(tok, messages, add_generation_prompt):
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def sid(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get('source_id') or x.get('id') or x.get('source') or json.dumps(x, ensure_ascii=False, sort_keys=True)
    return str(x)


def sanitize_completion(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def item(row):
    sources = j(row, 'sources_json', row.get('sources', []))
    target = j(row, 'target_json', row.get('target', {}))
    must = [sid(x) for x in target.get('selected_sources', [])]
    avoid = [sid(s) for s in sources if isinstance(s, dict) and s.get('status') in {'avoid', 'contradicted'}]
    return {
        'sources': [{'source_id': sid(s), 'status': s.get('status', 'unknown') if isinstance(s, dict) else 'unknown', 'text': s.get('text', '') if isinstance(s, dict) else ''} for s in sources],
        'verifier_hints': {'must_cite_source_ids': must, 'avoid_source_ids': avoid, 'requires_uncertainty': bool(target.get('uncertainties')), 'requires_conflict_detection': bool(target.get('conflicts'))},
    }


def extract_label_from_answer(text: str) -> str | None:
    text = text.lower().strip()
    text = re.sub(r'^[^a-z]+', '', text)
    for lab in ['maybe', 'yes', 'no']:
        if re.match(rf'^{lab}\b', text):
            return lab
    # fallback: unique label mention in first 80 chars
    head = text[:80]
    found = [lab for lab in LABELS if re.search(rf'\b{lab}\b', head)]
    return found[0] if len(found) == 1 else None


def extract_pred_label(completion: str) -> str | None:
    obj, ok = parse_json_output(completion)
    if obj is None:
        return extract_label_from_answer(completion)
    ans = obj.get('answer', '')
    pred = extract_label_from_answer(str(ans))
    if pred:
        return pred
    # fallback to high-importance claim text
    claims = obj.get('claims', []) if isinstance(obj.get('claims'), list) else []
    blob = ' '.join(str(c.get('claim', '')) for c in claims if isinstance(c, dict))
    return extract_label_from_answer(blob)


def load_model(base, adapter):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def gen(model, tok, row, max_new_tokens):
    import torch
    msgs = j(row, 'messages_json', row.get('messages', []))
    msgs = [m for m in msgs if m.get('role') != 'assistant']
    prompt = chat_text(tok, msgs, True)
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.08, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return sanitize_completion(tok.decode(out[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True))


def evaluate(name, base, adapter, rows, max_new_tokens):
    model, tok = load_model(base, adapter)
    results, totals, correct, label_attempts = [], [], 0, 0
    for row in rows:
        c = gen(model, tok, row, max_new_tokens)
        try:
            s = score_output(c, item(row))
        except Exception as e:
            s = {'json_validity': 0.0, 'total': -0.35, 'error': repr(e)}
        pred = extract_pred_label(c)
        gold = str(row.get('gold_label', '')).lower().strip()
        is_correct = bool(pred == gold and gold in LABELS)
        if pred in LABELS:
            label_attempts += 1
        correct += int(is_correct)
        totals.append(float(s.get('total', -0.35)))
        results.append({'id': row.get('id'), 'gold_label': gold, 'pred_label': pred, 'label_correct': is_correct, 'completion': c, 'scores': s})
    return {
        'name': name,
        'adapter': adapter,
        'n': len(rows),
        'mean_total': sum(totals) / max(1, len(totals)),
        'label_accuracy': correct / max(1, len(rows)),
        'label_attempt_rate': label_attempts / max(1, len(rows)),
        'results': results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--eval-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare-base', action='store_true')
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--limit', type=int, default=100)
    args = ap.parse_args()
    rows = list(read_jsonl(Path(args.eval_file)))[:args.limit]
    out = {'eval_file': args.eval_file, 'runs': []}
    if args.compare_base:
        out['runs'].append(evaluate('base', args.base_model, None, rows, args.max_new_tokens))
    out['runs'].append(evaluate('sft_adapter', args.base_model, args.adapter, rows, args.max_new_tokens))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({'ok': True, 'out': args.out, 'summary': [{'name': r['name'], 'n': r['n'], 'mean_total': r['mean_total'], 'label_accuracy': r['label_accuracy'], 'label_attempt_rate': r['label_attempt_rate']} for r in out['runs']]}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
