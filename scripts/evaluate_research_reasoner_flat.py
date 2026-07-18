#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from research_rewards import score_output

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def j(row, key, default):
    v = row.get(key)
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return default
    return v if v is not None else default

def chat_text(tok, messages, add_generation_prompt):
    try: return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

def item(row):
    sources = j(row, 'sources_json', row.get('sources', []))
    target = j(row, 'target_json', row.get('target', {}))
    return {'sources': [{'source_id': s.get('source_id'), 'status': s.get('status', 'unknown'), 'text': s.get('text', '')} for s in sources], 'verifier_hints': {'must_cite_source_ids': target.get('selected_sources', []), 'avoid_source_ids': [s.get('source_id') for s in sources if s.get('status') in {'avoid','contradicted'}], 'requires_uncertainty': bool(target.get('uncertainties')), 'requires_conflict_detection': bool(target.get('conflicts'))}}

def load_model(base, adapter):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval(); return model, tok

def gen(model, tok, row, max_new_tokens):
    import torch
    msgs = j(row, 'messages_json', row.get('messages', []))
    msgs = [m for m in msgs if m.get('role') != 'assistant']
    prompt = chat_text(tok, msgs, True)
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.08, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

def evaluate(name, base, adapter, rows, max_new_tokens):
    model, tok = load_model(base, adapter)
    results, totals = [], []
    for row in rows:
        c = gen(model, tok, row, max_new_tokens)
        s = score_output(c, item(row)); totals.append(s['total'])
        results.append({'id': row.get('id'), 'completion': c, 'scores': s})
    return {'name': name, 'adapter': adapter, 'n': len(rows), 'mean_total': sum(totals)/max(1,len(totals)), 'results': results}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base-model', default='Qwen/Qwen3-1.7B'); ap.add_argument('--adapter', required=True); ap.add_argument('--eval-file', required=True); ap.add_argument('--out', required=True); ap.add_argument('--compare-base', action='store_true'); ap.add_argument('--max-new-tokens', type=int, default=1024); ap.add_argument('--limit', type=int, default=8)
    args = ap.parse_args(); rows = list(read_jsonl(Path(args.eval_file)))[:args.limit]
    out = {'eval_file': args.eval_file, 'runs': []}
    if args.compare_base: out['runs'].append(evaluate('base', args.base_model, None, rows, args.max_new_tokens))
    out['runs'].append(evaluate('sft_adapter', args.base_model, args.adapter, rows, args.max_new_tokens))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f: json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({'ok': True, 'out': args.out, 'summary': [{'name': r['name'], 'n': r['n'], 'mean_total': r['mean_total']} for r in out['runs']]}, ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
