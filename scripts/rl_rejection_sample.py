#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from typing import Any, Dict, Iterable, List
from research_rewards import score_output


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1
    return n


def clean(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    text = text.strip()
    # Strip markdown fences if present.
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()
    return text


def split_prompt(prompt: str) -> List[Dict[str, str]]:
    parts = prompt.split('\n\n', 1)
    if len(parts) == 2:
        system, user = parts
    else:
        system, user = 'You are ResearchReasoner. Output only valid compact JSON.', prompt
    return [{'role': 'system', 'content': system.strip()}, {'role': 'user', 'content': user.strip()}]


def chat_text(tok, messages, add_generation_prompt):
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def load_model(base: str, adapter: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def generate_one(model, tok, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float, top_p: float, seed: int) -> str:
    import torch
    torch.manual_seed(seed)
    prompt = chat_text(tok, messages, True)
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.08,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return clean(tok.decode(out[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--adapter', default='outputs/research_reasoner_1p7b_v2_schema_hard')
    ap.add_argument('--prompts', default='data/processed/rl_prompts_v2.jsonl')
    ap.add_argument('--out-flat', default='data/processed/rl_rejection_sft_v2_flat.jsonl')
    ap.add_argument('--out-report', default='eval_outputs/rl_rejection_report_v2.jsonl')
    ap.add_argument('--limit', type=int, default=160)
    ap.add_argument('--num-candidates', type=int, default=3)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--min-score', type=float, default=0.75)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    rows = list(read_jsonl(Path(args.prompts)))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.limit]
    model, tok = load_model(args.base_model, args.adapter)

    sft_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    totals: List[float] = []
    kept = 0
    for i, row in enumerate(rows):
        messages = split_prompt(row['prompt'])
        candidates = []
        for k in range(args.num_candidates):
            temp = 0.55 + 0.15 * k
            completion = generate_one(model, tok, messages, args.max_new_tokens, temp, 0.9, args.seed + i * 17 + k)
            try:
                score = score_output(completion, row)
            except Exception as e:
                score = {'total': -0.35, 'json_validity': 0.0, 'error': repr(e)}
            candidates.append({'completion': completion, 'score': score})
        best = max(candidates, key=lambda c: float(c['score'].get('total', -999)))
        best_total = float(best['score'].get('total', -0.35))
        totals.append(best_total)
        report_rows.append({'id': row.get('id'), 'task_family': row.get('task_family'), 'best_total': best_total, 'best_score': best['score'], 'candidates': candidates})
        if best_total >= args.min_score and best['score'].get('json_validity', 0.0) >= 1.0:
            train_messages = messages + [{'role': 'assistant', 'content': best['completion']}]
            sft_rows.append({
                'id': str(row.get('id', f'rl_{i}')) + '_best',
                'task_family': str(row.get('task_family', 'rl_rejection')),
                'difficulty': 'reward_selected',
                'language': 'unknown',
                'messages_json': json.dumps(train_messages, ensure_ascii=False, separators=(',', ':')),
                'sources_json': json.dumps(row.get('sources', []), ensure_ascii=False, separators=(',', ':')),
                'target_json': '{}',
                'reward_total': best_total,
            })
            kept += 1
        if (i + 1) % 20 == 0:
            print(json.dumps({'progress': i + 1, 'kept': kept, 'mean_best': sum(totals) / len(totals)}, ensure_ascii=False), flush=True)

    n_sft = write_jsonl(Path(args.out_flat), sft_rows)
    n_rep = write_jsonl(Path(args.out_report), report_rows)
    print(json.dumps({'ok': True, 'prompts': len(rows), 'candidates_per_prompt': args.num_candidates, 'kept': n_sft, 'report_rows': n_rep, 'mean_best_total': sum(totals)/max(1,len(totals)), 'out_flat': args.out_flat, 'out_report': args.out_report}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
