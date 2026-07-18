#!/usr/bin/env python3
from __future__ import annotations

import argparse, inspect, json, os
from pathlib import Path
from typing import Any, Dict, List


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_messages(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    if 'messages_json' in ex:
        return json.loads(ex['messages_json'])
    return ex['messages']


def chat_text(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--max-seq-length', type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    base_model = cfg.get('base_model') or cfg.get('base_model_or_adapter')
    output_dir = args.output_dir or cfg['output_dir']
    max_seq_length = args.max_seq_length or int(cfg.get('max_seq_length', 4096))
    data_files = {'train': cfg['data']['train_file']}
    ev = cfg['data'].get('eval_file')
    if ev and Path(ev).exists():
        data_files['validation'] = ev
    ds = load_dataset('json', data_files=data_files)

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation=os.environ.get('ATTN_IMPL', 'sdpa'))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lc = cfg['lora']
    model = get_peft_model(model, LoraConfig(r=int(lc.get('r', 32)), lora_alpha=int(lc.get('alpha', 64)), lora_dropout=float(lc.get('dropout', 0.05)), bias='none', task_type='CAUSAL_LM', target_modules=lc.get('target_modules')))
    model.print_trainable_parameters()

    def encode(ex: Dict[str, Any]) -> Dict[str, Any]:
        msgs = get_messages(ex)
        prompt = chat_text(tok, msgs[:-1], True)
        full = chat_text(tok, msgs, False)
        pids = tok(prompt, add_special_tokens=False)['input_ids']
        fids = tok(full, add_special_tokens=False)['input_ids']
        if len(fids) > max_seq_length:
            overflow = len(fids) - max_seq_length
            fids = fids[overflow:]
            plen = max(0, len(pids) - overflow)
        else:
            plen = len(pids)
        labels = fids.copy()
        for i in range(min(plen, len(labels))): labels[i] = -100
        if all(x == -100 for x in labels): labels[-1] = fids[-1]
        return {'input_ids': fids, 'attention_mask': [1]*len(fids), 'labels': labels}

    tokenized = ds.map(encode, remove_columns=ds['train'].column_names)

    class Collator:
        def __call__(self, features):
            mx = max(len(f['input_ids']) for f in features)
            pad = tok.pad_token_id
            ids, masks, labs = [], [], []
            for f in features:
                n = mx - len(f['input_ids'])
                ids.append(f['input_ids'] + [pad]*n)
                masks.append(f['attention_mask'] + [0]*n)
                labs.append(f['labels'] + [-100]*n)
            return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(masks), 'labels': torch.tensor(labs)}

    opt, batching, logging = cfg.get('optimizer', {}), cfg.get('batching', {}), cfg.get('logging', {})
    kwargs = dict(output_dir=output_dir, per_device_train_batch_size=int(batching.get('per_device_train_batch_size', 1)), gradient_accumulation_steps=int(batching.get('gradient_accumulation_steps', 8)), num_train_epochs=float(batching.get('num_train_epochs', 1)), learning_rate=float(opt.get('learning_rate', 2e-5)), warmup_ratio=float(opt.get('warmup_ratio', 0.03)), weight_decay=float(opt.get('weight_decay', 0.01)), lr_scheduler_type=opt.get('lr_scheduler_type', 'cosine'), optim=opt.get('optim', 'paged_adamw_8bit'), logging_steps=int(logging.get('logging_steps', 10)), save_steps=int(logging.get('save_steps', 250)), save_total_limit=2, bf16=True, gradient_checkpointing=True, report_to='none', max_steps=args.max_steps)
    sig = inspect.signature(TrainingArguments.__init__)
    if 'validation' in tokenized:
        if 'eval_strategy' in sig.parameters: kwargs['eval_strategy'] = 'steps'
        elif 'evaluation_strategy' in sig.parameters: kwargs['evaluation_strategy'] = 'steps'
        kwargs['eval_steps'] = int(logging.get('eval_steps', 100))
    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized['train'], eval_dataset=tokenized.get('validation'), data_collator=Collator())
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / 'training_config_snapshot.json', 'w', encoding='utf-8') as f: json.dump(cfg, f, ensure_ascii=False, indent=2)
    print('SFT_FLAT_OK', output_dir)

if __name__ == '__main__': main()
