#!/usr/bin/env python3
"""Chat-template QLoRA SFT trainer for ResearchReasoner without TRL.

Fixes the smoke failure by:
1. using tokenizer.apply_chat_template for Qwen-style chat formatting;
2. masking prompt tokens with -100 so loss is only on assistant JSON;
3. supporting a small overfit/smoke mode and a real train mode.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    base_model = cfg.get("base_model") or cfg.get("base_model_or_adapter")
    output_dir = args.output_dir or cfg["output_dir"]
    max_seq_length = args.max_seq_length or int(cfg.get("max_seq_length", 8192))
    train_file = cfg["data"]["train_file"]
    eval_file = cfg["data"].get("eval_file")

    data_files = {"train": train_file}
    if eval_file and Path(eval_file).exists():
        data_files["validation"] = eval_file
    ds = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("alpha", 64)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def encode_example(ex: Dict[str, Any]) -> Dict[str, Any]:
        messages = ex["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"Expected final assistant message in {ex.get('id')}")
        prompt_messages = messages[:-1]
        assistant_message = messages[-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if len(full_ids) > max_seq_length:
            # Keep the end because assistant JSON target is at the end.
            overflow = len(full_ids) - max_seq_length
            full_ids = full_ids[overflow:]
            prompt_len = max(0, len(prompt_ids) - overflow)
        else:
            prompt_len = len(prompt_ids)

        labels = full_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # If truncation removed all assistant labels, drop via empty label marker.
        if all(x == -100 for x in labels):
            labels[-1] = full_ids[-1]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    remove_cols = ds["train"].column_names
    tokenized = ds.map(encode_example, remove_columns=remove_cols)

    class CausalCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            max_len = max(len(f["input_ids"]) for f in features)
            input_ids, attention_mask, labels = [], [], []
            pad_id = self.tokenizer.pad_token_id
            for f in features:
                pad = max_len - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [pad_id] * pad)
                attention_mask.append(f["attention_mask"] + [0] * pad)
                labels.append(f["labels"] + [-100] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    opt = cfg.get("optimizer", {})
    batching = cfg.get("batching", {})
    logging = cfg.get("logging", {})

    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=int(batching.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(batching.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(batching.get("num_train_epochs", 2)),
        learning_rate=float(opt.get("learning_rate", 2e-5)),
        warmup_ratio=float(opt.get("warmup_ratio", 0.03)),
        weight_decay=float(opt.get("weight_decay", 0.01)),
        lr_scheduler_type=opt.get("lr_scheduler_type", "cosine"),
        optim=opt.get("optim", "paged_adamw_8bit"),
        logging_steps=int(logging.get("logging_steps", 10)),
        save_steps=int(logging.get("save_steps", 500)),
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_steps=args.max_steps,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "validation" in tokenized:
        if "eval_strategy" in sig.parameters:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in sig.parameters:
            kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = int(logging.get("eval_steps", 250))
    else:
        if "eval_strategy" in sig.parameters:
            kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in sig.parameters:
            kwargs["evaluation_strategy"] = "no"

    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "training_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("SFT_CHAT_QLORA_OK", output_dir)


if __name__ == "__main__":
    main()
