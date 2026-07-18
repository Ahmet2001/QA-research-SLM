#!/usr/bin/env python3
"""TRL-free QLoRA SFT trainer for ResearchReasoner.

Uses only transformers + datasets + peft + bitsandbytes, because the global
python3 environment may not include TRL.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    base_model = cfg.get("base_model") or cfg.get("base_model_or_adapter")
    output_dir = cfg["output_dir"]
    max_seq_length = int(cfg.get("max_seq_length", 8192))
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

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

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
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "training_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("SFT_TRANSFORMERS_OK", output_dir)


if __name__ == "__main__":
    main()
