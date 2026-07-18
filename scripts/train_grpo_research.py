#!/usr/bin/env python3
"""GRPO/MGPO-style RL entrypoint for ResearchReasoner.

This script is intentionally conservative: exact TRL GRPO APIs may differ by
version, so it isolates reward logic and keeps config clean. If the installed TRL
version has a changed signature, this file is the only place that needs adapting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    from research_rewards import scalar_reward

    base_model = cfg.get("base_model_or_adapter") or cfg.get("base_model")
    output_dir = cfg["output_dir"]
    train_file = cfg["data"]["train_file"]
    ds = load_dataset("json", data_files={"train": train_file})["train"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    )
    model.config.use_cache = False

    def reward_fn(completions: List[str], **kwargs: Any) -> List[float]:
        # TRL versions may pass prompts, prompt_ids, or batch metadata differently.
        # The dataset row is kept under kwargs when available. For strict production,
        # adapt this wrapper to the exact installed TRL version.
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or []
        if prompts and isinstance(prompts[0], dict):
            items = prompts
        else:
            # Fallback: neutral prompt item, still rewards JSON/format validity.
            items = [{"sources": [], "verifier_hints": {}} for _ in completions]
        return [float(scalar_reward(c, item)) for c, item in zip(completions, items)]

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=int(cfg.get("num_generations", 8)),
        max_prompt_length=int(cfg.get("max_seq_length", 16384)) // 2,
        max_completion_length=int(cfg.get("max_seq_length", 16384)) // 2,
        logging_steps=5,
        save_steps=200,
        report_to="tensorboard",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "rl_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
