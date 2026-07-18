#!/usr/bin/env python3
"""Chat-template evaluator for ResearchReasoner adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from research_rewards import score_output


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_prompt_item(row: Dict[str, Any]) -> Dict[str, Any]:
    sources = row.get("sources", [])
    target = row.get("target", {})
    return {
        "sources": [
            {"source_id": s.get("source_id"), "status": s.get("status", "unknown"), "text": s.get("text", "")}
            for s in sources
        ],
        "verifier_hints": {
            "must_cite_source_ids": target.get("selected_sources", []),
            "avoid_source_ids": [s.get("source_id") for s in sources if s.get("status") in {"avoid", "contradicted"}],
            "requires_uncertainty": bool(target.get("uncertainties")),
            "requires_conflict_detection": bool(target.get("conflicts")),
        },
    }


def load_model(base_model: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def build_prompt(row: Dict[str, Any], tokenizer) -> str:
    prompt_messages = [m for m in row.get("messages", []) if m.get("role") != "assistant"]
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def evaluate(name: str, base_model: str, adapter: str | None, rows: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
    model, tokenizer = load_model(base_model, adapter)
    results = []
    totals = []
    for row in rows:
        prompt = build_prompt(row, tokenizer)
        completion = generate(model, tokenizer, prompt, max_new_tokens)
        scores = score_output(completion, make_prompt_item(row))
        totals.append(scores["total"])
        results.append({"id": row.get("id"), "completion": completion, "scores": scores})
    return {
        "name": name,
        "adapter": adapter,
        "n": len(rows),
        "mean_total": sum(totals) / max(1, len(totals)),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval-file", default="data/processed/eval_research_reasoner.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-base", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    rows = list(read_jsonl(Path(args.eval_file)))
    out = {"eval_file": args.eval_file, "runs": []}
    if args.compare_base:
        out["runs"].append(evaluate("base", args.base_model, None, rows, args.max_new_tokens))
    out["runs"].append(evaluate("sft_adapter", args.base_model, args.adapter, rows, args.max_new_tokens))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "ok": True,
        "out": args.out,
        "summary": [{"name": r["name"], "n": r["n"], "mean_total": r["mean_total"]} for r in out["runs"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
