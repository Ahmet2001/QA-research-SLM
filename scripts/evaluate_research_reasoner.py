#!/usr/bin/env python3
"""Tiny evaluator for ResearchReasoner smoke checkpoints.

It evaluates generated JSON outputs with deterministic research_rewards.py.
Use this for smoke/regression checks, not as a real benchmark.
"""

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


def build_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row:
        return row["prompt"]
    messages = row.get("messages", [])
    return "\n\n".join(m["content"] for m in messages if m.get("role") in {"system", "user"})


def make_prompt_item(row: Dict[str, Any]) -> Dict[str, Any]:
    if "verifier_hints" in row:
        return row
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def evaluate(name: str, base_model: str, adapter: str | None, rows: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
    model, tokenizer = load_model(base_model, adapter)
    results = []
    totals = []
    for row in rows:
        prompt = build_prompt(row)
        completion = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        scores = score_output(completion, make_prompt_item(row))
        totals.append(scores["total"])
        results.append({
            "id": row.get("id", "unknown"),
            "completion": completion,
            "scores": scores,
        })
    mean_total = sum(totals) / max(1, len(totals))
    return {"name": name, "adapter": adapter, "n": len(rows), "mean_total": mean_total, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default="outputs/research_reasoner_1p7b_sft_stage1")
    ap.add_argument("--eval-file", default="data/processed/eval_research_reasoner.jsonl")
    ap.add_argument("--out", default="eval_outputs/research_reasoner_eval_smoke.json")
    ap.add_argument("--compare-base", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    rows = list(read_jsonl(Path(args.eval_file)))
    if not rows:
        raise SystemExit("No eval rows")
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
