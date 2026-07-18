#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from research_rewards import parse_json_output, score_output

LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def json_field(row: Dict[str, Any], key: str, fallback: Any) -> Any:
    value = row.get(key)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return fallback
    return fallback if value is None else value


def source_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("source_id") or value.get("id") or value.get("source") or "")
    return str(value)


def chat_text(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sanitize_completion(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def verifier_item(row: Dict[str, Any]) -> Dict[str, Any]:
    sources = json_field(row, "sources_json", row.get("sources", []))
    target = json_field(row, "target_json", row.get("target", {}))
    must = [source_id(x) for x in target.get("selected_sources", [])]
    return {
        "sources": [
            {
                "source_id": source_id(s),
                "status": s.get("status", "unknown") if isinstance(s, dict) else "unknown",
                "text": s.get("text", "") if isinstance(s, dict) else "",
            }
            for s in sources
        ],
        "verifier_hints": {
            "must_cite_source_ids": must,
            "avoid_source_ids": [],
            "requires_uncertainty": bool(target.get("uncertainties")),
            "requires_conflict_detection": bool(target.get("conflicts")),
        },
    }


def normalize_label(value: Any) -> Optional[str]:
    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z]+", " ", text)
    tokens = set(text.split())
    if "NOINFO" in tokens or "INSUFFICIENT" in tokens or "UNKNOWN" in tokens or "UNSUPPORTED" in tokens:
        return "NOINFO"
    if "CONTRADICT" in tokens or "CONTRADICTED" in tokens or "REFUTE" in tokens or "REFUTED" in tokens:
        return "CONTRADICT"
    if "SUPPORT" in tokens or "SUPPORTED" in tokens or "ENTAILMENT" in tokens:
        return "SUPPORT"
    return None


def extract_prediction(completion: str) -> Tuple[Optional[str], List[str], bool]:
    obj, ok = parse_json_output(completion)
    if isinstance(obj, dict):
        answer_label = normalize_label(obj.get("answer", ""))
        selected = obj.get("selected_sources", [])
        if not isinstance(selected, list):
            selected = []
        if answer_label is None:
            claims = obj.get("claims", [])
            if isinstance(claims, list):
                for claim in claims:
                    if isinstance(claim, dict):
                        answer_label = normalize_label(claim.get("status", ""))
                        if answer_label:
                            break
        return answer_label, [source_id(x) for x in selected], bool(ok)
    return normalize_label(completion[:200]), [], bool(ok)


def set_prf(pred: Sequence[str], gold: Sequence[str]) -> Tuple[float, float, float]:
    pset = {x for x in pred if x}
    gset = {x for x in gold if x}
    if not pset and not gset:
        return 1.0, 1.0, 1.0
    if not pset:
        return 0.0, 0.0, 0.0
    overlap = len(pset & gset)
    precision = overlap / len(pset)
    recall = overlap / len(gset) if gset else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def classification_summary(golds: Sequence[str], preds: Sequence[Optional[str]]) -> Dict[str, Any]:
    per_label: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for g in golds if g == label),
        }
    n = max(1, len(golds))
    accuracy = sum(1 for g, p in zip(golds, preds) if g == p) / n
    attempt_rate = sum(1 for p in preds if p in LABELS) / n
    return {
        "label_accuracy": accuracy,
        "label_macro_f1": sum(f1_values) / len(LABELS),
        "label_attempt_rate": attempt_rate,
        "per_label": per_label,
    }


def load_model(base_model: str, adapter: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
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


def generate(model: Any, tokenizer: Any, row: Dict[str, Any], max_input_tokens: int, max_new_tokens: int) -> str:
    import torch

    messages = json_field(row, "messages_json", row.get("messages", []))
    messages = [m for m in messages if isinstance(m, dict) and m.get("role") != "assistant"]
    prompt = chat_text(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[-1]:]
    return sanitize_completion(tokenizer.decode(generated, skip_special_tokens=True))


def evaluate_run(name: str, base_model: str, adapter: Optional[str], rows: Sequence[Dict[str, Any]], max_input_tokens: int, max_new_tokens: int) -> Dict[str, Any]:
    model, tokenizer = load_model(base_model, adapter)
    results: List[Dict[str, Any]] = []
    golds: List[str] = []
    preds: List[Optional[str]] = []
    rewards: List[float] = []
    evidence_all: List[float] = []
    evidence_labeled: List[float] = []
    valid_json = 0

    for index, row in enumerate(rows, start=1):
        completion = generate(model, tokenizer, row, max_input_tokens, max_new_tokens)
        try:
            scores = score_output(completion, verifier_item(row))
        except Exception as exc:
            scores = {"json_validity": 0.0, "total": -0.35, "error": repr(exc)}

        pred_label, pred_sources, json_ok = extract_prediction(completion)
        gold_label = str(row.get("gold_label", "")).upper().strip()
        target = json_field(row, "target_json", row.get("target", {}))
        gold_sources = row.get("gold_source_ids", target.get("selected_sources", []))
        if not isinstance(gold_sources, list):
            gold_sources = []
        gold_sources = [source_id(x) for x in gold_sources]
        ep, er, ef1 = set_prf(pred_sources, gold_sources)

        golds.append(gold_label)
        preds.append(pred_label)
        rewards.append(float(scores.get("total", -0.35)))
        evidence_all.append(ef1)
        if gold_sources:
            evidence_labeled.append(ef1)
        valid_json += int(json_ok)
        results.append({
            "id": row.get("id"),
            "claim": row.get("claim"),
            "gold_label": gold_label,
            "pred_label": pred_label,
            "label_correct": pred_label == gold_label,
            "gold_sources": gold_sources,
            "pred_sources": pred_sources,
            "evidence_source_precision": ep,
            "evidence_source_recall": er,
            "evidence_source_f1": ef1,
            "completion": completion,
            "scores": scores,
        })
        if index % 25 == 0:
            print(json.dumps({"run": name, "done": index, "n": len(rows)}, ensure_ascii=False), flush=True)

    n = max(1, len(rows))
    cls = classification_summary(golds, preds)
    summary: Dict[str, Any] = {
        "name": name,
        "adapter": adapter,
        "n": len(rows),
        **cls,
        "evidence_source_f1_all": sum(evidence_all) / n,
        "evidence_source_f1_supported_or_contradicted": (
            sum(evidence_labeled) / len(evidence_labeled) if evidence_labeled else 0.0
        ),
        "json_valid_rate": valid_json / n,
        "mean_verifier_reward": sum(rewards) / n,
        "results": results,
    }

    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def parse_adapter(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Adapter must use NAME=PATH format")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Adapter must use non-empty NAME=PATH format")
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.eval_file)))[:args.limit]
    output: Dict[str, Any] = {
        "benchmark": "SciFact validation diagnostic",
        "notes": "Label accuracy and macro-F1 use SUPPORT/CONTRADICT/NOINFO. Evidence is measured at abstract source-document level, not official sentence-rationale level.",
        "eval_file": args.eval_file,
        "runs": [],
    }
    if args.compare_base:
        output["runs"].append(evaluate_run("base", args.base_model, None, rows, args.max_input_tokens, args.max_new_tokens))
    for name, adapter_path in args.adapter:
        output["runs"].append(evaluate_run(name, args.base_model, adapter_path, rows, args.max_input_tokens, args.max_new_tokens))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    summary_keys = [
        "name", "adapter", "n", "label_accuracy", "label_macro_f1", "label_attempt_rate",
        "evidence_source_f1_all", "evidence_source_f1_supported_or_contradicted",
        "json_valid_rate", "mean_verifier_reward",
    ]
    print(json.dumps({
        "ok": True,
        "out": str(out_path),
        "summary": [{key: run.get(key) for key in summary_keys} for run in output["runs"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
