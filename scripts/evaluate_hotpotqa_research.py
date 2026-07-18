#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from research_rewards import parse_json_output, score_output


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


def chat_text(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def sanitize_completion(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def source_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("source_id") or value.get("id") or value.get("source") or "")
    return str(value)


def verifier_item(row: Dict[str, Any]) -> Dict[str, Any]:
    sources = json_field(row, "sources_json", row.get("sources", []))
    target = json_field(row, "target_json", row.get("target", {}))
    must = [source_id(x) for x in target.get("selected_sources", [])]
    avoid = [
        source_id(s)
        for s in sources
        if isinstance(s, dict) and s.get("status") in {"avoid", "contradicted"}
    ]
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
            "avoid_source_ids": avoid,
            "requires_uncertainty": bool(target.get("uncertainties")),
            "requires_conflict_detection": bool(target.get("conflicts")),
        },
    }


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def answer_em(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


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


def extract_prediction(completion: str) -> Tuple[str, List[str], bool]:
    obj, ok = parse_json_output(completion)
    if not isinstance(obj, dict):
        return completion.strip(), [], bool(ok)
    answer = str(obj.get("answer", "")).strip()
    selected = obj.get("selected_sources", [])
    if not isinstance(selected, list):
        selected = []
    return answer, [source_id(x) for x in selected], bool(ok)


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
    totals: List[float] = []
    em_values: List[float] = []
    answer_f1_values: List[float] = []
    evidence_p_values: List[float] = []
    evidence_r_values: List[float] = []
    evidence_f1_values: List[float] = []
    joint_values: List[float] = []
    valid_json = 0

    for index, row in enumerate(rows, start=1):
        completion = generate(model, tokenizer, row, max_input_tokens, max_new_tokens)
        try:
            scores = score_output(completion, verifier_item(row))
        except Exception as exc:
            scores = {"json_validity": 0.0, "total": -0.35, "error": repr(exc)}

        prediction, predicted_sources, json_ok = extract_prediction(completion)
        target = json_field(row, "target_json", row.get("target", {}))
        gold_answer = str(row.get("gold_answer", target.get("answer", "")))
        gold_sources = [source_id(x) for x in target.get("selected_sources", [])]

        em = answer_em(prediction, gold_answer)
        af1 = answer_f1(prediction, gold_answer)
        ep, er, ef1 = set_prf(predicted_sources, gold_sources)
        joint = af1 * ef1

        valid_json += int(json_ok)
        totals.append(float(scores.get("total", -0.35)))
        em_values.append(em)
        answer_f1_values.append(af1)
        evidence_p_values.append(ep)
        evidence_r_values.append(er)
        evidence_f1_values.append(ef1)
        joint_values.append(joint)
        results.append({
            "id": row.get("id"),
            "gold_answer": gold_answer,
            "pred_answer": prediction,
            "gold_sources": gold_sources,
            "pred_sources": predicted_sources,
            "answer_em": em,
            "answer_f1": af1,
            "evidence_source_precision": ep,
            "evidence_source_recall": er,
            "evidence_source_f1": ef1,
            "joint_f1_proxy": joint,
            "completion": completion,
            "scores": scores,
        })
        if index % 25 == 0:
            print(json.dumps({"run": name, "done": index, "n": len(rows)}, ensure_ascii=False), flush=True)

    n = max(1, len(rows))
    summary = {
        "name": name,
        "adapter": adapter,
        "n": len(rows),
        "answer_em": sum(em_values) / n,
        "answer_f1": sum(answer_f1_values) / n,
        "evidence_source_precision": sum(evidence_p_values) / n,
        "evidence_source_recall": sum(evidence_r_values) / n,
        "evidence_source_f1": sum(evidence_f1_values) / n,
        "joint_f1_proxy": sum(joint_values) / n,
        "json_valid_rate": valid_json / n,
        "mean_verifier_reward": sum(totals) / n,
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
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.eval_file)))[:args.limit]
    output: Dict[str, Any] = {
        "benchmark": "HotpotQA distractor diagnostic",
        "notes": "Answer EM/F1 uses Hotpot-style normalization. Evidence metrics are source-document level, not official supporting-sentence level; joint_f1_proxy is answer_f1 * evidence_source_f1.",
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

    summary_keys = ["name", "adapter", "n", "answer_em", "answer_f1", "evidence_source_f1", "joint_f1_proxy", "json_valid_rate", "mean_verifier_reward"]
    print(json.dumps({"ok": True, "out": str(out_path), "summary": [{key: run.get(key) for key in summary_keys} for run in output["runs"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
