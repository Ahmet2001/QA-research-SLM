#!/usr/bin/env python3
"""Build compact chat-template-friendly SFT files for ResearchReasoner.

Compared with the first builder:
- compact JSON targets instead of pretty printed long JSON;
- system prompt explicitly disables thinking tags and requires JSON-only;
- optional no-eval split for overfit checks;
- keeps all schema fields but encourages short arrays/strings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

SYSTEM = (
    "You are ResearchReasoner. Output ONLY valid minified JSON. "
    "Do not use markdown. Do not use <think>. Do not explain outside JSON. "
    "Use only provided sources when sources are present. Required keys: "
    "task_type, research_plan, evidence_needed, selected_sources, claims, conflicts, uncertainties, answer. "
    "Keep values concise. claims[].status must be supported, contradicted, or insufficient."
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def source_block(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    parts = []
    for s in sources:
        parts.append(
            f"[source_id={s.get('source_id','')} status={s.get('status','unknown')} "
            f"type={s.get('source_type','')} reliability={s.get('reliability','')} date={s.get('date','')}]\n"
            f"title: {s.get('title','')}\ntext: {s.get('text','')}"
        )
    return "\n\nSOURCES:\n" + "\n\n".join(parts)


def compact_target(target: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure required fields and concise structures.
    return {
        "task_type": target.get("task_type", "research"),
        "research_plan": list(target.get("research_plan", []))[:6],
        "evidence_needed": list(target.get("evidence_needed", []))[:6],
        "selected_sources": list(target.get("selected_sources", []))[:8],
        "claims": list(target.get("claims", []))[:8],
        "conflicts": list(target.get("conflicts", []))[:4],
        "uncertainties": list(target.get("uncertainties", []))[:5],
        "answer": str(target.get("answer", ""))[:1200],
    }


def to_row(trace: Dict[str, Any]) -> Dict[str, Any]:
    user_msgs = [m.get("content", "") for m in trace.get("messages", []) if m.get("role") == "user"]
    user = "\n".join(user_msgs).strip() + source_block(trace.get("sources", []))
    target = compact_target(trace["target"])
    assistant = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant},
    ]
    return {
        "id": trace["id"],
        "task_family": trace.get("task_family", "unknown"),
        "difficulty": trace.get("difficulty", "medium"),
        "language": trace.get("language", "unknown"),
        "messages": messages,
        "target": target,
        "sources": trace.get("sources", []),
    }


def is_hard(row: Dict[str, Any]) -> bool:
    target = row.get("target", {})
    return (
        row.get("difficulty") in {"hard", "long_horizon"}
        or bool(target.get("conflicts"))
        or bool(target.get("uncertainties"))
        or any(c.get("status") in {"contradicted", "insufficient"} for c in target.get("claims", []))
        or len(row.get("sources", [])) >= 2
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["data/seed/research_seed.jsonl"])
    ap.add_argument("--out-stage1", default="data/processed/sft_stage1_compact.jsonl")
    ap.add_argument("--out-stage2", default="data/processed/sft_stage2_compact_hard.jsonl")
    ap.add_argument("--out-eval", default="data/processed/eval_compact.jsonl")
    ap.add_argument("--eval-count", type=int, default=8)
    args = ap.parse_args()

    traces: List[Dict[str, Any]] = []
    for p in args.inputs:
        path = Path(p)
        if path.exists():
            traces.extend(read_jsonl(path))
    rows = [to_row(t) for t in traces]
    if not rows:
        raise SystemExit("No input rows")
    # Deterministic split: keep at least some train rows for tiny datasets.
    eval_count = min(args.eval_count, max(1, len(rows) // 5))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:] or rows
    hard_rows = [r for r in train_rows if is_hard(r)] or train_rows
    print(json.dumps({
        "input_traces": len(traces),
        "train": write_jsonl(Path(args.out_stage1), train_rows),
        "hard": write_jsonl(Path(args.out_stage2), hard_rows),
        "eval": write_jsonl(Path(args.out_eval), eval_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
