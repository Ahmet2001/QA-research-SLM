#!/usr/bin/env python3
"""Build SFT JSONL files for ResearchReasoner.

This script intentionally starts with seed/local traces so the pipeline is testable
without network access. Later, public datasets can be converted into the same trace
schema and mixed here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

SYSTEM = (
    "You are ResearchReasoner, an evidence-grounded research reasoning model. "
    "Use only provided evidence when evidence is present. Return strict JSON with: "
    "task_type, research_plan, evidence_needed, selected_sources, claims, conflicts, uncertainties, answer."
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def source_block(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    blocks = []
    for s in sources:
        blocks.append(
            "\n".join(
                [
                    f"[source_id: {s.get('source_id', '')}]",
                    f"title: {s.get('title', '')}",
                    f"type: {s.get('source_type', '')}",
                    f"date: {s.get('date', '')}",
                    f"reliability: {s.get('reliability', '')}",
                    f"status: {s.get('status', 'unknown')}",
                    "text:",
                    str(s.get("text", "")),
                ]
            )
        )
    return "\n\nSources:\n" + "\n\n".join(blocks)


def user_content(trace: Dict[str, Any]) -> str:
    user_msgs = [m["content"] for m in trace.get("messages", []) if m.get("role") == "user"]
    content = "\n".join(user_msgs).strip()
    content += source_block(trace.get("sources", []))
    return content.strip()


def to_sft_row(trace: Dict[str, Any]) -> Dict[str, Any]:
    assistant_json = json.dumps(trace["target"], ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content(trace)},
        {"role": "assistant", "content": assistant_json},
    ]
    return {
        "id": trace["id"],
        "task_family": trace.get("task_family", "unknown"),
        "difficulty": trace.get("difficulty", "medium"),
        "language": trace.get("language", "unknown"),
        "messages": messages,
        "text": render_simple_chat(messages),
        "target": trace["target"],
        "sources": trace.get("sources", []),
    }


def render_simple_chat(messages: List[Dict[str, str]]) -> str:
    # Generic fallback text for SFTTrainer when tokenizer chat template is unavailable.
    chunks = []
    for m in messages:
        role = m["role"].upper()
        chunks.append(f"<{role}>\n{m['content']}")
    return "\n\n".join(chunks) + "\n"


def is_hard(trace: Dict[str, Any]) -> bool:
    sources = trace.get("sources", [])
    target = trace.get("target", {})
    source_chars = sum(len(str(s.get("text", ""))) for s in sources)
    has_conflict = bool(target.get("conflicts"))
    has_uncertainty = bool(target.get("uncertainties"))
    has_contradicted = any(
        c.get("status") == "contradicted" for c in target.get("claims", [])
    )
    multi_source = len(sources) >= 2
    return (
        trace.get("difficulty") in {"hard", "long_horizon"}
        or (multi_source and (has_conflict or has_uncertainty or has_contradicted))
        or source_chars >= 2500
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="data/seed/research_seed.jsonl")
    ap.add_argument("--out-stage1", default="data/processed/sft_stage1_broad.jsonl")
    ap.add_argument("--out-stage2", default="data/processed/sft_stage2_hard_long.jsonl")
    ap.add_argument("--out-eval", default="data/processed/eval_research_reasoner.jsonl")
    ap.add_argument("--eval-ratio", type=float, default=0.2)
    args = ap.parse_args()

    seed_path = Path(args.seed)
    traces = list(read_jsonl(seed_path))
    if not traces:
        raise SystemExit(f"No traces found in {seed_path}")

    sft_rows = [to_sft_row(t) for t in traces]
    hard_rows = [to_sft_row(t) for t in traces if is_hard(t)]

    # Tiny deterministic split. In real training, use a larger fixed eval set.
    eval_count = max(1, int(round(len(sft_rows) * args.eval_ratio)))
    eval_rows = sft_rows[:eval_count]
    train_rows = sft_rows[eval_count:] or sft_rows

    n1 = write_jsonl(Path(args.out_stage1), train_rows)
    n2 = write_jsonl(Path(args.out_stage2), hard_rows or train_rows)
    ne = write_jsonl(Path(args.out_eval), eval_rows)

    print(json.dumps({
        "stage1_rows": n1,
        "stage2_rows": n2,
        "eval_rows": ne,
        "seed_rows": len(traces),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
