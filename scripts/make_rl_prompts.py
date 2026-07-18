#!/usr/bin/env python3
"""Convert SFT rows into RL prompts with verifier hints.

The RL prompt keeps the question/context and strips the gold answer. Rewards are
computed by research_rewards.py using source metadata and verifier hints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def to_rl(row: Dict[str, Any]) -> Dict[str, Any]:
    sources = row.get("sources", [])
    target = row.get("target", {})
    avoid_ids = [s.get("source_id") for s in sources if s.get("status") in {"avoid", "contradicted"}]
    selected_ids = target.get("selected_sources", [])
    prompt = "\n\n".join(
        m["content"] for m in row.get("messages", []) if m.get("role") in {"system", "user"}
    )
    requires_uncertainty = bool(target.get("uncertainties")) or any(
        c.get("status") == "insufficient" for c in target.get("claims", [])
    )
    requires_conflict = bool(target.get("conflicts"))
    statuses = {c.get("status") for c in target.get("claims", [])}
    if "contradicted" in statuses and "supported" in statuses:
        gold_status = "mixed"
    elif "contradicted" in statuses:
        gold_status = "contradicted"
    elif "insufficient" in statuses:
        gold_status = "insufficient"
    elif "supported" in statuses:
        gold_status = "supported"
    else:
        gold_status = "unknown"
    return {
        "id": row.get("id", "unknown") + "_rl",
        "task_family": row.get("task_family", "unknown"),
        "prompt": prompt,
        "sources": [
            {"source_id": s.get("source_id"), "status": s.get("status", "unknown"), "text": s.get("text", "")}
            for s in sources
        ],
        "verifier_hints": {
            "must_cite_source_ids": selected_ids,
            "avoid_source_ids": avoid_ids,
            "requires_uncertainty": requires_uncertainty,
            "requires_conflict_detection": requires_conflict,
            "gold_status": gold_status,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/processed/sft_stage1_broad.jsonl")
    ap.add_argument("--out", default="data/processed/rl_prompts.jsonl")
    args = ap.parse_args()
    rows = [to_rl(r) for r in read_jsonl(Path(args.sft))]
    n = write_jsonl(Path(args.out), rows)
    print(json.dumps({"ok": True, "rl_rows": n}, indent=2))


if __name__ == "__main__":
    main()
