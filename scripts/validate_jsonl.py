#!/usr/bin/env python3
"""Lightweight validator for ResearchReasoner JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_TARGET = {
    "task_type",
    "research_plan",
    "evidence_needed",
    "selected_sources",
    "claims",
    "conflicts",
    "uncertainties",
    "answer",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield idx, json.loads(line)


def validate_row(row: Dict[str, Any], line_no: int) -> List[str]:
    errors: List[str] = []
    if "messages" not in row or not isinstance(row["messages"], list):
        errors.append(f"line {line_no}: missing messages[]")
    target = row.get("target")
    if not isinstance(target, dict):
        errors.append(f"line {line_no}: missing target object")
        return errors
    missing = REQUIRED_TARGET - set(target)
    if missing:
        errors.append(f"line {line_no}: target missing fields {sorted(missing)}")
    source_status = {s.get("source_id"): s.get("status") for s in row.get("sources", [])}
    selected = set(target.get("selected_sources", []))
    bad_selected = [sid for sid in selected if source_status.get(sid) in {"avoid", "contradicted"}]
    if bad_selected:
        errors.append(f"line {line_no}: selected bad sources {bad_selected}")
    known_ids = set(source_status)
    for ci, claim in enumerate(target.get("claims", [])):
        if claim.get("status") not in {"supported", "contradicted", "insufficient"}:
            errors.append(f"line {line_no}: claim {ci} invalid status")
        for eid in claim.get("evidence", []):
            # Allow source_id#span style.
            sid = str(eid).split("#", 1)[0]
            if sid and sid not in known_ids:
                errors.append(f"line {line_no}: claim {ci} cites unknown evidence {eid}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    total = 0
    all_errors: List[str] = []
    for file_name in args.files:
        path = Path(file_name)
        for line_no, row in read_jsonl(path):
            total += 1
            all_errors.extend(validate_row(row, line_no))
    if all_errors:
        print("VALIDATION_FAILED")
        for e in all_errors:
            print(e)
        raise SystemExit(1)
    print(json.dumps({"ok": True, "rows": total}, indent=2))


if __name__ == "__main__":
    main()
