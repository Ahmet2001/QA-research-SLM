#!/usr/bin/env python3
"""Robust deterministic rewards for ResearchReasoner."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

REQUIRED = [
    "task_type",
    "research_plan",
    "evidence_needed",
    "selected_sources",
    "claims",
    "conflicts",
    "uncertainties",
    "answer",
]


def parse_json_output(text: str) -> Tuple[Dict[str, Any] | None, float]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        obj = json.loads(text)
        return (obj, 1.0) if isinstance(obj, dict) else (None, 0.0)
    except Exception:
        return None, 0.0


def mgpo_prompt_weight(binary_rewards: List[int], gamma: float = 4.0, target: float = 0.5) -> float:
    if not binary_rewards:
        return 1.0
    p = sum(binary_rewards) / len(binary_rewards)
    return float(math.exp(-gamma * abs(p - target)))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _source_id(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("source_id") or value.get("id") or ""
    return str(value).split("#", 1)[0].strip()


def score_output(completion: str, prompt_item: Dict[str, Any]) -> Dict[str, float]:
    obj, json_score = parse_json_output(completion)
    scores: Dict[str, float] = {"json_validity": json_score}
    if obj is None:
        scores.update({
            "required_fields": 0.0,
            "evidence_id_validity": 0.0,
            "citation_precision": 0.0,
            "supported_claim_ratio": 0.0,
            "avoid_bad_sources": 0.0,
            "conflict_detection": 0.0,
            "uncertainty_calibration": 0.0,
            "brevity_efficiency": 0.0,
            "total": -0.35,
        })
        return scores

    sources = _as_list(prompt_item.get("sources", []))
    known_ids = {
        str(s.get("source_id"))
        for s in sources
        if isinstance(s, dict) and s.get("source_id") is not None
    }
    hints = prompt_item.get("verifier_hints", {})
    if not isinstance(hints, dict):
        hints = {}
    bad_ids = {str(x) for x in _as_list(hints.get("avoid_source_ids", [])) if x is not None}
    must_ids = {str(x) for x in _as_list(hints.get("must_cite_source_ids", [])) if x is not None}

    scores["required_fields"] = sum(1 for f in REQUIRED if f in obj) / len(REQUIRED)

    selected_ids = {_source_id(x) for x in _as_list(obj.get("selected_sources", []))}
    selected_ids.discard("")
    selected_bad = selected_ids & bad_ids
    scores["avoid_bad_sources"] = 0.0 if selected_bad else 1.0

    raw_claims = _as_list(obj.get("claims", []))
    claim_list = [c for c in raw_claims if isinstance(c, dict)]
    malformed_claims = len(raw_claims) - len(claim_list)

    cited_ids: List[str] = []
    supported_claims = 0
    important_unsupported = 0
    valid_evidence_count = 0
    evidence_count = 0

    for c in claim_list:
        status = str(c.get("status", "")).lower()
        evidence = _as_list(c.get("evidence", []))
        if status == "supported" and evidence:
            supported_claims += 1
        if c.get("importance") == "high" and status == "supported" and not evidence:
            important_unsupported += 1
        for eid in evidence:
            sid = _source_id(eid)
            if sid:
                evidence_count += 1
                cited_ids.append(sid)
                if sid in known_ids and sid not in bad_ids:
                    valid_evidence_count += 1

    scores["evidence_id_validity"] = (
        valid_evidence_count / evidence_count
        if evidence_count
        else (1.0 if not must_ids else 0.0)
    )

    cited_set = set(cited_ids)
    if must_ids:
        precision = len(cited_set & must_ids) / max(1, len(cited_set)) if cited_set else 0.0
        recall = len(cited_set & must_ids) / len(must_ids)
        scores["citation_precision"] = 0.5 * precision + 0.5 * recall
    else:
        scores["citation_precision"] = 1.0 if evidence_count == 0 else scores["evidence_id_validity"]

    denominator = max(1, len(raw_claims))
    scores["supported_claim_ratio"] = supported_claims / denominator

    scores["conflict_detection"] = (
        1.0 if (not hints.get("requires_conflict_detection") or obj.get("conflicts")) else 0.0
    )
    scores["uncertainty_calibration"] = (
        1.0 if (not hints.get("requires_uncertainty") or obj.get("uncertainties")) else 0.0
    )

    length = len(completion.split())
    scores["brevity_efficiency"] = max(0.0, min(1.0, 1.0 - max(0, length - 450) / 700))

    total = (
        0.12 * scores["json_validity"]
        + 0.10 * scores["required_fields"]
        + 0.16 * scores["evidence_id_validity"]
        + 0.16 * scores["citation_precision"]
        + 0.16 * scores["supported_claim_ratio"]
        + 0.12 * scores["avoid_bad_sources"]
        + 0.08 * scores["conflict_detection"]
        + 0.06 * scores["uncertainty_calibration"]
        + 0.04 * scores["brevity_efficiency"]
    )

    if selected_bad:
        total -= 0.25
    if important_unsupported:
        total -= 0.30 * important_unsupported
    if malformed_claims:
        total -= min(0.30, 0.05 * malformed_claims)
    if evidence_count and scores["evidence_id_validity"] < 1.0:
        total -= 0.25 * (1.0 - scores["evidence_id_validity"])

    scores["total"] = max(-1.0, min(1.0, total))
    return scores


def scalar_reward(completion: str, prompt_item: Dict[str, Any]) -> float:
    return score_output(completion, prompt_item)["total"]


def reward_func(completions: List[str], prompts: List[Dict[str, Any]], **kwargs: Any) -> List[float]:
    return [scalar_reward(c, p) for c, p in zip(completions, prompts)]


if __name__ == "__main__":
    item = {
        "sources": [{"source_id": "src_001", "status": "usable", "text": "x"}],
        "verifier_hints": {"must_cite_source_ids": ["src_001"], "avoid_source_ids": []},
    }
    tests = [
        {"claims": ["malformed claim"]},
        {"claims": [{"claim": "x", "status": "supported", "evidence": ["src_001"]}]},
    ]
    for t in tests:
        obj = {
            "task_type": "document_qa",
            "research_plan": [],
            "evidence_needed": [],
            "selected_sources": ["src_001"],
            "claims": t["claims"],
            "conflicts": [],
            "uncertainties": [],
            "answer": "x",
        }
        print(json.dumps(score_output(json.dumps(obj), item), indent=2))
