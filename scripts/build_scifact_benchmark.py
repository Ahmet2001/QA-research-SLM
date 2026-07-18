#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

SYSTEM = (
    "You are ResearchReasoner. Output ONLY valid minified JSON. No markdown. No <think>. "
    "Strict schema: task_type string; research_plan array of short strings; evidence_needed array of short strings; "
    "selected_sources array of source_id strings only; claims array of objects with claim,status,confidence,evidence,importance; "
    "conflicts array; uncertainties array of strings; answer string only. "
    "For this task answer must be exactly SUPPORT, CONTRADICT, or NOINFO. "
    "Evidence values must be source_id strings from SOURCES."
)


def dumpj(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(as_text(x) for x in value if as_text(x)).strip()
    return str(value).strip() if value is not None else ""


def ensure_data(cache_dir: Path) -> Path:
    data_dir = cache_dir / "data"
    corpus = data_dir / "corpus.jsonl"
    claims = data_dir / "claims_dev.jsonl"
    if corpus.exists() and claims.exists():
        return data_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "data.tar.gz"
    if not archive.exists():
        print(json.dumps({"download": DATA_URL, "to": str(archive)}), flush=True)
        urllib.request.urlretrieve(DATA_URL, archive)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(cache_dir)

    if not corpus.exists() or not claims.exists():
        raise FileNotFoundError(f"SciFact files were not found under {data_dir}")
    return data_dir


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_label(label: Any) -> str:
    text = str(label).upper().strip()
    if "SUPPORT" in text:
        return "SUPPORT"
    if "CONTRADICT" in text or "REFUTE" in text:
        return "CONTRADICT"
    return "NOINFO"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/domain_benchmarks/scifact_validation_flat.jsonl")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--cache-dir", default="data/raw/scifact")
    args = parser.parse_args()

    data_dir = ensure_data(Path(args.cache_dir))
    corpus_rows = read_jsonl(data_dir / "corpus.jsonl")
    claim_rows = read_jsonl(data_dir / "claims_dev.jsonl")

    corpus: Dict[str, Dict[str, str]] = {}
    for row in corpus_rows:
        did = str(row.get("doc_id", "")).strip()
        if not did:
            continue
        corpus[did] = {
            "title": as_text(row.get("title", "")),
            "text": as_text(row.get("abstract", "")),
        }

    rng = random.Random(args.seed)
    rng.shuffle(claim_rows)
    all_doc_ids = list(corpus.keys())
    output_rows: List[Dict[str, Any]] = []

    for raw_index, row in enumerate(claim_rows):
        claim = as_text(row.get("claim", ""))
        if not claim:
            continue

        evidence_map = row.get("evidence", {}) or {}
        gold_label = "NOINFO"
        gold_doc_ids: List[str] = []
        if isinstance(evidence_map, dict):
            labels_seen: List[str] = []
            for did_raw, annotations in evidence_map.items():
                did = str(did_raw)
                if did in corpus and did not in gold_doc_ids:
                    gold_doc_ids.append(did)
                if isinstance(annotations, list):
                    for ann in annotations:
                        if isinstance(ann, dict):
                            labels_seen.append(normalize_label(ann.get("label", "")))
            if "CONTRADICT" in labels_seen:
                gold_label = "CONTRADICT"
            elif "SUPPORT" in labels_seen:
                gold_label = "SUPPORT"

        cited_ids = [str(x) for x in (row.get("cited_doc_ids", []) or [])]
        candidate_ids: List[str] = []
        for did in gold_doc_ids + cited_ids:
            if did in corpus and did not in candidate_ids:
                candidate_ids.append(did)

        while len(candidate_ids) < args.max_sources and all_doc_ids:
            did = rng.choice(all_doc_ids)
            if did not in candidate_ids:
                candidate_ids.append(did)
        candidate_ids = candidate_ids[: args.max_sources]

        sources: List[Dict[str, Any]] = []
        for did in candidate_ids:
            record = corpus[did]
            sources.append({
                "source_id": f"sf_{did}",
                "title": record["title"],
                "text": record["text"][:4500],
                "status": "usable",
            })

        selected = [f"sf_{did}" for did in gold_doc_ids if did in candidate_ids]
        source_block = "\n".join(
            f"[{s['source_id']}] {s['title']}: {s['text']}" for s in sources
        )
        user = (
            "Classify the scientific claim using only the provided abstracts. "
            "Return SUPPORT when the evidence supports it, CONTRADICT when the evidence refutes it, "
            "and NOINFO when the supplied evidence is insufficient. Cite supporting source IDs in selected_sources and evidence.\n\n"
            f"CLAIM: {claim}\n\nSOURCES:\n{source_block}"
        )
        status = {
            "SUPPORT": "supported",
            "CONTRADICT": "contradicted",
            "NOINFO": "insufficient",
        }[gold_label]
        uncertainties = ["The supplied abstracts do not establish the claim."] if gold_label == "NOINFO" else []
        target = {
            "task_type": "scientific_claim_verification",
            "research_plan": ["read the claim", "inspect relevant abstracts", "classify the evidence"],
            "evidence_needed": ["abstract evidence that supports or refutes the claim"],
            "selected_sources": selected,
            "claims": [{
                "claim": claim,
                "status": status,
                "confidence": 0.9 if gold_label != "NOINFO" else 0.7,
                "evidence": selected,
                "importance": "high",
            }],
            "conflicts": [],
            "uncertainties": uncertainties,
            "answer": gold_label,
        }
        sample_id = str(row.get("id", raw_index))
        output_rows.append({
            "id": f"scifact_{sample_id}",
            "task_family": "scientific_claim_verification",
            "difficulty": "public_benchmark",
            "language": "en",
            "messages_json": dumpj([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": dumpj(target)},
            ]),
            "sources_json": dumpj(sources),
            "target_json": dumpj(target),
            "gold_label": gold_label,
            "gold_source_ids": selected,
            "claim": claim,
        })
        if len(output_rows) >= args.limit:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(dumpj(row) + "\n")

    label_counts = {"SUPPORT": 0, "CONTRADICT": 0, "NOINFO": 0}
    for row in output_rows:
        label_counts[row["gold_label"]] += 1
    print(json.dumps({
        "ok": True,
        "out": str(out_path),
        "rows": len(output_rows),
        "split": "claims_dev",
        "label_counts": label_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
