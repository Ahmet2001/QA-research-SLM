#!/usr/bin/env python3
"""Build real ResearchReasoner SFT traces from SciFact raw JSONL.

Uses the public SciFact repository raw files instead of Hugging Face dataset scripts,
because the installed datasets version may not support legacy dataset scripts.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

URLS = {
    "corpus": "https://raw.githubusercontent.com/allenai/scifact/master/data/corpus.jsonl",
    "claims_train": "https://raw.githubusercontent.com/allenai/scifact/master/data/claims_train.jsonl",
    "claims_dev": "https://raw.githubusercontent.com/allenai/scifact/master/data/claims_dev.jsonl",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"download {url} -> {path}")
    with urllib.request.urlopen(url, timeout=60) as r, path.open("wb") as f:
        f.write(r.read())


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def as_text_abstract(doc: Dict[str, Any]) -> Tuple[str, List[str]]:
    title = str(doc.get("title") or "")
    abstract = doc.get("abstract") or doc.get("abstract_text") or []
    if isinstance(abstract, str):
        sentences = [abstract]
    elif isinstance(abstract, list):
        sentences = [str(x) for x in abstract]
    else:
        sentences = [str(abstract)]
    text = (title + "\n" + " ".join(sentences)).strip()
    return text, sentences


def normalize_label(label: str) -> str:
    l = str(label).lower()
    if "support" in l:
        return "supported"
    if "refute" in l or "contradict" in l:
        return "contradicted"
    return "insufficient"


def extract_evidence_items(claim_row: Dict[str, Any]) -> List[Tuple[str, str, List[int]]]:
    """Return list of (doc_id, label, sentence_indices)."""
    out: List[Tuple[str, str, List[int]]] = []
    evidence = claim_row.get("evidence") or {}
    if isinstance(evidence, dict):
        for doc_id, evs in evidence.items():
            if isinstance(evs, dict):
                evs = [evs]
            for ev in evs or []:
                if not isinstance(ev, dict):
                    continue
                label = ev.get("label") or ev.get("evidence_label") or claim_row.get("label") or "insufficient"
                sent = ev.get("sentences") or ev.get("sentence_indices") or ev.get("rationale") or []
                if isinstance(sent, int):
                    sent = [sent]
                out.append((str(doc_id), normalize_label(label), [int(x) for x in sent if str(x).isdigit()]))
    elif isinstance(evidence, list):
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            doc_id = ev.get("doc_id") or ev.get("document_id") or ev.get("evidence_doc_id")
            if doc_id is None:
                continue
            label = ev.get("label") or claim_row.get("label") or "insufficient"
            sent = ev.get("sentences") or ev.get("sentence_indices") or []
            if isinstance(sent, int):
                sent = [sent]
            out.append((str(doc_id), normalize_label(label), [int(x) for x in sent if str(x).isdigit()]))
    return out


def build_trace(claim_row: Dict[str, Any], corpus: Dict[str, Dict[str, Any]], split: str, idx: int) -> Dict[str, Any] | None:
    claim = str(claim_row.get("claim") or "").strip()
    if not claim:
        return None
    ev_items = extract_evidence_items(claim_row)
    if not ev_items:
        # Keep a small number of insufficient examples using cited docs if available.
        cited = claim_row.get("cited_doc_ids") or claim_row.get("doc_ids") or []
        if not cited:
            return None
        ev_items = [(str(cited[0]), "insufficient", [])]

    sources = []
    claims = []
    selected_sources = []
    seen = set()
    for doc_id, status, sent_ids in ev_items[:3]:
        doc = corpus.get(str(doc_id))
        if not doc:
            continue
        full_text, sentences = as_text_abstract(doc)
        source_id = f"scifact_{doc_id}"
        if source_id not in seen:
            status_source = "usable" if status in {"supported", "contradicted"} else "unknown"
            sources.append({
                "source_id": source_id,
                "title": str(doc.get("title") or f"SciFact doc {doc_id}"),
                "source_type": "scientific_abstract",
                "date": "2020",
                "reliability": 0.86,
                "status": status_source,
                "text": full_text[:5000],
            })
            seen.add(source_id)
        if status != "insufficient":
            selected_sources.append(source_id)
        evidence_refs = [source_id]
        if sent_ids:
            evidence_refs = [f"{source_id}#sent_{i}" for i in sent_ids[:4]]
        rationale = " ".join(sentences[i] for i in sent_ids if 0 <= i < len(sentences))[:600]
        claims.append({
            "claim": claim,
            "status": status,
            "confidence": 0.88 if status != "insufficient" else 0.65,
            "evidence": evidence_refs if status != "insufficient" else [],
            "importance": "high",
            "evidence_summary": rationale,
        })

    if not sources:
        return None
    statuses = {c["status"] for c in claims}
    if "supported" in statuses and "contradicted" in statuses:
        task_type = "conflict_resolution"
    else:
        task_type = "claim_verification"
    answer_status = ", ".join(sorted(statuses))
    uncertainties = [] if "insufficient" not in statuses else ["Provided evidence is not enough to verify the claim fully."]
    answer = f"Claim status: {answer_status}. Use the cited SciFact abstract evidence and do not add unsupported claims."
    trace_id = f"scifact_{split}_{claim_row.get('id', idx)}"
    return {
        "id": trace_id,
        "task_family": "claim_verification",
        "difficulty": "medium" if len(sources) == 1 else "hard",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are ResearchReasoner. Output only valid compact JSON."},
            {"role": "user", "content": f"Verify this scientific claim using the provided SciFact evidence. Claim: {claim}"},
        ],
        "sources": sources,
        "target": {
            "task_type": task_type,
            "research_plan": ["Identify the claim", "Read the SciFact evidence abstract", "Assign supported/contradicted/insufficient", "Cite source IDs only"],
            "evidence_needed": ["SciFact evidence abstract", "Rationale sentences when available"],
            "selected_sources": sorted(set(selected_sources)),
            "claims": claims[:3],
            "conflicts": [],
            "uncertainties": uncertainties,
            "answer": answer,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/scifact")
    ap.add_argument("--out", default="data/real/scifact_research_traces.jsonl")
    ap.add_argument("--limit-train", type=int, default=400)
    ap.add_argument("--limit-dev", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    for k, url in URLS.items():
        download(url, raw / f"{k}.jsonl")

    corpus: Dict[str, Dict[str, Any]] = {}
    for doc in read_jsonl(raw / "corpus.jsonl"):
        doc_id = doc.get("doc_id") or doc.get("id")
        if doc_id is not None:
            corpus[str(doc_id)] = doc

    rows: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)
    for split, limit in [("train", args.limit_train), ("dev", args.limit_dev)]:
        claims_path = raw / ("claims_train.jsonl" if split == "train" else "claims_dev.jsonl")
        claims = list(read_jsonl(claims_path))
        rng.shuffle(claims)
        used = 0
        for i, c in enumerate(claims):
            tr = build_trace(c, corpus, split, i)
            if tr is None:
                continue
            rows.append(tr)
            used += 1
            if used >= limit:
                break
    n = write_jsonl(Path(args.out), rows)
    print(json.dumps({"ok": True, "corpus_docs": len(corpus), "traces": n, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
