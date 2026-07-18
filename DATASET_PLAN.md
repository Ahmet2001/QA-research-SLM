# Dataset plan: VibeThinker-style ResearchReasoner data

The training data mirrors the VibeThinker idea of moving from a broad solution spectrum to high-value verifiable signals, but replaces math/code supervision with research-verifiable supervision.

## SFT Stage 1: Broad research spectrum

Task families:

1. `research_planning`: task decomposition, source requirements, search plan.
2. `document_qa`: answer only from provided document chunks.
3. `evidence_extraction`: extract decision-relevant evidence spans.
4. `claim_verification`: supported / contradicted / insufficient.
5. `conflict_resolution`: resolve source conflicts using reliability, recency and source status.
6. `cited_synthesis`: synthesize with selected_sources, claims, uncertainty and final answer.

Reliable supervision signals:

- explicit source IDs
- evidence spans or source references
- support/refute/insufficient labels
- contradicted/avoid source flags
- conflict objects
- strict JSON schema

## SFT Stage 2: Hard and long evidence curriculum

Keep examples where at least one is true:

- multi-source reasoning
- conflict present
- uncertainty required
- contradicted claim present
- long source context
- multi-hop evidence needed

Reject examples where:

- selected_sources contains an avoid or contradicted source
- answer has important claims without evidence
- output violates JSON contract

## RL prompt data

RL prompts strip the gold target and keep verifier hints:

```json
{
  "must_cite_source_ids": ["src_001"],
  "avoid_source_ids": ["src_bad"],
  "requires_uncertainty": true,
  "requires_conflict_detection": true,
  "gold_status": "mixed"
}
```

## MGPO-style prompt weighting

For each prompt, sample G completions. Convert reward into binary correctness using a threshold, then compute empirical group accuracy `p(q)`. Prompts near `p(q)=0.5` get higher weight; prompts at 0.0 or 1.0 may be filtered for later rounds.

## Long2Short research RL

After grounding accuracy stabilizes, apply a second RL pass that only rewards shorter outputs among already-correct completions. Incorrect outputs do not receive length-based reward. This prevents shallow short answers from being rewarded.

## Later data expansion

Add public datasets through converters into the same schema:

- claim verification: FEVER-style and SciFact-style data
- scientific document QA: QASPER-style data
- multi-hop evidence QA: HotpotQA-style data
- document-grounded summarization: evidence-labelled synthetic traces
- Turkish research tasks: manually curated + translated/evidence-preserved traces

The key rule: do not add data unless it has a verifier signal or can be converted into one.
