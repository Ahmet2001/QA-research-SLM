# Benchmark Results

This document records public-benchmark diagnostics for the ResearchReasoner adapters built on `Qwen/Qwen3-1.7B`.

> These are reproducible diagnostic runs, not official leaderboard submissions. Sample counts, splits, seeds, prompts, decoding settings, and metric limitations are listed below. Results should not be presented as full-benchmark scores.

## Models

| Name | Adapter |
|---|---|
| Base | `Qwen/Qwen3-1.7B` |
| RL-lite | `outputs/research_reasoner_1p7b_v2_rl_lite` |
| Real-compact-flat | `outputs/research_reasoner_1p7b_real_compact_flat` |

## PubMedQA

- Dataset configuration: `pubmed_qa`, `pqa_labeled`
- Split used: train
- Diagnostic size: 80 examples

| Model | Label accuracy | Label attempt rate | Mean verifier reward |
|---|---:|---:|---:|
| Base | 1.25% | 16.25% | -0.1900 |
| **RL-lite** | **52.50%** | 85.00% | **0.7750** |
| Real-compact-flat | 22.50% | **93.75%** | 0.7218 |

The task label is extracted as `yes`, `no`, or `maybe`. This is an 80-example diagnostic from the labeled training split and is not the official PubMedQA test result.

## HotpotQA Distractor

- Dataset configuration: `hotpot_qa`, `distractor`
- Split used: validation
- Diagnostic size: 200 examples
- Sampling seed: `2026`

| Model | Answer EM | Answer F1 | Evidence source F1 | Joint F1 proxy | Valid JSON | Mean verifier reward |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.50% | 15.25% | **64.85%** | 11.23% | 93.50% | 0.6756 |
| **RL-lite** | **27.00%** | **36.91%** | 55.61% | **23.79%** | **95.50%** | **0.8757** |
| Real-compact-flat | 15.00% | 22.01% | 53.94% | 14.05% | 89.50% | 0.6649 |

`Evidence source F1` measures selected source-document IDs. It is not the official HotpotQA supporting-sentence metric. `Joint F1 proxy` combines answer F1 and source-document F1 and must not be confused with the official HotpotQA joint score.

## SciFact

- Dataset source: official SciFact release archive
- Split used: `claims_dev`
- Diagnostic size: 200 examples
- Sampling seed: `2026`

| Model | Label accuracy | Macro-F1 | Evidence source F1* | Valid JSON | Mean verifier reward |
|---|---:|---:|---:|---:|---:|
| **Base** | **53.50%** | **41.46%** | 58.79% | 89.00% | 0.6045 |
| RL-lite | 40.00% | 26.21% | 45.07% | **92.00%** | **0.7870** |
| Real-compact-flat | 49.00% | 41.02% | **63.27%** | 79.00% | 0.6111 |

\* Evidence source F1 is calculated on SUPPORT or CONTRADICT examples at abstract-document level. It is not the official sentence-rationale SciFact score.

SciFact reveals an important limitation: RL-lite improves structured output and verifier reward but does not improve scientific claim-label accuracy over the base model. Real-compact-flat gives the strongest evidence-document selection in this run.

## Interpretation

The strongest public-diagnostic result is RL-lite on HotpotQA, where answer F1 increases from 15.25% for the base model to 36.91%. PubMedQA also favors RL-lite at 52.50% label accuracy on the 80-example diagnostic.

The adapters remain task-specialized. Results vary substantially across datasets, so no single number should be described as general QA accuracy.

## Reproduction

Relevant scripts and jobs:

- `scripts/build_pubmedqa_benchmark.py`
- `scripts/evaluate_pubmedqa_research.py`
- `scripts/build_hotpotqa_benchmark.py`
- `scripts/evaluate_hotpotqa_research.py`
- `scripts/build_scifact_benchmark.py`
- `scripts/evaluate_scifact_research.py`
- `jobs/27_pubmedqa_compare_best.sh`
- `jobs/28_hotpotqa_compare_public.sh`
- `jobs/29_scifact_compare_public.sh`

Compact machine-readable summaries are available in `docs/benchmark_results.json`.
