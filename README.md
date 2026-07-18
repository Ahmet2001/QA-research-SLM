# QA-Research-SLM

**QA-Research-SLM** is an experimental open-source training workspace for small language models that perform evidence-grounded research reasoning.

The project focuses on structured planning, evidence selection, claim verification, conflict detection, uncertainty calibration, and cited synthesis. It is designed for local or HPC training with compact base models such as Qwen3-1.7B and other models in the 1B–3B range.

> Status: research prototype. The repository contains training, evaluation, reward, data-building, and SLURM utilities. Published checkpoints will be released separately on Hugging Face.

## Core idea

Instead of producing only free-form answers, the model learns a structured research state:

```json
{
  "task_type": "comparative_research",
  "research_plan": ["identify claims", "collect evidence", "verify conflicts"],
  "evidence_needed": ["primary sources", "independent confirmation"],
  "selected_sources": ["src_001"],
  "claims": [
    {
      "claim": "Example claim",
      "status": "supported",
      "confidence": 0.84,
      "evidence": ["src_001#span_001"],
      "importance": "high"
    }
  ],
  "conflicts": [],
  "uncertainties": [],
  "answer": "Evidence-grounded final answer"
}
```

## Training pipeline

1. **Broad SFT** for planning, evidence extraction, document QA, verification, and synthesis.
2. **Schema-hard SFT** for strict structured output and longer research traces.
3. **Verifiable RL / rejection sampling** using deterministic research rewards.
4. **Evaluation** on internal structured tasks and public QA/research benchmarks.

The current implementation uses Transformers, PEFT, bitsandbytes, and QLoRA. TRL is optional for some experimental RL scripts.

## Repository layout

```text
configs/       Training and evaluation configurations
schemas/       JSON schemas for research traces and RL prompts
scripts/       Dataset builders, trainers, evaluators, and rewards
jobs/          Example SLURM jobs for TRUBA/HPC environments
data/seed/     Small seed examples when available
requirements.txt
```

Generated datasets, logs, checkpoints, caches, and evaluation artifacts are intentionally excluded from Git.

## Installation

```bash
git clone https://github.com/Ahmet2001/QA-research-SLM.git
cd QA-research-SLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable environment is recommended for QLoRA training.

## Build and validate data

```bash
python3 scripts/build_research_sft.py \
  --input data/seed/research_seed.jsonl \
  --output data/processed/research_sft.jsonl

python3 scripts/validate_jsonl.py \
  --input data/processed/research_sft.jsonl
```

Available builder scripts cover compact research QA, SQuAD-derived tasks, SciFact, PubMedQA, HotpotQA, and QASPER experiments. Dataset licenses and terms must be reviewed before redistributing generated data.

## QLoRA SFT

Example chat-template training:

```bash
python3 scripts/train_sft_chat_qlora.py \
  --base-model Qwen/Qwen3-1.7B \
  --train-file data/processed/research_sft.jsonl \
  --output-dir outputs/qa-research-slm-sft
```

Configuration-driven alternatives are available under `configs/` and `jobs/`.

## Evaluation

```bash
python3 scripts/evaluate_research_reasoner_chat_v2.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/qa-research-slm-sft \
  --eval-file data/processed/eval_research_reasoner.jsonl \
  --out eval_outputs/eval.json \
  --compare-base \
  --max-new-tokens 1536
```

The deterministic reward module checks:

- valid JSON and required fields
- valid evidence identifiers
- citation precision and supported-claim ratio
- avoidance of known bad sources
- conflict detection
- uncertainty calibration
- output efficiency

## SLURM / TRUBA

The `jobs/` directory contains development and experiment scripts used on TRUBA. Paths, partitions, account names, and resource requests may need editing for another cluster.

```bash
sbatch jobs/05_sft_chat_overfit_smoke.sh
```

Some files are retained as experiment history rather than polished one-command recipes.

## Public benchmark diagnostics

The current public-benchmark results are **diagnostic runs rather than official leaderboard submissions**. Full settings, limitations, and machine-readable summaries are available in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

| Benchmark | Best adapter result | Base result | Diagnostic size |
|---|---:|---:|---:|
| PubMedQA PQA-L | RL-lite: **52.50% label accuracy** | 1.25% | 80 |
| HotpotQA Distractor | RL-lite: **27.00 EM / 36.91 F1** | 0.50 EM / 15.25 F1 | 200 |
| SciFact claims-dev | Real-compact-flat: **63.27% evidence-source F1**; base retained the best label accuracy | 53.50% label accuracy | 200 |

These results show task specialization rather than uniform general-QA improvement. See the benchmark report before quoting any score.

## Current limitations

- The project is experimental and not production-ready.
- Small smoke-test datasets are insufficient for generalization.
- Structured generation can still fail through truncation or malformed fields.
- Reward scores measure format and evidence behavior, not complete factual correctness.
- Public benchmark results should be independently reproduced before comparison.

## Roadmap

- publish reproducible SFT datasets or generation recipes
- release adapters and model cards on Hugging Face
- add unit tests and continuous integration
- standardize benchmark reports
- improve constrained JSON generation
- expand multilingual research reasoning

## License

Code is released under the MIT License. External datasets and base models remain subject to their original licenses.

## Citation

```bibtex
@software{qa_research_slm_2026,
  title  = {QA-Research-SLM},
  author = {Ahmet2001 and contributors},
  year   = {2026},
  url    = {https://github.com/Ahmet2001/QA-research-SLM}
}
```

Contributions, reproducibility reports, and issue reports are welcome.
