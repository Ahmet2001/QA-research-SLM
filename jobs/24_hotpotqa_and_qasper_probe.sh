#!/bin/bash
#SBATCH --job-name=rr_hotpot_qasper
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_hotpot_qasper_%j.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=kolyoz14

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs data/domain_benchmarks eval_outputs/domain_benchmarks

python3 scripts/probe_qasper_hf_files.py > eval_outputs/domain_benchmarks/qasper_hf_files_probe.json || true
cat eval_outputs/domain_benchmarks/qasper_hf_files_probe.json

python3 scripts/build_hotpotqa_benchmark.py \
  --out data/domain_benchmarks/hotpotqa_distractor_flat.jsonl \
  --limit 80

python3 scripts/evaluate_open_answer_research.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_unified_v0 \
  --eval-file data/domain_benchmarks/hotpotqa_distractor_flat.jsonl \
  --out eval_outputs/domain_benchmarks/hotpotqa_unified_v0_eval.json \
  --compare-base \
  --max-new-tokens 768 \
  --limit 40

echo HOTPOTQA_AND_QASPER_PROBE_OK
