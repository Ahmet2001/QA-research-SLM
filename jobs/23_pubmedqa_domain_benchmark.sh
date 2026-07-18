#!/bin/bash
#SBATCH --job-name=rr_pubmedqa
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_pubmedqa_%j.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=kolyoz14

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs data/domain_benchmarks eval_outputs/domain_benchmarks

python3 scripts/build_pubmedqa_benchmark.py \
  --out data/domain_benchmarks/pubmedqa_labeled_flat.jsonl \
  --limit 120

python3 scripts/evaluate_pubmedqa_research.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_unified_v0 \
  --eval-file data/domain_benchmarks/pubmedqa_labeled_flat.jsonl \
  --out eval_outputs/domain_benchmarks/pubmedqa_unified_v0_eval.json \
  --compare-base \
  --max-new-tokens 768 \
  --limit 80

echo PUBMEDQA_DOMAIN_BENCHMARK_OK
