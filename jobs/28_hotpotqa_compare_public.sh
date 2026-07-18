#!/bin/bash
#SBATCH --job-name=rr_hotpotqa200
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_hotpotqa200_%j.out
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --exclude=kolyoz14

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs data/domain_benchmarks eval_outputs/public_benchmarks /arf/scratch/egitimg16u5/hf_cache
export HF_HOME=/arf/scratch/egitimg16u5/hf_cache
export TOKENIZERS_PARALLELISM=false

python3 scripts/build_hotpotqa_benchmark.py \
  --out data/domain_benchmarks/hotpotqa_distractor_flat_n300.jsonl \
  --limit 300 \
  --seed 2026

python3 scripts/evaluate_hotpotqa_research.py \
  --base-model Qwen/Qwen3-1.7B \
  --eval-file data/domain_benchmarks/hotpotqa_distractor_flat_n300.jsonl \
  --out eval_outputs/public_benchmarks/hotpotqa_compare_best_n200.json \
  --compare-base \
  --adapter rl_lite=outputs/research_reasoner_1p7b_v2_rl_lite \
  --adapter real_compact_flat=outputs/research_reasoner_1p7b_real_compact_flat \
  --max-input-tokens 8192 \
  --max-new-tokens 512 \
  --limit 200

echo HOTPOTQA_PUBLIC_COMPARE_OK
