#!/bin/bash
#SBATCH --job-name=rr_eval_full40
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_eval_full40_%j.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs eval_outputs
python3 scripts/evaluate_research_reasoner_flat_safe.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_real_compact_flat \
  --eval-file data/processed/eval_real_compact_flat.jsonl \
  --out eval_outputs/research_reasoner_real_compact_flat_eval_full40_safe.json \
  --compare-base \
  --max-new-tokens 1024 \
  --limit 40
