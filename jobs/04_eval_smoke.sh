#!/bin/bash
#SBATCH --job-name=rr_eval_smoke
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_eval_smoke_%j.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs eval_outputs
python3 scripts/evaluate_research_reasoner.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_sft_stage1 \
  --eval-file data/processed/eval_research_reasoner.jsonl \
  --out eval_outputs/research_reasoner_eval_smoke.json \
  --compare-base \
  --max-new-tokens 512
