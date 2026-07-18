#!/bin/bash
#SBATCH --job-name=rr_eval_v2
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_eval_v2_%j.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs eval_outputs
python3 scripts/evaluate_research_reasoner_chat_v2.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_sft_chat_smoke \
  --eval-file data/processed/eval_research_reasoner.jsonl \
  --out eval_outputs/research_reasoner_eval_chat_v2.json \
  --compare-base \
  --max-new-tokens 1536
