#!/bin/bash
#SBATCH --job-name=rr_rl_rollout
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_rl_rollout_%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs eval_outputs data/processed
python3 scripts/rl_rejection_sample.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_v2_schema_hard \
  --prompts data/processed/rl_prompts_v2.jsonl \
  --out-flat data/processed/rl_rejection_sft_v2_flat.jsonl \
  --out-report eval_outputs/rl_rejection_report_v2.jsonl \
  --limit 160 \
  --num-candidates 3 \
  --max-new-tokens 512 \
  --min-score 0.75
