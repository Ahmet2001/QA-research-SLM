#!/bin/bash
#SBATCH --job-name=rr_rl_lite2
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_rl_lite2_%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=kolyoz14

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs eval_outputs

python3 scripts/train_sft_chat_masked_flat_continue.py \
  --config configs/sft_rl_lite_v2.yaml \
  --output-dir outputs/research_reasoner_1p7b_v2_rl_lite \
  --max-seq-length 4096

python3 scripts/evaluate_research_reasoner_flat_safe.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_v2_rl_lite \
  --eval-file data/processed/eval_v2_flat.jsonl \
  --out eval_outputs/research_reasoner_v2_rl_lite_eval.json \
  --compare-base \
  --max-new-tokens 1024 \
  --limit 40

echo RL_LITE_TRAIN_AND_EVAL_OK
