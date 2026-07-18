#!/bin/bash
#SBATCH --job-name=rr_real_sft
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_real_sft_%j.out
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --dependency=afterok:1317337

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs
python3 scripts/train_sft_chat_masked.py \
  --config configs/sft_real_compact.yaml \
  --output-dir outputs/research_reasoner_1p7b_real_compact \
  --max-seq-length 4096

python3 scripts/evaluate_research_reasoner_masked.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_real_compact \
  --eval-file data/processed/eval_real_compact.jsonl \
  --out eval_outputs/research_reasoner_real_compact_eval.json \
  --compare-base \
  --max-new-tokens 1024 \
  --limit 8
