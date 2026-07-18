#!/bin/bash
#SBATCH --job-name=rr_v2_sft
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_v2_sft_%j.out
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --dependency=afterok:1317430

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs eval_outputs

python3 scripts/train_sft_chat_masked_flat.py \
  --config configs/sft_v2_schema_hard.yaml \
  --output-dir outputs/research_reasoner_1p7b_v2_schema_hard \
  --max-seq-length 4096

python3 scripts/evaluate_research_reasoner_flat_safe.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_v2_schema_hard \
  --eval-file data/processed/eval_v2_flat.jsonl \
  --out eval_outputs/research_reasoner_v2_schema_hard_eval.json \
  --compare-base \
  --max-new-tokens 1024 \
  --limit 40
