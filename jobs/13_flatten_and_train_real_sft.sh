#!/bin/bash
#SBATCH --job-name=rr_flat_sft
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_flat_sft_%j.out
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs eval_outputs

python3 scripts/flatten_sft_jsonl.py \
  --input data/processed/sft_stage1_real_compact.jsonl \
  --output data/processed/sft_stage1_real_compact_flat.jsonl
python3 scripts/flatten_sft_jsonl.py \
  --input data/processed/eval_real_compact.jsonl \
  --output data/processed/eval_real_compact_flat.jsonl

python3 scripts/train_sft_chat_masked_flat.py \
  --config configs/sft_real_compact_flat.yaml \
  --output-dir outputs/research_reasoner_1p7b_real_compact_flat \
  --max-seq-length 4096

python3 scripts/evaluate_research_reasoner_flat.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_real_compact_flat \
  --eval-file data/processed/eval_real_compact_flat.jsonl \
  --out eval_outputs/research_reasoner_real_compact_flat_eval.json \
  --compare-base \
  --max-new-tokens 1024 \
  --limit 8
