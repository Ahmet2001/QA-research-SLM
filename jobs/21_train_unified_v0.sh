#!/bin/bash
#SBATCH --job-name=rr_unified
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_unified_%j.out
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=kolyoz14

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs eval_outputs/benchmarks data/processed

python3 scripts/build_unified_flat.py --out data/processed/sft_unified_flat.jsonl

python3 scripts/train_sft_chat_masked_flat_continue.py \
  --config configs/sft_unified_v0.yaml \
  --output-dir outputs/research_reasoner_1p7b_unified_v0 \
  --max-seq-length 4096

# Evaluate unified adapter on both benchmark families.
BASE=Qwen/Qwen3-1.7B
python3 scripts/evaluate_research_reasoner_flat_safe.py \
  --base-model $BASE \
  --adapter outputs/research_reasoner_1p7b_unified_v0 \
  --eval-file data/processed/eval_v2_flat.jsonl \
  --out eval_outputs/benchmarks/unified_v0_on_eval_v2.json \
  --compare-base \
  --max-new-tokens 1024 \
  --limit 40

python3 scripts/evaluate_research_reasoner_flat_safe.py \
  --base-model $BASE \
  --adapter outputs/research_reasoner_1p7b_unified_v0 \
  --eval-file data/processed/eval_real_compact_flat.jsonl \
  --out eval_outputs/benchmarks/unified_v0_on_eval_real.json \
  --compare-base \
  --max-new-tokens 1024 \
  --limit 40

echo UNIFIED_TRAIN_AND_EVAL_OK
