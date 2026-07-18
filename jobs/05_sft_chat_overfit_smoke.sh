#!/bin/bash
#SBATCH --job-name=rr_chat_smoke
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_chat_smoke_%j.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs
python3 scripts/train_sft_chat_qlora.py \
  --config configs/sft_stage1_broad.yaml \
  --output-dir outputs/research_reasoner_1p7b_sft_chat_smoke \
  --max-steps 20 \
  --max-seq-length 4096
python3 scripts/evaluate_research_reasoner_chat.py \
  --base-model Qwen/Qwen3-1.7B \
  --adapter outputs/research_reasoner_1p7b_sft_chat_smoke \
  --eval-file data/processed/eval_research_reasoner.jsonl \
  --out eval_outputs/research_reasoner_eval_chat_smoke.json \
  --compare-base \
  --max-new-tokens 512
