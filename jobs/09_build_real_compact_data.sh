#!/bin/bash
#SBATCH --job-name=rr_real_data
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_real_data_%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs data/raw data/real data/processed

python3 scripts/build_scifact_real_data.py \
  --raw-dir data/raw/scifact \
  --out data/real/scifact_research_traces.jsonl \
  --limit-train 400 \
  --limit-dev 80

python3 scripts/build_research_sft_compact.py \
  --inputs data/seed/research_seed.jsonl data/real/scifact_research_traces.jsonl \
  --out-stage1 data/processed/sft_stage1_real_compact.jsonl \
  --out-stage2 data/processed/sft_stage2_real_compact_hard.jsonl \
  --out-eval data/processed/eval_real_compact.jsonl \
  --eval-count 40

python3 scripts/validate_jsonl.py \
  data/processed/sft_stage1_real_compact.jsonl \
  data/processed/sft_stage2_real_compact_hard.jsonl \
  data/processed/eval_real_compact.jsonl

python3 scripts/make_rl_prompts.py \
  --sft data/processed/sft_stage1_real_compact.jsonl \
  --out data/processed/rl_prompts_real_compact.jsonl

echo REAL_DATA_BUILD_OK
