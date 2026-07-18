#!/bin/bash
#SBATCH --job-name=rr_v2_data
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_v2_data_%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs data/real data/processed

python3 scripts/build_squad_hard_v2.py \
  --input data/real/squad_research_traces.jsonl \
  --out data/real/squad_hard_v2_traces.jsonl \
  --limit 320

python3 scripts/build_research_sft_v2.py \
  --inputs data/seed/research_seed.jsonl data/real/squad_research_traces.jsonl data/real/squad_hard_v2_traces.jsonl \
  --out-stage1 data/processed/sft_v2_stage1.jsonl \
  --out-stage2 data/processed/sft_v2_stage2_hard.jsonl \
  --out-eval data/processed/eval_v2.jsonl \
  --eval-count 80

python3 scripts/flatten_sft_jsonl.py --input data/processed/sft_v2_stage1.jsonl --output data/processed/sft_v2_stage1_flat.jsonl
python3 scripts/flatten_sft_jsonl.py --input data/processed/sft_v2_stage2_hard.jsonl --output data/processed/sft_v2_stage2_hard_flat.jsonl
python3 scripts/flatten_sft_jsonl.py --input data/processed/eval_v2.jsonl --output data/processed/eval_v2_flat.jsonl

python3 scripts/validate_jsonl.py data/processed/sft_v2_stage1.jsonl data/processed/sft_v2_stage2_hard.jsonl data/processed/eval_v2.jsonl
python3 scripts/make_rl_prompts.py --sft data/processed/sft_v2_stage1.jsonl --out data/processed/rl_prompts_v2.jsonl

echo V2_DATA_OK
