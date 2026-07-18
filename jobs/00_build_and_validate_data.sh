#!/bin/bash
#SBATCH --job-name=rr_data_v0
#SBATCH --output=research_reasoner_v0/logs/rr_data_v0_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs data/processed

python3 scripts/build_research_sft.py --seed data/seed/research_seed.jsonl --out-stage1 data/processed/sft_stage1_broad.jsonl --out-stage2 data/processed/sft_stage2_hard_long.jsonl --out-eval data/processed/eval_research_reasoner.jsonl
python3 scripts/validate_jsonl.py data/processed/sft_stage1_broad.jsonl data/processed/sft_stage2_hard_long.jsonl data/processed/eval_research_reasoner.jsonl
python3 scripts/make_rl_prompts.py --sft data/processed/sft_stage1_broad.jsonl --out data/processed/rl_prompts.jsonl
python3 scripts/research_rewards.py

echo DATA_PIPELINE_OK
