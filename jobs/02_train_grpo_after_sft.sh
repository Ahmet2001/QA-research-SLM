#!/bin/bash
#SBATCH --job-name=rr_grpo
#SBATCH --output=research_reasoner_v0/logs/rr_grpo_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --dependency=afterok:1317226

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs

# This runs after the SFT job succeeds. If the installed TRL GRPO API differs,
# adapt scripts/train_grpo_research.py in one place.
python3 scripts/train_grpo_research.py --config configs/rl_mgpo_research.yaml
