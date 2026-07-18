#!/bin/bash
#SBATCH --job-name=rr_sft1
#SBATCH --output=research_reasoner_v0/logs/rr_sft1_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs

# Activate your environment here if needed, for example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rr_train

python3 scripts/train_sft_qlora.py --config configs/sft_stage1_broad.yaml
