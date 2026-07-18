#!/bin/bash
#SBATCH --job-name=rr_sft_global
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_sft_global_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --dependency=afterok:1317287

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs

which python3
python3 --version
python3 scripts/train_sft_qlora.py --config configs/sft_stage1_broad.yaml
