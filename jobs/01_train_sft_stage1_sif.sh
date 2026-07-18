#!/bin/bash
#SBATCH --job-name=rr_sft_sif
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_sft_sif_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs

# Put your container path here or export RR_SIF before submitting.
: ${RR_SIF:=/arf/home/egitimg16u5/research_train.sif}
export RR_SIF

bash jobs/run_with_sif.sh "python3 scripts/train_sft_qlora.py --config configs/sft_stage1_broad.yaml"
