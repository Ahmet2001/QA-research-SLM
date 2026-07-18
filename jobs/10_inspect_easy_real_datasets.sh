#!/bin/bash
#SBATCH --job-name=rr_easy_data
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_easy_data_%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs
python3 scripts/inspect_easy_real_datasets.py
