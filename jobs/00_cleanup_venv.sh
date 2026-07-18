#!/bin/bash
#SBATCH --job-name=rr_clean
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_clean_%j.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
rm -rf .venv
rm -rf __pycache__ scripts/__pycache__
echo CLEAN_OK
