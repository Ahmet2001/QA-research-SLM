#!/bin/bash
#SBATCH --job-name=rr_find_sif
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_find_sif_%j.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs
find /arf/home/egitimg16u5 /arf/scratch/egitimg16u5 -maxdepth 4 -type f -name '*.sif' 2>/dev/null | head -50
