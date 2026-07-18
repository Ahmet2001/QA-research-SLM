#!/bin/bash
#SBATCH --job-name=rr_find_sif2
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_find_sif2_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs

echo 'SEARCH_ROOTS'
echo '/arf/home/egitimg16u5'
echo '/arf/scratch/egitimg16u5'
echo '/arf/home/egitimg16u5/chatgpt_workspaces'

echo 'SIF_RESULTS'
find /arf/home/egitimg16u5 /arf/scratch/egitimg16u5 /arf/home/egitimg16u5/chatgpt_workspaces \
  -type f \( -name '*.sif' -o -name '*.simg' \) 2>/dev/null | head -100
