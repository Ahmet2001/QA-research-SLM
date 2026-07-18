#!/bin/bash
#SBATCH --job-name=rr_sif_smoke
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_sif_smoke_%j.out
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs

: ${RR_SIF:=/arf/home/egitimg16u5/research_train.sif}
export RR_SIF

bash jobs/run_with_sif.sh "python3 - <<'PY'
import torch
import transformers
import datasets
import peft
import trl
print('SIF_IMPORT_OK')
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('trl', trl.__version__)
PY"
