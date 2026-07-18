#!/bin/bash
#SBATCH --job-name=rr_env
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_env_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import torch
import transformers
import datasets
import peft
import trl
print('ENV_OK')
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('trl', trl.__version__)
PY
