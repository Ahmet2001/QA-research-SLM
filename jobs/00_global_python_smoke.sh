#!/bin/bash
#SBATCH --job-name=rr_py_smoke
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_py_smoke_%j.out
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs

which python3
python3 --version
python3 - <<'PY'
mods = ['torch', 'transformers', 'datasets', 'peft', 'trl', 'bitsandbytes', 'yaml']
for m in mods:
    try:
        mod = __import__(m)
        print(m, 'OK', getattr(mod, '__version__', 'no_version'))
    except Exception as e:
        print(m, 'FAIL', repr(e))
PY
python3 scripts/research_rewards.py
