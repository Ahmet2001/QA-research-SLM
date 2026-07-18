#!/bin/bash
#SBATCH --job-name=rr_sft_smoke
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_sft_smoke_%j.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs outputs
python3 - <<'PY'
import json, pathlib
for p in ['data/processed/sft_stage1_broad.jsonl','data/processed/rl_prompts.jsonl']:
    path=pathlib.Path(p)
    print(p, 'exists=', path.exists(), 'bytes=', path.stat().st_size if path.exists() else 0)
    if path.exists():
        with path.open(encoding='utf-8') as f:
            first=json.loads(next(f))
        print('first_id=', first.get('id'), 'keys=', sorted(first.keys()))
PY
python3 scripts/train_sft_qlora.py --config configs/sft_stage1_broad.yaml
