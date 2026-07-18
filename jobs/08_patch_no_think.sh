#!/bin/bash
#SBATCH --job-name=rr_patch_no_think
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_patch_no_think_%j.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
python3 - <<'PY'
from pathlib import Path
repls = [
    (Path('scripts/train_sft_chat_qlora.py'),
     "tokenizer.apply_chat_template(\n            prompt_messages,\n            tokenize=False,\n            add_generation_prompt=True,\n        )",
     "tokenizer.apply_chat_template(\n            prompt_messages,\n            tokenize=False,\n            add_generation_prompt=True,\n            enable_thinking=False,\n        )"),
    (Path('scripts/train_sft_chat_qlora.py'),
     "tokenizer.apply_chat_template(\n            messages,\n            tokenize=False,\n            add_generation_prompt=False,\n        )",
     "tokenizer.apply_chat_template(\n            messages,\n            tokenize=False,\n            add_generation_prompt=False,\n            enable_thinking=False,\n        )"),
    (Path('scripts/evaluate_research_reasoner_chat.py'),
     "tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)",
     "tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)"),
]
for path, old, new in repls:
    s = path.read_text(encoding='utf-8')
    if old not in s:
        print('pattern_missing', path, old[:40])
    s = s.replace(old, new)
    path.write_text(s, encoding='utf-8')
print('PATCH_NO_THINK_OK')
PY
