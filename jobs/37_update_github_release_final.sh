#!/bin/bash
#SBATCH --job-name=rr_git_release_final
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_git_release_final_%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs docs

python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
expected = {
    'Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL': 'f7df3ab0679fd2713e93d393d8d93430a535c1fd',
    'Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL-GGUF': 'd4471928db2fc9cd71d9a81943aae03f76303996',
}
for repo, published_sha in expected.items():
    info = api.model_info(repo)
    if not info.sha:
        raise RuntimeError(f'Missing release SHA for {repo}')
    print('HF_RELEASE_OK', repo, info.sha, 'initial_release_sha', published_sha)
PY

git fetch origin main
git checkout main
git pull --ff-only origin main

python3 - <<'PY'
from pathlib import Path

path = Path('README.md')
text = path.read_text(encoding='utf-8')
old_status = '> Status: research prototype. The repository contains training, evaluation, reward, data-building, and SLURM utilities. Published checkpoints will be released separately on Hugging Face.'
new_status = '> Status: research prototype. The RL-lite adapter and GGUF quantizations are published under the Ethosoft organization on Hugging Face.'
text = text.replace(old_status, new_status)

release_block = '''
## Published models

- **PEFT adapter:** [Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL](https://huggingface.co/Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL)
- **GGUF quantizations:** [Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL-GGUF](https://huggingface.co/Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL-GGUF)

The released model is based on `Qwen/Qwen3-1.7B` and is specialized for evidence-grounded research reasoning, source-aware QA, claim verification, and strict JSON schema generation. The project name **RL-lite** refers to verifier-guided rejection sampling followed by supervised fine-tuning; it is not a full online policy-gradient RL run.

The model is schema-sensitive. For best results, explicitly require valid JSON, list every required field, disable markdown, and restrict evidence entries to source IDs present in the prompt.
'''.strip()

if '## Published models' not in text:
    anchor = new_status + '\n'
    text = text.replace(anchor, anchor + '\n' + release_block + '\n')

inspiration = '''
## VibeThinker inspiration and attribution

QA-Research-SLM is **conceptually inspired by the VibeThinker line of work** on eliciting verifiable reasoning capabilities in compact language models. This project was independently trained for evidence-grounded research reasoning and strict JSON generation; it does **not** reuse VibeThinker weights, training data, or code.

Related work:

- Sen Xu et al., *Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B*, arXiv:2511.06221, 2025.
- Sen Xu et al., *VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models*, arXiv:2606.16140, 2026.
'''.strip()

if '## VibeThinker inspiration and attribution' not in text:
    text = text.replace('## Current limitations', inspiration + '\n\n## Current limitations')

text = text.replace(
    '- release adapters and model cards on Hugging Face',
    '- run full official benchmark evaluations and quant-specific GGUF diagnostics',
)
path.write_text(text, encoding='utf-8')
PY

cp release/hf_adapter/README.md docs/HUGGING_FACE_MODEL_CARD.md
cp /arf/scratch/egitimg16u5/research_reasoner_gguf/hf_gguf_docs/README.md docs/HUGGING_FACE_GGUF_CARD.md
cp /arf/scratch/egitimg16u5/research_reasoner_gguf/hf_gguf_docs/structural_validation.json docs/GGUF_STRUCTURAL_VALIDATION.json
cp release/hf_adapter/research_output_schema.json docs/research_output_schema.json

cat > docs/HUGGING_FACE_RELEASE.md <<'EOF'
# Hugging Face release

## Repositories

- Adapter: https://huggingface.co/Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL
- GGUF: https://huggingface.co/Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL-GGUF

## Adapter package

The adapter repository contains PEFT weights, tokenizer assets, the strict research-output JSON schema, training configuration, benchmark diagnostics, and the full model card.

The final stage is named **RL-lite** in this project. It uses verifier-guided rejection sampling followed by supervised fine-tuning and should not be represented as full online policy-gradient reinforcement learning.

## GGUF package

The GGUF repository contains Q8_0, Q5_K_M, and Q4_K_M quantizations generated after merging the adapter into Qwen3-1.7B. All three files passed binary structural validation:

- GGUF magic and version 3
- 310 tensors
- 29 metadata key-value entries

A CPU-only llama.cpp generation attempt on the TRUBA node exceeded a five-minute startup timeout for Q8_0. Therefore the release does not claim completed quant-specific runtime, strict-JSON, or public-benchmark results. Adapter benchmark numbers must not be presented as GGUF-specific results.

## Evaluation note

The published PubMedQA, HotpotQA, and SciFact numbers are fixed-seed diagnostics using the project's strict JSON research prompt. They are not official leaderboard submissions. See `BENCHMARK_RESULTS.md` before quoting them.
EOF

cat > docs/GGUF_EXPORT.md <<'EOF'
# GGUF export notes

The released GGUF files were built by:

1. Loading `Qwen/Qwen3-1.7B` in FP16.
2. Loading the `research_reasoner_1p7b_v2_rl_lite` PEFT adapter.
3. Merging it with `merge_and_unload(safe_merge=True)`.
4. Saving a merged Hugging Face model.
5. Converting the merged model with llama.cpp `convert_hf_to_gguf.py` using F16 output.
6. Quantizing with llama.cpp to Q8_0, Q5_K_M, and Q4_K_M.
7. Verifying each binary's GGUF header, version, tensor count, and metadata count before upload.

The exact final structural validation output is stored in `GGUF_STRUCTURAL_VALIDATION.json`.
EOF

python3 - <<'PY'
import json
for path in [
    'docs/benchmark_results.json',
    'docs/research_output_schema.json',
    'docs/GGUF_STRUCTURAL_VALIDATION.json',
]:
    with open(path, encoding='utf-8') as f:
        json.load(f)
    print(path, 'OK')
PY

git diff --check
git add \
  README.md \
  docs/HUGGING_FACE_RELEASE.md \
  docs/HUGGING_FACE_MODEL_CARD.md \
  docs/HUGGING_FACE_GGUF_CARD.md \
  docs/GGUF_STRUCTURAL_VALIDATION.json \
  docs/GGUF_EXPORT.md \
  docs/research_output_schema.json \
  jobs/34_publish_hf_adapter.sh \
  jobs/35g_structural_publish_gguf.sh \
  jobs/37_update_github_release_final.sh

if git diff --cached --quiet; then
  echo GITHUB_RELEASE_NO_CHANGES
  exit 0
fi

git commit -m 'Document Hugging Face adapter and GGUF releases'
git push origin main

echo GITHUB_HF_RELEASE_PUSH_OK commit=$(git rev-parse HEAD)
