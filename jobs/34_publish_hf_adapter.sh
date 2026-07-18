#!/bin/bash
#SBATCH --job-name=rr_hf_adapter
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_hf_adapter_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs release/hf_adapter_stage

python3 - <<'PY'
from pathlib import Path
import json, shutil
from huggingface_hub import HfApi, hf_hub_download

repo_id = "Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL"
root = Path.cwd()
src = root / "outputs/research_reasoner_1p7b_v2_rl_lite"
release = root / "release/hf_adapter"
stage = root / "release/hf_adapter_stage"

if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)

required = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]
for name in required:
    path = src / name
    if not path.exists():
        raise FileNotFoundError(path)
    shutil.copy2(path, stage / name)

optional = ["merges.txt", "vocab.json"]
for name in optional:
    path = src / name
    if path.exists():
        shutil.copy2(path, stage / name)

for name in ["README.md", "research_output_schema.json", "release_metadata.json"]:
    shutil.copy2(release / name, stage / name)

shutil.copy2(src / "training_config_snapshot.json", stage / "training_config.json")
shutil.copy2(root / "docs/benchmark_results.json", stage / "benchmark_results.json")

license_path = Path(hf_hub_download("Qwen/Qwen3-1.7B", "LICENSE"))
shutil.copy2(license_path, stage / "LICENSE")

for name in ["adapter_config.json", "training_config.json", "benchmark_results.json", "research_output_schema.json", "release_metadata.json"]:
    json.loads((stage / name).read_text(encoding="utf-8"))

api = HfApi()
who = api.whoami()
orgs = {o.get("name") for o in who.get("orgs", [])}
if "Ethosoft" not in orgs:
    raise PermissionError("Authenticated user is not a member of Ethosoft")

api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
commit = api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(stage),
    commit_message="Publish ResearchReasoning JSON RL adapter",
)
info = api.model_info(repo_id)
print(json.dumps({
    "ok": True,
    "repo_id": repo_id,
    "url": f"https://huggingface.co/{repo_id}",
    "sha": info.sha,
    "commit_url": getattr(commit, "commit_url", None),
    "files": sorted(p.name for p in stage.iterdir()),
}, ensure_ascii=False, indent=2))
PY

echo HF_ADAPTER_PUBLISH_OK
