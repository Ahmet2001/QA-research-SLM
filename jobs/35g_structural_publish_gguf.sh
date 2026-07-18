#!/bin/bash
#SBATCH --job-name=rr_gguf_upload
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_gguf_upload_%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
WORK=/arf/scratch/egitimg16u5/research_reasoner_gguf
OUT=$WORK/gguf
DOCS=$WORK/hf_gguf_docs
REPO=Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL-GGUF

cd "$PROJECT"
mkdir -p logs
rm -rf "$DOCS"
mkdir -p "$DOCS"

python3 - <<'PY'
from pathlib import Path
import json, struct

out = Path('/arf/scratch/egitimg16u5/research_reasoner_gguf/gguf')
models = [
    ('Q8_0', out / 'Qwen3-1.7B-ResearchReasoning-JSON-RL-Q8_0.gguf'),
    ('Q5_K_M', out / 'Qwen3-1.7B-ResearchReasoning-JSON-RL-Q5_K_M.gguf'),
    ('Q4_K_M', out / 'Qwen3-1.7B-ResearchReasoning-JSON-RL-Q4_K_M.gguf'),
]
results = []
for quant, path in models:
    if not path.exists() or path.stat().st_size < 100_000_000:
        raise RuntimeError(f'Missing or too-small GGUF: {path}')
    with path.open('rb') as f:
        magic = f.read(4)
        version_raw = f.read(4)
        tensor_raw = f.read(8)
        metadata_raw = f.read(8)
    if len(version_raw) != 4 or len(tensor_raw) != 8 or len(metadata_raw) != 8:
        raise RuntimeError(f'Truncated GGUF header: {path}')
    version = struct.unpack('<I', version_raw)[0]
    tensor_count = struct.unpack('<Q', tensor_raw)[0]
    metadata_count = struct.unpack('<Q', metadata_raw)[0]
    valid = magic == b'GGUF' and version in {2, 3} and tensor_count > 0 and metadata_count > 0
    result = {
        'quantization': quant,
        'file': path.name,
        'size_bytes': path.stat().st_size,
        'magic': magic.decode('ascii', errors='replace'),
        'gguf_version': version,
        'tensor_count': tensor_count,
        'metadata_kv_count': metadata_count,
        'structural_header_ok': valid,
    }
    results.append(result)
    print(json.dumps(result), flush=True)
    if not valid:
        raise RuntimeError(f'Invalid GGUF header: {path}')

payload = {
    'validation_type': 'GGUF binary header and structural-count validation',
    'runtime_note': 'A CPU-only llama.cpp generation test on TRUBA exceeded a five-minute startup timeout for Q8_0. Quant-specific runtime and JSON benchmarks are therefore not claimed.',
    'results': results,
}
path = Path('/arf/scratch/egitimg16u5/research_reasoner_gguf/hf_gguf_docs/structural_validation.json')
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('GGUF_STRUCTURAL_VALIDATION_OK', path, flush=True)
PY

python3 - <<'PY'
from pathlib import Path
import json, shutil
from huggingface_hub import HfApi, hf_hub_download

project = Path('/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0')
work = Path('/arf/scratch/egitimg16u5/research_reasoner_gguf')
out = work / 'gguf'
docs = work / 'hf_gguf_docs'
repo_id = 'Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL-GGUF'

validation = json.loads((docs / 'structural_validation.json').read_text(encoding='utf-8'))
shutil.copy2(project / 'docs/benchmark_results.json', docs / 'benchmark_results.json')
shutil.copy2(project / 'release/hf_adapter/research_output_schema.json', docs / 'research_output_schema.json')
license_path = Path(hf_hub_download('Qwen/Qwen3-1.7B', 'LICENSE'))
shutil.copy2(license_path, docs / 'LICENSE')

template = (project / 'release/hf_gguf/README.template.md').read_text(encoding='utf-8')
template = template.replace('## JSON smoke tests', '## GGUF structural validation')
template = template.replace(
    'The same fixed schema-constrained prompt was generated once with each quantization using llama.cpp. This is only a load/generation/schema smoke test, not a replacement for the adapter benchmark suite.\n\n<!-- SMOKE_TABLE -->\n\nRaw results are available in `smoke_test_results.json`.',
    'Each file was checked for a valid GGUF binary header, supported GGUF version, nonzero tensor count, and nonzero metadata count. The files were produced successfully with the llama.cpp converter and quantizer. This is a structural file validation, not a quant-specific inference or JSON benchmark.\n\n<!-- SMOKE_TABLE -->\n\nRaw results are available in `structural_validation.json`.'
)
rows = [
    '| Quantization | Size | GGUF version | Tensors | Metadata entries | Structural check |',
    '|---|---:|---:|---:|---:|---:|',
]
for result in validation['results']:
    gib = result['size_bytes'] / (1024 ** 3)
    rows.append(
        f"| `{result['quantization']}` | {gib:.2f} GiB | {result['gguf_version']} | "
        f"{result['tensor_count']} | {result['metadata_kv_count']} | "
        f"{'Passed' if result['structural_header_ok'] else 'Failed'} |"
    )
readme = template.replace('<!-- SMOKE_TABLE -->', '\n'.join(rows))
readme += (
    '\n\n## Quant-specific evaluation status\n\n'
    'The GGUF files passed structural validation after conversion and quantization. A CPU-only llama.cpp generation attempt on the TRUBA node exceeded a five-minute startup timeout for the Q8_0 file, so this release does not claim completed quant-specific runtime, JSON-adherence, or public-benchmark results. The benchmark table above belongs to the PEFT adapter before GGUF conversion. Re-run the evaluation suite in your target llama.cpp, LM Studio, or Ollama environment before reliability-sensitive deployment.\n'
)
(docs / 'README.md').write_text(readme, encoding='utf-8')

api = HfApi()
who = api.whoami()
if 'Ethosoft' not in {org.get('name') for org in who.get('orgs', [])}:
    raise PermissionError('Authenticated user is not a member of Ethosoft')
api.create_repo(repo_id=repo_id, repo_type='model', private=False, exist_ok=True)

api.upload_folder(
    repo_id=repo_id,
    repo_type='model',
    folder_path=str(docs),
    commit_message='Add GGUF model card and validation metadata',
)

quant_files = [
    out / 'Qwen3-1.7B-ResearchReasoning-JSON-RL-Q8_0.gguf',
    out / 'Qwen3-1.7B-ResearchReasoning-JSON-RL-Q5_K_M.gguf',
    out / 'Qwen3-1.7B-ResearchReasoning-JSON-RL-Q4_K_M.gguf',
]
for path in quant_files:
    print('UPLOAD_GGUF', path.name, path.stat().st_size, flush=True)
    api.upload_file(
        repo_id=repo_id,
        repo_type='model',
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        commit_message=f'Upload {path.name}',
    )
    print('UPLOAD_GGUF_OK', path.name, flush=True)

info = api.model_info(repo_id)
print(json.dumps({
    'ok': True,
    'repo_id': repo_id,
    'url': f'https://huggingface.co/{repo_id}',
    'sha': info.sha,
    'validation': validation['results'],
    'uploaded_files': [path.name for path in quant_files],
}, ensure_ascii=False, indent=2), flush=True)
PY

echo HF_GGUF_PUBLISH_OK
