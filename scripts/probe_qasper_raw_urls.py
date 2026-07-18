#!/usr/bin/env python3
from __future__ import annotations
import json, re, urllib.request, urllib.error
from pathlib import Path


def try_url(url: str):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read(2000)
            return {'ok': True, 'status': getattr(r, 'status', None), 'content_type': r.headers.get('content-type'), 'head': data[:200].decode('utf-8', 'replace')}
    except Exception as e:
        return {'ok': False, 'error': repr(e)}


def main():
    from huggingface_hub import hf_hub_download
    out = {'ok': False}
    try:
        p = hf_hub_download(repo_id='allenai/qasper', repo_type='dataset', filename='qasper.py')
        text = Path(p).read_text(encoding='utf-8')
        urls = sorted(set(re.findall(r'https?://[^\"\'\s)]+', text)))
        # also try likely historical URL patterns from the script name/project.
        candidates = urls + [
            'https://allennlp.s3.amazonaws.com/datasets/qasper/qasper-train-v0.3.json',
            'https://allennlp.s3.amazonaws.com/datasets/qasper/qasper-dev-v0.3.json',
            'https://allennlp.s3.amazonaws.com/datasets/qasper/qasper-test-v0.3.json',
            'https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/qasper/qasper-train-v0.3.json',
            'https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/qasper/qasper-dev-v0.3.json',
            'https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/qasper/qasper-test-v0.3.json',
            'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/qasper/qasper-train-v0.3.json',
            'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/qasper/qasper-dev-v0.3.json',
            'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/qasper/qasper-test-v0.3.json',
        ]
        results = []
        for u in sorted(set(candidates)):
            results.append({'url': u, **try_url(u)})
        out = {'ok': True, 'script_path': p, 'urls_from_script': urls, 'results': results, 'script_excerpt': text[:5000]}
    except Exception as e:
        out = {'ok': False, 'error': repr(e)}
    Path('eval_outputs/domain_benchmarks').mkdir(parents=True, exist_ok=True)
    Path('eval_outputs/domain_benchmarks/qasper_raw_url_probe.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
