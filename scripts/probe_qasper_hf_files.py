#!/usr/bin/env python3
from __future__ import annotations
import json


def main():
    out = {'ok': False}
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files('allenai/qasper', repo_type='dataset')
        out = {'ok': True, 'repo': 'allenai/qasper', 'files': files[:200], 'n_files': len(files)}
    except Exception as e:
        out = {'ok': False, 'error': repr(e)}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
