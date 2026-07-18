#!/bin/bash
# Shared helper for running ResearchReasoner commands inside a SIF container.
# Usage from a job script:
#   export RR_SIF=/absolute/path/to/container.sif
#   bash jobs/run_with_sif.sh "python3 scripts/train_sft_qlora.py --config configs/sft_stage1_broad.yaml"

set -euo pipefail

cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0

if [ -z "${RR_SIF:-}" ]; then
  echo "RR_SIF is not set. Set it to your .sif image path."
  exit 2
fi

if [ ! -f "$RR_SIF" ]; then
  echo "SIF not found: $RR_SIF"
  exit 3
fi

RUN_CMD="$1"

if command -v apptainer >/dev/null 2>&1; then
  apptainer exec --nv \
    --bind /arf:/arf \
    --bind /tmp:/tmp \
    "$RR_SIF" \
    bash -lc "cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0 && $RUN_CMD"
elif command -v singularity >/dev/null 2>&1; then
  singularity exec --nv \
    --bind /arf:/arf \
    --bind /tmp:/tmp \
    "$RR_SIF" \
    bash -lc "cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0 && $RUN_CMD"
else
  echo "Neither apptainer nor singularity found on this node."
  exit 4
fi
