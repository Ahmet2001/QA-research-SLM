#!/bin/bash
#SBATCH --job-name=rr_benchmark
#SBATCH --output=/arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0/logs/rr_benchmark_%j.out
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --dependency=afterok:1317549

set -euo pipefail
cd /arf/home/egitimg16u5/chatgpt_workspaces/research_reasoner_v0
mkdir -p logs eval_outputs/benchmarks
BASE=Qwen/Qwen3-1.7B

# Benchmark set A: v2 schema/hard eval set.
python3 scripts/evaluate_research_reasoner_flat_safe.py --base-model $BASE --adapter outputs/research_reasoner_1p7b_real_compact_flat --eval-file data/processed/eval_v2_flat.jsonl --out eval_outputs/benchmarks/v1_on_eval_v2.json --compare-base --max-new-tokens 1024 --limit 40
python3 scripts/evaluate_research_reasoner_flat_safe.py --base-model $BASE --adapter outputs/research_reasoner_1p7b_v2_schema_hard --eval-file data/processed/eval_v2_flat.jsonl --out eval_outputs/benchmarks/v2_on_eval_v2.json --compare-base --max-new-tokens 1024 --limit 40
python3 scripts/evaluate_research_reasoner_flat_safe.py --base-model $BASE --adapter outputs/research_reasoner_1p7b_v2_rl_lite --eval-file data/processed/eval_v2_flat.jsonl --out eval_outputs/benchmarks/rl_lite_on_eval_v2.json --compare-base --max-new-tokens 1024 --limit 40

# Benchmark set B: original compact real eval set.
python3 scripts/evaluate_research_reasoner_flat_safe.py --base-model $BASE --adapter outputs/research_reasoner_1p7b_real_compact_flat --eval-file data/processed/eval_real_compact_flat.jsonl --out eval_outputs/benchmarks/v1_on_eval_real.json --compare-base --max-new-tokens 1024 --limit 40
python3 scripts/evaluate_research_reasoner_flat_safe.py --base-model $BASE --adapter outputs/research_reasoner_1p7b_v2_schema_hard --eval-file data/processed/eval_real_compact_flat.jsonl --out eval_outputs/benchmarks/v2_on_eval_real.json --compare-base --max-new-tokens 1024 --limit 40
python3 scripts/evaluate_research_reasoner_flat_safe.py --base-model $BASE --adapter outputs/research_reasoner_1p7b_v2_rl_lite --eval-file data/processed/eval_real_compact_flat.jsonl --out eval_outputs/benchmarks/rl_lite_on_eval_real.json --compare-base --max-new-tokens 1024 --limit 40

python3 scripts/summarize_benchmarks.py \
  --items \
    v2bench_v1=eval_outputs/benchmarks/v1_on_eval_v2.json \
    v2bench_v2=eval_outputs/benchmarks/v2_on_eval_v2.json \
    v2bench_rl_lite=eval_outputs/benchmarks/rl_lite_on_eval_v2.json \
    realbench_v1=eval_outputs/benchmarks/v1_on_eval_real.json \
    realbench_v2=eval_outputs/benchmarks/v2_on_eval_real.json \
    realbench_rl_lite=eval_outputs/benchmarks/rl_lite_on_eval_real.json \
  --out eval_outputs/benchmarks/research_reasoner_benchmark_summary.json

echo BENCHMARK_SUITE_OK
