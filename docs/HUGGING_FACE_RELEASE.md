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
