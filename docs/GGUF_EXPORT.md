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
