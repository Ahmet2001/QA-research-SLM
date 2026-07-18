---
license: apache-2.0
base_model: Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL
library_name: gguf
pipeline_tag: text-generation
language:
- en
- tr
tags:
- qwen3
- gguf
- llama.cpp
- research-reasoning
- question-answering
- evidence-grounded
- structured-generation
- json
- quantized
---

# Qwen3-1.7B ResearchReasoning JSON RL — GGUF

GGUF quantizations of **Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL**, a Qwen3-1.7B fine-tune for evidence-grounded research reasoning, source-aware QA, claim verification, and strict JSON schema generation.

- Adapter repository: https://huggingface.co/Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL
- Base model: https://huggingface.co/Qwen/Qwen3-1.7B
- Source code: https://github.com/Ahmet2001/QA-research-SLM

The PEFT adapter was merged into the base model before GGUF conversion. The project calls the final training stage **RL-lite**: verifier-guided rejection sampling followed by supervised fine-tuning, not full online policy-gradient RL.

## Files

| Quantization | Suggested use |
|---|---|
| `Q8_0` | Highest fidelity among the uploaded quantizations; largest file |
| `Q5_K_M` | Recommended balance for schema-sensitive local use |
| `Q4_K_M` | Smaller and faster; validate JSON carefully |

## GGUF structural validation

Each file was checked for a valid GGUF binary header, supported GGUF version, nonzero tensor count, and nonzero metadata count. The files were produced successfully with the llama.cpp converter and quantizer. This is a structural file validation, not a quant-specific inference or JSON benchmark.

| Quantization | Size | GGUF version | Tensors | Metadata entries | Structural check |
|---|---:|---:|---:|---:|---:|
| `Q8_0` | 1.71 GiB | 3 | 310 | 29 | Passed |
| `Q5_K_M` | 1.17 GiB | 3 | 310 | 29 | Passed |
| `Q4_K_M` | 1.03 GiB | 3 | 310 | 29 | Passed |

Raw results are available in `structural_validation.json`.

## llama.cpp

```bash
llama-cli \
  -m Qwen3-1.7B-ResearchReasoning-JSON-RL-Q5_K_M.gguf \
  -cnv \
  -n 768
```

For programmatic use, supply a strong system prompt that requires valid JSON, lists all required fields, disables markdown, and restricts evidence entries to source IDs present in the prompt.

## Recommended output schema

```json
{
  "task_type": "document_qa",
  "research_plan": ["read the question", "inspect the sources", "answer with evidence"],
  "evidence_needed": ["supporting source text"],
  "selected_sources": ["src_001"],
  "claims": [
    {
      "claim": "The answer is supported by src_001.",
      "status": "supported",
      "confidence": 0.9,
      "evidence": ["src_001"],
      "importance": "high"
    }
  ],
  "conflicts": [],
  "uncertainties": [],
  "answer": "Evidence-grounded answer"
}
```

## Benchmark diagnostics

The full diagnostic results were produced with the PEFT adapter before GGUF conversion. They are included here for model context, but should not be treated as quant-specific measurements.

| Benchmark | Adapter result | Diagnostic size |
|---|---:|---:|
| PubMedQA PQA-L | 52.50% label accuracy | 80 |
| HotpotQA Distractor | 27.00 EM / 36.91 F1 | 200 |
| SciFact claims-dev | 40.00% label accuracy / 26.21 macro-F1 | 200 |

These are fixed-seed custom diagnostics using a strict JSON research prompt, not official leaderboard submissions. See `benchmark_results.json` and the adapter model card for full settings and base-model comparisons.

## Quantization caution

This model is optimized for strict structured generation. Quantization can preserve semantic answers while still damaging commas, quotes, closing braces, field names, or source-ID arrays. Validate every response. For reliability-sensitive applications, prefer `Q5_K_M` or `Q8_0` and use constrained decoding or application-level JSON repair.

## VibeThinker inspiration and attribution

This project is conceptually inspired by the VibeThinker line of work on verifiable reasoning in compact models, but it was independently trained for research-oriented structured generation. It does not reuse VibeThinker weights, data, or code.

- Sen Xu et al., *Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B*, arXiv:2511.06221, 2025. https://arxiv.org/abs/2511.06221
- Sen Xu et al., *VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models*, arXiv:2606.16140, 2026. https://arxiv.org/abs/2606.16140

## License

Apache-2.0, following the Qwen3 base model license. Review the original dataset licenses before redistributing derived training data.


## Quant-specific evaluation status

The GGUF files passed structural validation after conversion and quantization. A CPU-only llama.cpp generation attempt on the TRUBA node exceeded a five-minute startup timeout for the Q8_0 file, so this release does not claim completed quant-specific runtime, JSON-adherence, or public-benchmark results. The benchmark table above belongs to the PEFT adapter before GGUF conversion. Re-run the evaluation suite in your target llama.cpp, LM Studio, or Ollama environment before reliability-sensitive deployment.
