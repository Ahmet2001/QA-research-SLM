---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
language:
- en
- tr
tags:
- qwen3
- peft
- lora
- research-reasoning
- question-answering
- evidence-grounded
- structured-generation
- json
- rejection-sampling
- reinforcement-learning
---

# Qwen3-1.7B ResearchReasoning JSON RL

**Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL** is a PEFT/LoRA adapter for `Qwen/Qwen3-1.7B`, fine-tuned for evidence-grounded research reasoning, source-aware question answering, claim verification, uncertainty handling, and strict JSON schema generation.

The project calls the final stage **RL-lite**. More precisely, it is a verifier-guided rejection-sampling and supervised fine-tuning stage initialized from a schema-focused adapter. It is not a full online policy-gradient RL run.

- Base model: `Qwen/Qwen3-1.7B`
- Adapter type: LoRA / PEFT
- Languages represented in the project: English and Turkish
- Fine-tuning context length: 4,096 tokens
- Primary output mode: non-thinking, minified structured JSON
- Source code and reproducibility files: https://github.com/Ahmet2001/QA-research-SLM
- GGUF release: `Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL-GGUF`

## Intended behavior

The model is trained to return a structured research state rather than only a free-form answer:

```json
{
  "task_type": "multi_hop_document_qa",
  "research_plan": ["identify relevant sources", "connect evidence", "answer concisely"],
  "evidence_needed": ["supporting facts"],
  "selected_sources": ["src_001", "src_002"],
  "claims": [
    {
      "claim": "The answer is supported by the supplied documents.",
      "status": "supported",
      "confidence": 0.86,
      "evidence": ["src_001", "src_002"],
      "importance": "high"
    }
  ],
  "conflicts": [],
  "uncertainties": [],
  "answer": "Concise evidence-grounded answer"
}
```

The model is especially sensitive to the requested schema and performs best when the system prompt explicitly requires valid JSON, names all required fields, and restricts evidence values to supplied source IDs.

## Usage with Transformers and PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL"
base_id = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

system = (
    "You are ResearchReasoner. Output ONLY valid minified JSON. No markdown. "
    "Required fields: task_type, research_plan, evidence_needed, selected_sources, "
    "claims, conflicts, uncertainties, answer. Use only supplied source IDs."
)
user = """Question: Which source supports the claim?

SOURCES:
[src_001] The official report states that the system uses a 32K context window.
[src_002] An unverified repost claims 128K without an official reference."""

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=768,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

completion = tokenizer.decode(
    output[0, inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(completion)
```

For reliability-sensitive JSON workflows, validate the generated object against `research_output_schema.json` and retry or repair invalid generations in the application layer.

## Training details

The released adapter was initialized from the project's schema-hard SFT adapter and trained for one additional epoch on verifier-filtered rejection-sampling data.

| Setting | Value |
|---|---:|
| Base model | Qwen/Qwen3-1.7B |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o projections and gate/up/down projections |
| Learning rate | 5e-6 |
| Scheduler | cosine |
| Optimizer | paged AdamW 8-bit |
| Effective batch size | 4 |
| Fine-tuning epochs | 1 |
| Maximum sequence length | 4,096 |

The base model advertises a longer context window, but this adapter was fine-tuned and evaluated primarily at 4,096 tokens. Long-context quality beyond that range is not established by this release.

## Benchmark diagnostics

These are **fixed-seed diagnostic evaluations**, not official leaderboard submissions. The prompts require the project's strict research JSON schema, so results are not directly comparable with standard zero-shot leaderboard numbers.

### PubMedQA PQA-L — 80-example diagnostic

The 1,000-example labeled train split was shuffled with seed 2026; the first 80 generated examples were evaluated.

| Model | Label accuracy | Label attempt rate | Mean verifier reward |
|---|---:|---:|---:|
| Qwen3-1.7B base | 1.25% | 16.25% | -0.1900 |
| **ResearchReasoning JSON RL** | **52.50%** | **85.00%** | **0.7750** |

### HotpotQA Distractor validation — 200-example diagnostic

The validation split was shuffled with seed 2026. Evidence is scored at source-document level, not with the official supporting-sentence metric.

| Model | Answer EM | Answer F1 | Evidence-source F1 | JSON valid | Verifier reward |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B base | 0.50% | 15.25% | 64.85% | 93.50% | 0.6756 |
| **ResearchReasoning JSON RL** | **27.00%** | **36.91%** | 55.61% | **95.50%** | **0.8757** |

### SciFact claims-dev — 200-example diagnostic

Evidence is measured at abstract/source-document level rather than official sentence-rationale level.

| Model | Label accuracy | Macro-F1 | Evidence-source F1* | JSON valid | Verifier reward |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B base | **53.50%** | **41.46%** | 58.79% | 89.00% | 0.6045 |
| **ResearchReasoning JSON RL** | 40.00% | 26.21% | 45.07% | **92.00%** | **0.7870** |

\* Evidence-source F1 for examples labeled SUPPORT or CONTRADICT.

The SciFact result shows an important limitation: the adapter improves schema compliance and verifier reward but does not consistently improve scientific claim classification over the base model.

Full diagnostic reports and evaluator scripts are available in the GitHub repository.

## Limitations

- This is an experimental research adapter, not a production fact-checking system.
- Generated citations are source IDs from the prompt; the model does not browse or verify external URLs by itself.
- High verifier reward does not guarantee factual correctness.
- The model can over-specialize to the requested schema and may underperform the base model on some out-of-domain classifications.
- PubMedQA, HotpotQA, and SciFact numbers above are custom fixed-sample diagnostics rather than official leaderboard scores.
- JSON can still be malformed or truncated, especially with insufficient generation length or aggressive quantization.
- Lower-bit GGUF quantization may reduce schema adherence. Prefer Q5_K_M or Q8_0 for reliability-sensitive use.

## VibeThinker inspiration and attribution

This project is **conceptually inspired by the VibeThinker line of work** on eliciting verifiable reasoning in compact language models. The released adapter was developed independently for evidence-grounded research reasoning and strict JSON generation. It does **not** reuse VibeThinker weights, training data, or code.

Related work:

- Sen Xu et al., *Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B*, arXiv:2511.06221, 2025. https://arxiv.org/abs/2511.06221
- Sen Xu et al., *VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models*, arXiv:2606.16140, 2026. https://arxiv.org/abs/2606.16140

## Citation

```bibtex
@software{ethosoft_research_reasoner_2026,
  title        = {Qwen3-1.7B ResearchReasoning JSON RL},
  author       = {Ethosoft and QA-Research-SLM contributors},
  year         = {2026},
  url          = {https://huggingface.co/Ethosoft/Qwen3-1.7B-ResearchReasoning-JSON-RL},
  base_model   = {Qwen/Qwen3-1.7B}
}
```

Please also cite the Qwen3 technical report and the original benchmark datasets when using this model in research.
