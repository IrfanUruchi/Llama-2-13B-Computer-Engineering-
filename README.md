---
license: llama2
base_model: meta-llama/Llama-2-13b-hf
tags:
- llama2
- computer-engineering
- computer-architecture
- algorithms
- systems
- quantized
language:
- en
library_name: transformers
datasets:
- cais/mmlu
- sahil2801/CodeAlpaca-20k
- Open-Orca/OpenOrca
model_type: llama
---

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/IrfanUruchi/Llama-2-13B-Computer-Engineering-)
[![Model Weights](https://img.shields.io/badge/🤗-Model_Weights-FFD21F?style=for-the-badge)](https://huggingface.co/Irfanuruchi/Llama-2-13B-Computer-Engineering)
[![Meta AI Llama 2 Licence](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://huggingface.co/meta-llama/Llama-2-13b-hf)

---

# Llama-2-13B-Computer-Engineering

### Overview

**Llama-2-13B-Computer-Engineering** is a fine‑tuned variant of **LLaMA‑2‑13B**, adapted for **computer engineering, computer architecture, systems, and algorithms**.  
The model was trained using **QLoRA (4‑bit quantization)**, then merged into a single checkpoint.  
This allows **13B‑scale reasoning** to run in ~6.6 GB of storage and ~16GB of GPU memory, making it usable on a single modern GPU.

---

### Training Setup
- **Base model:** [LLaMA‑2‑13B](https://huggingface.co/meta-llama/Llama-2-13b-hf)  
- **Fine‑tuning method:** QLoRA (4‑bit NF4) + LoRA adapters (`r=16`, `α=32`)  
- **Optimized Layers:** Attention projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`)  
- **Final merge:** LoRA weights merged into the base model → single merged checkpoint  
- **Resulting size:** ~6.6 GB (`safetensors` sharded files) vs. ~24 GB fp16 
(Model Safetensor files are avaiable on the Model weights here just the tokenizers and configs files are avaiable)

---

## Dataset
The dataset was curated from multiple sources to emphasize **reasoning, explanations, and code writing in computer engineering contexts**.  

Included sources:
- **MMLU (Computer Security subset)** → exam‑style questions on systems and security  
- **CodeAlpaca‑20k (filtered)** → algorithm, data structures, complexity, trees, sorting/searching, graphs  
- **OpenOrca subset** → reasoning tasks mentioning computer systems and architecture  
- **Custom technical examples** (hand‑crafted) on:
  - CPU pipelining & instruction‑level parallelism  
  - Cache coherency and MESI protocol  
  - Compiler optimizations (instruction scheduling, inlining, loop unrolling)  
  - RISC vs. CISC architectures  
  - Memory hierarchies (registers, caches, RAM, storage)  
  - Branch prediction  
  - Example algorithms (binary search, stacks, etc.)  


---

### Usage


```python

model_id = "Irfanuruchi/Llama-2-13B-Computer-Engineering"

)

```

---

## Example Responses

Q: What is cache coherency in multicore systems?
A: Cache coherence ensures that all processor cores observe a consistent view of memory. Protocols such as MESI handle invalidation and updates when one core modifies data, preventing stale values and race conditions.

Q: Implement a stack in Python.
A: Produces a Stack class with methods for push, pop, peek, is_empty, and size.

---


## Limitations

- While optimized for computer engineering, performance outside this scope is similar to the base LLaMA‑2‑13B.

---

## License

- Base model:[Meta's LLaMA 2 license](https://huggingface.co/meta-llama/Llama-2-13b-hf).
- Fine‑tuned weights: Distributed under the same license.
- Datasets: Combination of open academic sets (MMLU, CodeAlpaca, OpenOrca) and custom educational material.
