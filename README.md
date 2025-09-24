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
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Irfanuruchi/Llama-2-13B-Computer-Engineering",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Irfanuruchi/Llama-2-13B-Computer-Engineering")

prompt = """### Instruction:
Explain CPU pipelining and its advantages.

### Response:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

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
