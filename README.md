# Dynamic Multi-Adapter LoRA Rank Ablation & vLLM Serving Pipeline

## Parameter-Efficient Fine-Tuning (PEFT) and Multi-LoRA Serving on a Tesla T4

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B--Instruct-6F42C1)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Notebooks](https://img.shields.io/badge/Notebooks-Colab%20%7C%20Kaggle-F9AB00)](#reproducing-the-experiment)

### 🚀 At a Glance

* **Core Achievement:** Trained three LoRA adapters (rank 4, 8, 16) on `Qwen/Qwen2.5-1.5B-Instruct` and served all of them from **a single vLLM engine instance**, switching adapters per request without reloading the base weights.
* **Mechanism:** vLLM keeps one copy of the base model in GPU memory and applies the selected adapter's low-rank matrices at request time (Punica-style batched LoRA kernels).
* **Environment:** Fine-tuning on a Kaggle T4; serving on a Google Colab T4. Both runs fit on a 16 GB GPU.

This project is the practical follow-up to my [vLLM PagedAttention vs. Hugging Face benchmark](https://github.com/adriennie). It asks two questions: (1) how much does LoRA rank actually matter for a small instruction-tuned model on a narrow task, and (2) how do you serve several fine-tuned variants of the same base model without paying for one model copy per variant?

> **Key takeaway:** Ranks 4, 8, and 16 reach nearly identical final training loss (2.4458 → 2.4438 → 2.4248), while trainable parameters double with each step (0.54M → 1.09M → 2.18M). All three adapters were then routed dynamically through a single vLLM session. The outputs are a small qualitative comparison on one prompt, not a controlled quality benchmark; see [Limitations](#interpretation-and-limitations).

## Project overview

LoRA (Low-Rank Adaptation) freezes the base model and learns a low-rank update `ΔW = B·A` for selected weight matrices, so only a tiny fraction of parameters is trained. The rank `r` controls the capacity of that update: higher ranks can express more, but cost more parameters and memory.

In production-style serving, the interesting problem shifts from *training one adapter* to *hosting many adapters on one base model*. A naive setup loads a full model per adapter, so VRAM scales roughly with `N × (base + adapter)`. vLLM's multi-LoRA support instead hosts a single base model and attaches adapter weights per request, so the incremental cost of each extra adapter is small.

This repository covers both halves:

1. **Fine-tuning phase (Kaggle):** a rank ablation over `r ∈ {4, 8, 16}` using Hugging Face `trl` (`SFTTrainer`) and `peft`.
2. **Serving phase (Colab):** loading the resulting adapters from Google Drive into vLLM and generating with each adapter (and the unmodified base model) in one continuous engine session.

## Architecture and workflow

```text
[ Kaggle T4 Training ] ──> Rank Ablation (r = 4, 8, 16) ──> Google Drive artifacts (/results-2/)
                                                                      │
[ Google Colab T4 ]  ──> vLLM engine + Multi-LoRA routing  <──────────┘
                              ├─ base model (no adapter)
                              ├─ adapter_r4   (LoRARequest id 1)
                              ├─ adapter_r8   (LoRARequest id 2)
                              └─ adapter_r16  (LoRARequest id 3)
```

### 1. Fine-tuning phase (Kaggle)

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct`, loaded in `torch.float16` |
| Task | Instruction fine-tuning for casual → formal/academic text conversion |
| Libraries | Hugging Face `trl` (`SFTTrainer`) and `peft` |
| Target modules | `q_proj`, `v_proj` |
| Ranks evaluated | `r ∈ {4, 8, 16}` |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Learning rate | 2 × 10⁻⁴ |

### 2. Serving and inference phase (Colab)

| Setting | Value |
| --- | --- |
| Serving engine | vLLM (v0.31.0) |
| LoRA configuration | `enable_lora=True`, `max_lora_rank=16` |
| Context limit | `max_model_len=512` |
| GPU memory utilization | 0.85 |
| Hardware adaptation | FlashAttention-2 requires compute capability ≥ 8.0, so vLLM falls back to Triton attention on the T4 (compute capability 7.5) |
| Routing | One `LoRARequest` handle per adapter (`adapter_r4`, `adapter_r8`, `adapter_r16`), selected per `generate()` call in a single engine session |

**Technology:** Python, PyTorch, Hugging Face Transformers, TRL, PEFT, vLLM, Google Colab, Kaggle.

## Quantitative rank ablation results

| Rank (r) | Trainable parameters | Trainable (%) | Training time (s) | Peak VRAM (GB) | Final training loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 544,768 | 0.0353% | 55.50 | 6.55 | 2.4458 |
| 8 | 1,089,536 | 0.0705% | 47.83 | 4.76 | 2.4438 |
| 16 | 2,179,072 | 0.1410% | 48.55 | 4.77 | 2.4248 |

Trainable parameters scale linearly with rank, as expected (`4,864 × r × 28` layers for `q_proj` + `v_proj` on this model). Final loss improves slightly with rank, with the largest gain between r=8 and r=16.

## Research baseline vs. this serving pipeline

| Dimension | Standard LoRA workflow (static loading) | This pipeline (vLLM multi-adapter) |
| --- | --- | --- |
| Serving architecture | Merge or load one adapter per model instance; switching adapters means reloading weights | One base model instance; adapters are injected per request via `LoRARequest` |
| VRAM footprint | Grows with the number of concurrent adapter endpoints (`N × base + adapter`) | Base weights shared across adapters; each adapter adds only its small low-rank matrices |
| Throughput and concurrency | Sequential Hugging Face `generate()` batching | vLLM continuous batching with PagedAttention |

## Implementation details

The serving notebook (`qwen_vllm_multi_adapter_serving.ipynb`) mounts Google Drive, initializes vLLM with LoRA enabled, registers the three adapters, and runs the same greedy prompt through the base model and each adapter. Results are written to `vllm_evaluation_results.json`.

> **Colab note:** vLLM's V1 engine expects `sys.stdout`/`sys.stderr` to expose `fileno()`, which Colab's IPython streams do not by default. The notebook patches `ipykernel.iostream.OutStream.fileno` (and falls back to fd 1/2) before importing vLLM.

## Qualitative inference comparison

**Input:** *"PagedAttention stops GPU memory from being wasted when generating text."*

| Model | Output |
| --- | --- |
| Base model | "The software PagedAttention effectively prevents GPU memory from being inefficiently utilized during the generation of text." |
| Adapter (r=4) | "PagedAttention effectively prevents GPU memory leaks during the generation of textual data by ensuring that the allocated memory is reused efficiently." |
| Adapter (r=8) | "PagedAttention effectively prevents GPU memory wastage by ensuring that only necessary model parameters are loaded into memory during text generation." |
| Adapter (r=16) | "PagedAttention effectively prevents GPU memory wastage during text generation by ensuring that only necessary data is retained in memory." |

### Interpretation and limitations

- **Loss differences are small.** The gap between r=4 and r=8 in final loss is about 0.002, and r=16 improves it by roughly 0.02. With a single training run per rank, this should not be read as a statistically meaningful ranking.
- **Training time and VRAM are not a clean scaling curve.** r=4 shows the slowest run and highest peak VRAM (55.50 s, 6.55 GB), which is likely due to warmup or session effects rather than rank itself. Repeated runs would be needed to separate noise from signal.
- **The qualitative comparison uses one prompt.** All adapters add technical detail not present in the input (for example, "memory leaks" in r=4 and "only necessary model parameters are loaded" in r=8), which is a sign of imperfect faithfulness to the source sentence. A proper evaluation needs a held-out set with automatic metrics (e.g., ROUGE/BERTScore) and human or LLM-judge review.
- **No serving benchmark here.** The "constant VRAM overhead" benefit of multi-LoRA serving is architectural; this repo demonstrates the routing but does not measure memory or throughput against a per-adapter-reload baseline. See the roadmap.
- **Environment.** Training and serving ran on different notebook runtimes (Kaggle and Colab). Library versions are those recorded in the notebooks, not pinned requirements.

## Repository structure

```text
.
├── <training_notebook>.ipynb                 # Kaggle: LoRA rank ablation (r = 4, 8, 16)
├── qwen_vllm_multi_adapter_serving.ipynb     # Colab: vLLM multi-adapter serving
├── vllm_evaluation_results.json              # Base vs. adapter outputs for the test prompt
├── results-2/
│   ├── lora_adapter_r4/                      # PEFT adapter, rank 4
│   ├── lora_adapter_r8/                      # PEFT adapter, rank 8
│   └── lora_adapter_r16/                     # PEFT adapter, rank 16
├── LICENSE
└── README.md
```

## Reproducing the experiment

1. **Train the adapters (Kaggle, T4 GPU).** Run the training notebook to fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with `r ∈ {4, 8, 16}`, `alpha=16`, `dropout=0.05`, `lr=2e-4`, targeting `q_proj` and `v_proj`. Save each adapter as `lora_adapter_r{4,8,16}`.
2. **Sync artifacts.** Upload the adapter folders to Google Drive under `MyDrive/results-2/` (or edit `ADAPTER_BASE_PATH` in the serving notebook).
3. **Serve (Colab, T4 GPU).** Open `qwen_vllm_multi_adapter_serving.ipynb`, install vLLM, mount Drive, and run all cells. Keep `max_lora_rank` at or above the largest adapter rank.
4. **Inspect outputs.** The notebook writes `vllm_evaluation_results.json` with the base model's output and each adapter's output for the test prompt.

Model access can require a Hugging Face account or token. Supply credentials through the notebook runtime's secret manager or environment variables; never commit tokens to source control.

## Future iterations and roadmap

- [ ] Evaluate adapters on a held-out set with ROUGE/BERTScore and an LLM-judge for faithfulness.
- [ ] Repeat each rank over multiple seeds to quantify variance in loss, time, and VRAM.
- [ ] Benchmark multi-LoRA serving vs. per-adapter model reloading (device-level VRAM via `pynvml`, throughput, and latency under mixed-adapter traffic).
- [ ] Extend ablation to more target modules (`k_proj`, `o_proj`, MLP projections) and different `alpha/r` ratios.
- [ ] Serve through vLLM's OpenAI-compatible server with per-request adapter selection and asynchronous clients.

## References

> Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
>
> Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023.* [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
>
> Chen, L. et al. "Punica: Multi-Tenant LoRA Serving." *MLSys 2024.* [arXiv:2310.18547](https://arxiv.org/abs/2310.18547).

## License

Released under the [MIT License](LICENSE).
