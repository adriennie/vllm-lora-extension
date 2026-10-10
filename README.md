# LoRA Rank Ablation and Multi-Adapter Serving with vLLM

This repository explores two related steps in parameter-efficient fine-tuning: training LoRA adapters at different ranks, then serving those adapters through one vLLM engine using per-request adapter selection.

The experiment fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` for casual-to-formal sentence conversion with ranks 4, 8, and 16. It records training metrics and demonstrates generation from the base model and each adapter. The included generation comparison is a single-prompt demonstration, not a quality benchmark.

## What is included

- A Kaggle-oriented notebook for rank ablation: [`qwen_lora_rank_ablation.ipynb`](qwen_lora_rank_ablation.ipynb).
- A Colab-oriented notebook for multi-adapter vLLM inference: [`qwen_vllm_multi_adapter_serving.ipynb`](qwen_vllm_multi_adapter_serving.ipynb).
- PEFT adapter checkpoints for ranks 4, 8, and 16 in [`results/`](results/).
- Recorded rank metrics in [`results/lora_rank_ablation.csv`](results/lora_rank_ablation.csv).
- Example serving outputs in [`vllm_evaluation_results.json`](vllm_evaluation_results.json).

## Workflow

```text
Kaggle training notebook
  └─ Qwen2.5-1.5B-Instruct + four hand-written examples repeated 25 times
       └─ LoRA adapters (r=4, 8, 16) and rank metrics
            └─ copy adapter folders to Google Drive
                 └─ Colab vLLM notebook loads one base model and routes requests
                    to the base model or one of the adapters
```

### Training configuration

| Setting | Value |
| --- | --- |
| Base model | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Task | Convert casual technical sentences into formal academic sentences |
| Training examples | Four hand-written instruction/input/output examples, repeated 25 times (100 rows; four unique examples) |
| Trainer | Hugging Face TRL `SFTTrainer` with PEFT |
| Target modules | `q_proj`, `v_proj` |
| Ranks | 4, 8, 16 |
| LoRA alpha / dropout | 16 / 0.05 |
| Learning rate | `2e-4` |
| Epochs | 3 |
| Per-device batch / gradient accumulation | 2 / 2 |
| Precision | FP16 |
| Sequence length | 256 tokens |

The small repeated dataset makes this a notebook-scale demonstration. The results should not be interpreted as general task performance or evidence that one rank is reliably better.

### Recorded rank results

| Rank | Trainable parameters | Trainable (%) | Training time (s) | Peak VRAM (GB) | Final training loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 544,768 | 0.0353 | 55.50 | 6.55 | 2.4458 |
| 8 | 1,089,536 | 0.0705 | 47.83 | 4.76 | 2.4438 |
| 16 | 2,179,072 | 0.1410 | 48.55 | 4.77 | 2.4248 |

The parameter counts grow with rank, while the recorded final training losses are close. There is one run per rank, so small differences in loss, runtime, or peak memory may reflect run-to-run or notebook-session effects. These values are the recorded outputs in the CSV; they are not a repeated-trial estimate.

### Multi-adapter serving

The serving notebook mounts Google Drive, initializes a vLLM `LLM` with LoRA enabled and `max_lora_rank=16`, and issues generation requests for the base model and each adapter. Adapter selection uses `LoRARequest` handles in the same engine instance. The notebook currently installs vLLM without a version pin and prints the installed version at runtime; actual compatibility can therefore depend on the environment when the notebook is run. It also includes a workaround for IPython output streams in Colab.

The notebook expects adapter directories named `lora_adapter_r4`, `lora_adapter_r8`, and `lora_adapter_r16` inside a Google Drive directory configured by `ADAPTER_BASE_PATH` (currently `/content/drive/MyDrive/results-2`). Copy the tracked folders from `results/` there, or change that setting. The notebook writes the example generations to `vllm_evaluation_results.json` in its current working directory; the JSON included in this repository is a saved example.

The demo shows adapter routing through one engine, but does not measure serving throughput, latency, or GPU memory against a model-per-adapter baseline. It therefore demonstrates the serving workflow, not a quantified efficiency improvement.

## Example generation

Prompt: “PagedAttention stops GPU memory from being wasted when generating text.”

| Model | Recorded output |
| --- | --- |
| Base model | The software PagedAttention effectively prevents GPU memory from being inefficiently utilized during the generation of text. |
| Adapter, r=4 | PagedAttention effectively prevents GPU memory leaks during the generation of textual data by ensuring that the allocated memory is reused efficiently. |
| Adapter, r=8 | PagedAttention effectively prevents GPU memory wastage by ensuring that only necessary model parameters are loaded into memory during text generation. |
| Adapter, r=16 | PagedAttention effectively prevents GPU memory wastage during text generation by ensuring that only necessary data is retained in memory. |

These outputs are included as qualitative examples only. Some add technical claims not present in the input, so they do not establish faithfulness or correctness.

## Reproduce

1. Open `qwen_lora_rank_ablation.ipynb` in Kaggle with a GPU runtime. Run the cells to train the three adapters and produce the metrics CSV. The notebook installs current package versions rather than reading a pinned environment file.
2. Copy `lora_adapter_r4`, `lora_adapter_r8`, and `lora_adapter_r16` into the configured Google Drive directory. The adapter files tracked in this repository are under `results/`.
3. Open `qwen_vllm_multi_adapter_serving.ipynb` in Colab, set `ADAPTER_BASE_PATH` to that directory, and run the cells. The notebook installs vLLM at runtime.
4. Inspect the generation output or the produced `vllm_evaluation_results.json`.

Model downloads may require access to Hugging Face. Use the runtime's secret manager or environment variables for credentials; do not commit access tokens.

## Limitations and next steps

- The training set has only four unique examples, repeated to form 100 rows. A larger, held-out dataset is needed for meaningful evaluation.
- Only final training loss is reported for each single run. There are no validation metrics or repeated seeds.
- The generation comparison uses one prompt and produces some unsupported details.
- Multi-adapter routing is demonstrated, but serving memory, throughput, and latency are not benchmarked.
- Notebook dependencies are installed dynamically and are not pinned to a reproducible environment.

Potential next steps include evaluating on a held-out set, repeating ranks with multiple seeds, pinning dependencies, and benchmarking mixed-adapter serving against a clearly defined baseline.

## Repository structure

```text
.
├── qwen_lora_rank_ablation.ipynb
├── qwen_vllm_multi_adapter_serving.ipynb
├── results/
│   ├── lora_adapter_r4/
│   ├── lora_adapter_r8/
│   ├── lora_adapter_r16/
│   ├── lora_rank_ablation.csv
│   └── lora_rank_ablation.png
├── vllm_evaluation_results.json
├── LICENSE
└── README.md
```

## References

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv preprint arXiv:2106.09685. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).

Chen, L. et al. (2023). *Punica: Multi-Tenant LoRA Serving*. [arXiv:2310.18547](https://arxiv.org/abs/2310.18547).

## License

Released under the [MIT License](LICENSE).
