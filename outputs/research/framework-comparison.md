# Compatibility matrix

| Framework | Llama 3.2 1B | MoE | Training | Inference | Quantization | Ease |
|---|---|---|---|---|---|---|
| HF Transformers | High—native Llama loader | High—Mixtral/Qwen source patterns | High | High, normal generation | ecosystem-wide; custom MoE needs validation | **Highest** |
| DeepSpeed-MoE | Medium—replace HF FFN / integrate engine | High | High / EP | Medium | framework-dependent | Medium |
| Megatron-Core | Medium-low—conversion/config work | Very high | Very high | High at cluster scale | FP8 etc. | Low for 1B prototype |
| vLLM | Low until custom architecture registration/weight mapper exists | High for supported formats | No training | High | strong, architecture-dependent | Low-medium |
| llama.cpp | Low until GGUF metadata/tensor loader/kernels know new architecture | supported MoE families | No | High local | strong GGUF support | Low |
| FairScale | Medium, manually wrap blocks | Top-2 | legacy | not serving-oriented | none central | Low |

## Reasons

HF already owns both the exact Llama decoder/block contract and a maintained sparse-MoE reference. A subclass/wrapper can load dense Llama, replace `model.layers[i].mlp`, preserve `generate`, and save a standard safetensors checkpoint plus an explicit custom config. That is the smallest integration boundary.

DeepSpeed can accept the Llama FFN as its expert, but introduces distributed process-group/checkpoint/runtime assumptions. Megatron adds high-performance EP, token permutation, and checkpoint conversion, but its own model/config stack makes it disproportionate for one 1B model. Its documentation explicitly notes dynamic expert dispatch complicates CUDA graph capture, a real operational constraint. [Evidence](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/cuda_graph.md).

vLLM and llama.cpp are targets after a stable checkpoint schema exists, not sources for the initial transformation. Both need model-specific loader/mapping work; llama.cpp's model-add guide explicitly requires architecture/model-saver metadata coverage. [Guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md).
