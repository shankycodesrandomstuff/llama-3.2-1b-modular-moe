# MoE landscape: implementation evidence

| Project | Router/expert/placement | Training & inference | Status / license |
|---|---|---|---|
| HF Transformers | Mixtral linear top-k FP32 softmax; per-expert FFNs; local gather/scatter | Trainer ecosystem; standard `generate` | Active; Apache-2.0 ([repo](https://github.com/huggingface/transformers)) |
| DeepSpeed-MoE | `TopKGate`, capacity and all-to-all EP; arbitrary `nn.Module` experts | purpose-built distributed training; inference possible but not a lightweight serving target | Active; Apache-2.0 ([source](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/layer.py)) |
| Megatron-Core | configurable top-k/group top-k/balancing; dropless and EP | production-scale pretraining/checkpoints; conversion tooling | Active; Apache-2.0 ([MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)) |
| FairScale | Top-2 gate / capacity dispatch | training reference | Archived; BSD-style ([source](https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/moe/top2gate.py)) |
| Fairseq | configurable sparse MoE language-model recipe, sharded checkpoints | distributed training/evaluation | Maintained toolkit; MIT ([MoE model card](https://github.com/facebookresearch/fairseq/blob/main/examples/moe_lm/model_card.md)) |
| vLLM | architecture-specific fused MoE loading/execution | high-throughput serving and supported quantized MoE models | Active; Apache-2.0 ([MoE mapping source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/transformers/moe.py)) |
| llama.cpp | GGUF architecture loader/kernel paths for many MoE architectures | local CPU/GPU inference + quantization | Active; MIT ([model factory](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)) |
| Qwen MoE | production MoE family (e.g. 128 routed / 8 active in Qwen3 30B-A3B) | HF/vLLM support | code Apache-2.0; model terms vary ([HF docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_moe.md)) |
| DeepSeek | fine-grained routed experts, auxiliary-loss-free balancing in V3 | reference architecture, very different scale/attention | code MIT; model license separate ([README](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README.md)) |

**DOCUMENTED FACT:** these implementations prove distinct mechanisms, not direct interchangeability. Mixtral/Qwen/DeepSeek checkpoints have their own configs and tensor names; they cannot load a dense Llama checkpoint as MoE without conversion.
