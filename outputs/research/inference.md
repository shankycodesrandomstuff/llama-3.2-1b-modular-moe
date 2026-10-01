# Inference, conversion, and quantization

## Verified constraints

HF local Mixtral inference is available through ordinary `AutoModelForCausalLM`/`generate`; router logits are optionally returned, and the cache remains attention-owned. This makes the proposed FFN-only swap compatible with existing causal generation in principle. [Mixtral source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modular_mixtral.py).

vLLM's [MoE model mapping source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/transformers/moe.py) illustrates why serving is not automatic: it maps architecture-specific expert tensor names and router dtype/weights. llama.cpp similarly selects specific architectures in its [model factory](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp). A novel `LlamaMoE` config will not be accepted merely because its tensors are Llama-like.

Meta's card documents 4-bit groupwise linear weights (group size 32) and dynamic int8 activations in its quantized Llama release, plus QAT/LoRA. That applies to its dense release; **UNKNOWN:** whether a third-party MoE dispatcher plus independently quantized experts attains the same accuracy/performance.

## Checkpoint plan

1. Load official dense HF checkpoint revision.
2. Emit a new declared architecture/config with `moe_layers`, expert count, top-k, router dtype and routing policy; never masquerade it as unmodified `LlamaForCausalLM`.
3. Store router tensors plus `experts.<e>.{gate,up,down}_proj` deterministically in safetensors; retain original non-MLP names unchanged.
4. Provide a dense→MoE conversion manifest containing base model revision, initialization policy and tool version. Round-trip save/load and dense-equivalence tests are mandatory.
5. Only then design a vLLM mapper or GGUF conversion. Keep a pure HF inference path as the reference oracle.

**ENGINEERING RECOMMENDATION:** quantize after training and evaluate each scheme, not before. Sparse activation does not mean sparse storage: full experts still consume memory.
