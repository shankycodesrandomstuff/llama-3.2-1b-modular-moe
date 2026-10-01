# Routing and dispatch

## Reusable reference implementations

Hugging Face's [Mixtral modular source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modular_mixtral.py) is the closest semantic reference: `MixtralTopKRouter` applies a bias-free linear projection from hidden state to expert logits, FP32 softmax, `topk`, and renormalizes the selected scores. `MixtralSparseMoeBlock` invokes per-expert MLPs only for tokens selected for each expert and uses `index_add_` to accumulate outputs. This is a compact single-node PyTorch reference, not an expert-parallel runtime.

Its [generated implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py) computes the Switch-style auxiliary load-balancing term from router logits when requested. Mask handling is present; include padding-mask tests.

DeepSpeed's Apache-2.0 [`MoE`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/layer.py) accepts an arbitrary expert `nn.Module`, wraps local copies in `Experts`, and constructs `MOELayer(TopKGate(...))`. The gate/runtime source is [`sharded_moe.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py). It supports top-1/top-2, train/eval capacity factors, minimum capacity, token dropping, noisy gating and expert-parallel groups; `MOELayer` returns/records `l_aux`. Its gate file preserves FairScale-derived BSD notices—do not copy individual code without preserving those notices.

Megatron-Core's current [MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md) documents configurable top-k, softmax/sigmoid, group top-k, aux/sinkhorn/sequence auxiliary balancing, dropless operation, expert parallelism, distributed checkpointing and upcycling. Router behavior lives under `megatron/core/transformer/moe/router.py`; balancing helpers under [`moe_utils.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py). It is a large-scale training stack, not a drop-in Llama HF module.

## Recommended first router

**ENGINEERING RECOMMENDATION:** use HF Mixtral-compatible linear top-2 routing, FP32 score computation, normalized selected weights, and an auxiliary balancing loss. Begin with no capacity limit / no token dropping for correctness at 1B scale. Add capacity and distributed dispatch only when profiling demonstrates a need. The resulting layer contract is:

```
hidden [B,S,H] → Router → (topk ids, normalized scores, diagnostics)
                 → Dispatcher → Expert(hidden selected tokens) → weighted combine [B,S,H]
```

**UNKNOWN:** the best `k`, number of experts, selected layers, coefficient, capacity factor and initialization cannot be inferred from the Llama checkpoint; they need ablations.
