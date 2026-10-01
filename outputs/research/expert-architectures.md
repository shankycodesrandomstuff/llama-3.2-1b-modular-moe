# Expert architectures and abstraction boundary

## What to preserve

Each first-generation expert should be Llama's existing SwiGLU FFN: `gate_proj: H→I`, `up_proj: H→I`, SiLU product, `down_proj: I→H`, with H=2048 and I=5632. This preserves the tensor contract at the FFN seam. Mixtral's sparse block is the analogous design: experts are feed-forward blocks and the router chooses experts per token.

DeepSpeed's `expert: nn.Module` constructor is evidence that the expert interface can remain a generic PyTorch module; no bespoke expert base class is required. [DeepSpeed source](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/layer.py).

## Modular API proposal

```
Expert: forward(tokens[N,H]) -> [N,H]              # WRAP existing LlamaMLP
Router: forward(tokens[N,H]) -> RoutePlan           # ADAPT HF Mixtral semantics
Dispatcher: execute(plan, tokens, experts) -> output # REUSE HF local dispatch initially
MoEFeedForward: forward([B,S,H]) -> [B,S,H]         # IMPLEMENT thin glue only
```

`RoutePlan` owns ids, scores, optional raw logits and metrics. It deliberately prevents a router from knowing whether execution is local, DeepSpeed EP, or another backend. This is the small abstraction layer, not a new training framework.

## Component classification

| Component | Classification | Basis |
|---|---|---|
| Llama backbone/tokenizer/cache | REUSE | Transformers Llama implementation |
| Expert weights / SwiGLU | WRAP | existing `LlamaMLP` |
| local sparse router/dispatch | ADAPT | HF Mixtral top-k/block |
| balancing-loss calculation | REUSE/ADAPT | HF Mixtral function |
| distributed EP | WRAP later | DeepSpeed MoE or Megatron-Core |
| model config + checkpoint naming | IMPLEMENT minimally | no upstream Llama-MoE variant exists |

**INFERENCE:** adapters/LoRA can be expert-local modules and may be more storage-efficient than full FFN copies. Meta documents QAT + LoRA for Llama 3.2, but that does not demonstrate routed adapter experts; validate that separately.
