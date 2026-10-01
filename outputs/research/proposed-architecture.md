# Proposed architecture — after research

## Decision

Use **Transformers as the first implementation host**, with a thin, explicit MoE FFN wrapper at selected Llama decoder layers. Do not start with DeepSpeed/Megatron/vLLM/llama.cpp. They remain optional execution/export backends once semantic correctness and checkpoint format are stable.

```
Llama 3.2 tokenizer/embedding
          │
16 unchanged Llama decoder layers
          │ (selected `post_attention_layernorm → mlp` seam)
          ▼
     MoEFeedForward abstraction
       ├─ Router (HF Mixtral-derived top-k semantics)
       ├─ RoutePlan / local dispatcher (thin glue + HF-style gather/scatter)
       └─ N experts (wrapped Llama SwiGLU MLPs)
          │
unchanged residual → final RMSNorm → LM head → HF generation/cache
```

| Component | Existing project / implementation | Adaptation | Why |
|---|---|---|---|
| Backbone | HF `LlamaModel`, `LlamaDecoderLayer` | none outside MLP slots | directly loads Llama and preserves attention semantics |
| Expert | HF `LlamaMLP` | wrap/clone per expert | exact input/output/activation contract |
| Router | HF `MixtralTopKRouter` | parameterize independently of Mixtral config | compact, readable top-k reference |
| Dispatch/combine | HF `MixtralSparseMoeBlock` | extract behind `RoutePlan` | correct local sparse reference |
| Balancing | HF `load_balancing_loss_func` | expose diagnostics/mask tests | maintained baseline |
| EP scale-out | DeepSpeed `MoE` or Megatron-Core | backend adapter, later | reuse only if profiling requires it |
| serving/export | vLLM / llama.cpp | new mapper/export support, later | neither recognizes a custom checkpoint automatically |

## Smallest new engineering

1. A `LlamaMoEConfig`/model wrapper and deterministic checkpoint schema.
2. `RoutePlan` plus adapter that inserts router + selected LlamaMLP experts at the MLP seam.
3. Dense-to-MoE initialization/conversion, validation, metrics and training scripts.
4. Tests/evaluation described in `training.md`.

Everything else is reused/wrapped. This is intentionally not a novel router, kernel, expert-parallel fabric, tokenizer, attention implementation or inference engine.

## Gates before implementation

Implementation is authorized only after: exact official checkpoint revision and config are acquired; intended redistribution/licensing is approved; 2/4/8-expert memory budget chosen; selected layers/top-k/evaluation dataset specified; and the first target is declared as HF-only training/inference. If a local llama.cpp/vLLM artifact is required initially, its adapter becomes an additional approved scope.
