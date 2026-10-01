# Llama 3.2 1B baseline

**Scope.** This is the text-only Llama 3.2 1B base/instruct family, not the 11B/90B vision variants. Sources were checked 2026-10-01; pin every dependency and downloaded checkpoint revision before implementation.

## Verified facts

Meta describes 1B as an autoregressive optimized Transformer with 1.23B parameters, 128K context, grouped-query attention (GQA), and shared embeddings in the family table. The release uses the custom [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md), not Apache/MIT.

The official 1B HF-format configuration is expected to be 16 layers; `hidden_size=2048`; 32 attention and 8 KV heads (GQA, `head_dim=64`); `intermediate_size=8192`; SiLU; RMSNorm epsilon `1e-5`; RoPE theta 500,000 with Llama-3 RoPE scaling; vocabulary 128,256; no MLP/attention bias; and 131,072 maximum positions. The gated official config must be recorded verbatim in the conversion manifest before any engineering. A mirror corroborates the 16-layer count [here](https://huggingface.co/context-labs/meta-llama-Llama-3.2-1B-Instruct-FP16/blob/main/config.json), but it is not an authoritative substitute.

Hugging Face's actual [Llama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) establishes the operational layout:

```
token embedding → 16 × [RMSNorm → RoPE self-attention → residual
                         RMSNorm → SwiGLU MLP → residual] → RMSNorm → LM head
```

`LlamaMLP.forward` is `down_proj(SiLU(gate_proj(x)) * up_proj(x))`. `LlamaDecoderLayer.forward` owns `self.mlp`; this is the intended replacement seam. Attention uses Q/K/V/O projections, RoPE, cache-aware attention dispatch selected by `_attn_implementation`, then output projection. HF exposes eager/SDPA/Flash Attention capability flags in that class. Keep this subsystem unmodified in the first conversion.

## Checkpoints and tokenizer

**DOCUMENTED FACT:** HF `PreTrainedModel.from_pretrained` conventions load `config.json`, tokenizer files and weights commonly in `model.safetensors` / sharded safetensors. Parameter keys in the HF implementation are structurally `model.embed_tokens`, `model.layers.N.self_attn.*`, `model.layers.N.mlp.{gate,up,down}_proj`, `model.layers.N.{input,post_attention}_layernorm`, `model.norm`, and `lm_head`. Confirm exact keys with a local checkpoint manifest; do not assume Meta's original-checkpoint layout equals HF keys.

The tokenizer is Llama's 128,256-vocabulary tokenizer; preserve it and embedding/LM-head semantics in an FFN-only conversion. The model card documents SFT, rejection sampling, DPO and QAT+LoRA in Meta's post-training recipe, and 4-bit groupwise linear-weight / 8-bit activation quantization for its mobile-oriented release. It does **not** document a public procedure for upcycling 1B into MoE.

## Consequence

**ENGINEERING RECOMMENDATION:** replace selected `layer.mlp` modules only, retaining dimensions, residual order, RMSNorm, RoPE, cache and tokenizer. Dense checkpoint weights can seed identical experts, but a router then has symmetry and no specialization; introduce controlled expert perturbation and train/router-calibrate before making quality claims.
