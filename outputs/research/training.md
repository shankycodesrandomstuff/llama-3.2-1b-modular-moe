# Training and evaluation

## Training implications

Duplicating a dense FFN into E experts exactly preserves a dense-style output only if every expert returns the same result and selected router weights sum to one. It yields no specialization and gives a symmetric router/expert state. This is a mathematical inference, not an upstream claim.

Hugging Face Mixtral uses router logits plus a Switch-style balancing loss when router-logit output is enabled; [source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py). DeepSpeed documents capacity factors, min capacity, noisy gating, dropped tokens and top-1/2. [Source](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/layer.py). Megatron supports aux, Sinkhorn, sequence auxiliary, or none; select it only when moving to that stack.

## Minimum staged evaluation

1. **Structural:** strict base-weight loading report; forward shapes; generation/cache; save/load; router padding-mask test.
2. **Equivalence:** all experts cloned, deterministic top-k, compare selected MoE FFN output to dense FFN within dtype tolerance.
3. **Optimization:** short domain SFT with loss, LM perplexity and router auxiliary loss; plot per-layer expert load, entropy, overflow/drop rate, dead experts.
4. **Task quality:** fixed held-out task/eval suite against original 1B, same tokenizer/prompting/decoding.
5. **Systems:** tokens/s, prefill/decode latency, peak VRAM/RAM, checkpoint size for dense, MoE BF16, and each quantized candidate.
6. **Ablation:** expert count {2,4,8}, top-k {1,2}, selected-layer schedule, initialization perturbation, aux coefficient, adapter-only versus full-expert fine-tuning.

## Fine-tuning choices

**RECOMMENDATION:** first train router plus expert-local LoRA/adapters (and optionally final selected FFN weights) while freezing attention/embeddings. This contains cost and limits catastrophic drift. Meta confirms LoRA/QAT was used in its own dense post-training, but routed-expert LoRA remains a project hypothesis. Escalate to full fine-tuning only if this baseline fails.
