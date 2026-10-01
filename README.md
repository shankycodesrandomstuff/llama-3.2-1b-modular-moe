# Llama 3.2 1B modular MoE

Small Hugging Face/PyTorch experiment. Replace selected Llama decoder FFNs with locally dispatched Llama-style SwiGLU experts. Attention, norms, residuals, tokenization, KV cache, and parallelism stay out of it.

## Install and validate

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

Tests use a tiny random Llama config. No gated weights. No credentials.

## Baseline and conversion

Run `python scripts/record_baseline.py --output experiments/baseline` where `torch` and `transformers` are installed. It records runtime and accelerator facts. It does not download weights.

`llama_moe.convert.convert_single_layer_to_moe` clones a selected dense MLP into experts on an instantiated `LlamaForCausalLM`. Save the result with Hugging Face `save_pretrained`; reload it through `LlamaMoEForCausalLM.from_pretrained`.

Do not put weights, datasets, credentials, private paths, or local artifacts in Git. See `docs/phase-1.md` before claiming the conversion works.
