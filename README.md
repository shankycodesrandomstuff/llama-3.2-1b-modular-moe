# Llama 3.2 1B modular MoE

This is a deliberately small Hugging Face/PyTorch experiment. It replaces one or more Llama decoder FFNs with locally dispatched, Llama-style SwiGLU experts. It does not change attention, normalization, residual paths, tokenization, KV-cache behavior, or introduce expert parallelism.

## Install and validate

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

Tests use a tiny randomly initialized Llama configuration; they do not download gated weights or require credentials.

## Baseline and conversion

Run `python scripts/record_baseline.py --output experiments/baseline` in an environment with `torch` and `transformers` installed. It records runtime versions and accelerator availability without downloading model weights.

`llama_moe.convert.convert_single_layer_to_moe` converts an instantiated `LlamaForCausalLM` by cloning a selected dense MLP into two or more experts. The resulting `LlamaMoEForCausalLM` saves with Hugging Face `save_pretrained` and reloads through `LlamaMoEForCausalLM.from_pretrained`.

No model weights, datasets, credentials, private paths, or local experiment artifacts belong in Git. See `docs/phase-1.md` for validation gates.
