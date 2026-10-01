# Phase 1 validation gates

## Invariants

- Convert exactly one `model.layers[index].mlp` at first.
- Each expert is a cloned Hugging Face `LlamaMLP`; no Llama attention/cache/norm/residual code is changed.
- The router uses an FP32 linear score calculation, softmax, top-k selection, and selected-score normalization.
- Dispatch invokes only experts selected for tokens. It is local PyTorch dispatch, not expert parallelism.

## Required evidence before training

1. Record `experiments/baseline/config.json`, `metrics.json`, and `README.md` for the exact checkpoint revision.
2. Run dense-to-MoE identity tests after cloning experts and forcing a router to expert 0.
3. Run checkpoint save/reload output-equivalence test.
4. Log expert loads, routing entropy, router auxiliary loss, latency, and peak memory for `top_k=1` and `top_k=2`.

## Experiment record

Use immutable directories such as `experiments/EXP-001/` and include ID, hypothesis, configuration, variables, method, expected result, measured result, and conclusion. Do not overwrite prior results.

## Security release gate

Before a commit or push, inspect `git status`, `git diff`, and `git diff --cached`; scan staged content with the CI secret-pattern check; verify remote, branch, staged files, and commit range. Never publish weights, `.env`, credentials, private data, or machine-specific paths.
