# Development TODO

- [x] Select a CPU-compatible Python runtime and install project dependencies.
- [x] Validate local sparse two-expert routing with a synthetic Llama model.
- [ ] Record a baseline for an exact official Llama 3.2 1B checkpoint revision.
- [ ] Assess available RAM before attempting a real checkpoint load.
- [ ] Validate one-layer real-checkpoint conversion, save/reload, and generation.
- [ ] Measure CPU-only routing diagnostics for top-1 and top-2.
- [ ] Convert a second layer only after the real one-layer gate passes.
