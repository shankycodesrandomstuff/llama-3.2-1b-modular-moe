"""Explicit dense-Llama to single-layer local-MoE conversion helpers."""

from __future__ import annotations

from typing import Any

from transformers import LlamaForCausalLM

from .moe import SparseMoEFeedForward


def _apply_moe_layers(model: LlamaForCausalLM, settings: dict[str, Any], clone_dense: bool) -> None:
    hidden_size = model.config.hidden_size
    for layer_index in settings["layers"]:
        layer = model.model.layers[layer_index]
        if isinstance(layer.mlp, SparseMoEFeedForward):
            continue
        layer.mlp = SparseMoEFeedForward(
            layer.mlp if clone_dense else layer.mlp,
            hidden_size,
            settings["num_experts"],
            settings["top_k"],
        )


class LlamaMoEForCausalLM(LlamaForCausalLM):
    """Llama CausalLM that creates MoE FFN modules from `config.moe_config`."""

    def __init__(self, config):
        super().__init__(config)
        settings = getattr(config, "moe_config", None)
        if settings:
            _apply_moe_layers(self, settings, clone_dense=False)


def convert_single_layer_to_moe(
    model: LlamaForCausalLM,
    layer_index: int,
    num_experts: int = 2,
    top_k: int = 1,
) -> LlamaMoEForCausalLM:
    """Replace one dense FFN in-place and declare its reconstruction config."""
    if not 0 <= layer_index < len(model.model.layers):
        raise IndexError("layer_index is outside the decoder layer range")
    if num_experts != 2:
        raise ValueError("Phase 1 is intentionally limited to two experts")
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    settings = {
        "layers": [layer_index],
        "num_experts": num_experts,
        "top_k": top_k,
        "router_dtype": "float32",
    }
    model.config.moe_config = settings
    model.config.architectures = ["LlamaMoEForCausalLM"]
    _apply_moe_layers(model, settings, clone_dense=True)
    return model  # runtime class remains compatible with LlamaForCausalLM methods
