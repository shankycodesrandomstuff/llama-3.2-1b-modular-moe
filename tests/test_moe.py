from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from llama_moe.convert import LlamaMoEForCausalLM, convert_single_layer_to_moe  # noqa: E402
from llama_moe.moe import SparseMoEFeedForward, load_balancing_loss  # noqa: E402


def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(config).eval()


def force_expert_zero(moe: SparseMoEFeedForward) -> None:
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0].fill_(1.0)


def test_dense_to_moe_identity_top1() -> None:
    model = tiny_model()
    dense_mlp = copy.deepcopy(model.model.layers[0].mlp)
    convert_single_layer_to_moe(model, 0, top_k=1)
    moe = model.model.layers[0].mlp
    force_expert_zero(moe)
    states = torch.ones(2, 3, 16)
    torch.testing.assert_close(moe(states), dense_mlp(states), rtol=1e-5, atol=1e-6)


def test_dense_to_moe_identity_top2() -> None:
    model = tiny_model()
    dense_mlp = copy.deepcopy(model.model.layers[0].mlp)
    convert_single_layer_to_moe(model, 0, top_k=2)
    states = torch.randn(2, 3, 16)
    torch.testing.assert_close(
        model.model.layers[0].mlp(states), dense_mlp(states), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("top_k", [1, 2])
def test_route_plan_and_sparse_dispatch(top_k: int) -> None:
    model = tiny_model()
    convert_single_layer_to_moe(model, 0, top_k=top_k)
    moe = model.model.layers[0].mlp
    states = torch.randn(2, 4, 16)
    output = moe(states)
    plan = moe.last_route_plan
    assert output.shape == states.shape
    assert plan is not None
    assert plan.expert_indices.shape == (8, top_k)
    torch.testing.assert_close(plan.routing_weights.sum(-1), torch.ones(8))


def test_router_is_deterministic_and_scores_in_fp32() -> None:
    model = tiny_model()
    convert_single_layer_to_moe(model, 0, top_k=2)
    moe = model.model.layers[0].mlp
    states = torch.randn(1, 3, 16, dtype=torch.bfloat16)
    first = moe.route(states)
    second = moe.route(states)
    assert first.router_logits.dtype is torch.float32
    assert torch.equal(first.expert_indices, second.expert_indices)
    torch.testing.assert_close(first.routing_weights, second.routing_weights)


def test_padded_mask_balancing_loss_backpropagates() -> None:
    logits = torch.randn(2, 3, 2, requires_grad=True)
    loss = load_balancing_loss(
        logits, num_experts=2, top_k=2, attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_checkpoint_round_trip(tmp_path) -> None:
    model = tiny_model()
    convert_single_layer_to_moe(model, 0, top_k=2)
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = model(inputs).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = LlamaMoEForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(inputs).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_synthetic_backward_reaches_router_and_experts() -> None:
    model = tiny_model().train()
    convert_single_layer_to_moe(model, 0, top_k=2)
    result = model(torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]))
    result.loss.backward()
    moe = model.model.layers[0].mlp
    assert moe.router.weight.grad is not None
    assert all(expert.mlp.gate_proj.weight.grad is not None for expert in moe.experts)


def test_synthetic_generation_uses_converted_layer() -> None:
    model = tiny_model()
    convert_single_layer_to_moe(model, 0, top_k=1)
    tokens = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False)
    assert tokens.shape == (1, 5)


def test_only_selected_experts_execute() -> None:
    model = tiny_model()
    convert_single_layer_to_moe(model, 0, top_k=1)
    moe = model.model.layers[0].mlp
    force_expert_zero(moe)
    calls = [0, 0]
    for index, expert in enumerate(moe.experts):
        original = expert.forward

        def counted(states, original=original, index=index):
            calls[index] += 1
            return original(states)

        expert.forward = counted
    moe(torch.ones(1, 2, 16))
    assert calls == [1, 0]
