"""Local Mixtral-style routing and sparse dispatch without distributed execution."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as functional
from torch import nn

from .route_plan import RoutePlan


class LlamaExpert(nn.Module):
    """A named wrapper around an existing Llama-compatible SwiGLU MLP."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class TopKRouter(nn.Module):
    """Bias-free FP32 linear top-k router with normalized selected probabilities."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 1) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, hidden_states: torch.Tensor) -> RoutePlan:
        if hidden_states.ndim != 3:
            raise ValueError("router expects [batch, sequence, hidden] states")
        batch, sequence, hidden = hidden_states.shape
        tokens = hidden_states.reshape(-1, hidden)
        logits = functional.linear(tokens.float(), self.weight.float())
        probabilities = functional.softmax(logits, dim=-1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return RoutePlan(indices, weights, logits, (batch, sequence))


class SparseMoEFeedForward(nn.Module):
    """Local sparse mixture over cloned Llama FFN experts."""

    def __init__(
        self, dense_mlp: nn.Module, hidden_size: int, num_experts: int = 2, top_k: int = 1
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("Phase 1 requires at least two experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(LlamaExpert(deepcopy(dense_mlp)) for _ in range(num_experts))
        self.last_route_plan: RoutePlan | None = None

    def route(self, hidden_states: torch.Tensor) -> RoutePlan:
        plan = self.router(hidden_states)
        plan.validate(self.num_experts)
        return plan

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        plan = self.route(hidden_states)
        self.last_route_plan = plan
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = torch.zeros_like(flat_states)
        for expert_index, expert in enumerate(self.experts):
            token_indices, route_columns = torch.where(plan.expert_indices == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat_states.index_select(0, token_indices))
            weighted_output = expert_output * plan.routing_weights[token_indices, route_columns].to(
                expert_output.dtype
            ).unsqueeze(-1)
            output.index_add_(0, token_indices, weighted_output)
        return output.reshape_as(hidden_states)


def load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Switch-style router auxiliary loss compatible with local top-k routing.

    This follows the public Hugging Face Mixtral loss semantics while keeping a
    one-layer tensor API. `attention_mask`, if present, must be `[batch, seq]`.
    """
    if router_logits.ndim != 3:
        raise ValueError("router_logits must be [batch, sequence, num_experts]")
    batch, sequence, observed_experts = router_logits.shape
    if observed_experts != num_experts:
        raise ValueError("router_logits expert dimension does not match num_experts")
    probabilities = functional.softmax(router_logits.float(), dim=-1)
    selected = torch.topk(probabilities, top_k, dim=-1).indices
    expert_mask = functional.one_hot(selected, num_classes=num_experts).float()
    if attention_mask is None:
        token_mask = torch.ones((batch, sequence, 1), device=router_logits.device)
    else:
        if attention_mask.shape != (batch, sequence):
            raise ValueError("attention_mask must be [batch, sequence]")
        token_mask = attention_mask.to(probabilities.dtype).unsqueeze(-1)
    valid_tokens = token_mask.sum().clamp_min(1)
    tokens_per_expert = (expert_mask * token_mask.unsqueeze(-1)).sum((0, 1, 2)) / (
        valid_tokens * top_k
    )
    probability_per_expert = (probabilities * token_mask).sum((0, 1)) / valid_tokens
    return (tokens_per_expert * probability_per_expert).sum() * num_experts
