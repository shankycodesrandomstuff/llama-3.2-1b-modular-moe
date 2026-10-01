"""Route data. Execution backend stays elsewhere."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutePlan:
    """Routes for flattened `[batch * sequence, hidden]` tokens."""

    expert_indices: torch.Tensor
    routing_weights: torch.Tensor
    router_logits: torch.Tensor
    batch_shape: tuple[int, int]

    def validate(self, num_experts: int) -> None:
        if self.expert_indices.ndim != 2 or self.routing_weights.shape != self.expert_indices.shape:
            raise ValueError(
                "expert_indices and routing_weights must have equal [tokens, top_k] shape"
            )
        if self.router_logits.shape != (self.expert_indices.shape[0], num_experts):
            raise ValueError("router_logits must have shape [tokens, num_experts]")
        if self.expert_indices.numel() and (
            self.expert_indices.min() < 0 or self.expert_indices.max() >= num_experts
        ):
            raise ValueError("route plan contains an invalid expert index")
