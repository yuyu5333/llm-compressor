from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule

if TYPE_CHECKING:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )


@MoECalibrationModule.register("GlmMoeDsaMoE")
class CalibrationGlmMoeDsaMoE(MoECalibrationModule):
    """
    Calibration version of GlmMoeDsaMoE that uses the original packed expert
    weights directly via F.linear, avoiding memory-heavy weight cloning.
    """

    is_permanent = True

    def __init__(
        self,
        original,
        config: "GlmMoeDsaConfig",
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.experts = original.experts
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def _expert_forward(self, hidden_states, expert_idx):
        gate, up = F.linear(
            hidden_states, self.experts.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        return F.linear(self.experts.act_fn(gate) * up, self.experts.down_proj[expert_idx])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(self.num_experts):
            token_idx, top_k_pos = torch.where(expert_mask[expert_idx])
            has_tokens = token_idx.numel() > 0

            if self.calibrate_all_experts:
                expert_output = self._expert_forward(hidden_states, expert_idx)

                if has_tokens:
                    expert_weights = topk_weights[token_idx, top_k_pos]
                    routed_output = expert_output[token_idx] * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_idx, routed_output)
            else:
                if has_tokens:
                    expert_output = self._expert_forward(hidden_states[token_idx], expert_idx)
                    expert_weights = topk_weights[token_idx, top_k_pos]
                    routed_output = expert_output * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_idx, routed_output)

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
