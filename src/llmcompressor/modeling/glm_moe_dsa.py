from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from compressed_tensors.offload import disable_onloading
from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )


@MoECalibrationModule.register("GlmMoeDsaMoE")
class CalibrationGlmMoeDsaMoE(MoECalibrationModule):
    """
    Calibration version of GlmMoeDsaMoE that unpacks packed expert weights into
    individual MLP modules so GPTQ/AWQ can match `Linear` layers.
    """

    is_permanent = True

    def __init__(
        self,
        original,
        config: "GlmMoeDsaConfig",
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.experts = SequentialGlmMoeDsaExperts(
            config, original.experts, original.shared_experts.__class__
        )
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                topk_indices, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            has_tokens = token_idx.numel() > 0

            if self.calibrate_all_experts:
                expert_out_all = self.experts[expert_idx](hidden_states)
                if not has_tokens:
                    continue
                expert_out = expert_out_all[token_idx]
            else:
                if not has_tokens:
                    continue
                expert_out = self.experts[expert_idx](hidden_states[token_idx])

            weighted_output = expert_out * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, weighted_output.to(final_hidden_states.dtype)
            )

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class SequentialGlmMoeDsaExperts(torch.nn.ModuleList):
    def __init__(self, config: "GlmMoeDsaConfig", original, mlp_cls):
        self.num_experts = config.num_local_experts
        with skip_weights_initialize():
            super().__init__(
                [
                    mlp_cls(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        with disable_onloading():
            gate_up = original.gate_up_proj
            down = original.down_proj

            for i in range(self.num_experts):
                gate_up_i = gate_up[i]
                down_i = down[i]

                gate_proj, up_proj = gate_up_i.chunk(2, dim=0)

                self[i].gate_proj.weight.data = gate_proj.contiguous().clone()
                self[i].up_proj.weight.data = up_proj.contiguous().clone()
                self[i].down_proj.weight.data = down_i.contiguous().clone()
