#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ACM that runs a Mamba decoder and an ACT decoder and gates between them.

Subclass of acm. Unlike the other variants this one is a real architectural
change rather than a decoder swap: with use_moe_decoder_fusion on, both decoders
run on every forward, each gets its own action head, and a small MLP over the
pooled encoder output mixes their action predictions.

    mamba_actions = action_head(decoder_mamba(...))
    act_actions   = action_head_act(decoder_act(...))
    g             = sigmoid(moe_gate(encoder_out.mean(0)))    # (B, 1, 1)
    actions       = g * act_actions + (1 - g) * mamba_actions

`g` starts at moe_gate_init (0.8, favouring the ACT branch) via the last Linear's
bias, weight zeroed. Training adds two terms: a branch L1 that supervises each
expert directly, and an MSE pulling the gate toward moe_gate_prior_target. The
prior is computed from a detached encoder context so it updates the gate MLP
only, never the encoder -- that detach is load-bearing and is preserved here.

It hangs off three hooks on acm, all no-ops there:

  _build_decoders     allocate two decoders instead of one, at the same point
  _build_extra_heads  the second action head and the gate MLP, right after
                      action_head -- acm's last RNG-consuming module
  _decode_to_actions  the dual decode and the mixture
  _extra_losses       the branch-L1 and gate-prior terms

The first two exist for RNG order: building the extra modules after
super().__init__ instead would shift every draw after them and change the initial
weights for a given seed. _build_extra_heads also has to run before
_reset_parameters, which reaches into action_head_act.

Note the pre-refactor file could not be constructed: its config never declared
mamba_d_state, mamba_d_conv or mamba_expand while its decoder reads all three.
The frozen copy used by the equivalence test has them restored at ACMConfig's
values.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.acm.modeling_acm import ACM, ACMPolicy, ACTDecoder
from lerobot.policies.acm_moe.configuration_acm_moe import ACMMoEConfig
from lerobot.utils.constants import ACTION


class ACMMoE(ACM):
    """ACM with a Mamba expert, an ACT expert, and a gate between them."""

    def _build_decoders(self, config: ACMMoEConfig) -> None:
        self.use_moe_decoder_fusion = getattr(config, "use_moe_decoder_fusion", False)

        if self.use_moe_decoder_fusion:
            # Expert 1: Mamba/ACM decoder
            self.decoder_mamba = self._build_mamba_decoder(config)

            # Expert 2: Transformer/ACT decoder
            self.decoder_act = ACTDecoder(config)

            # In MoE mode, self.decoder is unused.
            self.decoder = None
        else:
            super()._build_decoders(config)

            self.decoder_mamba = None
            self.decoder_act = None

    def _build_extra_heads(self, config: ACMMoEConfig) -> None:
        if not getattr(config, "use_moe_decoder_fusion", False):
            self.action_head_act = None
            self.moe_gate = None
            return

        self.action_head_act = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        gate_hidden_dim = getattr(config, "moe_gate_hidden_dim", 128)

        self.moe_gate = nn.Sequential(
            nn.LayerNorm(config.dim_model),
            nn.Linear(config.dim_model, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )

        # Initialize gate to prefer ACT/Transformer branch.
        # gate = sigmoid(logit), so logit(0.8) ≈ 1.386
        gate_init = float(getattr(config, "moe_gate_init", 0.8))
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        gate_bias = math.log(gate_init / (1.0 - gate_init))

        nn.init.zeros_(self.moe_gate[-1].weight)
        nn.init.constant_(self.moe_gate[-1].bias, gate_bias)

    def _reset_parameters(self):
        """Xavier initialization for Transformer parts only.
        Do not overwrite Mamba's internal initialization.
        """

        if not getattr(self.config, "use_moe_decoder_fusion", False):
            # Single-decoder mode is plain acm.
            super()._reset_parameters()
            return

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Do not overwrite Mamba internal initialization.
        # Initialize Transformer/ACT decoder branch.
        for p in self.decoder_act.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize both action heads consistently.
        for p in self.action_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.action_head_act.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Do not touch moe_gate final layer after bias initialization.
        # Its final layer is intentionally initialized to fixed prior gate.

    def _decode_to_actions(self, decoder_in, encoder_out, encoder_in_pos_embed) -> Tensor:
        if not getattr(self.config, "use_moe_decoder_fusion", False):
            return super()._decode_to_actions(decoder_in, encoder_out, encoder_in_pos_embed)

        mamba_decoder_out = self.decoder_mamba(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        act_decoder_out = self.decoder_act(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, T, D).
        mamba_decoder_out = mamba_decoder_out.transpose(0, 1)
        act_decoder_out = act_decoder_out.transpose(0, 1)

        # Branch-wise action predictions.
        mamba_actions = self.action_head(mamba_decoder_out)
        act_actions = self.action_head_act(act_decoder_out)

        # Main gate: used for final action.
        # Gradient from final L1 can update encoder + gate MLP.
        gate_context = encoder_out.mean(dim=0)  # (B, D)
        gate = torch.sigmoid(self.moe_gate(gate_context)).view(-1, 1, 1)

        raw_actions = gate * act_actions + (1.0 - gate) * mamba_actions

        # Prior gate: used only for gate-prior regularization.
        # Detach encoder context so prior loss updates gate MLP only, not encoder.
        gate_context_for_prior = encoder_out.mean(dim=0).detach()
        gate_for_prior = torch.sigmoid(self.moe_gate(gate_context_for_prior)).view(-1, 1, 1)

        self.last_mamba_actions = mamba_actions
        self.last_act_actions = act_actions
        self.last_moe_gate = gate
        self.last_moe_gate_for_prior = gate_for_prior

        return raw_actions


class ACMMoEPolicy(ACMPolicy):
    config_class = ACMMoEConfig
    name = "acm_moe"

    def _build_model(self, config: ACMMoEConfig) -> ACMMoE:
        return ACMMoE(config)

    def _extra_losses(self, actions_hat, batch, weights, loss_dict) -> list[Tensor]:
        if not getattr(self.config, "use_moe_decoder_fusion", False):
            return []

        mamba_actions = getattr(self.model, "last_mamba_actions", None)
        act_actions = getattr(self.model, "last_act_actions", None)
        moe_gate = getattr(self.model, "last_moe_gate", None)
        moe_gate_for_prior = getattr(self.model, "last_moe_gate_for_prior", None)

        moe_branch_l1_loss = None
        moe_gate_prior_loss = None

        if mamba_actions is not None and act_actions is not None:
            branch_weights = (
                (~batch["action_is_pad"]).unsqueeze(-1).to(actions_hat.dtype) * weights
            )
            mamba_l1 = (
                F.l1_loss(batch[ACTION], mamba_actions, reduction="none") * branch_weights
            ).mean()
            act_l1 = (
                F.l1_loss(batch[ACTION], act_actions, reduction="none") * branch_weights
            ).mean()
            moe_branch_l1_loss = 0.5 * (mamba_l1 + act_l1)

            loss_dict["moe_mamba_l1_loss"] = mamba_l1.item()
            loss_dict["moe_act_l1_loss"] = act_l1.item()
            loss_dict["moe_branch_l1_loss"] = moe_branch_l1_loss.item()

        if moe_gate is not None:
            loss_dict["moe_gate_mean"] = moe_gate.mean().item()
            loss_dict["moe_gate_min"] = moe_gate.min().item()
            loss_dict["moe_gate_max"] = moe_gate.max().item()

        if moe_gate_for_prior is not None:
            gate_target = float(getattr(self.config, "moe_gate_prior_target", 0.8))
            gate_target_tensor = torch.full_like(moe_gate_for_prior, gate_target)
            moe_gate_prior_loss = F.mse_loss(moe_gate_for_prior, gate_target_tensor)
            loss_dict["moe_gate_prior_loss"] = moe_gate_prior_loss.item()

        terms = []
        if moe_branch_l1_loss is not None:
            terms.append(self.config.moe_branch_l1_weight * moe_branch_l1_loss)
        if moe_gate_prior_loss is not None:
            terms.append(self.config.moe_gate_prior_weight * moe_gate_prior_loss)
        return terms
