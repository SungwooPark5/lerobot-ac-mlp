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
"""ACM with a pre-norm residual Mamba decoder and a jerk smoothness loss.

Subclass of the `acm` policy. Two things differ, and nothing else:

1. The decoder. `acm` runs its Mamba layers back to back and applies one
   LayerNorm at the end. This one wraps every layer in pre-norm + residual:

       acm            for layer in layers:  seq = layer(seq)
                      out = norm(seq[-chunk:])

       acm_refiner    for norm, layer in zip(norms, layers):
                          seq = seq + layer(norm(seq))
                      out = norm_final(seq[-chunk:])

   That is a different parameter set (`norms.*` per layer plus `norm_final`,
   versus a single `norm`), so it cannot be folded into acm behind a flag.

2. The smoothness penalty. `acm` penalises acceleration, this one penalises
   jerk -- a third-order difference, which is what gives the action-space conv
   refiner something to optimise against. Both weights live in ACMConfig; which
   one is read is decided here, by overriding _action_smoothness_loss.

Everything else -- backbone, VAE, encoder, l1/KLD, the conv refiner itself and
its delta losses, inference -- is inherited unchanged. Features acm grew after
this variant was branched off (BiMamba, pre-cross attention, action
self-attention, MoE fusion) come along with the inheritance but stay off: each
is gated on a `use_*` flag that defaults to False, and none allocate parameters
or draw from the RNG while off.
"""

import torch
from torch import Tensor, nn

from lerobot.policies.acm.modeling_acm import ACM, ACMPolicy
from lerobot.policies.acm_refiner.configuration_acm_refiner import ACMRefineConfig

try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class PreNormResidualMambaDecoder(nn.Module):
    """Mamba decoder with per-layer pre-norm and a residual connection.

    Not a subclass of acm's MambaACMDecoder: the module layout differs (`norms`
    + `norm_final` here, a single `norm` there), so there is no shared __init__
    to inherit -- only the signature is shared.
    """

    def __init__(self, config: ACMRefineConfig):
        super().__init__()

        if not HAS_MAMBA:
            raise ImportError("Mamba-ssm is not installed. Please install it to use 'use_mamba=true'")

        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=config.dim_model,  # Model dimension
                    d_state=config.mamba_d_state,  # SSM state expansion factor
                    d_conv=config.mamba_d_conv,  # Local convolution width
                    expand=config.mamba_expand,  # Block expansion factor
                )
                for _ in range(config.n_decoder_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.dim_model) for _ in range(config.n_decoder_layers)]
        )

        self.norm_final = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,  # Action Queries (Chunk Size, Batch, Dim)
        encoder_out: Tensor,  # Context (Encoder Seq, Batch, Dim)
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed

        x = x.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)

        combined_seq = torch.cat([encoder_out, x], dim=1)

        for norm, layer in zip(self.norms, self.layers):
            residual = combined_seq
            combined_seq = norm(combined_seq)
            combined_seq = layer(combined_seq)
            combined_seq = combined_seq + residual

        chunk_size = x.shape[1]
        out = combined_seq[:, -chunk_size:, :]

        if self.norm_final is not None:
            out = self.norm_final(out)

        return out.transpose(0, 1)


class ACMRefine(ACM):
    """ACM with the pre-norm residual decoder swapped in."""

    def _build_mamba_decoder(self, config: ACMRefineConfig) -> PreNormResidualMambaDecoder:
        return PreNormResidualMambaDecoder(config)


class ACMRefinePolicy(ACMPolicy):
    """ACM policy with a jerk smoothness penalty instead of acceleration."""

    config_class = ACMRefineConfig
    name = "acm_refiner"

    def _build_model(self, config: ACMRefineConfig) -> ACMRefine:
        return ACMRefine(config)

    def _action_smoothness_loss(self, actions_hat, batch, loss_dict) -> Tensor | None:
        # ------------------------------------------------------------
        # Final action jerk loss
        # Purpose:
        #   Give the refiner a smoothness objective.
        #
        # actions_hat is the final action output from:
        #   decoder_out -> action_head -> raw_actions -> action_conv_refiner
        #
        # Therefore, this loss backpropagates through the refiner.
        #
        # jerk[t] = a[t+3] - 3a[t+2] + 3a[t+1] - a[t]
        # ------------------------------------------------------------
        action_jerk_weight = getattr(self.config, "action_jerk_weight", 0.0)

        if not (action_jerk_weight > 0.0 and actions_hat.shape[1] >= 4):
            return None

        pad_mask = ~batch["action_is_pad"]  # (B, T)

        action_jerk = (
            actions_hat[:, 3:]
            - 3.0 * actions_hat[:, 2:-1]
            + 3.0 * actions_hat[:, 1:-2]
            - actions_hat[:, :-3]
        )

        jerk_mask = (
            pad_mask[:, 3:]
            & pad_mask[:, 2:-1]
            & pad_mask[:, 1:-2]
            & pad_mask[:, :-3]
        )

        jerk_denom = jerk_mask.sum().clamp(min=1) * actions_hat.shape[-1]

        action_jerk_loss = (
            action_jerk.pow(2) * jerk_mask.unsqueeze(-1)
        ).sum() / jerk_denom

        loss_dict["action_jerk_loss"] = action_jerk_loss.item()

        return action_jerk_weight * action_jerk_loss
