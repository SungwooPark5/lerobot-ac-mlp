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
"""BiMamba decoder with a Bi-Mamba+ style complementary forget gate.

Subclass of acm_bimamba, not of acm: this variant's backward stack reverses the
whole concatenated sequence (`combined_seq.flip(1)`), the same scan acm_bimamba
uses and not the action-only flip acm uses. So the parent is
BiMambaWholeFlipDecoder, and the only thing added on top is the gate.

The gate runs on the sliced action tokens, before the decoder's final LayerNorm:

    local = SiLU(Conv1d(LayerNorm(out)))     local temporal candidate
    g     = sigmoid(Linear(LayerNorm(out)))  how much to keep the BiMamba path
    out   = g * out + (1 - g) * local

`g` is initialised to bimamba_forget_gate_init via the Linear's bias, with the
weight zeroed, so it starts as a constant mix and learns to become token
dependent. It is [B,T,1] when bimamba_gate_scalar, [B,T,D] otherwise.

Note the gate is a *mixture*, not a residual refiner -- the local path competes
with the BiMamba output for the same mass rather than being added on top.

The policy is inherited wholesale: the pre-refactor ACMBiMambaGatePolicy was a
strict subset of ACMPolicy, differing only in which model it constructed.
"""

import math

import torch
from torch import Tensor, nn

from lerobot.policies.acm.modeling_acm import ACM, ACMPolicy
from lerobot.policies.acm_bimamba.modeling_acm_bimamba import BiMambaWholeFlipDecoder
from lerobot.policies.acm_bimamba_gate.configuration_acm_bimamba_gate import ACMBiMambaGateConfig


class BiMambaForgetGateDecoder(BiMambaWholeFlipDecoder):
    """Whole-flip BiMamba decoder plus a complementary local/global forget gate."""

    def __init__(self, config: ACMBiMambaGateConfig):
        super().__init__(config)

        self.use_bimamba_forget_gate = getattr(config, "use_bimamba_forget_gate", False)

        if self.use_bimamba_forget_gate:
            bimamba_local_kernel_size = int(getattr(config, "bimamba_local_kernel_size", 3))
            assert bimamba_local_kernel_size % 2 == 1, "bimamba_local_kernel_size must be odd"

            self.bimamba_fg_norm = nn.LayerNorm(config.dim_model)

            self.bimamba_fg_local_conv = nn.Conv1d(
                in_channels=config.dim_model,
                out_channels=config.dim_model,
                kernel_size=bimamba_local_kernel_size,
                padding=bimamba_local_kernel_size // 2,
                groups=1,
            )
            self.bimamba_fg_local_act = nn.SiLU()

            self.bimamba_gate_scalar = getattr(config, "bimamba_gate_scalar", True)
            gate_dim = 1 if self.bimamba_gate_scalar else config.dim_model
            self.bimamba_fg_gate = nn.Linear(config.dim_model, gate_dim)

            gate_init = float(getattr(config, "bimamba_forget_gate_init", 0.5))
            gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
            gate_bias = math.log(gate_init / (1.0 - gate_init))

            # Start from a constant gate. The model then learns token-dependent values.
            nn.init.zeros_(self.bimamba_fg_gate.weight)
            nn.init.constant_(self.bimamba_fg_gate.bias, gate_bias)
        else:
            self.bimamba_fg_norm = None
            self.bimamba_fg_local_conv = None
            self.bimamba_fg_local_act = None
            self.bimamba_fg_gate = None
            self.bimamba_gate_scalar = True

    def _post_scan(self, out: Tensor) -> Tensor:
        if not (self.use_bimamba_decoder and self.use_bimamba_forget_gate):
            return out

        bi_out = out

        fg_norm = self.bimamba_fg_norm(out)  # [B, T, D]

        local_out = fg_norm.transpose(1, 2)  # [B, D, T]
        local_out = self.bimamba_fg_local_conv(local_out)
        local_out = self.bimamba_fg_local_act(local_out)
        local_out = local_out.transpose(1, 2)  # [B, T, D]

        gate = torch.sigmoid(self.bimamba_fg_gate(fg_norm))  # [B,T,1] or [B,T,D]

        return gate * bi_out + (1.0 - gate) * local_out


class ACMBiMambaGate(ACM):
    """ACM with the gated whole-flip BiMamba decoder."""

    def _build_mamba_decoder(self, config: ACMBiMambaGateConfig) -> BiMambaForgetGateDecoder:
        return BiMambaForgetGateDecoder(config)


class ACMBiMambaGatePolicy(ACMPolicy):
    config_class = ACMBiMambaGateConfig
    name = "acm_bimamba_gate"

    def _build_model(self, config: ACMBiMambaGateConfig) -> ACMBiMambaGate:
        return ACMBiMambaGate(config)
