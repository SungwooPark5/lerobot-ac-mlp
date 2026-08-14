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
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm.configuration_acm import ACMConfig


@PreTrainedConfig.register_subclass("acm_moe")
@dataclass
class ACMMoEConfig(ACMConfig):
    """Config for the two-decoder mixture variant of ACM.

    The MoE fields -- use_moe_decoder_fusion, moe_gate_init, moe_gate_hidden_dim,
    moe_branch_l1_weight, moe_gate_prior_weight, moe_gate_prior_target -- are all
    declared in ACMConfig already, where nothing reads them. acm has no MoE code
    at all; this is the only place they mean anything.

    use_moe_decoder_fusion defaults to False, so selecting this policy type does
    not by itself turn the mixture on -- it falls back to a single decoder, which
    is plain acm. That was true before the refactor too.
    """
