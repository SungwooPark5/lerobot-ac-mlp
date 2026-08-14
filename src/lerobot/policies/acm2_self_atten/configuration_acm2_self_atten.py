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
from lerobot.policies.acm2.configuration_acm2 import ACM2Config


@PreTrainedConfig.register_subclass("acm2_self_atten")
@dataclass
class ACM2SelfAttenConfig(ACM2Config):
    """ACM2 with action-token self-attention on by default.

    Mamba2ACMDecoder already implements the block -- gated residual attention over
    the last chunk_size tokens, scaled by tanh(gamma) -- and ACM2Config just
    defaults the flag to False. Selecting this policy type is what turns it on.

    action_self_attention_use_gate is left at ACM2Config's default of True, so the
    attention delta is scaled by tanh(gamma) with gamma starting near zero: the
    block begins as almost a no-op and has to earn its contribution.

    The BiMamba decoder stays at its default (off): use acm2_bimamba for that, or
    set the flag here to combine the two.
    """

    use_action_self_attention: bool = True
