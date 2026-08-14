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


@PreTrainedConfig.register_subclass("acm2_bimamba")
@dataclass
class ACM2BiMambaConfig(ACM2Config):
    """ACM2 with the bidirectional Mamba-2 decoder on by default.

    Mamba2ACMDecoder already implements the scan; ACM2Config just defaults the
    flag to False. Selecting this policy type is what turns it on.

    The backward stack reverses the whole concatenated sequence, encoder context
    included -- the same scan acm_bimamba uses, not the action-only flip in acm.
    ACM2Config declares bimamba_action_only_flip but nothing reads it.

    Action self-attention stays at its ACM2Config default (off): use
    acm2_self_atten for that, or set the flag here to combine the two.
    """

    use_bimamba_decoder: bool = True
