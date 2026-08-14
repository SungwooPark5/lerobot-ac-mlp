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


@PreTrainedConfig.register_subclass("acm_self_atten")
@dataclass
class ACMSelfConfig(ACMConfig):
    """ACM with action-token self-attention on by default.

    This is the whole variant. Both fields exist in ACMConfig; the only thing the
    copy did differently was default them to True, which is what selecting this
    policy type buys you. Unlike acm_cross_atten and acm_moe -- where the feature
    flag stayed False and the type on its own changed nothing -- this one is
    self-enabling.
    """

    use_action_self_attention: bool = True
    action_self_attention_use_gate: bool = True
