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


@PreTrainedConfig.register_subclass("acm_cross_atten")
@dataclass
class ACMCrossAttenConfig(ACMConfig):
    """Config for the pre-cross-attention variant of ACM.

    Adds nothing. use_pre_cross_attention and pre_cross_attention_gamma_init --
    the two fields this variant exists for -- are already in ACMConfig with the
    same defaults, and ACMConfig has 48 further fields this copy never picked up.

    Both defaults are off (use_pre_cross_attention=False), which is worth saying
    plainly: selecting this policy type does not by itself turn cross attention
    on. That was true before the refactor too.
    """
