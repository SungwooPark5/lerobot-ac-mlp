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


@PreTrainedConfig.register_subclass("acm_refiner")
@dataclass
class ACMRefineConfig(ACMConfig):
    """Config for the pre-norm-residual Mamba decoder variant of ACM.

    Field-for-field this used to be a copy of ACMConfig that had fallen behind:
    every field it declared was already in ACMConfig with the same default, and
    ACMConfig had since grown a set the copy never picked up (the BiMamba block,
    pre-cross attention, action self-attention, MoE fusion). Inheriting adds
    exactly those, and every one of them is gated on its own `use_*` flag that
    defaults to False -- so a config built here behaves as before.

    Note the two are not interchangeable even so: `acm` reads action_accel_weight
    and this variant reads action_jerk_weight, and both fields live in ACMConfig.
    Which one is actually applied is decided by the policy, not the config --
    see ACMRefinePolicy._action_smoothness_loss.
    """
