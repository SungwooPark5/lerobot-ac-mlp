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


@PreTrainedConfig.register_subclass("acm_accel_loss")
@dataclass
class ACMAccelConfig(ACMConfig):
    """ACM with the action-acceleration penalty.

    Adds nothing and changes no default. action_accel_weight and
    action_accel_front_steps are in ACMConfig with the same defaults, and
    ACMConfig has 39 further fields this copy never picked up.

    action_accel_weight defaults to 0.0, so this policy type on its own applies
    no penalty -- it is plain acm until the weight is set. That was true before
    the refactor too.
    """
