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
"""ACM with the action-acceleration penalty.

No model code, and no config change either -- this variant is a duplicate of acm.

Its acceleration loss is character for character the block that now lives in
ACMPolicy._action_smoothness_loss: same second difference, same front_steps
handling, same masking, same two loss_dict entries, added to the loss at the same
point. Its decoder, ACM and policy were all strict subsets of acm's, and its
config declared 39 fields fewer and not one acm lacks.

Kept as a registered name so `--policy.type=acm_accel_loss` keeps resolving.
Delete the directory if that name is not wanted; nothing else references it.

The pre-refactor file could not be constructed: its config never declared
mamba_d_state, mamba_d_conv or mamba_expand while its decoder reads all three.
"""

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.acm_accel_loss.configuration_acm_accel import ACMAccelConfig


class ACMAccelPolicy(ACMPolicy):
    config_class = ACMAccelConfig
    name = "acm_accel_loss"
