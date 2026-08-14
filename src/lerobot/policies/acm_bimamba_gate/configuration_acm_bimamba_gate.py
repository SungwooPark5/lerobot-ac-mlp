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


@PreTrainedConfig.register_subclass("acm_bimamba_gate")
@dataclass
class ACMBiMambaGateConfig(ACMConfig):
    """Config for the BiMamba decoder with a complementary forget gate.

    The fields this variant actually uses -- use_bimamba_forget_gate,
    bimamba_forget_gate_init, bimamba_local_kernel_size, bimamba_gate_scalar --
    are already declared in ACMConfig, which is where the copy got them. ACMConfig
    declares them but never reads them: no decoder in acm implements the gate, so
    they are dead there and live only here.

    Inheriting also brings the fields the copy had fallen behind on (the action
    conv refiner, the accel/jerk and delta penalties, action self-attention,
    pre-cross attention, MoE fusion). All are gated on flags defaulting to False.
    """
