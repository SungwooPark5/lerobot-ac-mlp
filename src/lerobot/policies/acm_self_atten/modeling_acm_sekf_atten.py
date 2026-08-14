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
"""ACM with action-token self-attention after the Mamba decoder.

No model code. acm's MambaACMDecoder already contains this block, identical down
to `residual + tanh(gamma) * dropout(attn)`; the copy simply lacked the BiMamba
branch, the pre-cross attention and the forget-gate path, all of which acm skips
at their defaults. ACM and ACMSelfPolicy were strict subsets of acm's. The whole
variant is the two config defaults in ACMSelfConfig.

The file name keeps its typo (sekf) so imports elsewhere do not break.

One structural note. acm allocates action_self_attn_gamma whenever
self-attention is on; the copy allocated it only when the gate was also on, and
set it to None otherwise. With action_self_attention_use_gate=True -- this
config's default, and the only setting the copy shipped -- the two agree. Turn
the gate off and acm carries one extra parameter the copy did not.

The pre-refactor file could not be constructed: its config never declared
mamba_d_state, mamba_d_conv or mamba_expand while its decoder reads all three.
"""

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.acm_self_atten.configuration_acm_self_atten import ACMSelfConfig


class ACMSelfPolicy(ACMPolicy):
    config_class = ACMSelfConfig
    name = "acm_self_atten"
