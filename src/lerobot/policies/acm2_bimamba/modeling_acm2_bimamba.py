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
"""ACM2 with the BiMamba decoder enabled.

No model code: Mamba2ACMDecoder already branches on use_bimamba_decoder, so the
variant is the config default and the name.

Replaces acm2_bimamba_self_atten, which was a byte-identical copy of acm2 apart
from its `name` string -- both flags it is named after were left at False, so it
enabled neither. This half enables BiMamba; acm2_self_atten enables the other.
"""

from lerobot.policies.acm2.modeling_acm2 import ACM2Policy
from lerobot.policies.acm2_bimamba.configuration_acm2_bimamba import ACM2BiMambaConfig


class ACM2BiMambaPolicy(ACM2Policy):
    config_class = ACM2BiMambaConfig
    name = "acm2_bimamba"
