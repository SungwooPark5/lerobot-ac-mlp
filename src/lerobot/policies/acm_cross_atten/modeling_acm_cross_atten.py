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
"""ACM with pre-cross-attention in the decoder.

This variant contributes no model code. Its decoder was acm's MambaACMDecoder
with the BiMamba branch and the action self-attention block dropped, and its
pre-cross-attention block was identical to acm's line for line -- so with
use_bimamba_decoder=False (the default) acm's decoder already takes the same
path, over the same parameters, in the same order. ACMCrossPolicy was likewise a
strict subset of ACMPolicy.

All that is left is the name and the config class, which is why there is no ACM
subclass and no decoder here.

Two things about the pre-refactor file worth recording, since neither is visible
once it is deleted:

  - It could not be imported. It did `from ...configuration_acm_cross_atten
    import ACMCrossConfig`, and the config module defines ACMCrossAttenConfig.
    No such name has ever existed in the tree.

  - Even past that, it could not be constructed: its config never declared
    mamba_d_state, mamba_d_conv or mamba_expand while its decoder read all three.

So this policy has never run. The equivalence test compares against a repaired
copy -- import fixed, the three fields restored at ACMConfig's values -- to check
what it would have computed had it been runnable.

One behavioural note: use_bimamba_decoder was not a field on the old config, so
the BiMamba path was unreachable. It is reachable now through inheritance, and
setting it True gives a different decoder than the pre-refactor file could ever
produce. At its default the two agree.
"""

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.acm_cross_atten.configuration_acm_cross_atten import ACMCrossAttenConfig


class ACMCrossPolicy(ACMPolicy):
    config_class = ACMCrossAttenConfig
    name = "acm_cross_atten"
