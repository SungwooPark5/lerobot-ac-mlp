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
"""ACM with a whole-sequence-flip BiMamba decoder.

Subclass of the `acm` policy. Everything -- backbone, VAE, encoder, losses,
inference -- is inherited; only the decoder's backward scan differs.

The difference is one line, and it is NOT cosmetic:

    acm (MambaACMDecoder)      backward_input = cat([encoder_tokens,
                                                     action_tokens.flip(1)])
    acm_bimamba (this file)    backward_seq   = combined_seq.flip(1)

`acm` keeps the encoder context in reading order and reverses only the action
queries. This file reverses the entire concatenated sequence, encoder tokens
included -- the Vim-style external bidirectional scan the variant was written
to test. Neither is a special case of the other, so this decoder cannot be
folded into `acm` behind a flag without changing one of the two models.

Everything `acm` gained after this variant was branched off (action-space conv
refiner, action-acceleration loss, decoder action self-attention, pre-cross
attention) is inherited but stays inert: each is gated on a config field that
ACMBiMambaConfig does not define, so `getattr(config, ..., False)` disables it.
Set any of them on the config and it will switch on -- that is a change in
behaviour relative to the pre-refactor file, not an accident of inheritance.
"""

import torch
from torch import Tensor

from lerobot.policies.acm.modeling_acm import (
    ACM,
    ACMPolicy,
    MambaACMDecoder,
)
from lerobot.policies.acm_bimamba.configuration_acm_bimamba import ACMBiMambaConfig


class BiMambaWholeFlipDecoder(MambaACMDecoder):
    """MambaACMDecoder whose backward stack scans the whole sequence reversed.

    __init__ is inherited unchanged -- the forward/backward Mamba stacks, the
    LayerNorm, and the (unused here) attention blocks are all built identically.
    Only the scan is overridden.
    """

    def forward(
        self,
        x: Tensor,  # Action Queries (Chunk Size, Batch, Dim)
        encoder_out: Tensor,  # Context (Encoder Seq, Batch, Dim)
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed

        x = x.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)

        combined_seq = torch.cat([encoder_out, x], dim=1)

        if self.use_bimamba_decoder:
            # Vim-style external bidirectional Mamba.
            # Forward direction.
            forward_seq = combined_seq
            for layer in self.forward_layers:
                forward_seq = layer(forward_seq)

            # Backward direction.
            backward_seq = combined_seq.flip([1])
            for layer in self.backward_layers:
                backward_seq = layer(backward_seq)
            backward_seq = backward_seq.flip([1])

            # Use average instead of sum to keep activation scale close to the original decoder.
            combined_seq = 0.5 * (forward_seq + backward_seq)
        else:
            for layer in self.layers:
                combined_seq = layer(combined_seq)

        chunk_size = x.shape[1]
        out = combined_seq[:, -chunk_size:, :]  # (B, T, D)

        out = self._post_scan(out)

        if self.norm is not None:
            out = self.norm(out)

        return out.transpose(0, 1)

    def _post_scan(self, out: Tensor) -> Tensor:
        """Hook between slicing the action tokens and the final LayerNorm.

        Identity here. acm_bimamba_gate mixes in a local-conv candidate at this
        point, which is why it is a hook and not inlined -- the gate has to see
        the un-normalised scan output, exactly as it did before the refactor.
        """
        return out


class ACMBiMamba(ACM):
    """ACM with the whole-flip decoder swapped in."""

    def _build_mamba_decoder(self, config: ACMBiMambaConfig) -> BiMambaWholeFlipDecoder:
        return BiMambaWholeFlipDecoder(config)


class ACMBimambaPolicy(ACMPolicy):
    """ACM policy wrapping the whole-flip BiMamba decoder."""

    config_class = ACMBiMambaConfig
    name = "acm_bimamba"

    def _build_model(self, config: ACMBiMambaConfig) -> ACMBiMamba:
        return ACMBiMamba(config)
