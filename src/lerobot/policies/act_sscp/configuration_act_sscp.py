"""Configuration for ACT + SSM-State-Carryover-style carry (control experiment).

It applies the SAME carry mechanism used by acm3_sscp — the terminal decoder
output token from chunk n is made available to chunk n+1 — but to ACT's
Transformer decoder.

Expected result: little/no improvement over vanilla ACT, because for a Transformer
the carry is merely one extra cross-attention key/value; it does NOT warm up any
recurrent hidden state (Transformers are stateless across the sequence).  This
control is what licenses the paper's claim that the carry benefit is SSM-specific.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_sscp")
@dataclass
class ACTSSCPConfig(ACTConfig):
    """ACT + SSCP carry (control experiment for C2)."""

    # Inference
    sscp_enabled: bool = True   # if False, behaves identically to ACT

    # Training
    sscp_p_carry: float = 0.5   # probability of chunk-continuation training per batch
    sscp_detach: bool = True    # detach carry at chunk boundary (True = stable)
