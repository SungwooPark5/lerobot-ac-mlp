"""Configuration for ACM3 + SSM State Carryover Protocol (SSCP).

SSCP mechanism (inference):
  After chunk n, the Mamba3 decoder's terminal output token (B, 1, D) is saved
  as a carry vector.  At the start of chunk n+1 this carry is prepended to the
  combined sequence so the SSM warms up from the previous chunk's context instead
  of starting from h=0.

  SSM-specific: prepending carry_n causes its hidden state to propagate into all
  subsequent positions via Mamba3's sequential scan.  In a Transformer this would
  merely be an additional attention key with no hidden-state warmup effect.

Training with Chunk-Continuation pairs:
  When sscp_p_carry > 0 the trainer supplies consecutive chunk pairs.
  The carry extracted from chunk n is detached and prepended to chunk n+1.
  Loss is computed on both chunks.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3.configuration_acm3 import ACM3Config


@PreTrainedConfig.register_subclass("acm3_sscp")
@dataclass
class ACM3SSCPConfig(ACM3Config):
    """ACM3 + SSM State Carryover Protocol (no ICPE).

    Note: SSCP carries a summary token (previous chunk's terminal decoder output),
    not Mamba3's literal recurrent state tensor — it warms up the scan rather than
    transferring h verbatim.
    """

    # Inference
    sscp_enabled: bool = True     # if False, behaves identically to ACM3

    # Training
    sscp_p_carry: float = 0.5    # probability of chunk-continuation training per batch
    sscp_detach: bool = True     # detach carry at chunk boundary (True = stable)

    # Carry placement inside the combined sequence.
    #   "pre_query": [encoder_out, carry, queries]  — carry adjacent to the queries
    #                so the SSM state it seeds is NOT washed out by the (long) encoder
    #                token stream before reaching the action queries. (recommended)
    #   "prefix":    [carry, encoder_out, queries]  — original placement; carry is
    #                diluted across all encoder tokens. Kept for the placement ablation.
    sscp_carry_position: str = "pre_query"  # "pre_query" | "prefix"
