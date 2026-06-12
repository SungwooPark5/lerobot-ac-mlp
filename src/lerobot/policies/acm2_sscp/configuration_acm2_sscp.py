"""Configuration for ACM2 + SSM State Carryover Protocol (SSCP).

Mamba-2 (SSD) port of acm3_sscp. SSCP mechanism (inference):
  After chunk n, the Mamba-2 decoder's terminal output token (B, 1, D) is saved
  as a carry vector.  At the start of chunk n+1 this carry is prepended to the
  combined sequence so the SSM warms up from the previous chunk's context instead
  of starting from h=0.

  SSM-specific: prepending carry_n causes its hidden state to propagate into all
  subsequent positions via Mamba-2's sequential SSD scan.  In a Transformer this
  would merely be an additional attention key with no hidden-state warmup effect.

Training with Chunk-Continuation pairs:
  When sscp_p_carry > 0 the trainer supplies consecutive chunk pairs.
  The carry extracted from chunk n is detached and prepended to chunk n+1.
  Loss is computed on both chunks.

Note: this is the *summary-token* carry (re-derives state by scanning the token);
the *literal* recurrent-state carry lives in acm2_sscp_literal.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2.configuration_acm2 import ACM2Config


@PreTrainedConfig.register_subclass("acm2_sscp")
@dataclass
class ACM2SSCPConfig(ACM2Config):
    """ACM2 + SSM State Carryover Protocol (summary-token warm-up).

    Note: SSCP carries a summary token (previous chunk's terminal decoder output),
    not Mamba-2's literal recurrent state tensor — it warms up the scan rather than
    transferring the SSD state verbatim.
    """

    # Inference
    sscp_enabled: bool = True     # if False, behaves identically to ACM2

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
