"""Configuration for ACM3 + True SSM State Carryover Protocol (SSCP).

SSCP mechanism (inference):
  After chunk n, the Mamba3 decoder's terminal output token (B, 1, D) is saved
  as a carry vector.  At the start of chunk n+1 this carry is prepended to the
  combined sequence so the SSM warms up from the previous chunk's context instead
  of starting from h=0.

  This is structurally distinct from the v4 action-blending approach:
    - v4 aSSCP:  post-processes actions at the output level (ACT-compatible)
    - True SSCP: injects hidden-state context into Mamba3's sequential scan (SSM-only)

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
    """ACM3 + True SSM State Carryover Protocol (no ICPE)."""

    # Inference
    sscp_enabled: bool = True     # if False, behaves identically to ACM3

    # Training
    sscp_p_carry: float = 0.5    # probability of chunk-continuation training per batch
    sscp_detach: bool = True     # detach carry at chunk boundary (True = stable)
