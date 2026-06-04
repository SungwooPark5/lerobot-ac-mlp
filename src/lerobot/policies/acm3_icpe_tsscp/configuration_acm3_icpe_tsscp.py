"""Configuration for ACM3 + ICPE + True SSM State Carryover Protocol (tSSCP).

True SSCP mechanism:
  After chunk n, the Mamba3 decoder's terminal output token (the last decoder
  query position) is extracted as a dense carry vector (B, 1, D). At the start
  of chunk n+1 this carry token is prepended to the combined sequence so the SSM
  warms up from the previous chunk's context rather than h=0.

  This is structurally distinct from ACM3_SSCP (action blending):
    - ACM3_SSCP: post-processes actions with weighted average → ACT-compatible
    - tSSCP:     injects hidden-state context into Mamba3's sequential scan → SSM-only

Training with Chunk-Continuation pairs:
  When `tsscp_use_chunk_pairs` is True, the trainer expects batches with keys
  {obs_n, action_n, action_is_pad_n, obs_n1, action_n1, action_is_pad_n1}.
  The carry extracted from chunk n is detached (no gradient) and prepended to
  chunk n+1's sequence. Loss is computed on both chunks.
  When pairs are unavailable (p_carry=0), falls back to standard training.

  tsscp_p_carry  : probability of using chunk-continuation training per batch
  tsscp_detach   : whether to detach the carry tensor (True = stable, False = BPTT)
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_icpe.configuration_acm3_icpe import ACM3ICPEConfig


@PreTrainedConfig.register_subclass("acm3_icpe_tsscp")
@dataclass
class ACM3ICPETSSCPConfig(ACM3ICPEConfig):
    """ACM3 + ICPE + True SSM State Carryover Protocol."""

    # Inference carry
    tsscp_enabled: bool = True   # if False, behaves identically to ACM3ICPE

    # Training
    tsscp_p_carry: float = 0.5   # probability of chunk-continuation training per batch
    tsscp_detach: bool = True    # detach carry at chunk boundary (True = stable)
