"""Configuration for ACM2 + Intra-Chunk Phase Embedding (ICPE).

Mamba-2 (SSD) port of acm3_icpe. ICPE injects a continuous temporal position
signal into each decoder query token, enabling the SSM to modulate its generation
based on position within the chunk.

Signal φ(i, K):
    "full"   (4D): [sin(2πi/K), cos(2πi/K), i/K, (K-i)/K]
    "sincos" (2D): [sin(2πi/K), cos(2πi/K)]
    "linear" (2D): [i/K, (K-i)/K]

Why SSM-specific: ACM2 generates actions via the Mamba-2 SSD scan, so position i
is influenced by accumulated hidden state from 0..i-1. Making position explicit
allows the model to calibrate uncertainty at each step. ACT (Transformer) already
sees all positions simultaneously, so ICPE has no effect (control experiment).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2.configuration_acm2 import ACM2Config


@PreTrainedConfig.register_subclass("acm2_icpe")
@dataclass
class ACM2ICPEConfig(ACM2Config):
    """ACM2 with Intra-Chunk Phase Embedding.

    Adds:
        icpe_mode: Signal variant — "full" (4D), "sincos" (2D), "linear" (2D).
        icpe_scale_init: Initial scale of icpe_proj output relative to pos_embed.
                         Small init prevents disrupting pretrained pos embeddings.
    """

    icpe_mode: str = "full"        # "full" | "sincos" | "linear"
    icpe_scale_init: float = 0.1   # weight init scale for icpe_proj

    def __post_init__(self):
        super().__post_init__()
        if self.icpe_mode not in ("full", "sincos", "linear"):
            raise ValueError(
                f"icpe_mode must be 'full', 'sincos', or 'linear'. Got '{self.icpe_mode}'."
            )

    @property
    def icpe_dim(self) -> int:
        """Dimension of the raw phase signal before projection."""
        return 4 if self.icpe_mode == "full" else 2
