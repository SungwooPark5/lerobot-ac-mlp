"""Configuration for ACT + Intra-Chunk Phase Embedding (ICPE) — control experiment.

This adds the identical ICPE signal used in ACM3ICPE to ACT's Transformer decoder.
Expected result: no meaningful improvement over vanilla ACT.

Hypothesis: ACT generates all K actions in parallel via global self-attention,
so every position already has access to every other position. The ICPE signal is
therefore redundant — the Transformer can derive positional context from its
learnable pos embeddings and cross-attention. This is in contrast to Mamba3, which
processes positions sequentially and can benefit from an explicit phase signal.

This control experiment confirms that ICPE's effectiveness is SSM-specific.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_icpe")
@dataclass
class ACTICPEConfig(ACTConfig):
    """ACT with Intra-Chunk Phase Embedding (control experiment).

    Adds:
        icpe_mode:       Signal variant matching ACM3ICPE ("full" | "sincos" | "linear").
        icpe_scale_init: Same small-scale initialisation as in ACM3ICPE.
    """

    icpe_mode: str = "full"
    icpe_scale_init: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.icpe_mode not in ("full", "sincos", "linear"):
            raise ValueError(f"icpe_mode must be 'full', 'sincos', or 'linear'. Got '{self.icpe_mode}'.")

    @property
    def icpe_dim(self) -> int:
        return 4 if self.icpe_mode == "full" else 2
