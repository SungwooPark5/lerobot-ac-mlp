from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2.configuration_acm2 import ACM2Config


@PreTrainedConfig.register_subclass("acm2_icpe")
@dataclass
class ACM2ICPEConfig(ACM2Config):
    """ACM2 + Intra-Chunk Phase Embedding."""

    icpe_mode: str = "full"        # "full" | "sincos" | "linear"
    icpe_scale_init: float = 0.1

    def __post_init__(self):
        super().__post_init__()

        if self.icpe_mode not in ("full", "sincos", "linear"):
            raise ValueError(
                f"icpe_mode must be 'full', 'sincos', or 'linear'. Got '{self.icpe_mode}'."
            )

    @property
    def icpe_dim(self) -> int:
        return 4 if self.icpe_mode == "full" else 2