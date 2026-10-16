from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_icpe_sscp.configuration_acm2_icpe_sscp import ACM2ICPESSCPConfig


@PreTrainedConfig.register_subclass("acm2_bimamba_action_icpe_sscp")
@dataclass
class ACM2BiMambaActionICPESSCPConfig(ACM2ICPESSCPConfig):
    """ACM2 + Action-only BiMamba + ICPE + SSCP."""

    # 고정 의미: action token만 backward 방향으로 flip
    bimamba_flip_mode: str = "action_only"

    def __post_init__(self):
        super().__post_init__()

        if self.bimamba_flip_mode != "action_only":
            raise ValueError(
                f"ACM2BiMambaActionICPESSCPConfig only supports "
                f"bimamba_flip_mode='action_only'. Got {self.bimamba_flip_mode}."
            )