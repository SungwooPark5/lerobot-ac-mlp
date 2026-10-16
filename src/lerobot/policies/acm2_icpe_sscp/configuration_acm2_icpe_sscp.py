from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_icpe.configuration_acm2_icpe import ACM2ICPEConfig


@PreTrainedConfig.register_subclass("acm2_icpe_sscp")
@dataclass
class ACM2ICPESSCPConfig(ACM2ICPEConfig):
    """ACM2 + ICPE + SSCP."""

    sscp_enabled: bool = True
    sscp_p_carry: float = 0.5
    sscp_detach: bool = True
    sscp_carry_position: str = "pre_query"

    def __post_init__(self):
        super().__post_init__()

        if self.sscp_carry_position not in ("pre_query", "prefix"):
            raise ValueError(
                f"sscp_carry_position must be 'pre_query' or 'prefix'. Got '{self.sscp_carry_position}'."
            )