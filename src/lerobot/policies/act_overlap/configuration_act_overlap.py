"""ACT + overlap-add crossfade (inference-only) — the ACT counterpart of MOSAIC's overlap term.

ACT has no recurrent state, so it CANNOT carry state across chunk boundaries (SSCP). Only the
overlap-add crossfade — which operates on the predicted ACTION outputs, not on any internal
state — is applicable. At each boundary the last `sscp_overlap` steps of the previous chunk are
crossfaded with the first `sscp_overlap` steps of the freshly predicted chunk via a rising window.

Because the two overlapping chunks are predicted INDEPENDENTLY (no carry), their overlap regions
need not agree, so the blend is a smeared average rather than a clean continuation. This is the
intended ablation: it isolates "overlap without carry", showing that carry (MOSAIC) is what makes
the crossfade clean.

Inference-only: ACT's training pipeline does not do chunk-pair forwards, so there is no
overlap-consistency training term (no sscp_overlap_train_weight). sscp_overlap=0 -> identical to
plain ACT. Disabled automatically when temporal_ensemble_coeff is set.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_overlap")
@dataclass
class ACTOverlapConfig(ACTConfig):
    # Number of overlap steps crossfaded across each boundary. 0 = off. Must be <= chunk_size//2.
    sscp_overlap: int = 0

    # Crossfade window shape for the rising blend weight: "linear" or "hann".
    sscp_overlap_window: str = "linear"

    def __post_init__(self):
        super().__post_init__()
        if self.sscp_overlap < 0:
            raise ValueError("sscp_overlap must be >= 0.")
        if self.sscp_overlap > self.chunk_size // 2:
            raise ValueError(
                f"sscp_overlap ({self.sscp_overlap}) must be <= chunk_size//2 ({self.chunk_size // 2})."
            )
        if self.sscp_overlap_window not in ("linear", "hann"):
            raise ValueError(f"sscp_overlap_window must be 'linear' or 'hann', got {self.sscp_overlap_window}.")
