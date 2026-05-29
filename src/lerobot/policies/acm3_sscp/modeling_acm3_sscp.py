"""ACM-3 SSCP (SSM State Carryover Protocol) Policy

ACM3Policy를 상속하여 청크 경계에서 action blending을 적용한다.
모델 구조·학습 방식은 ACM3와 완전히 동일하며, select_action에서만 SSCP 로직 추가.

SSCP blending:
    청크 경계(step_in_chunk == 0)에서 이전 청크 마지막 action과 현재 예측을 혼합.
    α(t) = sscp_blend_alpha × (1 - t / sscp_blend_steps), t = 0..blend_steps-1

Stage 1 검증 결과: boundary jitter 40.5%↓, spike ratio 2.10x→1.25x, SR 80% 유지.
"""

from collections import deque

import torch
from torch import Tensor

from lerobot.policies.acm3.modeling_acm3 import ACM3Policy
from lerobot.policies.acm3_sscp.configuration_acm3_sscp import ACM3SSCPConfig


class ACM3SSCPPolicy(ACM3Policy):
    config_class = ACM3SSCPConfig
    name = "acm3_sscp"

    def __init__(self, config: ACM3SSCPConfig):
        super().__init__(config)
        self.config = config
        # SSCP state (에피소드마다 reset)
        self._step_in_chunk: int = 0
        self._prev_tail: Tensor | None = None

    def reset(self):
        """에피소드 시작 시 호출. action queue + SSCP state 초기화."""
        super().reset()
        self._step_in_chunk = 0
        self._prev_tail = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        # ── 표준 ACM3 action queue 로직 ────────────────────────────────────────
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
        else:
            if len(self._action_queue) == 0:
                actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
                self._action_queue.extend(actions.transpose(0, 1))
            action = self._action_queue.popleft()

        # ── SSCP: 청크 경계에서 이전 tail과 blending ───────────────────────────
        alpha = self.config.sscp_blend_alpha
        blend_steps = self.config.sscp_blend_steps

        if (self._step_in_chunk < blend_steps) and (self._prev_tail is not None):
            # linear decay: α_0 = blend_alpha, α_{blend_steps} = 0
            alpha_t = alpha * (1.0 - self._step_in_chunk / blend_steps)
            action = (1.0 - alpha_t) * action + alpha_t * self._prev_tail.to(action.device)

        # 현재 step action을 tail로 저장 (다음 청크 경계에서 사용)
        self._prev_tail = action.detach().clone()
        self._step_in_chunk = (self._step_in_chunk + 1) % self.config.n_action_steps

        return action
