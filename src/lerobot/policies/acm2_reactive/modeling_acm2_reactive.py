#!/usr/bin/env python
"""ACM2 reactive — receding-horizon state-refined re-planning (cor x BiMamba, v13).

Model == acm2_sscp_literal_bimamba (`_build_model` unchanged). v13 changes only control flow:

  inference (reset/select_action): receding-horizon loop — re-plan every `refine_period`
      executed steps from fresh obs (gate re-corrects carry toward o_t, BiMamba refines),
      commit the carry once per chunk.

  training (forward, reactive_train=True): the SAME loop, teacher-forced over a segment of M
      consecutive chunks (ReactiveSegmentDataset). carry committed at chunk boundaries; at
      sampled mid-chunk times we re-decode from the committed carry + fresh obs and supervise
      the executed head plan[:exec_len] against ground truth → trains the gate for per-step
      invocation (train == inference). No zero-shot path.
"""
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    detach_states,
    ema_states,
)
from lerobot.policies.acm2_sscp_literal_bimamba.modeling_acm2_sscp_literal_bimamba import (
    ACM2SSCPLiteralBiMamba,
    ACM2SSCPLiteralBiMambaPolicy,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES

from .configuration_acm2_reactive import ACM2ReactiveConfig


class ACM2ReactivePolicy(ACM2SSCPLiteralBiMambaPolicy):
    """cor x BiMamba with a receding-horizon, state-refined reactive loop (train == inference)."""

    config_class = ACM2ReactiveConfig
    name = "acm2_reactive"

    def _build_model(self, config):
        # Identical model to the parent → architecture unchanged.
        return ACM2SSCPLiteralBiMamba(config)

    # ── Inference state ──────────────────────────────────────────────────────────
    def reset(self):
        super().reset()                       # _action_queue / temporal_ensembler + self._carry = None
        self._plan: Tensor | None = None      # (B, K, action_dim) most recent decoded chunk
        self._carry_next: list | None = None  # carry_out stashed at the boundary → next chunk's carry_in
        self._t: int = 0                      # executed-step index within the chunk [0, commit_period)
        self._j: int = 0                      # index into the current plan (resets to 0 on each re-plan)

    @property
    def _commit_period(self) -> int:
        return self.config.commit_period or self.config.n_action_steps

    @property
    def _eff_refine_period(self) -> int:
        if not self.config.reactive_refine:
            return self._commit_period        # open-loop control
        return max(1, min(self.config.refine_period, self._commit_period))

    # ── Reactive select_action (receding-horizon) ────────────────────────────────
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        # Temporal ensembling = output-space reactive scheme, mutually exclusive with re-planning.
        if self.config.temporal_ensemble_coeff is not None:
            return super().select_action(batch)

        r = self._eff_refine_period
        at_boundary = (self._plan is None) or (self._t == 0)
        if at_boundary:
            # slow channel: decode with the committed carry_in; stash carry_out for the next chunk.
            self._plan, self._carry_next = self._decode(batch, self._carry)
            self._j = 0
        elif self._t % r == 0:
            # fast channel: re-plan from the SAME committed carry_in (do NOT advance the carry).
            self._plan, _ = self._decode(batch, self._carry)
            self._j = 0

        action = self._plan[:, self._j]
        self._j += 1
        self._t += 1
        if self._t >= self._commit_period:    # crossed a chunk boundary → commit the slow channel
            self._carry = self._carry_next
            self._t = 0
        return action

    @torch.no_grad()
    def _decode(self, batch: dict[str, Tensor], carry: list | None):
        """One decoder forward from a fixed carry_in, WITHOUT mutating self._carry."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        carry_in = carry if self.config.sscp_enabled else None
        actions, _, new_states = self.model(batch, carry=carry_in, return_state=True)
        carry_out = None
        if self.config.sscp_enabled:
            carry_out = detach_states(new_states)
            if self.config.carry_fusion == "ema":
                carry_out = ema_states(carry_out, self._carry, self.config.carry_ema_beta)
        return actions, carry_out

    # ── Training: teacher-forced reactive segment rollout (Phase R) ──────────────
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if not (self.config.reactive_train and "step0_action" in batch):
            raise RuntimeError(
                "acm2_reactive.forward expects reactive_train=True with a ReactiveSegmentDataset "
                "batch (keys 'step{j}_*'). Set --policy.reactive_train=true and use the reactive "
                "segment loader (lerobot_train wires it automatically)."
            )
        cfg = self.config
        M, S = cfg.reactive_train_chunks, cfg.reactive_train_samples
        per = 1 + S
        exec_len = cfg.reactive_exec_len

        carry = None
        l_boundary = l_react = kld_sum = 0.0
        n_b = n_r = 0
        for c in range(M):
            # ── chunk boundary: open-loop chunk loss + carry_out (committed for the next chunk) ──
            jb = c * per
            b_batch = self._step_batch(batch, jb)
            plan_b, (mu, log_sigma_x2), carry_out = self.model(b_batch, carry=carry, return_state=True)
            l_boundary = l_boundary + self._chunk_l1(plan_b, b_batch[ACTION], b_batch["action_is_pad"])
            if cfg.use_vae:
                kld_sum = kld_sum + self._kld(mu, log_sigma_x2)
            n_b += 1

            # ── mid-chunk re-plans: SAME committed carry + fresh obs → supervise executed head ──
            for s in range(S):
                js = c * per + 1 + s
                s_batch = self._step_batch(batch, js)
                plan_s, _, _ = self.model(s_batch, carry=carry, return_state=True)
                l_react = l_react + self._chunk_l1(
                    plan_s, s_batch[ACTION], s_batch["action_is_pad"], head=exec_len)
                n_r += 1

            carry = detach_states(carry_out)   # commit (truncated BPTT across chunk boundary)

        loss = l_boundary / max(n_b, 1)
        loss_dict = {"l1_loss": float((l_boundary / max(n_b, 1)).detach())}
        if n_r > 0:
            react = l_react / n_r
            loss = loss + cfg.reactive_consistency_weight * react
            loss_dict["reactive_head_loss"] = float(react.detach())
        if cfg.use_vae and n_b > 0:
            kld = kld_sum / n_b
            loss = loss + kld * cfg.kl_weight
            loss_dict["kld_loss"] = float(kld.detach())
        loss_dict["loss"] = float(loss.detach())
        return loss, loss_dict

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _step_batch(self, batch: dict[str, Tensor], j: int) -> dict[str, Tensor]:
        """Extract the j-th decode point (strip the 'step{j}_' prefix) into a standard batch."""
        p = f"step{j}_"
        out = {k[len(p):]: v for k, v in batch.items() if k.startswith(p)}
        if self.config.image_features:
            out[OBS_IMAGES] = [out[key] for key in self.config.image_features]
        return out

    def _chunk_l1(self, plan: Tensor, target: Tensor, pad: Tensor, head: int | None = None) -> Tensor:
        """Masked L1 over a chunk (or its first `head` steps = the executed receding-horizon part)."""
        if head is not None:
            h = min(head, plan.shape[1])
            plan, target, pad = plan[:, :h], target[:, :h], pad[:, :h]
        return (F.l1_loss(target, plan, reduction="none") * ~pad.unsqueeze(-1)).mean()

    @staticmethod
    def _kld(mu: Tensor, log_sigma_x2: Tensor) -> Tensor:
        return (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
