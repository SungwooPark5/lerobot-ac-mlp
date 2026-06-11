"""ACM3 + ICPE + SSM State Carryover Protocol (SSCP)

How SSCP works (inference):
  1. After chunk n, extract the Mamba3 decoder's LAST OUTPUT TOKEN:
       carry_n = decoder_out[:, -1:, :]   # (B, 1, D)
     This dense vector summarises the accumulated SSM context at the end of chunk n.

  2. Before chunk n+1's action queries, insert carry_n into the combined sequence
     inside Mamba3ICPEDecoder. Default placement is "pre_query":
       combined = cat([encoder_out_{n+1}, carry_n, decoder_queries_{n+1}], dim=1)
     Mamba3 scans carry_n immediately before the queries, re-deriving a non-zero
     hidden state from previous-chunk context instead of h=0. (This is a
     summary-token warm-up, not a literal transfer of Mamba3's recurrent state.)

  3. The K action tokens are still extracted from the LAST K positions of combined_out.

Why this is SSM-specific (not applicable to ACT Transformer):
  - Mamba3 processes carry_n sequentially, so carry_n's hidden state propagates
    into all subsequent positions.
  - In ACT (Transformer), adding carry_n as an extra attention key does not warm up
    any hidden state; it is merely another token in global attention.

Training strategy (Chunk-Continuation):
  With probability sscp_p_carry, the batch provides consecutive chunk pairs.
  chunk n → compute carry (detach) → use as prefix for chunk n+1.
  Loss = L1(chunk_n) + L1(chunk_n+1).  Both computed identically to standard ACM3ICPE.
  With probability (1 - sscp_p_carry), standard training without carry.
"""

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.acm3_icpe.modeling_acm3_icpe import ACM3ICPE, ACM3ICPEPolicy
from lerobot.policies.acm3_icpe_sscp.configuration_acm3_icpe_sscp import ACM3ICPESSCPConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACM3ICPESSCPPolicy(ACM3ICPEPolicy):
    """ACM3 + ICPE + SSCP.

    Extends ACM3ICPEPolicy with:
      - Inference: carry the terminal decoder output token across chunk boundaries.
      - Training:  chunk-continuation pairs (when available) to teach the model
                   to handle non-zero initial SSM context.
    """

    config_class = ACM3ICPESSCPConfig
    name = "acm3_icpe_sscp"

    def __init__(self, config: ACM3ICPESSCPConfig):
        super().__init__(config)
        self.config = config
        # Carry state: (B, 1, D) or None — updated after each chunk at inference
        self._carry: Tensor | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def reset(self):
        """Reset episode state: action queue + SSM carry."""
        super().reset()
        self._carry = None

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self._predict_with_carry(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self._predict_with_carry(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self._predict_with_carry(batch)

    def _predict_with_carry(self, batch: dict[str, Tensor]) -> Tensor:
        """Run model forward with the stored carry and update carry for next chunk."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        carry = self._carry if self.config.sscp_enabled else None

        actions, _, decoder_out = self.model(batch, carry=carry, return_decoder_out=True)

        if self.config.sscp_enabled:
            self._carry = decoder_out[-1:, :, :].transpose(0, 1).detach()  # (B, 1, D)

        return actions

    # ── Training ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Supports two batch formats:
          Standard:  keys match ACM3 (obs, action, action_is_pad)
          Pair mode: additionally has obs_n1, action_n1, action_is_pad_n1
                     for chunk-continuation training.
        """
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Check if batch provides consecutive chunk pairs for SSCP training
        has_pairs = "action_n1" in batch

        if has_pairs and self.config.sscp_p_carry > 0.0:
            return self._forward_chunk_pair(batch)
        else:
            return self._forward_single(batch, carry=None)

    def _forward_single(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None,
        return_decoder_out: bool = False,
    ) -> tuple:
        """Standard forward on a single chunk with optional carry."""
        if return_decoder_out:
            actions_hat, (mu, log_sigma_x2), decoder_out = self.model(batch, carry=carry, return_decoder_out=True)
        else:
            actions_hat, (mu, log_sigma_x2) = self.model(batch, carry=carry)
            decoder_out = None

        K      = actions_hat.shape[1]
        n_exec = self.config.n_action_steps
        device = actions_hat.device
        weights = torch.ones(K, device=device)

        if getattr(self.config, "use_temporal_weighting", False) and n_exec < K:
            exec_mass = getattr(self.config, "temporal_execution_weight", 0.9)
            weights[:n_exec] = (K * exec_mass) / n_exec
            weights[n_exec:] = (K * (1 - exec_mass)) / (K - n_exec)

        l1 = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
            * weights.view(1, -1, 1)
        ).mean()

        loss_dict = {"l1_loss": l1.item()}
        if self.config.use_vae:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = kld.item()
            loss = l1 + kld * self.config.kl_weight
        else:
            loss = l1
        if return_decoder_out:
            return loss, loss_dict, decoder_out
        return loss, loss_dict

    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Chunk-continuation training forward.

        1. Process chunk n (carry=None, standard).
        2. Extract terminal decoder output token as carry_n.
        3. Optionally detach carry_n to prevent gradient from flowing across boundary.
        4. Process chunk n+1 with carry=carry_n.
        5. Return combined loss.
        """
        # ── Chunk n ───────────────────────────────────────────────────────────
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, loss_dict_n, dec_out_n = self._forward_single(batch_n, carry=None, return_decoder_out=True)

        # ── Extract carry from chunk n ────────────────────────────────────────
        carry = dec_out_n[-1:, :, :].transpose(0, 1)  # (B, 1, D)
        if self.config.sscp_detach:
            carry = carry.detach()

        # ── Chunk n+1 ─────────────────────────────────────────────────────────
        _n1_candidates = {
            OBS_STATE:       batch.get("obs_state_n1",     batch.get(OBS_STATE)),
            OBS_ENV_STATE:   batch.get("obs_env_state_n1", batch.get(OBS_ENV_STATE)),
            ACTION:          batch["action_n1"],
            "action_is_pad": batch["action_is_pad_n1"],
        }
        batch_n1 = {k: v for k, v in _n1_candidates.items() if v is not None}
        if OBS_IMAGES in batch:
            batch_n1[OBS_IMAGES] = [
                batch[k + "_n1"] for k in self.config.image_features
                if k + "_n1" in batch
            ] or batch[OBS_IMAGES]

        loss_n1, loss_dict_n1 = self._forward_single(batch_n1, carry=carry)

        # ── Combined loss & metrics ───────────────────────────────────────────
        total_loss = loss_n + loss_n1
        combined = {
            "l1_loss":      (loss_dict_n["l1_loss"]  + loss_dict_n1["l1_loss"])  / 2,
            "l1_loss_n":    loss_dict_n["l1_loss"],
            "l1_loss_n1":   loss_dict_n1["l1_loss"],
        }
        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)) / 2

        return total_loss, combined
