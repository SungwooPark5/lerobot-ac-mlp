import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.acm2_icpe.modeling_acm2_icpe import ACM2ICPEPolicy
from lerobot.policies.acm2_icpe_sscp.configuration_acm2_icpe_sscp import ACM2ICPESSCPConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACM2ICPESSCPPolicy(ACM2ICPEPolicy):
    """ACM2 + ICPE + SSCP."""

    config_class = ACM2ICPESSCPConfig
    name = "acm2_icpe_sscp"

    def __init__(self, config: ACM2ICPESSCPConfig):
        super().__init__(config)
        self.config = config
        self._carry: Tensor | None = None

    def reset(self):
        super().reset()
        self._carry = None

    # -------------------------
    # Inference
    # -------------------------
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
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        carry = self._carry if self.config.sscp_enabled else None

        actions, _, decoder_out = self.model(
            batch,
            carry=carry,
            return_decoder_out=True,
        )

        if self.config.sscp_enabled:
            self._carry = decoder_out[-1:, :, :].transpose(0, 1).detach()

        return actions

    # -------------------------
    # Training
    # -------------------------
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        has_pairs = "action_n1" in batch

        if has_pairs and self.config.sscp_p_carry > 0.0:
            return self._forward_chunk_pair(batch)

        return self._forward_single(batch, carry=None)

    def _forward_single(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None,
        return_decoder_out: bool = False,
    ) -> tuple:
        if return_decoder_out:
            actions_hat, (mu, log_sigma_x2), decoder_out = self.model(
                batch,
                carry=carry,
                return_decoder_out=True,
            )
        else:
            actions_hat, (mu, log_sigma_x2) = self.model(batch, carry=carry)
            decoder_out = None

        K = actions_hat.shape[1]
        n_exec = self.config.n_action_steps
        device = actions_hat.device

        weights = torch.ones(K, device=device)

        if getattr(self.config, "use_temporal_weighting", False) and n_exec < K:
            exec_mass = getattr(self.config, "temporal_execution_weight", 0.9)
            weights[:n_exec] = (K * exec_mass) / n_exec
            weights[n_exec:] = (K * (1.0 - exec_mass)) / (K - n_exec)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
            * weights.view(1, -1, 1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}

        if self.config.use_vae:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = kld.item()
            loss = l1_loss + kld * self.config.kl_weight
        else:
            loss = l1_loss

        if return_decoder_out:
            return loss, loss_dict, decoder_out

        return loss, loss_dict

    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}

        loss_n, loss_dict_n, dec_out_n = self._forward_single(
            batch_n,
            carry=None,
            return_decoder_out=True,
        )

        carry = dec_out_n[-1:, :, :].transpose(0, 1)

        if self.config.sscp_detach:
            carry = carry.detach()

        batch_n1 = {
            OBS_STATE: batch.get("obs_state_n1", batch.get(OBS_STATE)),
            OBS_ENV_STATE: batch.get("obs_env_state_n1", batch.get(OBS_ENV_STATE)),
            ACTION: batch["action_n1"],
            "action_is_pad": batch["action_is_pad_n1"],
        }
        batch_n1 = {k: v for k, v in batch_n1.items() if v is not None}

        if OBS_IMAGES in batch:
            batch_n1[OBS_IMAGES] = [
                batch[k + "_n1"]
                for k in self.config.image_features
                if k + "_n1" in batch
            ] or batch[OBS_IMAGES]

        loss_n1, loss_dict_n1 = self._forward_single(batch_n1, carry=carry)

        total_loss = loss_n + loss_n1

        combined = {
            "l1_loss": (loss_dict_n["l1_loss"] + loss_dict_n1["l1_loss"]) / 2,
            "l1_loss_n": loss_dict_n["l1_loss"],
            "l1_loss_n1": loss_dict_n1["l1_loss"],
        }

        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (
                loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)
            ) / 2

        return total_loss, combined