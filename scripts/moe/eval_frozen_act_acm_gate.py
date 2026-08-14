#!/usr/bin/env python
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# repo src path
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.configs import AlohaEnv


ACT_POLICY_PATH = Path(
    "/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model"
)

# 60k peak ACM
ACM_POLICY_PATH = Path(
    "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_transfer_cube_test_100k/checkpoints/060000/pretrained_model"
)

GATE_PATH = Path(
    "/home1/eunji24/lerobot_project/outputs/frozen_act_acm_gate/gate_final.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_EPISODES = 50
MAX_STEPS = 400
BATCH_SIZE = 50

MODE = "fixed"  
# choices:
# "gate"
# "fixed"
# "act"
# "acm"

FIXED_W = 0.7


class FrozenACTACMGate(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, prior_w=0.7):
        super().__init__()
        in_dim = state_dim + action_dim * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        prior_logit = torch.logit(torch.tensor(prior_w))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, prior_logit.item())

    def forward(self, state, act_chunk, acm_chunk):
        B, L, A = act_chunk.shape

        state_rep = state[:, None, :].expand(B, L, -1)
        x = torch.cat([state_rep, act_chunk, acm_chunk], dim=-1)

        logit = self.net(x).squeeze(-1)
        w = torch.sigmoid(logit)

        final_chunk = w[..., None] * act_chunk + (1.0 - w[..., None]) * acm_chunk
        return final_chunk, w


def load_policy(policy_cls, path):
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path

    # 여기서는 내부 TE 쓰면 안 됨. chunk만 직접 뽑아서 우리가 섞음.
    cfg.temporal_ensemble_coeff = None
    cfg.n_action_steps = cfg.chunk_size

    policy = policy_cls.from_pretrained(path, config=cfg)
    policy.to(DEVICE)
    policy.eval()
    return policy, cfg


def to_device_batch(batch):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(DEVICE)
        else:
            out[k] = v
    return out

def obs_to_policy_batch(obs):
    """
    gym_aloha raw obs:
      obs["agent_pos"]: (B, 14), float64
      obs["pixels"]: dict or array

    policy expected:
      observation.state
      observation.images.top
    """
    batch = {}

    state = torch.as_tensor(obs["agent_pos"], dtype=torch.float32, device=DEVICE)
    batch["observation.state"] = state

    pixels = obs["pixels"]

    if isinstance(pixels, dict):
        # usually {"top": image}
        if "top" in pixels:
            img = pixels["top"]
        else:
            print("pixels keys:", pixels.keys())
            img = next(iter(pixels.values()))
    else:
        img = pixels

    img = torch.as_tensor(img, dtype=torch.float32, device=DEVICE)

    # possible raw shapes:
    # (B,H,W,C) uint8/float
    # (B,C,H,W)
    if img.ndim == 4 and img.shape[-1] == 3:
        img = img.permute(0, 3, 1, 2)

    # normalize if uint8-like
    if img.max() > 2:
        img = img / 255.0

    batch["observation.images.top"] = img
    return batch


def main():
    print("=" * 80)
    print("Frozen ACT/ACM gate eval")
    print("MODE:", MODE)
    print("ACT:", ACT_POLICY_PATH)
    print("ACM:", ACM_POLICY_PATH)
    print("GATE:", GATE_PATH)
    print("=" * 80)

    print("Loading ACT policy...")
    act_policy, act_cfg = load_policy(ACTPolicy, ACT_POLICY_PATH)

    print("Loading ACM policy...")
    acm_policy, acm_cfg = load_policy(ACMPolicy, ACM_POLICY_PATH)

    print("Loading gate...")
    ckpt = torch.load(GATE_PATH, map_location=DEVICE)

    gate = FrozenACTACMGate(
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        d_model=ckpt["d_model"],
        prior_w=ckpt.get("prior_w", 0.7),
    ).to(DEVICE)
    gate.load_state_dict(ckpt["model"])
    gate.eval()

    print("Making env...")
    env_cfg = AlohaEnv(
        task="AlohaTransferCube-v0",
        fps=50,
        episode_length=MAX_STEPS,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    env = make_env(env_cfg, n_envs=BATCH_SIZE)

    # Some LeRobot/gym_aloha versions return nested dict:
    # {'aloha': {0: vec_env}} instead of env directly.
    if isinstance(env, dict):
        print("make_env returned dict keys:", env.keys())
        env = next(iter(env.values()))
        if isinstance(env, dict):
            print("make_env returned inner dict keys:", env.keys())
            env = next(iter(env.values()))


    successes = np.zeros(BATCH_SIZE, dtype=bool)
    done_any = np.zeros(BATCH_SIZE, dtype=bool)

    gate_means = []
    gate_stds = []

    obs, info = env.reset()

    for step in tqdm(range(MAX_STEPS)):
        batch = obs_to_policy_batch(obs)

        with torch.no_grad():
            act_chunk = act_policy.predict_action_chunk(batch)
            acm_chunk = acm_policy.predict_action_chunk(batch)

            if MODE == "act":
                final_chunk = act_chunk
                w = None
            elif MODE == "acm":
                final_chunk = acm_chunk
                w = None
            elif MODE == "fixed":
                final_chunk = FIXED_W * act_chunk + (1.0 - FIXED_W) * acm_chunk
                w = torch.full(
                    act_chunk.shape[:2],
                    FIXED_W,
                    device=act_chunk.device,
                    dtype=act_chunk.dtype,
                )
            elif MODE == "gate":
                state = batch["observation.state"]
                final_chunk, w = gate(state, act_chunk, acm_chunk)
            else:
                raise ValueError(f"Unknown MODE: {MODE}")

            action = final_chunk[:, 0]
            action = action.detach().cpu().numpy()

        if w is not None:
            gate_means.append(w.mean().item())
            gate_stds.append(w.std().item())

        obs, reward, terminated, truncated, info = env.step(action)

        done = np.logical_or(terminated, truncated)
        done_any |= done

        if "final_info" in info:
            final_info = info["final_info"]

            if isinstance(final_info, dict):
                if "is_success" in final_info:
                    successes |= np.array(final_info["is_success"], dtype=bool)
            else:
                for i, fi in enumerate(final_info):
                    if isinstance(fi, dict) and fi.get("is_success", False):
                        successes[i] = True

        if step % 50 == 0:
            print(
                f"step {step}/{MAX_STEPS}, "
                f"success {successes.sum()}/{BATCH_SIZE}"
            )

    success_rate = successes.mean()

    print("=" * 80)
    print("RESULT")
    print("MODE:", MODE)
    print("success:", successes.sum(), "/", BATCH_SIZE)
    print("success_rate:", success_rate)

    if gate_means:
        print("gate_mean_avg:", float(np.mean(gate_means)))
        print("gate_std_avg:", float(np.mean(gate_stds)))

    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()