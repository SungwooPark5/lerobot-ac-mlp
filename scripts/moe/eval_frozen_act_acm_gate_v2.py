import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import os
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.envs.configs import AlohaEnv


ACT_DIR = Path("/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model")
ACM_DIR = Path("/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_transfer_cube_test_100k/checkpoints/060000/pretrained_model")
GATE_PATH = Path("/home1/eunji24/lerobot_project/outputs/frozen_act_acm_gate/gate_final.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="gate", choices=["gate", "fixed", "act", "acm"])
parser.add_argument("--eval_episodes", type=int, default=50)
parser.add_argument("--eval_batch_size", type=int, default=50)
parser.add_argument("--fixed_w", type=float, default=0.7)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, state, act_chunk, acm_chunk):
        B, L, A = act_chunk.shape
        state_rep = state[:, None, :].expand(B, L, -1)
        x = torch.cat([state_rep, act_chunk, acm_chunk], dim=-1)
        w = torch.sigmoid(self.net(x).squeeze(-1))
        final_chunk = w[..., None] * act_chunk + (1.0 - w[..., None]) * acm_chunk
        return final_chunk, w


class FrozenACTACMPolicy:
    def __init__(self, act_policy, acm_policy, gate, mode="gate", fixed_w=0.7):
        self.act_policy = act_policy
        self.acm_policy = acm_policy
        self.gate = gate
        self.mode = mode
        self.fixed_w = fixed_w
        self.config = act_policy.config
        self.gate_means = []
        self.gate_stds = []

    def reset(self):
        self.gate_means.clear()
        self.gate_stds.clear()

    def eval(self):
        self.act_policy.eval()
        self.acm_policy.eval()
        if self.gate is not None:
            self.gate.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch):
        act_chunk = self.act_policy.predict_action_chunk(batch)
        acm_chunk = self.acm_policy.predict_action_chunk(batch)

        if self.mode == "act":
            final_chunk = act_chunk
            w = None
        elif self.mode == "acm":
            final_chunk = acm_chunk
            w = None
        elif self.mode == "fixed":
            final_chunk = self.fixed_w * act_chunk + (1.0 - self.fixed_w) * acm_chunk
            w = torch.full(act_chunk.shape[:2], self.fixed_w, device=act_chunk.device)
        elif self.mode == "gate":
            state = batch["observation.state"]
            final_chunk, w = self.gate(state, act_chunk, acm_chunk)
        else:
            raise ValueError(self.mode)

        if w is not None:
            self.gate_means.append(w.mean().item())
            self.gate_stds.append(w.std().item())

        return final_chunk[:, 0]


def main():
    print("=" * 80)
    print("Frozen ACT/ACM gate eval v2")
    print("mode:", args.mode)
    print("ACT:", ACT_DIR)
    print("ACM:", ACM_DIR)
    print("GATE:", GATE_PATH)
    print("=" * 80)

    print("Loading ACT...")
    act = ACTPolicy.from_pretrained(str(ACT_DIR)).to(DEVICE)
    act.eval()
    act.config.temporal_ensemble_coeff = None
    act.config.n_action_steps = act.config.chunk_size

    print("Loading ACM...")
    acm = ACMPolicy.from_pretrained(str(ACM_DIR)).to(DEVICE)
    acm.eval()
    acm.config.temporal_ensemble_coeff = None
    acm.config.n_action_steps = acm.config.chunk_size

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

    policy = FrozenACTACMPolicy(
        act_policy=act,
        acm_policy=acm,
        gate=gate,
        mode=args.mode,
        fixed_w=args.fixed_w,
    )

    config = act.config

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(ACT_DIR),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
    env = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)

    while isinstance(env, dict):
        print("make_env returned dict keys:", env.keys())
        env = list(env.values())[0]

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    print(f"Running rollout: {args.eval_episodes} episodes, batch={args.eval_batch_size}")

    seeds = list(range(args.eval_episodes))[: args.eval_batch_size]
    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(args.eval_batch_size, dtype=bool)
    successes = np.zeros(args.eval_batch_size, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    policy.reset()

    step = 0
    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(obs_dict)
        obs = add_envs_task(env, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        action_np = action.detach().cpu().numpy()

        obs_dict, reward, terminated, truncated, info = env.step(action_np)
        done = done | terminated | truncated

        if "final_info" in info:
            final_infos = info.get("final_info", [None] * args.eval_batch_size)
            if isinstance(final_infos, dict):
                if "is_success" in final_infos:
                    successes |= np.array(final_infos["is_success"], dtype=bool)
            else:
                for i, fi in enumerate(final_infos):
                    if isinstance(fi, dict) and fi.get("is_success", False):
                        successes[i] = True

        step += 1
        if step % 50 == 0:
            print(f"step {step}/{max_steps}, done {done.sum()}/{args.eval_batch_size}, success {successes.sum()}/{args.eval_batch_size}")

    print("\nRESULT")
    print("mode =", args.mode)
    print(f"pc_success = {successes.mean():.3f} ({successes.sum()}/{args.eval_batch_size})")
    if policy.gate_means:
        print("gate_mean_avg =", float(np.mean(policy.gate_means)))
        print("gate_std_avg =", float(np.mean(policy.gate_stds)))

    env.close()


if __name__ == "__main__":
    main()
