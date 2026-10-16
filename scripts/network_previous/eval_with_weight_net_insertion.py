import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.envs.configs import AlohaEnv


BASELINE_DIR = Path("/home1/eunji24/lerobot_project/lerobot-ac-mlp/src/outputs/train/act_aloha_insertion_100k/checkpoints/060000/pretrained_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="residual", choices=["pure_te", "residual"])
parser.add_argument("--weight_net_path", type=str, default=None)
parser.add_argument("--eval_episodes", type=int, default=50)
parser.add_argument("--eval_batch_size", type=int, default=50)
parser.add_argument("--state_noise_std", type=float, default=0.0)
parser.add_argument("--state_noise_ratio", type=float, default=0.0)
parser.add_argument("--noise_seed", type=int, default=0)
parser.add_argument("--L", type=int, default=100)
parser.add_argument("--te_coeff", type=float, default=0.01)
args = parser.parse_args()


class ResidualWeightNetwork(nn.Module):
    def __init__(self, L, state_dim, action_dim, te_coeff=0.01,
                 d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.L = L

        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.queries = nn.Embedding(L, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, 1)

        self.register_buffer(
            "te_logits",
            -te_coeff * torch.arange(L, dtype=torch.float32),
        )

    def forward(self, state, contribs):
        B = state.shape[0]

        state_tok = self.state_proj(state).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)
        memory = torch.cat([state_tok, chunk_toks], dim=1)

        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)

        delta = self.head(x).squeeze(-1)
        final_logits = self.te_logits.unsqueeze(0) + delta
        return final_logits, delta


class EnsemblePolicy:
    def __init__(self, baseline_policy, weight_net, L, mode="residual", te_coeff=0.01):
        self.baseline = baseline_policy
        self.weight_net = weight_net
        self.L = L
        self.mode = mode
        self.te_coeff = te_coeff
        self.config = baseline_policy.config
        self.past_chunks = deque(maxlen=L)

    def reset(self):
        self.past_chunks.clear()

    def eval(self):
        self.baseline.eval()
        if self.weight_net is not None:
            self.weight_net.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch):
        chunk = self.baseline.predict_action_chunk(batch)
        self.past_chunks.append(chunk)
        n = len(self.past_chunks)

        contribs = torch.stack([
            self.past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        if self.mode == "pure_te" or n < self.L:
            decay_w = torch.exp(
                -self.te_coeff * torch.arange(n, device=contribs.device)
            )
            decay_w = decay_w / decay_w.sum()
            action = (decay_w.view(1, n, 1) * contribs).sum(dim=1)
            return action

        state = batch["observation.state"]
        final_logits, _ = self.weight_net(state, contribs)
        weights = F.softmax(final_logits, dim=1).unsqueeze(-1)
        action = (weights * contribs).sum(dim=1)
        return action


def add_state_noise(obs):
    if args.state_noise_std <= 0 and args.state_noise_ratio <= 0:
        return obs

    key = "observation.state"
    state = obs[key]

    noise = torch.zeros_like(state)

    if args.state_noise_std > 0:
        noise = noise + args.state_noise_std * torch.randn_like(state)

    if args.state_noise_ratio > 0:
        scale = state.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
        noise = noise + args.state_noise_ratio * scale * torch.randn_like(state)

    if not hasattr(add_state_noise, "printed"):
        add_state_noise.printed = True
        print("[NOISE DEBUG]")
        print("state mean abs:", state.abs().mean().item())
        print("state std mean:", state.std(dim=0).mean().item())
        print("noise mean abs:", noise.abs().mean().item())
        print("noise/state abs ratio:", (noise.abs().mean() / state.abs().mean().clamp_min(1e-6)).item())

    obs[key] = state + noise
    return obs


def run_one_batch(env, policy, preprocessor, postprocessor,
                  env_preprocessor, env_postprocessor, seeds):
    batch_size = len(seeds)
    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(batch_size, dtype=bool)
    successes = np.zeros(batch_size, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    policy.reset()

    step = 0
    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(obs_dict)
        obs = add_envs_task(env, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        # baseline chunk prediction과 residual state input 둘 다 noisy state를 봄
        obs = add_state_noise(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        action_np = action.detach().cpu().numpy()

        obs_dict, reward, terminated, truncated, info = env.step(action_np)
        done = done | terminated | truncated

        if "final_info" in info:
            final_infos = info.get("final_info", [None] * batch_size)

            if isinstance(final_infos, dict):
                if "is_success" in final_infos:
                    successes |= np.array(final_infos["is_success"], dtype=bool)
            else:
                for i, fi in enumerate(final_infos):
                    if isinstance(fi, dict) and fi.get("is_success", False):
                        successes[i] = True

        step += 1
        if step % 50 == 0:
            print(f"step {step}/{max_steps}, done {done.sum()}/{batch_size}")

    return successes


def main():
    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)

    print(f"Loading ACT baseline from {BASELINE_DIR}")
    baseline = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    weight_net = None
    L = args.L

    if args.mode == "residual":
        if args.weight_net_path is None:
            raise ValueError("--weight_net_path is required when --mode residual")

        weight_net_path = Path(args.weight_net_path)
        print(f"Loading weight network from {weight_net_path}")
        ckpt = torch.load(weight_net_path, map_location=DEVICE)

        L = ckpt["L"]

        weight_net = ResidualWeightNetwork(
            L=L,
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            te_coeff=ckpt["te_coeff"],
            d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"],
        ).to(DEVICE)

        weight_net.load_state_dict(ckpt["model"])
        weight_net.eval()

        print(
            f"Loaded residual net: L={L}, "
            f"lambda={ckpt.get('lambda_delta_reg')}, "
            f"base={ckpt.get('base_policy')}, "
            f"te_direction={ckpt.get('te_direction')}, "
            f"trained_noise_std={ckpt.get('noise_std')}, "
            f"trained_corrupt_prob={ckpt.get('corrupt_prob')}"
        )
    else:
        print(f"Running pure TE wrapper with L={L}")

    policy = EnsemblePolicy(
        baseline_policy=baseline,
        weight_net=weight_net,
        L=L,
        mode=args.mode,
        te_coeff=args.te_coeff,
    ).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    env_cfg = AlohaEnv(task="AlohaInsertion-v0")
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    print("\nEval config:")
    print(f"  mode={args.mode}")
    print(f"  eval_episodes={args.eval_episodes}")
    print(f"  eval_batch_size={args.eval_batch_size}")
    print(f"  L={L}")
    print(f"  te_coeff={args.te_coeff}")
    print(f"  state_noise_std={args.state_noise_std}")
    print(f"  state_noise_ratio={args.state_noise_ratio}")
    print(f"  noise_seed={args.noise_seed}")

    all_successes = []
    start = 0

    while start < args.eval_episodes:
        cur_batch = min(args.eval_batch_size, args.eval_episodes - start)
        seeds = list(range(start, start + cur_batch))

        print(f"\nRunning ACT rollout batch: seeds {seeds[0]}..{seeds[-1]}, batch={cur_batch}")

        env = make_env(env_cfg, n_envs=cur_batch, use_async_envs=False)
        while isinstance(env, dict):
            print("make_env returned dict keys:", env.keys())
            env = list(env.values())[0]

        successes = run_one_batch(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            seeds=seeds,
        )

        all_successes.extend(successes.tolist())
        env.close()
        start += cur_batch

    all_successes = np.array(all_successes, dtype=bool)
    pc_success = all_successes.mean()

    print("\nResults ACT insertion eval")
    print(f"mode = {args.mode}")
    print(f"state_noise_std = {args.state_noise_std}")
    print(f"state_noise_ratio = {args.state_noise_ratio}")
    print(f"pc_success = {pc_success:.3f} ({all_successes.sum()}/{len(all_successes)})")

    print("\nReference:")
    print("  ACT baseline n=50 commit:        62%")
    print("  LeRobot TE fresh-favored:        54%")
    print("  our wrapper pure TE no network:  58%")
    print("  residual clean old result:       check your ckpt")


if __name__ == "__main__":
    main()
