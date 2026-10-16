import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.envs.configs import AlohaEnv


BASELINE_DIR = Path(
    "/home1/eunji24/lerobot_project/outputs/train/"
    "act_network_baseline_100k/checkpoints/last/pretrained_model"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="residual", choices=["pure_te", "residual"])
parser.add_argument("--weight_net_path", type=str, default=None)
parser.add_argument("--eval_episodes", type=int, default=50)
parser.add_argument("--eval_batch_size", type=int, default=50)

parser.add_argument("--L", type=int, default=100)
parser.add_argument("--te_coeff", type=float, default=0.01)
parser.add_argument("--fresh_index", type=str, default="first", choices=["first", "last"])
parser.add_argument("--alpha_residual", type=float, default=0.1)
parser.add_argument("--max_delta", type=float, default=1.0)

parser.add_argument("--state_noise_std", type=float, default=0.0)
parser.add_argument("--state_noise_ratio", type=float, default=0.0)
parser.add_argument("--image_noise_std", type=float, default=0.0)
parser.add_argument("--image_noise_prob", type=float, default=1.0)
parser.add_argument("--noise_seed", type=int, default=0)

args = parser.parse_args()


class ActionResidualNetwork(nn.Module):
    """
    TE fixed + action residual only.

    Input:
        states:   [B, state_dim]
        contribs: [B, L, action_dim]
    Output:
        delta_a:  [B, action_dim]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_delta: float = 1.0,
    ):
        super().__init__()
        self.max_delta = max_delta

        self.state_proj = nn.Linear(state_dim, d_model)
        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.query = nn.Embedding(1, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, states: torch.Tensor, contribs: torch.Tensor) -> torch.Tensor:
        B = states.shape[0]

        state_tok = self.state_proj(states).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)
        memory = torch.cat([state_tok, chunk_toks], dim=1)

        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory).squeeze(1)

        raw_delta = self.head(x)
        delta_a = torch.tanh(raw_delta) * self.max_delta
        return delta_a


def make_fixed_te_weights(
    L: int,
    te_coeff: float,
    device: torch.device,
    fresh_index: str = "first",
):
    """
    fresh_index:
        "first" -> contribs[:, 0]이 최신 chunk
        "last"  -> contribs[:, -1]이 최신 chunk
    """
    if fresh_index == "first":
        age = torch.arange(0, L, device=device).float()
    elif fresh_index == "last":
        age = torch.arange(L - 1, -1, -1, device=device).float()
    else:
        raise ValueError(f"fresh_index must be 'first' or 'last', got {fresh_index}")

    weights = torch.exp(-te_coeff * age)
    weights = weights / weights.sum()
    return weights


class EnsemblePolicy:
    def __init__(
        self,
        baseline_policy,
        residual_net,
        L,
        mode="residual",
        te_coeff=0.01,
        fresh_index="first",
        alpha_residual=0.1,
    ):
        self.baseline = baseline_policy
        self.residual_net = residual_net
        self.L = L
        self.mode = mode
        self.te_coeff = te_coeff
        self.fresh_index = fresh_index
        self.alpha_residual = alpha_residual
        self.config = baseline_policy.config
        self.past_chunks = deque(maxlen=L)

    def reset(self):
        self.past_chunks.clear()

    def eval(self):
        self.baseline.eval()
        if self.residual_net is not None:
            self.residual_net.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch):
        chunk = self.baseline.predict_action_chunk(batch)
        self.past_chunks.append(chunk)
        n = len(self.past_chunks)

        # 현재 구조:
        # i=0 -> 가장 최근 chunk의 pos 0
        # i=1 -> 한 step 전 chunk의 pos 1
        # ...
        # 따라서 contribs[:, 0]이 freshest
        contribs = torch.stack(
            [self.past_chunks[-(i + 1)][:, i] for i in range(n)],
            dim=1,
        )

        te_weights = make_fixed_te_weights(
            L=n,
            te_coeff=self.te_coeff,
            device=contribs.device,
            fresh_index="first",
        )
        a_te = torch.sum(te_weights.view(1, n, 1) * contribs, dim=1)

        if self.mode == "pure_te" or n < self.L:
            return a_te

        state = batch["observation.state"]
        delta_a = self.residual_net(state, contribs)
        action = a_te + self.alpha_residual * delta_a

        if not hasattr(self, "_printed_residual_debug"):
            self._printed_residual_debug = True
            print("[ACTION RESIDUAL DEBUG]")
            print("n:", n)
            print("alpha_residual:", self.alpha_residual)
            print("a_te abs mean:", a_te.abs().mean().item())
            print("delta_a abs mean:", delta_a.abs().mean().item())
            print("alpha*delta abs mean:", (self.alpha_residual * delta_a).abs().mean().item())
            print("action-a_te abs mean:", (action - a_te).abs().mean().item())

        return action


def add_observation_noise(obs, args, step=None):
    state_key = "observation.state"

    if state_key in obs and torch.is_tensor(obs[state_key]):
        if args.state_noise_std > 0.0 or args.state_noise_ratio > 0.0:
            state = obs[state_key]
            noise = torch.zeros_like(state)

            if args.state_noise_std > 0.0:
                noise = noise + args.state_noise_std * torch.randn_like(state)

            if args.state_noise_ratio > 0.0:
                scale = state.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
                noise = noise + args.state_noise_ratio * scale * torch.randn_like(state)

            obs[state_key] = state + noise

            if step == 0:
                print("[STATE NOISE DEBUG]")
                print("state mean abs:", state.abs().mean().item())
                print("state std mean:", state.std(dim=0).mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print(
                    "noise/state abs ratio:",
                    (noise.abs().mean() / state.abs().mean().clamp_min(1e-6)).item(),
                )

    if args.image_noise_std > 0.0:
        for key in list(obs.keys()):
            if not key.startswith("observation.images"):
                continue
            if not torch.is_tensor(obs[key]):
                continue

            img = obs[key]

            if args.image_noise_prob < 1.0:
                B = img.shape[0]
                apply_mask = torch.rand(B, device=img.device) < args.image_noise_prob
                view_shape = [B] + [1] * (img.ndim - 1)
                apply_mask = apply_mask.view(*view_shape).float()
                noise = args.image_noise_std * torch.randn_like(img) * apply_mask
            else:
                noise = args.image_noise_std * torch.randn_like(img)

            noisy_img = img + noise

            img_min = img.min().item()
            img_max = img.max().item()
            if img_min >= -0.1 and img_max <= 1.1:
                noisy_img = noisy_img.clamp(0.0, 1.0)

            obs[key] = noisy_img

            if step == 0:
                denom = img.abs().mean().clamp_min(1e-6)
                print(f"[IMAGE NOISE DEBUG] {key}")
                print("image min/max/mean:", img.min().item(), img.max().item(), img.mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print("noise/image abs ratio:", (noise.abs().mean() / denom).item())

    return obs


def clone_obs(obs):
    out = {}
    for k, v in obs.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def run_one_batch(
    env,
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    seeds,
):
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

        obs_clean = clone_obs(obs)
        obs_noisy = add_observation_noise(obs, args, step=step)

        if step == 0 and (args.state_noise_std > 0 or args.state_noise_ratio > 0 or args.image_noise_std > 0):
            with torch.inference_mode():
                chunk_clean = policy.baseline.predict_action_chunk(obs_clean)
                chunk_noisy = policy.baseline.predict_action_chunk(obs_noisy)

            chunk_abs = chunk_clean.abs().mean().item()
            chunk_diff = (chunk_noisy - chunk_clean).abs().mean().item()

            print("[CHUNK NOISE DEBUG]")
            print("chunk_clean_abs_mean:", chunk_abs)
            print("chunk_diff_mean:", chunk_diff)
            print("chunk_diff / chunk_abs:", chunk_diff / max(chunk_abs, 1e-8))

        with torch.inference_mode():
            action = policy.select_action(obs_noisy)

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

    residual_net = None
    L = args.L
    te_coeff = args.te_coeff
    fresh_index = args.fresh_index
    alpha_residual = args.alpha_residual

    if args.mode == "residual":
        if args.weight_net_path is None:
            raise ValueError("--weight_net_path is required when --mode residual")

        weight_net_path = Path(args.weight_net_path)
        print(f"Loading action residual network from {weight_net_path}")
        ckpt = torch.load(weight_net_path, map_location=DEVICE)

        L = int(ckpt["L"])
        te_coeff = float(ckpt.get("te_coeff", args.te_coeff))
        fresh_index = ckpt.get("fresh_index", args.fresh_index)
        alpha_residual = float(ckpt.get("alpha_residual", args.alpha_residual))
        max_delta = float(ckpt.get("max_delta", args.max_delta))

        residual_net = ActionResidualNetwork(
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            d_model=ckpt.get("d_model", 128),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 2),
            max_delta=max_delta,
        ).to(DEVICE)

        state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
        residual_net.load_state_dict(state_dict)
        residual_net.eval()

        print(
            f"Loaded TE-fixed action residual net: "
            f"L={L}, te_coeff={te_coeff}, fresh_index={fresh_index}, "
            f"alpha_residual={alpha_residual}, max_delta={max_delta}, "
            f"lambda_residual={ckpt.get('lambda_residual')}, "
            f"base={ckpt.get('base_policy')}, "
            f"trained_noise_std={ckpt.get('noise_std')}, "
            f"trained_corrupt_prob={ckpt.get('corrupt_prob')}"
        )
    else:
        print(f"Running pure TE wrapper with L={L}")

    test_weights = make_fixed_te_weights(
        L=L,
        te_coeff=te_coeff,
        device=DEVICE,
        fresh_index="first",
    )
    print("\n[TE DEBUG]")
    print("Runtime contrib convention: contribs[:, 0] = freshest")
    print("te_coeff:", te_coeff)
    print("te_weights first 5:", test_weights[:5].detach().cpu().numpy())
    print("te_weights last 5:", test_weights[-5:].detach().cpu().numpy())
    print("te_weights argmax:", int(test_weights.argmax().item()))

    policy = EnsemblePolicy(
        baseline_policy=baseline,
        residual_net=residual_net,
        L=L,
        mode=args.mode,
        te_coeff=te_coeff,
        fresh_index=fresh_index,
        alpha_residual=alpha_residual,
    ).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    print("\nEval config:")
    print(f"  mode={args.mode}")
    print(f"  eval_episodes={args.eval_episodes}")
    print(f"  eval_batch_size={args.eval_batch_size}")
    print(f"  L={L}")
    print(f"  te_coeff={te_coeff}")
    print(f"  fresh_index={fresh_index}")
    print(f"  alpha_residual={alpha_residual}")
    print(f"  state_noise_std={args.state_noise_std}")
    print(f"  state_noise_ratio={args.state_noise_ratio}")
    print(f"  image_noise_std={args.image_noise_std}")
    print(f"  image_noise_prob={args.image_noise_prob}")
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

    print("\nResults ACT TE-fixed action residual eval")
    print(f"mode = {args.mode}")
    print(f"state_noise_std = {args.state_noise_std}")
    print(f"state_noise_ratio = {args.state_noise_ratio}")
    print(f"image_noise_std = {args.image_noise_std}")
    print(f"image_noise_prob = {args.image_noise_prob}")
    print(f"pc_success = {pc_success:.3f} ({all_successes.sum()}/{len(all_successes)})")

    print("\nReference:")
    print("  ACT baseline n=50 commit:        62%")
    print("  LeRobot TE fresh-favored:        54%")
    print("  our wrapper pure TE no network:  58%")


if __name__ == "__main__":
    main()
